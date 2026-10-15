"""Build ("evolve") a deep network whose untrained layers preserve dimensionality and task information.

    python reservoir_search.py OUT_DIR [--size small|full] [--layers 5] [--jobs 4] [--target-hz 20]

Fast CPU evaluation without the Snake training loop:
1. Record input frames once: Snake played by a random policy (no waiting between moves), each frame stored as the
   Poisson rates the network sees plus the task variables (food relative to the head, danger per direction, heading).
2. Per candidate network (snake_network spec): compile it with the inference compiler (fixed weights, no learning).
3. Calibrate layer by layer (LSUV-style, non-local pre-processing): run a short stretch of frames, measure each
   layer's E rate and rescale the excitatory feed-forward weights into that layer until it fires near --target-hz,
   then move on to the next layer; finally the fields.
4. Evaluate: for every frame, 30 simulation steps (one decision), then record each hidden population's membrane
   potentials and spike counts. Per population: effective dimensionality (participation ratio), decoding of the task
   variables above chance (linear probe, as probe_analyse.py), and rate.
   Also: input reconstruction (variance of the input frame a linear readout of the population explains, held out).
5. Perturbation reach (what the perturbation rule needs): for each layer k, every frame adds a known random pattern
   xi_f (amplitude --pert-amp per step) to the E membrane potentials of layer k; the score is the share of the
   fields' spike-count variance a linear readout of xi explains (held out). A good network gives similar values for
   every k: perturbations neither vanish nor explode on their way to the output.
Score: how well the last layer keeps layer 1's dimensionality and the input's information, and how evenly
perturbations from each layer reach the fields.
"""
import argparse
import ast
import contextlib
import io
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STEPS_PER_MOVE = 30
INPUT_RATE_SCALE = 0.7


# ---------------------------------------------------------------- frames
def snake_env_class():
    """SnakeEnv from snake.py without running the training script (the module runs on import)."""
    import cv2
    from scipy.ndimage import gaussian_filter
    tree = ast.parse((HERE / "snake.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SnakeEnv")
    ns = {"np": np, "cv2": cv2, "random": random, "gaussian_filter": gaussian_filter}
    exec(compile(ast.Module([node], []), "snake_env", "exec"), ns)
    return ns["SnakeEnv"]


DIRS = {"left": (0, -1), "up": (-1, 0), "right": (0, 1), "down": (1, 0)}


def task_variables(env):
    hy, hx = env.snake[0]
    ay, ax = env.apples[0] if env.apples else (hy, hx)
    danger = [float(not (0 <= hy + dy < env.size and 0 <= hx + dx < env.size) or (hy + dy, hx + dx) in env.snake[:-1])
              for dy, dx in DIRS.values()]
    return np.array([ay - hy, ax - hx, *danger, list(DIRS).index(env.direction)], dtype=np.float32)


def record_frames(n, seed=0, board_size=5):
    SnakeEnv = snake_env_class()
    random.seed(seed); np.random.seed(seed)
    env = SnakeEnv(size=board_size, visible_range=5, wait_inc=0, scale=4, inp_shape=(20, 20, 3),
                   bounded_camera=False)
    env.reset()
    rates, ys = [], []
    while len(rates) < n:
        obs = env.get_local_img_observation()
        rates.append((np.clip(obs, 0.0, 1.0) * INPUT_RATE_SCALE).astype(np.float32).ravel())
        ys.append(task_variables(env))
        # random policy that avoids immediate death when it can
        safe = [a for a, (d, (dy, dx)) in enumerate(DIRS.items())
                if 0 <= env.snake[0][0] + dy < env.size and 0 <= env.snake[0][1] + dx < env.size
                and (env.snake[0][0] + dy, env.snake[0][1] + dx) not in env.snake[:-1]]
        _, _, done = env.step(random.choice(safe) if safe and random.random() < 0.9 else random.randrange(4))
        if done:
            env.reset()
    return np.array(rates), np.array(ys)


# ---------------------------------------------------------------- network
def build(hp_over, size, layers):
    import snake_hparams
    import snake_network
    from ml_genn.compilers.eprop_compiler import default_params
    hp = {**snake_hparams.DEFAULTS, "ei_layers": layers, "explicit_feedback": False}
    if size == "small":
        hp.update(hid_e=6, hid_i=4, fan_in=50)
    hp.update(hp_over)
    np.random.seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        return snake_network.build_network(snake_network.spec_from_hparams(hp, (20, 20, 3), 4), default_params)


class Sim:
    def __init__(self, built, name, workdir, backend="single_threaded_cpu"):
        """backend: "single_threaded_cpu" or "cuda". Spikes are counted on the device (GeNN spike recording, one
        buffer per decision of STEPS_PER_MOVE steps, read once per frame), so the GPU runs a whole decision
        without host transfers."""
        from ml_genn.compilers import InferenceCompiler
        self.b = built
        for p in built.hidden:
            p.record_spikes = True
        os.chdir(workdir)
        kw = {"backend": backend}
        if backend == "single_threaded_cpu":
            kw["optimize_code"] = True
        with contextlib.redirect_stdout(io.StringIO()):
            self.net = InferenceCompiler(evaluate_timesteps=STEPS_PER_MOVE, reset_time_between_batches=False,
                                         reset_vars_between_batches=False, **kw).compile(built.network, name)
        self.pops = built.hidden
        self.labels = [built.labels[p] for p in self.pops]
        self.factors = {}           # calibration: cumulative weight factor per group (layer k, or "fields")

    def __enter__(self):
        self.net.__enter__(); return self

    def __exit__(self, *a):
        self.net.__exit__(*a)

    def run(self, rates, record=False, perturb=None):
        """Present each frame for one decision. Returns per-population spike counts per frame (and V, refractory).
        perturb: (population, amplitude, patterns [frames x neurons]): V += amplitude * pattern every step."""
        npops = [self.net.neuron_populations[p] for p in self.pops]
        tau = [float(p.neuron.tau_refrac) for p in self.pops]
        counts = [np.zeros((len(rates), int(np.prod(p.shape)))) for p in self.pops]
        states = []
        self.psi = []
        self.adapt = []                                  # per frame: the adaptation variable A of every population
        ppop = None
        if perturb is not None:
            ppop = self.net.neuron_populations[perturb[0]]
        model = self.net.genn_model
        for f, r in enumerate(rates):
            self.net.set_input({self.b.input: r.reshape(self.b.input.shape)})
            for _ in range(STEPS_PER_MOVE):
                if ppop is not None:                     # optional perturbation measure: per-step host access
                    ppop.vars["V"].pull_from_device()
                    ppop.vars["V"].view[:] += (perturb[1] * perturb[2][f]).reshape(ppop.vars["V"].view.shape)
                    ppop.vars["V"].push_to_device()
                self.net.step_time()
            # the recording buffer holds exactly one decision (STEPS_PER_MOVE steps): one transfer per frame
            model.pull_recording_buffers_from_device()
            for i, npop in enumerate(npops):
                _, ids = npop.spike_recording_data[0]
                counts[i][f] = np.bincount(np.asarray(ids, dtype=np.int64), minlength=counts[i].shape[1])
                npop.vars["RefracTime"].pull_from_device() if record else None
            if record:
                row, psi_row, a_row = [], [], []
                for p, npop in zip(self.pops, npops):
                    npop.vars["V"].pull_from_device(); npop.vars["A"].pull_from_device()
                    v = np.asarray(npop.vars["V"].view).ravel().copy()
                    ref = np.asarray(npop.vars["RefracTime"].view).ravel()
                    row += [v, (ref > 0).astype(np.float32)]
                    psi_row.append(pseudo_derivative(v, np.asarray(npop.vars["A"].view).ravel(), p.neuron.beta,
                                                     p.neuron.v_thresh, ref))
                    a_row.append(np.asarray(npop.vars["A"].view).ravel().copy())
                states.append(np.concatenate(row))
                self.psi.append(psi_row)
                self.adapt.append(a_row)
        return counts, (np.array(states) if record else None)

    def hz(self, counts):
        return [1000.0 * c.mean() / STEPS_PER_MOVE for c in counts]

    def scale_into(self, targets, factor, key=None):
        """Multiply the excitatory feed-forward weights into the target populations by factor."""
        for c, sg in self.net.connection_populations.items():
            if c.target() in targets and c.source() not in targets and getattr(c, "exc_inh_sign", None) == 1 \
                    and not c.is_feedback:
                sg.vars["g"].pull_from_device()
                sg.vars["g"].values = sg.vars["g"].values * factor
                sg.vars["g"].push_to_device()
        if key is not None:
            self.factors[key] = self.factors.get(key, 1.0) * factor


def calibrate(sim, rates, target_hz, moves=120, rounds=5, log=print):
    """Layer by layer (then the fields): rescale the excitatory input until the layer's E rate is near target."""
    groups = [list(l) for l in sim.b.layers] + ([list(sim.b.fields)] if sim.b.fields else [])
    for k, group in enumerate(groups):
        idx = sim.pops.index(group[0])
        for r in range(rounds):
            start = (k * rounds + r) * moves % max(1, len(rates) - moves)
            counts, _ = sim.run(rates[start:start + moves])
            hz = sim.hz(counts)[idx]
            if abs(np.log(max(hz, 1e-3) / target_hz)) < 0.15:
                break
            factor = float(np.clip((target_hz / max(hz, 0.5)) ** 0.5, 0.5, 3.0))
            sim.scale_into(set(group), factor, key=("fields" if k == len(sim.b.layers) else k + 1))
            log(f"    calibrate {sim.labels[idx]}: {hz:6.1f} Hz -> x{factor:.2f}")


def ridge_r2(x, y, ridge=1.0, cut=0.7, max_features=100):
    """Share of y's variance a ridge regression from x explains on the held-out last (1 - cut) of the samples.
    Large populations are first reduced to their top max_features principal components (fit on the training part),
    so the readout cannot overfit with more features than frames."""
    n = len(x); k = int(cut * n)
    if x.shape[1] > max_features:
        xc = x - x[:k].mean(0)
        _, _, vt = np.linalg.svd(xc[:k], full_matrices=False)
        x = xc @ vt[:max_features].T
    mu, sd = x[:k].mean(0), x[:k].std(0) + 1e-6
    xs = np.hstack([(x - mu) / sd, np.ones((n, 1))])
    ym = y[:k].mean(0)
    w = np.linalg.solve(xs[:k].T @ xs[:k] + ridge * np.eye(xs.shape[1]), xs[:k].T @ (y[:k] - ym))
    resid = y[k:] - ym - xs[k:] @ w
    total = ((y[k:] - y[k:].mean(0)) ** 2).sum()
    return float(1.0 - (resid ** 2).sum() / max(total, 1e-12))


MEMORY_LAGS = (1, 2, 3)


def _pcs(x, k, n=100):
    """Top-n principal components of x, fit on its first k rows."""
    xc = x - x[:k].mean(0)
    _, _, vt = np.linalg.svd(xc[:k], full_matrices=False)
    return xc @ vt[:n].T


def memory_r2(state, rates, lags=MEMORY_LAGS, ridge=10.0, cut=0.7):
    """Memory of a population (state: its V and adaptation A per frame; with tau_mem 10 ms and 30 ms frames, V and the
    spikes forget the previous frame and what survives is in A), per lag k (frames): how much of the input k frames ago its state recovers BEYOND what
    the current frame already says about it. Consecutive Snake frames are similar, so the past frame is first
    regressed on the current one (ridge on the top-100 input PCs, fit on the training part) and only the residual is
    the target; the score is the held-out share of that residual's variance a readout of the state explains
    (ridge_r2). ~0: the population only represents the present frame."""
    z = _pcs(rates, int(cut * len(rates)))
    out = []
    for lag in lags:
        now, past, st = z[lag:], z[:-lag], state[lag:]
        k = int(cut * len(now))
        xs = np.hstack([now, np.ones((len(now), 1))])
        w = np.linalg.solve(xs[:k].T @ xs[:k] + ridge * np.eye(xs.shape[1]), xs[:k].T @ past[:k])
        out.append(max(ridge_r2(st, past - xs @ w, ridge=ridge, cut=cut), 0.0))
    return out


def perturbation_reach(sim, rates, amp, frames_per_layer=1200, n_dirs=4, folds=5, seed=0):
    """Per layer k: how detectably a perturbation of layer k shows up at the fields, measured the way the
    perturbation rule sees it. Each frame, layer k's E potentials get amp * sum_j s_fj * u_j per step (u_j fixed
    random spatial patterns, unit s.d. per neuron; s_fj independent random scalars). For each direction j the
    field-response direction d_j = mean s_fj * centred counts is estimated on the training folds; the score is the
    held-out correlation between s_fj and the counts projected on d_j, averaged over directions and folds.
    ~0: the perturbation is lost on the way to the output; larger: it reaches it."""
    if not sim.b.fields:
        return {}
    rng = np.random.default_rng(seed)
    out = {}
    field_idx = [sim.pops.index(p) for p in sim.b.fields]
    for k, (e, _) in enumerate(sim.b.layers, 1):
        n = int(np.prod(e.shape))
        u = rng.standard_normal((n_dirs, n)).astype(np.float32)
        s_f = rng.standard_normal((frames_per_layer, n_dirs)).astype(np.float32)
        start = (k - 1) * frames_per_layer % max(1, len(rates) - frames_per_layer)
        frames = rates[start:start + frames_per_layer]
        if len(frames) < frames_per_layer:                       # wrap around the recorded frames
            frames = np.concatenate([frames, rates[:frames_per_layer - len(frames)]])
        counts, _ = sim.run(frames, perturb=(e, amp, s_f @ u))
        y = np.hstack([counts[i] for i in field_idx])
        idx = np.arange(len(y))
        scores = []
        for fold in range(folds):
            test = idx % folds == fold
            yc = y - y[~test].mean(0)
            for j in range(n_dirs):
                d = (s_f[~test, j:j + 1] * yc[~test]).mean(0)
                proj = yc[test] @ d
                if proj.std() > 0:
                    scores.append(np.corrcoef(s_f[test, j], proj)[0, 1])
        out[f"L{k}"] = float(np.mean(scores)) if scores else 0.0
    return out


def pseudo_derivative(v, a, beta, v_thresh, refrac):
    """psi of the ALIF rule (as snake_switch.pseudo_derivative): how close to threshold, zero when refractory."""
    psi = (1.0 / v_thresh) * 0.3 * np.maximum(0.0, 1.0 - np.abs((v - (v_thresh + beta * a)) / v_thresh))
    return np.where(refrac > 0, 0.0, psi)


def rf_spread(counts, rates, shape, in_hw=(20, 20), in_c=3):
    """Median receptive-field spread (input cells) of a population: per neuron, the correlation of its spike counts
    with each input pixel (channels summed) across frames; only |r| > 3 / sqrt(frames) counts; spread = RMS toroidal
    distance from the neuron's own position (mapped onto the input grid), weighted by |r|."""
    h, w = shape[0], shape[1]
    c = shape[2] if len(shape) == 3 else 1
    pix = rates.reshape(len(rates), in_hw[0], in_hw[1], in_c).sum(-1).reshape(len(rates), -1)
    pz = (pix - pix.mean(0)) / (pix.std(0) + 1e-9)
    cz = (counts - counts.mean(0)) / (counts.std(0) + 1e-9)
    corr = np.abs(cz.T @ pz) / len(rates)                       # neurons x pixels
    corr[corr < 3.0 / np.sqrt(len(rates))] = 0.0
    py, px = np.divmod(np.arange(in_hw[0] * in_hw[1]), in_hw[1])
    cells = np.arange(counts.shape[1]) // c
    ny = (cells // w + 0.5) / h * in_hw[0] - 0.5
    nx = (cells % w + 0.5) / w * in_hw[1] - 0.5
    dy = np.abs(py[None, :] - ny[:, None]); dy = np.minimum(dy, in_hw[0] - dy)
    dx = np.abs(px[None, :] - nx[:, None]); dx = np.minimum(dx, in_hw[1] - dx)
    wsum = corr.sum(1)
    ok = wsum > 0
    if not ok.any():
        return float("nan")
    spread = np.sqrt((corr * (dy ** 2 + dx ** 2)).sum(1)[ok] / wsum[ok])
    return float(np.median(spread))


def evaluate(sim, rates, ys):
    from probe_analyse import per_population
    from layer_dims import participation_ratio
    counts, states = sim.run(rates, record=True)
    sizes = [int(np.prod(p.shape)) for p in sim.pops]
    decode = per_population(states, ys, sizes, sim.labels)
    psi = [np.array([row[i] for row in sim.psi]) for i in range(len(sim.pops))]     # frames x neurons per pop
    adapt = [np.array([row[i] for row in sim.adapt]) for i in range(len(sim.pops))]   # adaptation A, same layout
    prev_e = None
    out, off = [], 0
    for lab, n, c, ps in zip(sim.labels, sizes, counts, psi):
        v = states[:, off:off + n]; off += 2 * n
        x = states[:, off - 2 * n:off]
        out.append({"pop": lab, "n": n, "hz": 1000.0 * c.mean() / STEPS_PER_MOVE,
                    "dim": participation_ratio(v), "dim_counts": participation_ratio(c),
                    "decode": decode[lab], "active": float(np.mean(c.sum(0) > 0)),
                    "input_r2": ridge_r2(x, rates, ridge=10.0),
                    # learnability for the rule: mean psi, and the fraction of neurons almost never near threshold
                    "psi": float(ps.mean()), "psi_dead": float(np.mean(ps.mean(0) < 0.01)),
                    # mixing: receptive-field spread in input cells, and (E of layers >= 2, fields) how much of the layer's
                    # spike counts a linear map from the previous layer's E counts explains (1 = a relay)
                    "rf_spread": rf_spread(c, rates, tuple(sim.pops[sim.labels.index(lab)].shape)),
                    "pred_from_prev": (ridge_r2(prev_e, c, ridge=10.0) if prev_e is not None
                                       and (lab.endswith("_E") or "field" in lab) else float("nan")),
                    # memory: the input 1, 2, 3 frames ago recovered beyond what the current frame predicts
                    "memory": (memory_r2(np.hstack([v, adapt[len(out)]]), rates)
                               if lab.endswith("_E") or "field" in lab else None)})
        if lab.endswith("_E"):
            prev_e = c
    return out


# ---------------------------------------------------------------- one candidate
def run_candidate(args):
    name, hp_over, size, layers, target_hz, calib, frames_path, out_dir, pert_amp = args
    work = Path(out_dir) / name
    work.mkdir(parents=True, exist_ok=True)
    lines = []
    log = lambda s: lines.append(s)
    d = np.load(frames_path)
    rates, ys = d["rates"], d["ys"]
    calib_rates, eval_rates, eval_ys = rates[:1500], rates[1500:], ys[1500:]
    t0 = time.time()
    built = build(hp_over, size, layers)
    with Sim(built, name.replace("-", "_").replace(".", "p"), work) as sim:
        if calib:
            calibrate(sim, calib_rates, target_hz, log=log)
        result = evaluate(sim, eval_rates, eval_ys)
        reach = perturbation_reach(sim, eval_rates, pert_amp)
    e = [r for r in result if r["pop"].endswith("_E")]
    r_vals = [v for v in reach.values()]
    score = {"dim_ratio": e[-1]["dim"] / max(e[0]["dim"], 1e-9), "decode_last": e[-1]["decode"],
             "decode_ratio": e[-1]["decode"] / max(e[0]["decode"], 1e-9),
             "input_r2_ratio": e[-1]["input_r2"] / max(e[0]["input_r2"], 1e-9),
             "reach_min_max": (max(min(r_vals), 0.0) / max(max(r_vals), 1e-9)) if r_vals else float("nan")}
    res = {"name": name, "hp": hp_over, "calibrated": calib, "per_pop": result, "reach": reach, "score": score,
           "calibration_factors": {str(k): v for k, v in sim.factors.items()},
           "snake_config": snake_config(hp_over, sim.factors, layers, size),
           "seconds": round(time.time() - t0, 1), "log": lines}
    (work / "result.json").write_text(json.dumps(res, indent=1))
    return res


def snake_config(hp_over, factors, layers, size):
    """Snake config keys that reproduce this candidate, with the calibration folded into the weight scales: the
    calibration multiplied every weight of the excitatory feed-forward connections into a layer, i.e. their
    mean_scale and sd_scale (input -> L1: the input ConnSpec; L(k-1) -> Lk: layer k-1's to_next; last layer ->
    fields: the fields' conn)."""
    import copy
    import snake_hparams
    import snake_network
    hp = {**snake_hparams.DEFAULTS, "ei_layers": layers}
    if size == "small":
        hp.update(hid_e=6, hid_i=4, fan_in=50)
    hp.update(hp_over)
    spec = snake_network.spec_from_hparams(hp, (20, 20, 3), 4)
    cfg = copy.deepcopy(hp_over)
    cfg["ei_layers"] = layers
    if size == "small":
        cfg.update(hid_e=6, hid_i=4, fan_in=50)
    net = cfg.setdefault("network", {})
    lay = net.get("layers") or [{} for _ in range(layers)]
    lay = [dict(l) for l in lay] + [{} for _ in range(layers - len(lay))]
    f1 = factors.get(1, 1.0)
    if f1 != 1.0:
        net["input"] = {**net.get("input", {}), "mean_scale": spec.input.mean_scale * f1,
                        "sd_scale": spec.input.sd_scale * f1}
    # the calibration scaled only the excitatory projections; with feed-forward inhibition (to_next_i / conn_i None:
    # the I projection shares the E settings) the I projection is written out with its own, unscaled weights
    for k in range(2, layers + 1):
        f = factors.get(k, 1.0)
        if f != 1.0:
            ls = spec.layers[k - 2]
            src, own = ls.to_next, dict(lay[k - 2].get("to_next", {}))
            if ls.to_next_i is None:
                lay[k - 2]["to_next_i"] = {**own, "mean_scale": src.mean_scale, "sd_scale": src.sd_scale}
            lay[k - 2]["to_next"] = {**own, "mean_scale": src.mean_scale * f, "sd_scale": src.sd_scale * f}
    net["layers"] = lay
    if factors.get("fields", 1.0) != 1.0 and spec.fields is not None:
        f = factors["fields"]
        fd = dict(net.get("fields", {}))
        own = dict(fd.get("conn", {}))
        if spec.fields.conn_i is None:
            fd["conn_i"] = {**own, "mean_scale": spec.fields.conn.mean_scale, "sd_scale": spec.fields.conn.sd_scale}
        fd["conn"] = {**own, "mean_scale": spec.fields.conn.mean_scale * f, "sd_scale": spec.fields.conn.sd_scale * f}
        net["fields"] = fd
    return cfg


def table(res):
    sc = res["score"]
    s = [f"\n{res['name']}  (calibrated: {res['calibrated']}, {res['seconds']:.0f} s)  "
         f"L_last/L1: dim {sc['dim_ratio']:.2f}, input R2 {sc['input_r2_ratio']:.2f}; "
         f"perturbation reach min/max {sc['reach_min_max']:.2f}",
         f"  {'pop':14s} {'n':>5s} {'Hz':>6s} {'eff.dim':>8s} {'dim(cnt)':>9s} {'inputR2':>8s} {'decode':>7s} "
         f"{'psi':>6s} {'psi~0':>6s} {'RF':>5s} {'R2prev':>6s}"]
    for r in res["per_pop"]:
        s.append(f"  {r['pop']:14s} {r['n']:5d} {r['hz']:6.1f} {r['dim']:8.2f} {r['dim_counts']:9.2f} "
                 f"{r['input_r2']:8.3f} {r['decode']:+7.3f} {r['psi']:6.3f} {r['psi_dead']:6.2f} "
                 f"{r.get('rf_spread', float('nan')):5.2f} {r.get('pred_from_prev', float('nan')):6.3f}")
    s.append("  perturbation reach (held-out corr. of an injected L_k perturbation with the fields): "
             + "  ".join(f"{k} {v:.3f}" for k, v in res["reach"].items()))
    return "\n".join(s)


SMALL_LOCAL = {"toroidal": "local", "local": {"sigma_cells": 1.0, "input_sigma_ref": "source"}}


def candidates_sweep2(layers):
    """Around the narrow feed-forward winner; recurrence strength is kept (only its E/I balance varies)."""
    def net(ff_sigma=0.6, field_sigma=None, field_shape=None, inh=None):
        d = {**SMALL_LOCAL, "network": {"layers": [{"to_next": {"sigma_cells": ff_sigma}}] * (layers - 1) + [{}]}}
        fields = {}
        if field_sigma is not None:
            fields["conn"] = {"sigma_cells": field_sigma}
        if field_shape is not None:
            fields["shape"] = field_shape
        if fields:
            d["network"]["fields"] = fields
        if inh is not None:
            d["local"] = {**SMALL_LOCAL["local"], "inhibition_scale": inh}
        return d
    c = {f"ff{s:g}": net(ff_sigma=s) for s in (0.3, 0.45, 0.6, 0.8, 1.0)}
    c["ff0.6_fieldnarrow"] = net(field_sigma=0.6)
    c["ff0.6_field6"] = net(field_shape=6)
    c["ff0.6_field6_narrow"] = net(field_shape=6, field_sigma=0.6)
    c["ff0.6_inh2"] = net(inh=2.0)
    c["ff0.6_inh4"] = net(inh=4.0)
    c["ff0.45_field6_narrow"] = net(ff_sigma=0.45, field_shape=6, field_sigma=0.45)
    c["ff1_field6_narrow"] = net(ff_sigma=1.0, field_shape=6, field_sigma=1.0)     # the default ff width
    return c


def candidates_full(layers):
    """Full size (20x20 E, 15x15 I): the default local sigma is 1.75 cells; narrower feed-forward and fields."""
    def net(ff_sigma=None, field_sigma=None):
        d = {"toroidal": "local"}
        if ff_sigma is not None:
            d["network"] = {"layers": [{"to_next": {"sigma_cells": ff_sigma}}] * (layers - 1) + [{}]}
        if field_sigma is not None:
            d.setdefault("network", {})["fields"] = {"conn": {"sigma_cells": field_sigma}}
        return d
    c = {"full_default": net()}
    for s in (0.45, 0.8, 1.2):
        c[f"full_ff{s:g}_fieldnarrow"] = net(ff_sigma=s, field_sigma=s)
    return c


def candidates_full1(layers):
    """One full-size layer: the user's current "matched" connectivity and legacy against principled local variants
    (neighbourhood width, density, inhibition balance, fields' size and projection)."""
    loc = lambda **kw: {"toroidal": "local", "local": kw} if kw else {"toroidal": "local"}
    fields = lambda d, shape=None, sigma=None: {**d, "network": {"fields": {
        **({"shape": shape} if shape else {}), **({"conn": {"sigma_cells": sigma}} if sigma else {})}}}
    return {
        "matched": {"toroidal": "matched"},
        "legacy": {"toroidal": "legacy"},
        "local": loc(),
        "local_sig1.25": loc(sigma_cells=1.25),
        "local_sig2.5": loc(sigma_cells=2.5),
        "local_sparse": loc(sigma_cells=2.5, p_max=0.5),
        "local_inh4": loc(inhibition_scale=4.0),
        "local_field20_narrow": fields(loc(), shape=20, sigma=0.6),
        "local_field15_narrow": fields(loc(), sigma=0.6),
    }


def candidates(layers):
    ff = lambda **kw: {"network": {"layers": [{"to_next": kw}] * (layers - 1) + [{}]}}
    c = {
        "legacy": {"toroidal": "legacy"},
        "local": SMALL_LOCAL,
        "local_sig1.75": {"toroidal": "local", "local": {"sigma_cells": 1.75, "input_sigma_ref": "source"}},
        "local_ff_sparse": {**SMALL_LOCAL, **ff(p_max=0.3, sigma_cells=1.75)},
        "local_ff_narrow": {**SMALL_LOCAL, **ff(sigma_cells=0.6)},
        "local_inh2": {"toroidal": "local", "local": {"sigma_cells": 1.0, "input_sigma_ref": "source",
                                                      "inhibition_scale": 2.0}},
    }
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--size", default="small", choices=["small", "full"])
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--target-hz", type=float, default=20.0)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--only", nargs="*", help="candidate names to run (default: all)")
    ap.add_argument("--pert-amp", type=float, default=0.03, help="perturbation added to V per step (x pattern)")
    ap.add_argument("--set", default="base", choices=["base", "sweep2", "full", "full1"], help="candidate set")
    ap.add_argument("--no-uncalibrated", action="store_true", help="skip the uncalibrated reference of each candidate")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    frames_path = out / "frames.npz"
    if not frames_path.exists():
        rates, ys = record_frames(args.frames)
        np.savez(frames_path, rates=rates, ys=ys)
        print(f"recorded {len(rates)} frames")
    cands = {"base": candidates, "sweep2": candidates_sweep2, "full": candidates_full,
             "full1": candidates_full1}[args.set](args.layers)
    names = args.only or list(cands)
    jobs = []
    for n in names:
        jobs.append((f"{n}-cal", cands[n], args.size, args.layers, args.target_hz, True, str(frames_path), str(out),
                     args.pert_amp))
        if not args.no_uncalibrated:
            jobs.append((f"{n}-raw", cands[n], args.size, args.layers, args.target_hz, False, str(frames_path),
                         str(out), args.pert_amp))
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        for res in pool.imap_unordered(run_candidate, jobs):
            print(table(res), flush=True)


if __name__ == "__main__":
    main()
