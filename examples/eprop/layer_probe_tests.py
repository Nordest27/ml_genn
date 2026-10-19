"""Do the layers and the recurrence do anything useful? Four tests on probe recordings (outputs/*_probe.npz, recorded
with the "probe" config option; needs the input frames the extended ProbeRecorder stores).

    python layer_probe_tests.py RUN_DIR_or_probe.npz [...] [--window first|last|both]

Alignment: a probe sample is taken one step after the next frame was presented, so the state at sample t is paired
with the frame and task variables of sample t-1 (same episode only).

1. Task decoding (does a layer compute task variables?): linear decoding of danger in 4 directions and the apple's
   side (y, x) from each population, as BALANCED accuracy above chance (mean recall over classes - 1 / classes, so a
   constant guess scores 0 however rare danger is), next to the same decoder on the raw input frame.
   Danger and the apple's side are non-linear functions of the frame: a layer decoding them better than the input
   (and better than layer 1) has computed something. Every feature set is reduced to its top 100 principal
   components first (fit on the training part), so populations of different sizes and the input compete fairly.
   Next to it, a NON-LINEAR decoder (random Fourier features of the top 50 components: an RBF-kernel ridge readout):
   information a layer still holds in a non-linear form shows up there even when the linear decoder misses it.
   Linear low and non-linear high: the layer TRANSFORMED the information; both low: it was LOST.
2. Transformation (copy or compute?): R2 of each layer's membrane potentials from the previous layer's E potentials.
   ~1: the layer is (linearly) a copy of the one below; training that lowers it made the layer transform.
3. Spatial transport (does information travel sideways?): small patches of each E population (and the fields)
   decode input pixels OUTSIDE the patch's feed-forward receptive field: the input cells reachable from the patch's
   neurons through feed-forward synapses only (input -> L1 -> ... , recurrence excluded), traced on the network
   rebuilt from the run's config (same statistics; the exact synapses are resampled, so the field is dilated by
   one cell to be conservative). The frames are very structured, so the input inside the field already predicts most
   outside pixels; the test therefore decodes the RESIDUAL of each outside pixel (what the inside input cannot
   predict) from the patch, and from the patch's mean potential alone (a global signal). A residual R2 above zero
   is information that reached the patch sideways, through recurrence.
4. Memory (is recurrence used over time?): how much of the input 1-3 frames back each population recovers beyond what
   the current frame already says (reservoir_search.memory_r2; with adaptation A when recorded), and decoding of the
   previous heading against the current frame as baseline.

5. What does the last layer represent? (needs the body / apple the recorder stores): a battery of variables, local to
   global, decoded linearly from the input and from each layer's E population and the fields: danger, trap and safety
   per direction (3 classes: lethal / trap = non-lethal but the tail becomes unreachable / safe; balanced accuracy over
   the non-lethal moves only, i.e. trap vs safe), free area reachable after each move, path distance to the apple
   around the body, snake length, tail position (R2). Traps are not linearly decodable from the frame (a global,
   non-linear property): a layer that decodes them has computed it. Plus an unsupervised view: the last population's
   top principal components, each with the variable it correlates with most.

6. Is the input-to-fields transformation linear? Each population's dominant activity (top 20 principal components of
   its potentials) predicted from the INPUT by (a) a linear map of the current frame, (b) a linear map of the input
   history (current frame, the 3 frames before, and an exponentially filtered history, time constant ~10 frames like
   the adaptation), (c) a non-linear map (random features) of the same history. (b) - (a): linear integration over
   time; (c) - (b): the non-linear part of the transformation; what (c) leaves: noise and input-independent dynamics.

Compare windows (first vs last moves: before vs after learning) and runs (rule vs reservoir; with lesions, later).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_analyse import probe, targets          # noqa: E402
from reservoir_search import memory_r2, ridge_r2  # noqa: E402

DECODE_KEYS = ("danger_l", "danger_u", "danger_r", "danger_d", "apple_y", "apple_x")
INPUT_HW, INPUT_C = (20, 20), 3


def pcs(x, cut, k=100):
    """Top-k principal components of x (fit on the first cut rows); x unchanged if it has at most k columns."""
    if x.shape[1] <= k:
        return x
    xc = x - x[:cut].mean(0)
    _, sv, vt = np.linalg.svd(xc[:cut], full_matrices=False)
    k = min(k, int(np.sum(sv > 1e-3 * sv[0])))           # drop degenerate components (near-constant inputs)
    return xc @ vt[:k].T


def load(path, window):
    d = np.load(path)
    if f"{window}_inp" not in d.files or d[f"{window}_inp"].size == 0:
        raise SystemExit(f"{path}: no input frames recorded (needs the extended ProbeRecorder; rerun with probe on)")
    x, y, extra, inp = (d[f"{window}_{k}"] for k in ("x", "y", "extra", "inp"))
    a = d[f"{window}_a"].astype(np.float32) if f"{window}_a" in d.files else None
    ep = extra[:, list(d["extra_names"]).index("episode")]
    t = np.flatnonzero(ep[1:] == ep[:-1]) + 1                     # samples whose previous sample is the same episode
    body = d[f"{window}_body"][t - 1] if f"{window}_body" in d.files and d[f"{window}_body"].size else None
    apple = d[f"{window}_apple"][t - 1] if body is not None else None
    run = {"body": body, "apple": apple, "board": int(d["board"]) if "board" in d.files else 0,
           "state": x[t], "adapt": None if a is None else a[t], "y": y[t - 1], "extra": extra[t - 1],
           "inp_all": inp, "t_index": t, "ep_all": ep,
           "frame": inp[t - 1].astype(np.float32) / 255.0, "sizes": [int(n) for n in d["sizes"]],
           "shapes": [tuple(int(v) for v in s) for s in d["shapes"]], "labels": [str(l) for l in d["labels"]],
           "extra_names": list(d["extra_names"]), "episode": ep[t]}
    return run


def blocks(run):
    """Per population: label, shape, V block, [V, refractory] block, adaptation block (or None)."""
    out, off, aoff = [], 0, 0
    for lab, n, shp in zip(run["labels"], run["sizes"], run["shapes"]):
        st = run["state"][:, off:off + 2 * n]; off += 2 * n
        a = None if run["adapt"] is None else run["adapt"][:, aoff:aoff + n]
        aoff += n
        out.append((lab, shp, st[:, :n], st, a))
    return out


def balanced_probe(x, labels, ridge=1.0):
    """Linear least-squares one-vs-rest classifier (as probe_analyse.probe), scored by BALANCED accuracy (mean recall
    over the classes present in the test part) minus its chance level 1 / classes: a constant guess scores 0, however
    unbalanced or time-shifted the classes are (danger is rare; plain accuracy rewards guessing 'no danger')."""
    n = len(labels)
    cut = int(0.7 * n)
    if cut < 50 or n - cut < 20:
        return float("nan")
    mu, sd = x[:cut].mean(0), x[:cut].std(0) + 1e-6
    xs = np.hstack([(x - mu) / sd, np.ones((n, 1))])
    classes = np.unique(labels[:cut])
    test_classes = [c for c in np.unique(labels[cut:]) if c in classes]
    if len(classes) < 2 or len(test_classes) < 2:
        return float("nan")
    onehot = (labels[:cut, None] == classes[None]).astype(float)
    a = xs[:cut]
    w = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]), a.T @ onehot)
    pred = classes[np.argmax(xs[cut:] @ w, 1)]
    recalls = [float(np.mean(pred[labels[cut:] == c] == c)) for c in test_classes]
    return float(np.mean(recalls)) - 1.0 / len(test_classes)


def random_features(f, cut, n_features=600, seed=0):
    """Random Fourier features approximating an RBF kernel (width: median distance between training samples)."""
    rng = np.random.default_rng(seed)
    f = (f - f[:cut].mean(0)) / (f[:cut].std(0) + 1e-6)
    sub = f[:cut][rng.choice(cut, size=min(cut, 300), replace=False)]
    d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
    width = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    w = rng.normal(0.0, 1.0 / width, (f.shape[1], n_features))
    b = rng.uniform(0, 2 * np.pi, n_features)
    return np.sqrt(2.0 / n_features) * np.cos(f @ w + b)


def decode_gain(feats, labels_by_key, cut, nonlinear=False):
    f = pcs(feats, cut, k=50 if nonlinear else 100)
    if nonlinear:
        f = random_features(f, cut)
    gains = {}
    for k in DECODE_KEYS:
        gains[k] = balanced_probe(f, labels_by_key[k], ridge=10.0 if nonlinear else 1.0)
    return gains


def test_decoding(run):
    t = targets(run["y"])
    cut = int(0.7 * len(run["y"]))
    feats = [("input frame", run["frame"])] + [(lab, st) for lab, _, _, st, _ in blocks(run)]
    print("  1. task decoding: balanced accuracy above chance, linear (lin) and non-linear (nl) decoder")
    print("     " + f"{'features':16s}" + "".join(f"{k:>14s}" for k in DECODE_KEYS) + f"{'mean lin/nl':>14s}")
    for name, x in feats:
        g, gn = decode_gain(x, t, cut), decode_gain(x, t, cut, nonlinear=True)
        cells = "".join(f"{g[k]:+7.3f}/{gn[k]:+.3f}"[:14].rjust(14) for k in DECODE_KEYS)
        print("     " + f"{name:16s}" + cells +
              f"{np.nanmean([g[k] for k in DECODE_KEYS]):+7.3f}/{np.nanmean([gn[k] for k in DECODE_KEYS]):+.3f}")


def test_transformation(run):
    bl = blocks(run)
    print("  2. transformation: R2 of each population's potentials from the previous layer's E (1 = a copy)")
    prev = None
    for lab, _, v, _, _ in bl:
        if prev is not None and (lab.endswith("_E") or "field" in lab):
            print(f"     {prev[0]:>12s} -> {lab:14s} R2 {ridge_r2(prev[1], v, ridge=10.0):6.3f}")
        if lab.endswith("_E"):
            prev = (lab, v)


def cell_positions(shape):
    """Input-grid coordinates (y, x) of each neuron of a population laid out (row, col, channel)."""
    h, w, c = shape
    cell = np.arange(h * w * c) // c
    r, col = cell // w, cell % w
    return (r + 0.5) / h * INPUT_HW[0] - 0.5, (col + 0.5) / w * INPUT_HW[1] - 0.5


def toroidal(d, size):
    d = np.abs(d)
    return np.minimum(d, size - d)


def ff_receptive_fields(config_path):
    """Per hidden population (by label): a boolean matrix neurons x input cells (20x20) of the input cells reachable
    through feed-forward synapses only, on the network rebuilt from the run's config."""
    import contextlib
    import io
    import scipy.sparse as sp
    import snake_hparams
    import snake_network
    from ml_genn.compilers.eprop_compiler import default_params
    cfg = {k: v for k, v in json.loads(Path(config_path).read_text()).items() if k != "_meta"}
    with contextlib.redirect_stdout(io.StringIO()):
        b = snake_network.build_network(snake_network.spec_from_hparams({**snake_hparams.DEFAULTS, **cfg},
                                                                        INPUT_HW + (INPUT_C,), 4), default_params)
    depth = {}
    for lab in b.labels.values():
        depth[lab] = 10 ** 6 if "field" in lab else int(lab[1:].split("_")[0])
    n_cells = INPUT_HW[0] * INPUT_HW[1]
    reach = {"input": sp.csr_matrix((np.ones(n_cells * INPUT_C, bool),
                                     (np.arange(n_cells * INPUT_C), np.arange(n_cells * INPUT_C) // INPUT_C)),
                                    shape=(n_cells * INPUT_C, n_cells))}
    ff = []
    for c in b.network.connections:
        con = c.connectivity
        if c.is_feedback or getattr(con, "pre_ind", None) is None:
            continue
        src = "input" if c.source() is b.input else b.labels.get(c.source())
        tgt = b.labels.get(c.target())
        if tgt is None or src is None or (src != "input" and depth[tgt] <= depth[src]):
            continue                                    # recurrent / within-layer, or into the heads
        ff.append((0 if src == "input" else depth[src], src, tgt, np.asarray(con.pre_ind), np.asarray(con.post_ind),
                   int(np.prod(c.source().shape)), int(np.prod(c.target().shape))))
    for _, src, tgt, pre, post, ns, nt in sorted(ff, key=lambda f: f[0]):
        adj = sp.csr_matrix((np.ones(len(pre), bool), (post, pre)), shape=(nt, ns))
        r = (adj.astype(np.int32) @ reach[src].astype(np.int32)) > 0
        reach[tgt] = (reach[tgt] + r) > 0 if tgt in reach else r
    return {k: v.toarray() for k, v in reach.items() if k != "input"}


def dilate(cells_mask, steps=1):
    m = cells_mask.reshape(INPUT_HW)
    for _ in range(steps):
        m = m | np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)
    return m.ravel()


def test_spatial(run, config_path, n_patches=24, seed=0):
    rng = np.random.default_rng(seed)
    pix = run["frame"].reshape(len(run["frame"]), *INPUT_HW, INPUT_C).sum(-1).reshape(len(run["frame"]), -1)
    keep = pix.std(0) > 1e-6
    try:
        rf = ff_receptive_fields(config_path)
    except Exception as e:                              # no config next to the probe: cannot trace the fields
        print(f"  3. spatial transport: skipped ({e!r})")
        return {}
    print("  3. spatial transport: pixels OUTSIDE a patch's feed-forward receptive field (FF RF). 'FF RF input' = R2 of "
          "those pixels from the input inside the field;")
    print("     'patch' / 'patch mean' = R2 of the RESIDUAL the inside input cannot predict, from the patch / its mean "
          "(> 0: information arrived sideways)")
    print(f"     {'patch from':14s} {'RF cells':>8s} {'FF RF input':>11s} {'patch':>7s} {'patch mean':>10s}")
    results = {}
    for lab, shp, v, _, _ in blocks(run):
        if not (lab.endswith("_E") or "field" in lab) or lab not in rf:
            continue
        ny, nx = cell_positions(shp)
        rad = max(1.0, INPUT_HW[0] / shp[0])               # about one cell of the population's own grid
        vals, sizes = {"patch": [], "rf": [], "mean": []}, []
        for cy, cx in rng.uniform(0, INPUT_HW[0], size=(n_patches, 2)):
            sel = np.flatnonzero(np.hypot(toroidal(ny - cy, INPUT_HW[0]), toroidal(nx - cx, INPUT_HW[1])) <= rad)
            if len(sel) == 0:
                continue
            field = dilate(rf[lab][sel].any(0))
            outside = np.flatnonzero(keep & ~field)
            inside = np.flatnonzero(keep & field)
            if len(outside) == 0:                       # the patch's FF RF covers the whole input
                continue
            sizes.append(int(field.sum()))
            tgt = pix[:, rng.choice(outside, size=min(12, len(outside)), replace=False)]
            vals["rf"].append(max(ridge_r2(pix[:, inside], tgt, ridge=10.0), 0.0) if len(inside) else 0.0)
            # the part of the outside pixels that the patch's own feed-forward input cannot predict (ridge on the
            # inside pixels, fit on the training part): only this residual can show sideways transport
            if len(inside):
                k = int(0.7 * len(tgt))
                xi = np.hstack([pcs(pix[:, inside], k), np.ones((len(tgt), 1))])
                w = np.linalg.solve(xi[:k].T @ xi[:k] + 10.0 * np.eye(xi.shape[1]), xi[:k].T @ tgt[:k])
                tgt = tgt - xi @ w
            vals["patch"].append(max(ridge_r2(v[:, sel], tgt, ridge=10.0), 0.0))
            vals["mean"].append(max(ridge_r2(v[:, sel].mean(1, keepdims=True), tgt, ridge=10.0), 0.0))
        if not sizes:
            print(f"     {lab:14s} feed-forward RF covers the whole input: no outside pixels to test")
            continue
        m = {k: float(np.mean(x)) for k, x in vals.items()}
        results[lab] = m
        print(f"     {lab:14s} {np.mean(sizes):8.0f} {m['rf']:11.3f} {m['patch']:7.3f} {m['mean']:10.3f}")
    return results


def test_memory(run):
    t = targets(run["y"])
    cut = int(0.7 * len(run["y"]))
    ex = run["extra"]
    prev_head = ex[:, run["extra_names"].index("prev_heading")].astype(int)
    ok = prev_head >= 0
    use_a = run["adapt"] is not None
    print(f"  4. memory: past input recovered beyond the current frame (lags 1, 2, 3 frames; state = V"
          f"{' + adaptation A' if use_a else ' only: record with probe adapt=true for A'}), and previous heading")
    b0 = balanced_probe(pcs(run["frame"][ok], int(0.7 * ok.sum())), prev_head[ok])
    print(f"     {'input frame':16s} prev heading {b0:+.3f}")
    for lab, _, v, st, a in blocks(run):
        if not (lab.endswith("_E") or "field" in lab):
            continue
        state = np.hstack([v, a]) if a is not None else v
        mem = memory_r2(pcs(state, cut), run["frame"])
        bh = balanced_probe(pcs(st[ok], int(0.7 * ok.sum())), prev_head[ok])
        print(f"     {lab:16s} prev heading {bh:+.3f} | memory lag 1/2/3: " + " ".join(f"{m:.3f}" for m in mem))


def test_representation(run):
    import snake_strategy as ss
    if run.get("body") is None or not run["board"]:
        print("  5. representation: skipped (no body / apple in this recording: record with the updated ProbeRecorder)")
        return
    heading = run["y"][:, 6].astype(int)
    cls, val = zip(*[ss.state_variables(b, a, h, run["board"]) for b, a, h in zip(run["body"], run["apple"], heading)])
    cls, val = np.array(cls), np.array(val)
    cut = int(0.7 * len(cls))
    feats = [("input frame", run["frame"])] + [(lab, st) for lab, _, _, st, _ in blocks(run)
                                               if lab.endswith("_E") or "field" in lab]
    print(f"  5. representation: trap vs safe among non-lethal moves (balanced acc. above chance), and R2 of global "
          f"values; traps among non-lethal moves: {np.mean(cls == 1) / max(np.mean(cls > 0), 1e-9):.3f}")
    names = ["trap_" + n.split("_")[1] for n in ss.STATE_CLASS_NAMES] + ss.STATE_VALUE_NAMES
    print("     " + f"{'features':16s}" + "".join(f"{n[:11]:>12s}" for n in names))
    for name, x in feats:
        f = pcs(x, cut)
        row = []
        for k in range(4):
            ok = cls[:, k] > 0                                   # non-lethal moves: trap (1) vs safe (2)
            row.append(balanced_probe(f[ok], cls[ok, k]) if ok.sum() > 100 and len(np.unique(cls[ok, k])) > 1
                       else float("nan"))
        for j in range(val.shape[1]):
            row.append(max(ridge_r2(f, val[:, j:j + 1], ridge=10.0), -1.0))
        print("     " + f"{name:16s}" + "".join(f"{v:+12.3f}" for v in row))
    # unsupervised: the last population's top principal components and their best-matching variables
    lab, st = feats[-1]
    z = pcs(st, cut, k=5)
    allv = np.hstack([val, (cls == 1).astype(np.float32), (cls == 0).astype(np.float32), run["y"][:, :2]])
    alln = ss.STATE_VALUE_NAMES + ["trap_" + n.split("_")[1] for n in ss.STATE_CLASS_NAMES] + \
        ["danger_" + n.split("_")[1] for n in ss.STATE_CLASS_NAMES] + ["apple_dy", "apple_dx"]
    print(f"     top principal components of {lab}: best-matching variables (|correlation|)")
    for i in range(z.shape[1]):
        c = [abs(np.corrcoef(z[:, i], allv[:, j])[0, 1]) if allv[:, j].std() > 0 else 0 for j in range(allv.shape[1])]
        top = np.argsort(c)[::-1][:3]
        print(f"       PC{i + 1}: " + ", ".join(f"{alln[j]} {c[j]:.2f}" for j in top))


def input_history(run, lags=3, tau=10.0):
    """Per aligned sample: the frame processed, the `lags` frames before it (zero across an episode start) and an
    exponentially filtered history of the frames (restarted at episode starts)."""
    inp, t, ep = run["inp_all"].astype(np.float32) / 255.0, run["t_index"], run["ep_all"]
    n = len(inp)
    cur = t - 1                                              # the frame each aligned state has processed
    feats = [inp[cur]]
    for k in range(1, lags + 1):
        idx = np.clip(cur - k, 0, n - 1)
        ok = (cur - k >= 0) & (ep[idx] == ep[cur])
        feats.append(np.where(ok[:, None], inp[idx], 0.0))
    a = np.exp(-1.0 / tau)
    hist = np.zeros_like(inp)                                # hist[i]: filtered frames up to and including i
    hist[0] = (1 - a) * inp[0]
    for i in range(1, n):
        hist[i] = (a * hist[i - 1] if ep[i] == ep[i - 1] else 0.0) + (1 - a) * inp[i]
    feats.append(hist[cur])
    return feats


def test_linearity(run, n_targets=20):
    cut = int(0.7 * len(run["frame"]))
    feats = input_history(run)
    cur = pcs(feats[0], cut)
    hist = np.hstack([pcs(f, cut, k=40) for f in feats])
    hist_nl = random_features(pcs(np.hstack(feats), cut, k=60), cut, n_features=1500)
    print("  6. input -> population: R2 of its top principal components from the input (current frame linear / "
          "input history linear / input history non-linear)")
    print(f"     {'population':16s} {'current lin':>11s} {'history lin':>11s} {'history nl':>11s} "
          f"{'non-linear part':>15s} {'unexplained':>11s}")
    for lab, _, v, _, _ in blocks(run):
        if not (lab.endswith("_E") or "field" in lab):
            continue
        y = pcs(v, cut, k=n_targets)
        r_cur = ridge_r2(cur, y, ridge=10.0)
        r_lin = ridge_r2(hist, y, ridge=10.0)
        r_nl = ridge_r2(hist_nl, y, ridge=10.0, max_features=10 ** 6)
        print(f"     {lab:16s} {r_cur:11.3f} {r_lin:11.3f} {r_nl:11.3f} {r_nl - r_lin:+15.3f} {1 - max(r_nl, r_lin):11.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--window", default="both", choices=["first", "last", "both"])
    ap.add_argument("--tests", default="123456")
    args = ap.parse_args()
    files = []
    for p in args.paths:
        files += sorted(glob.glob(str(Path(p) / "outputs/*_probe.npz"))) if Path(p).is_dir() else [p]
    for f in files:
        for w in (("first", "last") if args.window == "both" else (args.window,)):
            run = load(f, w)
            print(f"\n{f}  [{w} window, {len(run['y'])} aligned samples]")
            if "1" in args.tests:
                test_decoding(run)
            if "2" in args.tests:
                test_transformation(run)
            if "3" in args.tests:
                test_spatial(run, Path(f).parent.parent / "config.json")
            if "4" in args.tests:
                test_memory(run)
            if "5" in args.tests:
                test_representation(run)
            if "6" in args.tests:
                test_linearity(run)


if __name__ == "__main__":
    main()
