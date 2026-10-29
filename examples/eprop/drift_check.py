"""Check, in a trained network, that the drift is the weight derivative of the eligibility (paper, Appendix
"Numerical verification", there done in a toy network).

From a snapshot of the network state, many short rollouts are run. Every rollout reloads the GeNN model (which
reseeds its random number generator), advances the generator by k steps, and restores the snapshot. Every step
draws the same number of random numbers whatever the network does, so rollout k sees its own noise and Poisson
input spikes, and two rollouts with the same k differ only in what we change between them: common random numbers
for the finite differences. Expectations are over the injected noise and the input spikes, at fixed input rates.

Compared per sampled synapse and step t of the window:
- the Stein side: the mean over rollouts of the drift score, s = kappa g pi NoiseTrace / SynSig^2, with the rule's
  drift gate g = psi - psibar ("gate": "drift", default) or the paper's g = psi ("gate": "psi");
- the weight derivative of the mean eligibility, (E[e](w + h) - E[e](w - h)) / 2h with e = psi pi;
- the mean eligibility itself (which should not predict the score).
pi = ZFilter - Beta epsA (or ZFilter alone with score_adaptation false). No learning happens during the check.
"""
import numpy as np
from pygenn import SynapseMatrixConnectivity

from ml_genn.compilers.eprop import hidden_rule_from_dict


def _arrays(holder):
    return {n: v for n, v in holder.items() if getattr(v, "_array", None) is not None}


class _State:
    """Every neuron, synapse, pre and post variable of the loaded model (connectivity is regenerated identically
    by every load and checked)."""

    def __init__(self, net):
        self.items = []
        for pop in net.neuron_populations.values():
            for n, v in _arrays(pop.vars).items():
                v.pull_from_device()
                self.items.append((v, np.array(v.view, copy=True)))
        for sg in net.connection_populations.values():
            for holder in (sg.vars, sg.pre_vars, sg.post_vars):
                for n, v in _arrays(holder).items():
                    v.pull_from_device()
                    # sparse synapse variables are only readable as a copy ('values')
                    vals = v.values if holder is sg.vars else v.view
                    self.items.append((v, np.array(vals, copy=True)))
            if sg.matrix_type & SynapseMatrixConnectivity.SPARSE:
                sg.pull_connectivity_from_device()
        self.inds = {sg.name: (np.array(sg.get_sparse_pre_inds()), np.array(sg.get_sparse_post_inds()))
                     for sg in net.connection_populations.values()
                     if sg.matrix_type & SynapseMatrixConnectivity.SPARSE}

    def restore(self, net):
        for sg in net.connection_populations.values():
            if sg.name in self.inds:
                sg.pull_connectivity_from_device()
                pre, post = self.inds[sg.name]
                if not (np.array_equal(sg.get_sparse_pre_inds(), pre) and np.array_equal(sg.get_sparse_post_inds(), post)):
                    raise RuntimeError(f"{sg.name}: the reloaded connectivity differs from the snapshot")
        for v, val in self.items:
            try:
                v.view[...] = val
            except Exception:
                v.values = val
            v.push_to_device()


def load_weights(compiled_net, prefix):
    """Weights from a run's checkpoint (<prefix>-<synapse group>-g.npy, same network and seed): the connectivity
    must match the saved pre/post indices."""
    for sg in compiled_net.connection_populations.values():
        try:
            g = np.load(f"{prefix}-{sg.name}-g.npy")
        except FileNotFoundError:
            continue
        if sg.matrix_type & SynapseMatrixConnectivity.SPARSE:
            sg.pull_connectivity_from_device()
            pre, post = np.load(f"{prefix}-{sg.name}-pre_ind.npy"), np.load(f"{prefix}-{sg.name}-post_ind.npy")
            order = np.lexsort((np.asarray(sg.get_sparse_post_inds()), np.asarray(sg.get_sparse_pre_inds())))
            saved = np.lexsort((post, pre))
            if not (np.array_equal(np.asarray(sg.get_sparse_pre_inds())[order], pre[saved])
                    and np.array_equal(np.asarray(sg.get_sparse_post_inds())[order], post[saved])):
                raise RuntimeError(f"{sg.name}: checkpoint connectivity differs from this network")
            new = np.empty_like(g); new[order] = g[saved]
            sg.vars["g"].values = new
        else:
            sg.vars["g"].view[...] = g.reshape(sg.vars["g"].view.shape)
        sg.vars["g"].push_to_device()


def record(compiled_net, sg, conn, snapshot, set_input, step_callbacks, cfg, out_path):
    """Record, from the snapshot, cfg["windows"] windows of cfg["window"] steps (the random streams continue, so
    every window has its own input spikes and noise): the spikes of every population projecting to the target
    population, and the target neurons' state and noise samples, for an open-loop single-neuron replay
    (drift_toy.py). Spikes are read from the presynaptic traces, z_t = ZFilter_t - alpha ZFilter_{t-1}.

    Window layout: the snapshot is restored, one settling step is run (its input is whatever spikes were pending),
    and recording starts after it: state[0] is the target state after the settling step and spikes[t] are the
    spikes emitted in step t (t = 0 is the settling step), delivered in step t + 1."""
    tgt = conn.target()
    tpop = compiled_net.neuron_populations[tgt]
    W, T = int(cfg.get("windows", 256)), int(cfg.get("window", 200))
    alpha = float(np.exp(-1.0 / tgt.neuron.tau_mem))
    incoming = [(c, s) for c, s in compiled_net.connection_populations.items()
                if c.target() is tgt and not c.is_feedback]
    sources, zgroups = [], []
    for c, s in incoming:
        if c.source() in sources:
            continue
        z = next(o for o in compiled_net.connection_populations.values()
                 if o.src is s.src and "ZFilter" in o.pre_vars
                 and getattr(o.pre_vars["ZFilter"], "_array", None) is not None)
        sources.append(c.source()); zgroups.append(z)
    groups = []
    for c, s in incoming:
        for v in ("g", "SynSig"):
            if v in s.vars:
                s.vars[v].pull_from_device()
        groups.append(dict(name=s.name, source=sources.index(c.source()), pre=np.asarray(s.get_sparse_pre_inds()),
                           post=np.asarray(s.get_sparse_post_inds()), g=np.array(s.vars["g"].values),
                           syn_sig=np.array(s.vars["SynSig"].values) if "SynSig" in s.vars else None))
    names = ("V", "A", "RefracTime", "Noise1", "Beta")
    psi_vars = ("Psi", "PsiBar")
    spikes = [np.zeros((W, T + 1, z.src.num_neurons), dtype=bool) for z in zgroups]
    state = {n: np.zeros((W, T + 1, tpop.num_neurons), dtype=np.float32) for n in names + psi_vars}

    def zf():
        out = []
        for z in zgroups:
            z.pre_vars["ZFilter"].pull_from_device()
            out.append(np.array(z.pre_vars["ZFilter"].view, dtype=np.float64).ravel())
        return out

    def target_state(w, t):
        for n in names:
            tpop.vars[n].pull_from_device()
            state[n][w, t] = np.asarray(tpop.vars[n].view).ravel()
        for n in psi_vars:
            sg.post_vars[n].pull_from_device()
            state[n][w, t] = np.asarray(sg.post_vars[n].view).ravel()

    zf0 = None
    for w in range(W):
        snapshot.restore(compiled_net)
        set_input()
        prev = zf()
        zf0 = prev if zf0 is None else zf0
        for t in range(T + 1):
            compiled_net.step_time(step_callbacks)
            cur = zf()
            for k, (a, b) in enumerate(zip(prev, cur)):
                z = b - alpha * a
                if np.any((np.abs(z) > 1e-3) & (np.abs(z - 1) > 1e-3)):
                    raise RuntimeError("spike inference from ZFilter failed")
                spikes[k][w, t] = z > 0.5
            prev = cur
            target_state(w, t)
        if (w + 1) % 32 == 0:
            print(f"drift record: {w + 1}/{W} windows", flush=True)
    neuron = tgt.neuron
    np.savez_compressed(out_path, alpha=alpha, rho=float(np.exp(-1.0 / neuron.tau_adapt)), v_thresh=neuron.v_thresh,
                        tau_refrac=neuron.tau_refrac, groups=np.array(groups, dtype=object),
                        source_names=np.array([z.name for z in zgroups]),
                        **{f"spikes_{k}": z for k, z in enumerate(spikes)},
                        **{f"zf0_{k}": z for k, z in enumerate(zf0)}, n_sources=len(spikes),
                        **{f"state_{n}": v for n, v in state.items()}, target_group=sg.name,
                        trace_lambda=float(cfg.get("trace_lambda", 0.0)), beta1=float(cfg.get("beta1", 0.99)))
    print(f"drift record: saved {out_path}", flush=True)
    return "recorded"


def run(compiled_net, built, hidden_rule, step_callbacks, set_input, cfg, out_path):
    """cfg: {"group": "L3_E->L3_E", "n_syn": 30, "rollouts": 64, "window": 30, "h": 0.2, "seed": 0,
    "weights_from": None, "warmup": 2000} (h relative to the RMS weight of the group). set_input: re-applies the
    current input after a reload. weights_from: checkpoint prefix to load trained weights from, followed by
    warmup steps without learning."""
    group, n_syn = cfg.get("group", "L3_E->L3_E"), int(cfg.get("n_syn", 30))
    K, T, h_rel = int(cfg.get("rollouts", 64)), int(cfg.get("window", 30)), float(cfg.get("h", 0.2))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    rule = hidden_rule_from_dict(hidden_rule) if isinstance(hidden_rule, dict) else hidden_rule
    gm = compiled_net.genn_model

    labels = dict(zip(built.hidden, [built.labels[p] for p in built.hidden]))
    conn, sg = next((c, s) for c, s in compiled_net.connection_populations.items()
                    if not c.is_feedback and labels.get(c.target()) is not None
                    and f"{labels.get(c.source(), c.source().name)}->{labels[c.target()]}" == group)
    tgt = conn.target()
    pre_all, post_all = np.array(sg.get_sparse_pre_inds()), np.array(sg.get_sparse_post_inds())
    alpha = float(np.exp(-1.0 / tgt.neuron.tau_mem))
    kappa = (1.0 - alpha * alpha) if rule.kappa else 1.0
    beta = np.broadcast_to(np.asarray(tgt.neuron.beta, dtype=np.float64).ravel(), (sg.trg.num_neurons,))
    adapt = rule.score_adaptation in (True, "drift")
    zsg = next(o for o in [sg] + list(compiled_net.connection_populations.values())
               if o.src is sg.src and "ZFilter" in o.pre_vars and getattr(o.pre_vars["ZFilter"], "_array", None) is not None)

    if cfg.get("weights_from"):
        load_weights(compiled_net, cfg["weights_from"])
        set_input()
        for _ in range(int(cfg.get("warmup", 2000))):
            compiled_net.step_time(step_callbacks)
        print(f"drift check: loaded {cfg['weights_from']} and warmed up", flush=True)
    snapshot = _State(compiled_net)
    # every noise trace must hold only the controlled noise
    traces = [o.vars["NoiseTrace"] for o in compiled_net.connection_populations.values() if "NoiseTrace" in o.vars]
    snapshot.items = [(v, np.zeros_like(val) if any(v is tr for tr in traces) else val) for v, val in snapshot.items]
    if cfg.get("mode") == "record":
        return record(compiled_net, sg, conn, snapshot, set_input, step_callbacks, cfg, out_path)

    def rollout(k, syn=None, delta=0.0, sel=None, reload=True):
        if reload:
            gm.unload(); gm.load()
            for _ in range(k):                 # advance the random streams: rollout k's own noise and inputs
                gm.step_time()
        snapshot.restore(compiled_net)         # without reload: the random streams simply continue
        if syn is not None:
            sg.vars["g"].pull_from_device()
            g = np.array(sg.vars["g"].values); g[syn] += delta
            sg.vars["g"].values = g; sg.vars["g"].push_to_device()
        set_input()
        s_out, e_out = [], []
        for t in range(T):
            compiled_net.step_time(step_callbacks)
            for v in ("NoiseTrace", "SynSig", "epsilonA"):
                sg.vars[v].pull_from_device()
            sg.post_vars["Psi"].pull_from_device(); zsg.pre_vars["ZFilter"].pull_from_device()
            if drift_gate:
                sg.post_vars["PsiBar"].pull_from_device()
            nt, sig, eps = (np.asarray(sg.vars[v].values, dtype=np.float64) for v in ("NoiseTrace", "SynSig", "epsilonA"))
            psi = np.asarray(sg.post_vars["Psi"].view, dtype=np.float64).ravel()[post_all]
            zf = np.asarray(zsg.pre_vars["ZFilter"].view, dtype=np.float64).ravel()[pre_all]
            pi = zf - beta[post_all] * eps if adapt else zf
            if drift_gate:
                psi = psi - np.asarray(sg.post_vars["PsiBar"].view, dtype=np.float64).ravel()[post_all]
            e = psi * pi
            s = kappa * e * nt / (sig * sig + 1e-6)
            idx = sel if sel is not None else slice(None)
            s_out.append(s[idx]); e_out.append(e[idx])
        return np.array(s_out), np.array(e_out)

    drift_gate = cfg.get("gate", "drift") == "drift"
    # the rule's own averaging, applied after the expectation (both filters are linear): the eligibility trace
    # (lambda per step) and the optimiser's momentum (beta1)
    lam, beta1 = float(cfg.get("trace_lambda", 0.0)), float(cfg.get("beta1", 0.99))

    def rule_filter(x):
        """x: (..., T, n) per-step values -> the momentum average at the end of the window of the lambda-trace."""
        tr, out = np.zeros(x.shape[:-2] + x.shape[-1:]), []
        for t in range(x.shape[-2]):
            tr = lam * tr + x[..., t, :]
            out.append(tr)
        out = np.stack(out, axis=-2)
        w = (1 - beta1) * beta1 ** np.arange(x.shape[-2] - 1, -1, -1)
        return np.tensordot(w, out, axes=([0], [-2])) if out.ndim == 2 else np.einsum("t,...tn->...n", w, out)

    def save(stage, base_s, base_e, fd=None, done=0):
        out = dict(stage=stage, synapses=sel, h=h, trace_lambda=lam, beta1=beta1, gate=cfg.get("gate", "drift"),
                   score=base_s.mean(0), score_se=base_s.std(0) / np.sqrt(len(base_s)),
                   score_a=base_s[0::2].mean(0), score_b=base_s[1::2].mean(0), elig=base_e.mean(0),
                   score_f=rule_filter(base_s.mean(0)), score_fa=rule_filter(base_s[0::2].mean(0)),
                   score_fb=rule_filter(base_s[1::2].mean(0)), elig_f=rule_filter(base_e.mean(0)),
                   n_done=done)
        if fd is not None and done > 0:
            f = fd[:, :, :done]
            out.update(deriv=f.mean(0), deriv_se=f.std(0) / np.sqrt(len(f)), deriv_a=f[0::2].mean(0),
                       deriv_b=f[1::2].mean(0), deriv_f=rule_filter(f.mean(0)),
                       deriv_fa=rule_filter(f[0::2].mean(0)), deriv_fb=rule_filter(f[1::2].mean(0)))
        np.savez(out_path, **out)

    # common random numbers: the same k twice gives identical rollouts, different k different ones
    a, _ = rollout(1); b, _ = rollout(1); c, _ = rollout(2)
    if not np.array_equal(a, b) or np.array_equal(a, c):
        raise RuntimeError("rollouts are not reproducible per k / not different across k")
    # pilot: pick synapses that transmit (noise trace fills) and are eligible during the window
    s0, e0 = rollout(0)
    active = np.flatnonzero((np.abs(s0).mean(0) > 0) & (np.abs(e0).mean(0) > 0))
    if len(active) == 0:
        raise RuntimeError("no synapse of the group has eligibility in the window")
    sel = np.sort(rng.choice(active, size=min(n_syn, len(active)), replace=False))
    sg.vars["g"].pull_from_device()
    h = h_rel * float(np.sqrt(np.mean(np.asarray(sg.vars["g"].values, dtype=np.float64) ** 2)))

    # Stein side: needs only many independent rollouts (no common random numbers), so the snapshot is restored
    # without reloading and the random streams continue
    Ks = int(cfg.get("stein_rollouts", 8192))
    base_s, base_e = zip(*(rollout(0, sel=sel, reload=False) for _ in range(Ks)))
    base_s, base_e = np.array(base_s), np.array(base_e)          # (Ks, T, n)
    print(f"drift check: {Ks} Stein rollouts done", flush=True)
    save("stein", base_s, base_e)
    fd = np.zeros((K, T, len(sel)))
    for j, syn in enumerate(sel):
        for k in range(K):
            _, ep = rollout(k, syn, +h, sel)
            _, em = rollout(k, syn, -h, sel)
            fd[k, :, j] = (ep[:, j] - em[:, j]) / (2 * h)
        print(f"drift check: synapse {j + 1}/{len(sel)} done", flush=True)
        if (j + 1) % 4 == 0 or j + 1 == len(sel):
            save("derivative", base_s, base_e, fd, j + 1)

    S, D, E = base_s.mean(0).ravel(), fd.mean(0).ravel(), base_e.mean(0).ravel()
    S_se = (base_s.std(0) / np.sqrt(Ks)).ravel()

    def fit(x, y):
        c = np.corrcoef(x, y)[0, 1]
        slope = np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean())
        return c, slope

    c_d, sl_d = fit(D, S); c_e, _ = fit(E, S)
    summary = (f"drift check {group} ({len(sel)} synapses, {Ks} Stein / {K} derivative rollouts, {T} steps, h {h:.3g}): "
               f"mean score vs dE[e]/dW corr {c_d:.3f} slope {sl_d:.2f} R2 {c_d ** 2:.3f} | "
               f"vs E[e] corr {c_e:.3f} | median score s.e. / |score| {np.median(S_se / (np.abs(S) + 1e-12)):.2f}")
    print(summary, flush=True)
    return summary
