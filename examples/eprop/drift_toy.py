"""Open-loop single-neuron replay of a recorded network (drift_check.py, "mode": "record").

The target neuron is simulated alone, fed the recorded spikes of all its afferents, with the update order and
equations of the generated GeNN code:
1. synapse dynamics: the score with the drift gate psi - psibar and the noise trace (then the trace decays);
2. presynaptic delivery of the spikes emitted in the previous step: g + SynSig * Noise1 into the input, SynSig *
   Noise1 into the noise trace;
3. neuron update: V = alpha V + Isyn, A decays, refractoriness, a new Noise1, psi from the pre-reset voltage,
   psibar, spike and reset.
Its own noise is drawn afresh in every rollout; the afferent spikes are those of the recorded window (open loop:
the neuron's noise cannot change its inputs). With the recorded noise instead, the replay must reproduce the
recorded voltages (validate()).

drift(...) gives the Stein side and the finite-difference weight derivative of the eligibility for chosen
synapses of the target group, per step and after the rule's own averaging (trace lambda, momentum beta1).
"""
import numpy as np


class Recording:
    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        self.d = d
        self.alpha, self.rho, self.vth = float(d["alpha"]), float(d["rho"]), float(d["v_thresh"])
        self.refrac = float(d["tau_refrac"])
        self.groups = list(d["groups"])
        n = int(d["n_sources"])
        self.spikes = [d[f"spikes_{k}"] for k in range(n)]   # per source: (W, T+1, n_src), spikes emitted in step t
        self.zf0 = [d[f"zf0_{k}"] for k in range(n)]
        self.state = {k[6:]: d[k] for k in d.files if k.startswith("state_")}   # (W, T+1, n_tgt), after step t
        self.target_group = str(d["target_group"])
        self.lam, self.beta1 = float(d["trace_lambda"]), float(d["beta1"])
        self.W, self.T1 = self.state["V"].shape[:2]
        # presynaptic traces after every recorded step: ZF_t = alpha ZF_{t-1} + z_t
        self.zf = []
        for z, zf0 in zip(self.spikes, self.zf0):
            out = np.empty(z.shape, dtype=np.float64)
            prev = np.broadcast_to(np.asarray(zf0, dtype=np.float64), z.shape[::2]).copy()
            for t in range(z.shape[1]):
                prev = self.alpha * prev + z[:, t]
                out[:, t] = prev
            self.zf.append(out)

    def afferents(self, j):
        """All synapses onto target neuron j: source index, presynaptic neuron, weight, SynSig, (group, index)."""
        src, pre, g, sig, ids = [], [], [], [], []
        for gr in self.groups:
            for k in np.flatnonzero(gr["post"] == j):
                src.append(gr["source"]); pre.append(gr["pre"][k]); g.append(gr["g"][k])
                sig.append(0.0 if gr["syn_sig"] is None else gr["syn_sig"][k]); ids.append((gr["name"], k))
        return np.array(src), np.array(pre), np.array(g, dtype=np.float64), np.array(sig, dtype=np.float64), ids

    def inputs(self, j):
        """Per window and step: which afferents deliver (spiked in the previous step) and their presynaptic
        trace at the synapse-dynamics stage (after the previous step): (W, T+1, n_aff) each."""
        src, pre, *_ = self.afferents(j)
        deliver = np.stack([self.spikes[s][:, :, p] for s, p in zip(src, pre)], axis=-1)
        trace = np.stack([self.zf[s][:, :, p] for s, p in zip(src, pre)], axis=-1)
        return deliver, trace


def gate_shape(x, gate):
    """Gate as a function of x = (V - threshold) / Vthresh, in units of 0.3 / Vthresh:
    "drift": the code's triangle max(0, 1 - |x|) (zero outside |x| < 1);
    "exp<s>": exp(-|x| / s), long exponential tails (psi'/psi constant);
    "fs<s>": 1 / (1 + |x| / s)^2, the fast-sigmoid (SuperSpike) power-law tails."""
    if gate in ("drift", "psi"):
        return np.maximum(0.0, 1.0 - np.abs(x))
    if gate.startswith("exp"):
        return np.exp(-np.abs(x) / float(gate[3:]))
    if gate.startswith("fs"):
        return 1.0 / (1.0 + np.abs(x) / float(gate[2:])) ** 2
    raise ValueError(gate)


def simulate(rec, j, w, noise, g=None, steps=None, gate="drift", return_state=False):
    """Replay target neuron j in window w (an int, or one window per rollout as an (R,) array) for R rollouts.
    noise: (R, T+1) Noise1 samples (noise[:, t] is drawn in step t and delivered in step t+1); g: (R, n_aff)
    weights (default: the recorded ones). gate: "drift" (psi - psibar with the code's psi), "psi" (psi alone), or
    a tailed gate ("exp<s>" / "fs<s>", see gate_shape) used, with its own running mean, only in the score and the
    eligibility (the neuron's dynamics are unchanged). Returns per step t = 1..T: score s, eligibility e for every
    afferent (R, T, n_aff) and the voltage after the step (R, T)."""
    src, pre, g0, sig, _ = rec.afferents(j)
    deliver, trace = rec.inputs(j)
    R = noise.shape[0]
    T = (rec.T1 - 1) if steps is None else steps
    g = np.broadcast_to(g0, (R, len(g0))) if g is None else g
    w = np.broadcast_to(np.asarray(w), (R,))
    st = {k: rec.state[k][w, 0, j].astype(np.float64) for k in ("V", "A", "RefracTime", "Psi", "PsiBar")}
    tailed = gate not in ("drift", "psi")
    if tailed:   # the tailed gate and its running mean, started from the recorded voltage
        x0 = (st["V"] - (rec.vth + rec.state["Beta"][w, 0, j] * st["A"])) / rec.vth
        st["PsiT"] = np.where(st["RefracTime"] > 0, 0.0, (0.3 / rec.vth) * gate_shape(x0, gate))
        st["PsiTBar"] = st["PsiT"].copy()
    beta = rec.state["Beta"][w, 0, j].astype(np.float64)
    a, kappa = rec.alpha, 1.0 - rec.alpha ** 2
    nt = np.zeros((R, len(g0)))
    s_out, e_out, v_out = [], [], []
    for t in range(1, T + 1):
        # 1. synapse dynamics (psi and psibar from the previous neuron update, presynaptic trace after step t-1)
        gt = (st["PsiT"] - st["PsiTBar"] if tailed else
              st["Psi"] - st["PsiBar"] if gate == "drift" else st["Psi"])
        e = gt[:, None] * trace[w, t - 1]
        s_out.append(kappa * e * nt / (sig * sig + 1e-6)); e_out.append(e)
        nt *= a
        # 2. delivery of the spikes emitted in step t-1, with the noise drawn in step t-1
        d = deliver[w, t - 1]
        eps = sig[None, :] * noise[:, t - 1][:, None] * d
        isyn = (g * d).sum(1) + eps.sum(1)
        nt += eps
        # 3. neuron update
        st["V"] = a * st["V"] + isyn
        st["A"] *= rec.rho
        ref = st["RefracTime"]
        ref = np.where(ref > 0, ref - 1.0, ref)
        th = rec.vth + beta * st["A"]
        psi = (1.0 / rec.vth) * 0.3 * np.maximum(0.0, 1.0 - np.abs((st["V"] - th) / rec.vth))
        st["Psi"] = np.where(ref > 0, 0.0, psi)
        st["PsiBar"] = 0.99 * st["PsiBar"] + 0.01 * st["Psi"]
        if tailed:
            st["PsiT"] = np.where(ref > 0, 0.0, (0.3 / rec.vth) * gate_shape((st["V"] - th) / rec.vth, gate))
            st["PsiTBar"] = 0.99 * st["PsiTBar"] + 0.01 * st["PsiT"]
        spk = (st["V"] >= th) & (ref <= 0)
        st["V"] = np.where(spk, st["V"] - rec.vth, st["V"])
        st["A"] = np.where(spk, st["A"] + 1.0, st["A"])
        st["RefracTime"] = np.where(spk, rec.refrac, ref)
        v_out.append(st["V"].copy())
    if return_state:
        return st, nt
    return np.stack(s_out, 1), np.stack(e_out, 1), np.stack(v_out, 1)


def validate(rec, j, windows=8):
    """Replay with the recorded noise: maximum voltage error against the recording."""
    err = 0.0
    for w in range(min(windows, rec.W)):
        noise = rec.state["Noise1"][w, :, j][None, :].astype(np.float64)
        _, _, v = simulate(rec, j, w, noise)
        err = max(err, float(np.max(np.abs(v[0] - rec.state["V"][w, 1:, j]))))
    return err


def rule_filter(x, lam, beta1):
    """(..., T, n) -> momentum average (beta1) at the end of the window of the lambda-trace."""
    tr, out = np.zeros(x.shape[:-2] + x.shape[-1:]), []
    for t in range(x.shape[-2]):
        tr = lam * tr + x[..., t, :]
        out.append(tr)
    w = (1 - beta1) * beta1 ** np.arange(x.shape[-2] - 1, -1, -1)
    return np.einsum("t,...tn->...n", w, np.stack(out, axis=-2))


def drift(rec, synapses, h, rollouts=64, fd_rollouts=16, windows=None, seed=0, gate="drift"):
    """synapses: indices into the target group's synapse arrays. For each: the Stein side (mean score) and the
    finite-difference derivative of the mean eligibility (weight +-h, common random numbers), per step and
    rule-averaged, with split halves (alternate windows) for repeatability. Averages over windows (inputs) and
    rollouts (own noise); all windows run as one batch."""
    rng = np.random.default_rng(seed)
    tg = next(gr for gr in rec.groups if gr["name"] == rec.target_group)
    windows = np.arange(rec.W) if windows is None else np.asarray(windows)
    nW = len(windows)
    res = {k: [] for k in ("score", "deriv", "elig", "score_f", "deriv_f", "elig_f")}
    halves = {k: [[], []] for k in ("score", "deriv", "score_f", "deriv_f")}
    for syn in synapses:
        j = int(tg["post"][syn])
        _, _, g0, _, ids = rec.afferents(j)
        k = ids.index((rec.target_group, syn))
        wr = np.repeat(windows, rollouts)
        s, e, _ = simulate(rec, j, wr, rng.standard_normal((len(wr), rec.T1)), gate=gate)
        S = s[:, :, k].reshape(nW, rollouts, -1).mean(1); E = e[:, :, k].reshape(nW, rollouts, -1).mean(1)
        wf = np.repeat(windows, fd_rollouts)
        noise = rng.standard_normal((len(wf), rec.T1))
        gp = np.tile(g0, (len(wf), 1)); gp[:, k] += h
        gm = np.tile(g0, (len(wf), 1)); gm[:, k] -= h
        _, ep, _ = simulate(rec, j, wf, noise, g=gp, gate=gate)
        _, em, _ = simulate(rec, j, wf, noise, g=gm, gate=gate)
        D = ((ep[:, :, k] - em[:, :, k]) / (2 * h)).reshape(nW, fd_rollouts, -1).mean(1)
        for name, X in (("score", S), ("deriv", D), ("elig", E)):
            res[name].append(X.mean(0))
            res[name + "_f"].append(rule_filter(X.mean(0)[:, None], rec.lam, rec.beta1)[0])
            if name != "elig":
                for half in (0, 1):
                    Xh = X[half::2].mean(0)
                    halves[name][half].append(Xh)
                    halves[name + "_f"][half].append(rule_filter(Xh[:, None], rec.lam, rec.beta1)[0])
    out = {k: np.array(v).T if np.ndim(v[0]) == 1 else np.array(v) for k, v in res.items()}
    for k, (a, b) in halves.items():
        out[k + "_a"] = np.array(a).T if np.ndim(a[0]) == 1 else np.array(a)
        out[k + "_b"] = np.array(b).T if np.ndim(b[0]) == 1 else np.array(b)
    return out
