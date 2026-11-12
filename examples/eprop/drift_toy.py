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


def simulate(rec, j, w, noise, g=None, steps=None, gate="drift", return_state=False, noise2=None, sigma2=None,
             th_true=None, th_gate=None, gate_table=None, track=False, eif=None, resonator=None, plateau=None,
             parts_k=None):
    """Replay target neuron j in window w (an int, or one window per rollout as an (R,) array) for R rollouts.
    noise: (R, T+1) Noise1 samples (noise[:, t] is drawn in step t and delivered in step t+1); g: (R, n_aff)
    weights (default: the recorded ones). gate: "drift" (psi - psibar with the code's psi), "psi" (psi alone), or
    a tailed gate ("exp<s>" / "fs<s>", see gate_shape) used, with its own running mean, only in the score and the
    eligibility (the neuron's dynamics are unchanged). Model-free gates measured with Stein's identity:
    "stein2": gate = the neuron's last spike x its own total noise trace H / Var(H) (the same noise the score uses:
    the score then subtracts E[gate-free part] x Cov(H, NoiseTrace), the second-order Stein correction);
    "stein2x": gate = last spike x the trace of a second, independent per-neuron noise (noise2, s.d. sigma2, added
    to the membrane) / its variance (no correction). Both estimate the drift of the true eligibility (the neuron's
    measured spike sensitivity x presynaptic trace).
    th_true: the neuron's real threshold (before adaptation; (R,) or scalar; default the nominal Vthresh), hidden
    from the gate; th_gate: the threshold the gate is centred on (default th_true). The gate keeps the triangle's
    width Vthresh (a design choice). gate_table: (edges, values), a measured gate as a function of the noise-free
    voltage relative to th_gate (the bins option); replaces the triangle.
    eif: {"delta": D, "vt": VT, "vpeak": VP} - exponential integrate-and-fire upswing,
    V <- Alpha V + (1 - Alpha) D exp((V - VT) / D) + I, spike at V >= VP + Beta A (relative reset by Vthresh); the
    gate still assumes the nominal hard threshold.
    resonator: {"period": P, "gain": k} - resonate-and-fire membrane: (V, W) <- Alpha R(2 pi / P) (V, W) + (k I, 0),
    threshold and
    relative reset on V as usual. With track, the e-prop eligibilities for the spike of each step are also returned
    (aligned with it): psi x a single decaying input trace (the unknown-model eligibility) and psi x the V component
    of the eligibility vector propagated with the same damped rotation (the derived, known-model eligibility).
    plateau: {"gp": g, "thp": theta, "s": s, "tau": tau, "gain": k} - bistable persistent current P:
    P <- P + (sigmoid((V - theta) / s) - P) / tau, V <- Alpha V + g P + k I. With track the e-prop eligibilities are
    returned as for the resonator: single decaying trace vs the derived (V, P) eligibility vector (Jacobian with the
    sigmoid's slope), plus the plateau variable itself.
    track: also return, per step, the noise-free voltage relative to the real (adaptive) spike threshold,
    x = V - H - threshold (H = the neuron's own noise trace) before the
    spike decision, the Stein sensitivity sample w = z H / Var(H), and the spikes, for learning the centre online.
    Returns per step t = 1..T: score s, eligibility e for every afferent (R, T, n_aff) and the voltage after the
    step (R, T) (and, with track, (x, w, z), each (R, T)).
    parts_k: afferent indices; also returns a dict of per-step parts for them (to build other estimators offline):
    "xi" the normalised noise trace kappa * NoiseTrace / SynSig^2 and "pre" the presynaptic trace at the synapse
    stage (R, T, len(parts_k)), "psi" / "psibar" the gate values used there and "z" the spikes of the step (R, T)."""
    src, pre, g0, sig, _ = rec.afferents(j)
    deliver, trace = rec.inputs(j)
    R = noise.shape[0]
    T = (rec.T1 - 1) if steps is None else steps
    g = np.broadcast_to(g0, (R, len(g0))) if g is None else g
    w = np.broadcast_to(np.asarray(w), (R,))
    st = {k: rec.state[k][w, 0, j].astype(np.float64) for k in ("V", "A", "RefracTime", "Psi", "PsiBar")}
    stein = gate in ("stein2", "stein2x")
    tailed = gate not in ("drift", "psi") and not stein
    if stein:
        R0 = noise.shape[0]
        zlast = np.zeros(R0); H = np.zeros(R0); VarH = np.full(R0, 1e-12)
        C = None                                   # Cov(H, NoiseTrace) per synapse ("stein2")
        if gate == "stein2x" and (noise2 is None or sigma2 is None):
            raise ValueError("stein2x needs noise2 and sigma2")
    if tailed:   # the tailed gate and its running mean, started from the recorded voltage
        x0 = (st["V"] - (rec.vth + rec.state["Beta"][w, 0, j] * st["A"])) / rec.vth
        st["PsiT"] = np.where(st["RefracTime"] > 0, 0.0, (0.3 / rec.vth) * gate_shape(x0, gate))
        st["PsiTBar"] = st["PsiT"].copy()
    beta = rec.state["Beta"][w, 0, j].astype(np.float64)
    a, kappa = rec.alpha, 1.0 - rec.alpha ** 2
    R1 = noise.shape[0]
    th0 = np.broadcast_to(np.asarray(rec.vth if th_true is None else th_true, dtype=np.float64), (R1,))
    thg = th0 if th_gate is None else np.broadcast_to(np.asarray(th_gate, dtype=np.float64), (R1,))
    if track or gate_table is not None or th_true is not None or th_gate is not None:
        Ht = np.zeros(R1); VarHt = np.full(R1, 1e-12)
    x_out, w_out, z_out = [], [], []
    if plateau is not None:
        st["P"] = np.zeros(R1)
        epsV = np.zeros((R1, len(g0))); epsP = np.zeros((R1, len(g0))); eps1 = np.zeros((R1, len(g0)))
        e1_out, e2_out, p_out = [], [], []
    if resonator is not None:
        th_r = 2 * np.pi / float(resonator["period"])
        cr, sr = np.cos(th_r), np.sin(th_r)
        st["W"] = np.zeros(R1)
        epsV = np.zeros((R1, len(g0))); epsW = np.zeros((R1, len(g0))); eps1 = np.zeros((R1, len(g0)))
        e1_out, e2_out = [], []
    nt = np.zeros((R, len(g0)))
    s_out, e_out, v_out = [], [], []
    parts = {k: [] for k in ("xi", "pre", "psi", "psibar", "z")} if parts_k is not None else None
    for t in range(1, T + 1):
        if parts is not None:   # the synapse stage of step t (before the noise trace decays)
            pk = np.asarray(parts_k)
            parts["xi"].append(kappa * nt[:, pk] / (sig[pk] * sig[pk] + 1e-6))
            parts["pre"].append(np.broadcast_to(trace[w, t - 1][..., pk], (R, len(pk))).copy())
            parts["psi"].append(st["Psi"].copy()); parts["psibar"].append(st["PsiBar"].copy())
        # 1. synapse dynamics (psi and psibar from the previous neuron update, presynaptic trace after step t-1)
        if stein:
            gt = zlast * H / VarH
            e = gt[:, None] * trace[w, t - 1]
            score = kappa * e * nt / (sig * sig + 1e-6)
            if gate == "stein2" and C is not None:   # second-order Stein: remove the E[f] Cov(H, NT) part
                score = score - kappa * (zlast / VarH)[:, None] * trace[w, t - 1] * C / (sig * sig + 1e-6)
            s_out.append(score); e_out.append(e)
        else:
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
        if track or gate_table is not None or th_true is not None or th_gate is not None:
            S_t = (sig[None, :] * d).sum(1)
            Ht = a * Ht + noise[:, t - 1] * S_t; VarHt = a * a * VarHt + S_t * S_t
        if stein:
            S = (sig[None, :] * d).sum(1)                  # the shared sample's total weight this step
            if gate == "stein2":
                C = (a * a * C if C is not None else 0.0) + (sig[None, :] * d) * S[:, None]
                H = a * H + noise[:, t - 1] * S
                VarH = a * a * VarH + S * S
            else:
                eta = sigma2 * noise2[:, t - 1]
                isyn = isyn + eta
                H = a * H + eta
                VarH = a * a * VarH + sigma2 * sigma2
        # 3. neuron update
        if plateau is not None:
            pg, pth, ps, ptau = (float(plateau[k]) for k in ("gp", "thp", "s", "tau"))
            kin = float(plateau.get("gain", 1.0))
            v_old, p_old = st["V"], st["P"]
            sgm = 1.0 / (1.0 + np.exp(-(v_old - pth) / ps))
            st["P"] = p_old + (sgm - p_old) / ptau
            st["V"] = a * v_old + pg * p_old + kin * isyn
            dsg = (sgm * (1 - sgm) / ps)[:, None]               # the sigmoid's slope at the current voltage
            ev_old = epsV
            epsV = a * ev_old + pg * epsP + kin * d
            epsP = (1 - 1 / ptau) * epsP + dsg * ev_old / ptau
            eps1 = a * eps1 + kin * d
        elif resonator is not None:
            v_old, w_old = st["V"], st["W"]
            st["V"] = a * (cr * v_old - sr * w_old) + float(resonator.get("gain", 1.0)) * isyn
            st["W"] = a * (sr * v_old + cr * w_old)
            # eligibility traces of the input (per synapse): single decaying vs the derived damped rotation
            ev_old = epsV
            gr = float(resonator.get("gain", 1.0))
            epsV = a * (cr * ev_old - sr * epsW) + gr * d
            epsW = a * (sr * ev_old + cr * epsW)
            eps1 = a * eps1 + gr * d
        elif eif is not None:   # soft threshold and peak move with adaptation, like the hard threshold
            ba = beta * st["A"]
            up = (1 - a) * eif["delta"] * np.exp(np.minimum((st["V"] - eif["vt"] - ba) / eif["delta"], 20.0))
            # the runaway is cut at the spike peak (discrete EIF): it spikes this step instead of diverging
            st["V"] = np.minimum(a * st["V"] + up + isyn, eif["vpeak"] + ba + 0.05)
        else:
            st["V"] = a * st["V"] + isyn
        st["A"] *= rec.rho
        ref = st["RefracTime"]
        ref = np.where(ref > 0, ref - 1.0, ref)
        th = (eif["vpeak"] if eif is not None else th0) + beta * st["A"]   # the neuron's real (adaptive) threshold
        thg_t = thg + beta * st["A"]                       # where the gate is centred
        if gate_table is not None:
            edges, vals = gate_table
            xr = st["V"] - Ht - thg_t                     # noise-free voltage relative to the gate's centre
            psi = vals[np.clip(np.searchsorted(edges, xr) - 1, 0, len(vals) - 1)]
        else:
            psi = (1.0 / rec.vth) * 0.3 * np.maximum(0.0, 1.0 - np.abs((st["V"] - thg_t) / rec.vth))
        st["Psi"] = np.where(ref > 0, 0.0, psi)
        st["PsiBar"] = 0.99 * st["PsiBar"] + 0.01 * st["Psi"]
        if tailed:
            st["PsiT"] = np.where(ref > 0, 0.0, (0.3 / rec.vth) * gate_shape((st["V"] - th) / rec.vth, gate))
            st["PsiTBar"] = 0.99 * st["PsiTBar"] + 0.01 * st["PsiT"]
        spk = (st["V"] >= th) & (ref <= 0)
        if parts is not None:
            parts["z"].append(spk.astype(np.float64))
        if (resonator is not None or plateau is not None) and track:
            psi_now = np.where(ref > 0, 0.0, (1.0 / rec.vth) * 0.3 * np.maximum(0.0, 1.0 - np.abs((st["V"] - th) / rec.vth)))
            e1_out.append(psi_now[:, None] * eps1); e2_out.append(psi_now[:, None] * epsV)
        if track:
            x_out.append(st["V"] - Ht - th); z_out.append(spk.astype(np.float64))   # relative to the real threshold
            w_out.append(np.where(ref > 0, 0.0, spk * Ht / VarHt))
        st["V"] = np.where(spk, st["V"] - th0, st["V"])
        st["A"] = np.where(spk, st["A"] + 1.0, st["A"])
        st["RefracTime"] = np.where(spk, rec.refrac, ref)
        if stein:
            zlast = spk.astype(np.float64)
        if plateau is not None and track:
            p_out.append(st["P"].copy())
        v_out.append(st["V"].copy())
    if parts is not None:
        return np.stack(s_out, 1), np.stack(e_out, 1), np.stack(v_out, 1), {k: np.stack(v, 1) for k, v in parts.items()}
    if return_state:
        return st, nt
    if track:
        extra = (np.stack(x_out, 1), np.stack(w_out, 1), np.stack(z_out, 1))
        if resonator is not None or plateau is not None:
            extra = extra + (np.stack(e1_out, 1), np.stack(e2_out, 1))
        if plateau is not None:
            extra = extra + (np.stack(p_out, 1),)
        return np.stack(s_out, 1), np.stack(e_out, 1), np.stack(v_out, 1), extra
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
