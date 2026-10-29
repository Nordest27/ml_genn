"""Search for conditions where the drift is NOT the weight derivative of the eligibility.

A synthetic neuron (drift_toy.simulate, the generated GeNN update) receives synthetic afferent spike trains whose
statistics are set by a genome: number of afferents, rates and their spread, synchrony, slow rate modulation,
excitatory/inhibitory balance and drive, adaptation strength and noise level. For each genome, on up to 12
synapses with eligibility, the rule-averaged drift (score through the rule's trace and momentum) is compared with
the finite-difference weight derivative of the rule-averaged eligibility; the fitness is the noise-corrected
correlation across those synapses (lower = worse). A genome counts only if both sides are repeatable (split-half
>= 0.8); otherwise it is "unmeasurable" and kept apart.

MAP-Elites: genomes are binned by measured descriptors (mean psi, spread of the number of co-active afferents,
adaptation share of the threshold) and each bin keeps its worst case, so the archive holds distinct failure
"species". Evaluations run in a process pool.

Usage: python drift_search.py out_dir [evaluations] [workers]
"""
import json, os, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drift_toy as dt

ALPHA, RHO, VTH, REFRAC = float(np.exp(-0.1)), float(np.exp(-1 / 300)), 0.61, 3.0
LAM, BETA1 = 0.9925894760, 0.99
T, BURN, WINDOWS = 200, 300, 32

# genome: name -> (low, high, log scale)
GENES = {"n_aff": (8, 150, True), "rate": (2.0, 80.0, True), "rate_spread": (0.0, 1.5, False),
         "sync": (0.0, 0.8, False), "mod_depth": (0.0, 1.0, False), "mod_period": (20.0, 600.0, True),
         "inh_frac": (0.0, 0.6, False), "drive": (0.5, 4.0, True), "beta": (0.001, 0.3, True),
         "sigma": (0.001, 0.05, True)}


def random_genome(rng):
    return {k: float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if log else float(rng.uniform(lo, hi))
            for k, (lo, hi, log) in GENES.items()}


def mutate(genome, rng, scale=0.15):
    out = dict(genome)
    for k, (lo, hi, log) in GENES.items():
        if rng.random() < 0.5:
            if log:
                out[k] = float(np.clip(np.exp(np.log(out[k]) + rng.normal(0, scale * np.log(hi / lo))), lo, hi))
            else:
                out[k] = float(np.clip(out[k] + rng.normal(0, scale * (hi - lo)), lo, hi))
    return out


class Synthetic:
    """The drift_toy.Recording interface for one synthetic neuron (index 0)."""

    def __init__(self, genome, seed):
        rng = np.random.default_rng(seed)
        n = int(round(genome["n_aff"]))
        self.alpha, self.rho, self.vth, self.refrac = ALPHA, RHO, VTH, REFRAC
        self.lam, self.beta1 = LAM, BETA1
        steps = BURN + T + 1
        rates = genome["rate"] * np.exp(genome["rate_spread"] * rng.standard_normal(n) - genome["rate_spread"] ** 2 / 2)
        t = np.arange(steps)
        phase = rng.uniform(0, 2 * np.pi, (WINDOWS, 1))
        mod = 1 + genome["mod_depth"] * np.sin(2 * np.pi * t[None, :] / genome["mod_period"] + phase)   # (W, steps)
        p = np.clip(rates[None, None, :] * mod[:, :, None] / 1000.0, 0, 1)                            # (W, steps, n)
        own = rng.random((WINDOWS, steps, n)) < p * (1 - genome["sync"])
        events = rng.random((WINDOWS, steps, 1)) < genome["rate"] / 1000.0 * mod[:, :, None]
        shared = events & (rng.random((WINDOWS, steps, n)) < genome["sync"])
        spikes = own | shared
        inh = rng.random(n) < genome["inh_frac"]
        # weights: drive sets the mean excitatory input relative to the threshold per membrane time constant
        exc_rate = max(float(rates[~inh].sum()) / 1000.0, 1e-3)
        w_e = genome["drive"] * VTH * (1 - ALPHA) / exc_rate
        g = np.where(inh, -w_e * rng.uniform(0.5, 1.5, n), w_e * rng.uniform(0.5, 1.5, n))
        self.g, self.sig = g, np.full(n, genome["sigma"])
        self.beta = genome["beta"]
        self._spikes = spikes
        zf = np.zeros(spikes.shape)
        prev = np.zeros((WINDOWS, n))
        for k in range(steps):
            prev = ALPHA * prev + spikes[:, k]
            zf[:, k] = prev
        self._zf = zf
        # burn-in: start every window from the neuron's own state after BURN steps of its input
        self.W, self.T1 = WINDOWS, BURN + 1
        self.state = {k: np.zeros((WINDOWS, 1, 1)) for k in ("V", "A", "RefracTime", "Psi", "PsiBar")}
        self.state["Beta"] = np.full((WINDOWS, 1, 1), self.beta)
        self._deliver, self._trace = spikes[:, :BURN + 1], zf[:, :BURN + 1]
        st, _ = dt.simulate(self, 0, np.arange(WINDOWS), rng.standard_normal((WINDOWS, BURN + 1)), return_state=True)
        self.state = {k: v.reshape(WINDOWS, 1, 1) for k, v in st.items()}
        self.state["Beta"] = np.full((WINDOWS, 1, 1), self.beta)
        self._deliver, self._trace = spikes[:, BURN:], zf[:, BURN:]
        self.T1 = T + 1
        self.target_group = "synthetic"
        self.groups = [dict(name="synthetic", post=np.zeros(n, dtype=int))]

    def afferents(self, j):
        n = len(self.g)
        return np.zeros(n, int), np.arange(n), self.g, self.sig, [("synthetic", k) for k in range(n)]

    def inputs(self, j):
        return self._deliver, self._trace


def _rel(a, b):
    r = np.corrcoef(np.ravel(a), np.ravel(b))[0, 1]
    return 2 * r / (1 + r) if np.isfinite(r) else np.nan


def _precision(out, name):
    """Median over synapses of |value| / standard error, the standard error from the two halves."""
    a, b = out[name + "_f_a"], out[name + "_f_b"]
    se = np.abs(a - b) / 2 + 1e-12
    return float(np.median(np.abs((a + b) / 2) / se))


def evaluate(args):
    genome, seed = args
    t0 = time.perf_counter()
    try:
        syn = Synthetic(genome, seed)
        rng = np.random.default_rng(seed + 1)
        # synapses with eligibility: presynaptic activity in the window and a non-silent neuron
        s, e, v = dt.simulate(syn, 0, np.repeat(np.arange(WINDOWS), 4), rng.standard_normal((WINDOWS * 4, T + 1)))
        elig = np.abs(e).mean((0, 1))
        cand = np.flatnonzero(elig > 0)
        desc = descriptors(syn, e)
        if len(cand) < 4:
            return dict(genome=genome, seed=seed, status="no eligibility", desc=desc, time=time.perf_counter() - t0)
        sel = np.sort(rng.choice(cand, size=min(12, len(cand)), replace=False))
        h = 0.2 * float(np.sqrt(np.mean(syn.g ** 2)))
        # screening pass, then more samples only when that can reach the threshold (precision grows like sqrt(n):
        # 6x the samples gives ~2.4x, so a condition below ~1.25 standard errors cannot reach 3)
        for rollouts, fd in ((32, 12), (192, 72)):
            out = dt.drift(syn, sel, h, rollouts=rollouts, fd_rollouts=fd, seed=seed + 2 + rollouts)
            rs, rd = _rel(out["score_f_a"], out["score_f_b"]), _rel(out["deriv_f_a"], out["deriv_f_b"])
            zs, zd = _precision(out, "score"), _precision(out, "deriv")
            if (zs >= 3 and zd >= 3) or min(zs, zd) < 1.25:
                break
        c = float(np.corrcoef(out["deriv_f"], out["score_f"])[0, 1])
        ce = float(np.corrcoef(out["elig_f"], out["score_f"])[0, 1])
        # measurable: every side's per-synapse value is >= 3 of its own standard errors (median over synapses);
        # then the raw correlation is the fitness (split-half reliability across synapses also falls when the true
        # values are merely similar, so it is reported but not used)
        status = "ok" if (zs >= 3 and zd >= 3) else "unmeasurable"
        return dict(genome=genome, seed=seed, status=status, corr=c, corrected=c if status == "ok" else np.nan,
                    corr_elig=ce, z_score=float(zs), z_deriv=float(zd), rel_score=float(rs), rel_deriv=float(rd),
                    desc=desc, n_syn=len(sel), time=time.perf_counter() - t0)
    except Exception as ex:          # keep the search running; record what failed
        return dict(genome=genome, seed=seed, status=f"error: {ex!r}", time=time.perf_counter() - t0)


def descriptors(syn, e):
    d = syn._deliver
    active = d.sum(-1)                                       # co-active afferents per step
    co = [active[d[:, :, k]].mean() for k in range(d.shape[-1]) if d[:, :, k].any()]
    return dict(psi=float(syn.state["Psi"].mean() + 1e-6), coactive_cv=float(np.std(co) / (np.mean(co) + 1e-9)),
                adapt=float(syn.beta * syn.state["A"].mean() / VTH))


BINS = {"psi": [0.01, 0.05, 0.15], "coactive_cv": [0.1, 0.3, 0.6], "adapt": [0.01, 0.05, 0.2]}


def niche(desc):
    return tuple(int(np.searchsorted(BINS[k], desc[k])) for k in BINS)


def main(out_dir, evaluations=400, workers=6):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    # archive: measurable conditions, worst correlation per niche (relation failures); weak: unmeasurable
    # conditions, lowest precision per niche (estimation failures: the drift is buried in its own noise)
    archive, weak, log = {}, {}, open(os.path.join(out_dir, "evaluations.jsonl"), "a")
    n_init = min(96, evaluations)
    done = 0
    with Pool(workers) as pool:
        while done < evaluations:
            batch = workers * 4
            if done < n_init or not archive:
                genomes = [random_genome(rng) for _ in range(batch)]
            else:   # mostly refine relation failures, some estimation failures, some fresh exploration
                pools = [list(archive.values())] * 6 + ([list(weak.values())] * 2 if weak else [])
                genomes = []
                for _ in range(batch):
                    u = rng.random()
                    if u < 0.15:
                        genomes.append(random_genome(rng))
                    else:
                        pool_ = pools[rng.integers(len(pools))]
                        genomes.append(mutate(pool_[rng.integers(len(pool_))]["genome"], rng))
            for r in pool.imap_unordered(evaluate, [(g, int(rng.integers(1 << 30))) for g in genomes]):
                done += 1
                log.write(json.dumps(r) + "\n"); log.flush()
                if r["status"] == "ok":
                    key = niche(r["desc"])
                    if key not in archive or r["corrected"] < archive[key]["corrected"]:
                        archive[key] = r
                elif r["status"] == "unmeasurable":
                    key = niche(r["desc"])
                    z = min(r["z_score"], r["z_deriv"])
                    if key not in weak or z < min(weak[key]["z_score"], weak[key]["z_deriv"]):
                        weak[key] = r
            worst = sorted(archive.values(), key=lambda r: r["corrected"])[:3]
            print(f"{done} evaluations, {len(archive)} niches | worst corrected corr: "
                  + ", ".join(f"{r['corrected']:.2f}" for r in worst), flush=True)
            json.dump({str(k): v for k, v in archive.items()}, open(os.path.join(out_dir, "archive.json"), "w"),
                      indent=1)
            json.dump({str(k): v for k, v in weak.items()}, open(os.path.join(out_dir, "weak.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], *(int(x) for x in sys.argv[2:]))
