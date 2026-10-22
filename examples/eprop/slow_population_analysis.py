"""Task-agnostic analysis of networks with slow excitatory sub-populations (snake_network SlowSpec), untrained.

    python slow_population_analysis.py GENOMES.json OUT_DIR [--layers 5] [--size small] [--names a b c]

For every genome (architecture_evolution genome, built and calibrated as in the search), on recorded Snake frames:
1. intrinsic timescale: per population, the mean autocorrelation of each neuron's spike counts across frames at lags
   1..8, and the lag at which it falls below 1/e (in frames; 30 ms each);
2. memory curve: how much of the input k frames ago (k = 1..8) the population's state (V, adaptation, spike counts)
   recovers beyond what the current frame predicts (reservoir_search.memory_r2), and the current frame itself;
3. job split: how much of a slow population's spike counts a linear map from its own layer's E counts explains
   (1 = a copy of E, low = it carries something E does not);
4. lesion: the same frames again with every connection OUT of the slow populations set to zero; the fields' memory
   curve and current-frame decoding with and without them (what the slow neurons contribute to the readout).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LAGS = tuple(range(1, 9))


def autocorr_timescale(counts, lags=LAGS):
    """Mean over active neurons of the spike-count autocorrelation across frames, per lag; and the first lag where it
    drops below 1/e (linear interpolation; > max lag: nan)."""
    c = counts[:, counts.std(0) > 0]
    if c.shape[1] == 0:
        return [0.0] * len(lags), float("nan")
    z = (c - c.mean(0)) / c.std(0)
    ac = [float(np.mean(np.sum(z[l:] * z[:-l], 0) / (len(z) - l))) for l in lags]
    prev_l, prev_a = 0, 1.0
    for l, a in zip(lags, ac):
        if a < 1 / np.e:
            return ac, float(prev_l + (prev_a - 1 / np.e) / max(prev_a - a, 1e-9) * (l - prev_l))
        prev_l, prev_a = l, a
    return ac, float("nan")


def features(sim, states, counts, k):
    """The population's V, adaptation and spike counts per frame (as the search's measures), standardised."""
    sizes = [int(np.prod(p.shape)) for p in sim.pops]
    off = sum(2 * n for n in sizes[:k])
    v = states[:, off:off + sizes[k]]
    a = np.array([row[k] for row in sim.adapt])
    zs = lambda x: (x - x.mean(0)) / (x.std(0) + 1e-6)
    return np.hstack([zs(v), zs(a), zs(counts[k])])


def lesion_slow(sim):
    """Zero every connection whose source is a slow sub-population."""
    slow = {p for p in getattr(sim.b, "slow", []) if p is not None}
    for c, sg in sim.net.connection_populations.items():
        if c.source() in slow and not c.is_feedback:
            sg.vars["g"].pull_from_device()
            sg.vars["g"].values = sg.vars["g"].values * 0.0
            sg.vars["g"].push_to_device()


def analyse(genome, layers, size, frames, name, work):
    import architecture_evolution as ae
    import reservoir_search as rs
    d = np.load(frames)
    rates = d["rates"]
    calib, ev = rates[:1500], rates[1500:]
    hp = ae.genome_to_hp(genome, layers, size)
    built = rs.build(hp, size, layers)
    out = {"name": name, "genome": genome, "pops": {}}
    with rs.Sim(built, name, work) as sim:
        rs.calibrate(sim, calib, genome["target_hz"], log=lambda s: None)
        counts, states = sim.run(ev, record=True)
        for k, lab in enumerate(sim.labels):
            if lab.endswith("_I"):
                continue
            x = features(sim, states, counts, k)
            ac, tau = autocorr_timescale(counts[k])
            out["pops"][lab] = {"hz": float(1000.0 * counts[k].mean() / rs.STEPS_PER_MOVE),
                                "autocorr": ac, "timescale_frames": tau,
                                "current": rs.ridge_r2(x, ev, ridge=10.0),
                                "memory": rs.memory_r2(x, ev, lags=LAGS)}
        slow = [p for p in getattr(built, "slow", []) if p is not None]
        for s in slow:                                  # job split: S counts from its own layer's E counts
            k = next(i for i, sp in enumerate(built.slow) if sp is s)
            e = built.layers[k][0]
            out["pops"][sim.labels[sim.pops.index(s)]]["from_E"] = rs.ridge_r2(
                counts[sim.pops.index(e)], counts[sim.pops.index(s)], ridge=10.0)
        if slow:                                        # lesion: the fields without the slow populations' output
            lesion_slow(sim)
            counts_l, states_l = sim.run(ev, record=True)
            out["lesion"] = {}
            for k, lab in enumerate(sim.labels):
                if "field" in lab:
                    x = features(sim, states_l, counts_l, k)
                    out["lesion"][lab] = {"current": rs.ridge_r2(x, ev, ridge=10.0), "memory": rs.memory_r2(x, ev, lags=LAGS)}
    return out


def show(r):
    print(f"\n{r['name']}")
    print(f"  {'pop':13s} {'Hz':>5s} {'tau(fr)':>7s} {'cur':>5s}  memory lag 1..8                            from E")
    for lab, p in r["pops"].items():
        mem = " ".join(f"{m:.2f}" for m in p["memory"])
        fe = f"{p['from_E']:.2f}" if "from_E" in p else ""
        print(f"  {lab:13s} {p['hz']:5.1f} {p['timescale_frames']:7.2f} {p['current']:5.2f}  {mem}   {fe}")
    for lab, p in r.get("lesion", {}).items():
        mem = " ".join(f"{m:.2f}" for m in p["memory"])
        print(f"  {lab + ' -S':13s} {'':5s} {'':7s} {p['current']:5.2f}  {mem}   (slow populations' output removed)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("genomes"); ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=5); ap.add_argument("--size", default="small")
    ap.add_argument("--frames", required=True); ap.add_argument("--names", nargs="*")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    gs = json.load(open(args.genomes))
    names = args.names or [f"g{i}" for i in range(len(gs))]
    results = []
    for g, n in zip(gs, names):
        work = out / "work" / n; work.mkdir(parents=True, exist_ok=True)
        r = analyse(g, args.layers, args.size, args.frames, n, work)
        show(r); results.append(r)
        (out / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
