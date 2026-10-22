"""Lateral transport through depth, untrained networks: is the input still where it entered, or has it moved?

    python lateral_transport.py GENOMES.json OUT_DIR --frames FRAMES.npz [--layers 5] [--size full] [--names ...]

For every genome (built and calibrated as in architecture_evolution), per hidden E population and the value field:
- local: the input at each sheet position (its channels) decoded from the population's neurons in the 3 x 3 cells
  around that same position only (V and spike counts per frame; ridge, held-out R2), averaged over positions;
- global: the input decoded from the whole population (the search's input_r2 measure, top principal components).
Information that moves sideways without being lost keeps global high while local falls with depth.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def held_out_r2(x, y, ridge=1.0, cut=0.7):
    n = len(x); k = int(cut * n)
    mu, sd = x[:k].mean(0), x[:k].std(0) + 1e-6
    xs = np.hstack([(x - mu) / sd, np.ones((n, 1))])
    ym = y[:k].mean(0)
    w = np.linalg.solve(xs[:k].T @ xs[:k] + ridge * np.eye(xs.shape[1]), xs[:k].T @ (y[:k] - ym))
    resid = y[k:] - ym - xs[k:] @ w
    total = ((y[k:] - y[k:].mean(0)) ** 2).sum()
    return float(1.0 - (resid ** 2).sum() / max(total, 1e-12))


def local_r2(feats, shape, rates, in_hw=(20, 20), in_c=3, radius=1, max_positions=120, seed=0):
    """Mean over input positions of the held-out R2 of that position's channels from the population's neurons within
    `radius` cells of the same position (mapped onto the population's grid)."""
    h, w, c = shape
    f = feats.reshape(len(feats), h, w, c, -1)
    target = rates.reshape(len(rates), in_hw[0], in_hw[1], in_c)
    rng = np.random.default_rng(seed)
    pos = [(y, x) for y in range(in_hw[0]) for x in range(in_hw[1])]
    pos = [pos[i] for i in rng.permutation(len(pos))[:max_positions]]
    out = []
    for y, x in pos:
        t = target[:, y, x, :]
        if t.std() == 0:
            continue
        py, px = int((y + 0.5) / in_hw[0] * h), int((x + 0.5) / in_hw[1] * w)
        rows = [(py + d) % h for d in range(-radius, radius + 1)]
        cols = [(px + d) % w for d in range(-radius, radius + 1)]
        xf = f[:, rows][:, :, cols].reshape(len(f), -1)
        out.append(held_out_r2(xf, t))
    return float(np.mean(out)) if out else float("nan")


def analyse(genome, layers, size, frames, name, work, backend="cuda"):
    import architecture_evolution as ae
    import reservoir_search as rs
    d = np.load(frames)
    rates = d["rates"]
    calib, ev = rates[:1500], rates[1500:]
    built = rs.build(ae.genome_to_hp(genome, layers, size), size, layers)
    res = {"name": name, "pops": {}}
    with rs.Sim(built, name, work, backend=backend) as sim:
        rs.calibrate(sim, calib, genome["target_hz"], log=lambda s: None)
        counts, states = sim.run(ev, record=True)
        sizes = [int(np.prod(p.shape)) for p in sim.pops]
        off = 0
        for k, (lab, n) in enumerate(zip(sim.labels, sizes)):
            v = states[:, off:off + n]; off += 2 * n
            if not (lab.endswith("_E") or lab == "value_field"):
                continue
            feats = np.stack([v, counts[k]], axis=-1)              # per neuron: V and the frame's spike count
            shape = tuple(sim.pops[k].shape)
            glob = rs.ridge_r2(np.hstack([(v - v.mean(0)) / (v.std(0) + 1e-6),
                                          (counts[k] - counts[k].mean(0)) / (counts[k].std(0) + 1e-6)]), ev, ridge=10.0)
            res["pops"][lab] = {"local": local_r2(feats, shape, ev), "global": glob}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("genomes"); ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=5); ap.add_argument("--size", default="full")
    ap.add_argument("--frames", required=True); ap.add_argument("--names", nargs="*")
    ap.add_argument("--backend", default="cuda", choices=["cuda", "single_threaded_cpu"])
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    gs = json.load(open(args.genomes))
    names = args.names or [f"g{i}" for i in range(len(gs))]
    results = []
    for g, n in zip(gs, names):
        work = out / "work" / n; work.mkdir(parents=True, exist_ok=True)
        r = analyse(g, args.layers, args.size, args.frames, n, work, args.backend)
        print(f"\n{n}\n  {'pop':12s} {'local':>6s} {'global':>7s}")
        for lab, p in r["pops"].items():
            print(f"  {lab:12s} {p['local']:6.2f} {p['global']:7.2f}", flush=True)
        results.append(r)
        (out / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
