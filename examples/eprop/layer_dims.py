"""Dimensionality and task information per layer, from probe recordings (outputs/*_probe.npz).

    python layer_dims.py RUN_DIR [RUN_DIR ...]     (or probe files)

Per hidden population and window (first / last recorded moves):
- eff.dim: participation ratio of the membrane potentials (effective number of dimensions the population's
  activity uses; at most its number of neurons);
- frac: eff.dim / number of neurons;
- decode: mean decoding accuracy above chance (food side y/x, danger in 4 directions) from that population alone;
- active: fraction of neurons whose potential varies at all (silent layers have eff.dim ~ 0 and nothing to decode).
Ideal propagation: eff.dim and decode stay roughly constant from L1 to the last layer.
"""
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_analyse import label_of, per_population                  # noqa: E402


def participation_ratio(v):
    v = v[:, v.std(0) > 1e-6]
    if v.shape[1] < 2:
        return 0.0
    ev = np.clip(np.linalg.eigvalsh(np.cov(v, rowvar=False)), 0, None)
    return float(ev.sum() ** 2 / (np.square(ev).sum() + 1e-30))


def per_layer(x, y, sizes, labels):
    decode = per_population(x, y, sizes, labels)
    out, off = [], 0
    for n, lab in zip(sizes, labels):
        v = x[:, off:off + n]; off += 2 * n                         # [V, refractory] per population
        out.append((str(lab), n, participation_ratio(v), float(np.mean(v.std(0) > 1e-6)), decode[str(lab)]))
    return out


def main(args):
    files = []
    for a in args:
        if Path(a).is_dir():          # a study's runs directory, or a single run directory
            files += glob.glob(str(Path(a) / "*/outputs/*_probe.npz")) + glob.glob(str(Path(a) / "outputs/*_probe.npz"))
        else:
            files.append(a)
    for f in sorted(files, key=label_of):
        d = np.load(f)
        sizes, labels = [int(n) for n in d["sizes"]], list(d["labels"])
        for w in ("first", "last"):
            x, y = d[f"{w}_x"], d[f"{w}_y"]
            if len(x) == 0:
                continue
            print(f"\n{label_of(f)}  [{w} window]")
            print(f"  {'population':14s} {'n':>5s} {'eff.dim':>8s} {'frac':>6s} {'active':>7s} {'decode':>7s}")
            for lab, n, pr, act, dec in per_layer(x, y, sizes, labels):
                print(f"  {lab:14s} {n:5d} {pr:8.2f} {pr / n:6.3f} {act:7.2f} {dec:+7.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
