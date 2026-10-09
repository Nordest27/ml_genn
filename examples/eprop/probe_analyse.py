"""Linear probes on Snake's hidden representation (outputs/*_probe.npz from the "probe" config option).

For each run and window (first / last recorded moves): a linear least-squares one-vs-rest classifier from the hidden
state (membrane potentials + refractory flags of all hidden populations) to each task variable, trained on the first
70% of the window and tested on the last 30% (time order, no shuffling). Reported: test accuracy and the
majority-class rate (chance). Features created by learning show up as accuracy rising from the first to the last
window, beyond what a frozen reservoir achieves.
Usage: python3 probe_analyse.py <study runs dir or probe files ...>
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np


def targets(y):
    """Classes per variable: apple side (-1/0/+1) in y and x, danger in 4 directions (0/1), heading (0-3)."""
    return {"apple_y": np.sign(y[:, 0]).astype(int), "apple_x": np.sign(y[:, 1]).astype(int),
            "danger_l": y[:, 2].astype(int), "danger_u": y[:, 3].astype(int), "danger_r": y[:, 4].astype(int),
            "danger_d": y[:, 5].astype(int), "heading": y[:, 6].astype(int)}


def probe(x, labels, ridge=1.0):
    n = len(labels)
    cut = int(0.7 * n)
    if cut < 50 or n - cut < 20:
        return float("nan"), float("nan")
    mu, sd = x[:cut].mean(0), x[:cut].std(0) + 1e-6
    xs = np.hstack([(x - mu) / sd, np.ones((n, 1))])
    classes = np.unique(labels[:cut])
    if len(classes) < 2:
        return float("nan"), float("nan")
    onehot = (labels[:cut, None] == classes[None]).astype(float)
    a = xs[:cut]
    w = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]), a.T @ onehot)
    pred = classes[np.argmax(xs[cut:] @ w, 1)]
    acc = float((pred == labels[cut:]).mean())
    chance = float(max((labels[cut:] == c).mean() for c in np.unique(labels[cut:])))
    return acc, chance


def dimensionality(x, sizes):
    """Participation ratio of the hidden state's covariance (effective number of dimensions used), and the
    fraction of neurons whose membrane potential barely varies (frozen: s.d. below 1% of the median s.d.)."""
    # the recorded state is [V, refractory] per population; keep the membrane potentials only
    cols, off = [], 0
    for n in sizes:
        cols.append(np.arange(off, off + n)); off += 2 * n
    v = x[:, np.concatenate(cols)]
    ev = np.clip(np.linalg.eigvalsh(np.cov(v, rowvar=False)), 0, None)
    pr = float(ev.sum() ** 2 / (np.square(ev).sum() + 1e-30))
    sd = v.std(0)
    frozen = float((sd < 0.01 * np.median(sd[sd > 0]) if np.any(sd > 0) else np.ones_like(sd, bool)).mean())
    return pr, frozen


def per_population(x, y, sizes, labels):
    """Mean decoding accuracy above chance (food side y/x, danger in 4 directions) from each population alone."""
    out, off = {}, 0
    t = targets(y)
    for n, lab in zip(sizes, labels):
        block = x[:, off:off + 2 * n]; off += 2 * n
        gains = []
        for key in ("apple_y", "apple_x", "danger_l", "danger_u", "danger_r", "danger_d"):
            acc, ch = probe(block, t[key])
            if np.isfinite(acc):
                gains.append(acc - ch)
        out[str(lab)] = float(np.mean(gains)) if gains else float("nan")
    return out


def label_of(path):
    cfg = Path(path).parent.parent / "config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        return c.get("_meta", {}).get("variant", Path(path).parent.parent.name) + f" s{c.get('seed')}"
    return Path(path).name


def main(args):
    files = []
    for a in args:
        files += glob.glob(str(Path(a) / "*/outputs/*_probe.npz")) if Path(a).is_dir() else [a]
    names = list(targets(np.zeros((1, 7))))
    print(f"{'run':28s} {'window':6s} " + " ".join(f"{n:>13s}" for n in names) + f" {'eff.dim':>8s} {'frozen':>7s}")
    for f in sorted(files, key=label_of):
        d = np.load(f)
        for w in ("first", "last"):
            x, y = d[f"{w}_x"], d[f"{w}_y"]
            if len(x) == 0:
                continue
            cells = []
            for n, lab in targets(y).items():
                acc, ch = probe(x, lab)
                cells.append(f"{acc:.2f} ({ch:.2f})")
            pr, frozen = dimensionality(x, [int(n) for n in d["sizes"]])
            print(f"{label_of(f):28s} {w:6s} " + " ".join(f"{c:>13s}" for c in cells) + f" {pr:8.1f} {frozen:7.2f}")
    print("\nper population: mean decoding accuracy above chance (food side, danger), first -> last window")
    for f in sorted(files, key=label_of):
        d = np.load(f)
        sizes, labels = [int(n) for n in d["sizes"]], list(d["labels"])
        if len(d["first_x"]) == 0:
            continue
        a = per_population(d["first_x"], d["first_y"], sizes, labels)
        b = per_population(d["last_x"], d["last_y"], sizes, labels)
        print(f"{label_of(f):28s} " + "  ".join(f"{k}: {a[k]:+.2f}->{b[k]:+.2f}" for k in a))
    print("\naccuracy (majority-class chance); learned features: last > first, and above a frozen reservoir")
    print("eff.dim: participation ratio of the membrane potentials; frozen: fraction of neurons with ~constant V")


if __name__ == "__main__":
    main(sys.argv[1:])
