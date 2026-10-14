"""Unattended architecture search ("evolution") for Snake networks, meant to run for hours on one GPU.

    python architecture_evolution.py OUT_DIR --layers 1 --size full --backend cuda --hours 24
    python architecture_evolution.py OUT_DIR --report            # ranking so far, and export the best configs

Every candidate is an untrained network built from a "genome" of connectivity settings (local toroidal mode). Each
one is compiled, calibrated layer by layer to the target rate (non-local pre-processing; reservoir_search.py), and
driven with recorded Snake frames (all learning off). The measures are task-agnostic:
- per layer: effective dimensionality, and how much of the input frame a linear readout recovers;
- for the fields: the same;
- rates, and psi (how often neurons sit near threshold).

Score (higher is better; every metric is stored, so --report can re-rank with other weights):
    0.4 * input_kept + 0.3 * dim_kept + 0.3 * fields_quality - penalties
- input_kept: input reconstruction of the last layer / of layer 1 (1 layer: layer 1's own reconstruction);
- dim_kept: mean over layers 2.. of dim_k / dim_1, capped at 1.2 (1 layer: layer 1 dim / 15, capped at 1);
- fields_quality: fields' input reconstruction / layer 1's, plus their dimensionality / layer 1's, halved;
- penalties: any layer below 2 Hz or above 80 Hz after calibration, or more than half its neurons almost never
  near threshold (psi ~ 0).

Disk: each candidate's GeNN build folder is deleted once its results are saved.
GPU: needs a CUDA-enabled GeNN (CUDA_PATH set). One candidate at a time, in its own process.

Search: the first --random candidates are sampled at random from the space below; after that, each candidate is
either a mutation of one of the current top 5 (70%: 1-3 genes changed) or a fresh random sample (30%). Candidates run
one after another, each in its own process (GPU memory is released between candidates). Results are appended to
OUT_DIR/results.jsonl as they finish, so the search can be stopped and resumed at any time (finished candidates are
never rerun). Stop it with Ctrl-C.
"""
import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# gene: (kind, low, high) for continuous (log-uniform when kind == "log"), or ("choice", options)
SPACE = {
    "sigma_cells": ("log", 0.75, 3.0),          # recurrent / within-layer neighbourhood (cells of the sheet)
    "p_max": ("lin", 0.3, 1.0),                 # density of the neighbourhood
    "input_sigma_cells": ("log", 1.0, 4.0),     # input -> layer 1 neighbourhood
    "inhibition_scale": ("lin", 2.0, 5.0),      # inhibitory weight factor (E/I balance)
    "ff_sigma": ("log", 0.3, 2.0),              # layer k -> k+1 neighbourhood (only with >1 layer)
    "field_sigma": ("log", 0.3, 2.5),           # last layer -> fields neighbourhood
    "field_shape": ("choice", ["inhibitory", "excitatory"]),   # fields sized like the I or the E grid
    "target_hz": ("choice", [10.0, 20.0, 30.0]),               # calibration target
}


def sample_gene(rng, spec):
    kind = spec[0]
    if kind == "choice":
        return rng.choice(spec[1])
    lo, hi = spec[1], spec[2]
    if kind == "log":
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    return float(rng.uniform(lo, hi))


def random_genome(rng, layers):
    g = {k: sample_gene(rng, s) for k, s in SPACE.items()}
    if layers == 1:
        g.pop("ff_sigma")
    return g


def mutate(rng, genome, layers):
    g = dict(genome)
    keys = [k for k in SPACE if layers > 1 or k != "ff_sigma"]
    for k in rng.sample(keys, rng.randint(1, 3)):
        spec = SPACE[k]
        if spec[0] == "choice":
            g[k] = rng.choice(spec[1])
        else:                                   # multiplicative jitter within the bounds
            lo, hi = spec[1], spec[2]
            g[k] = float(min(hi, max(lo, g[k] * math.exp(rng.gauss(0.0, 0.25)))))
    return g


def genome_to_hp(g, layers, size):
    hid_e, hid_i = (20, 15) if size == "full" else (6, 4)
    local = {"sigma_cells": round(g["sigma_cells"], 3), "p_max": round(g["p_max"], 3),
             "input_sigma_cells": round(g["input_sigma_cells"], 3), "inhibition_scale": round(g["inhibition_scale"], 3)}
    if size != "full":
        local["input_sigma_ref"] = "source"
    net = {"fields": {"conn": {"sigma_cells": round(g["field_sigma"], 3)},
                      "shape": hid_i if g["field_shape"] == "inhibitory" else hid_e}}
    if layers > 1:
        net["layers"] = [{"to_next": {"sigma_cells": round(g["ff_sigma"], 3)}}] * (layers - 1) + [{}]
    return {"toroidal": "local", "local": local, "network": net}


def score(result, layers):
    e = [p for p in result["per_pop"] if p["pop"].endswith("_E")]
    f = [p for p in result["per_pop"] if "field" in p["pop"]]
    l1 = e[0]
    if layers > 1:
        input_kept = e[-1]["input_r2"] / max(l1["input_r2"], 1e-6)
        dim_kept = float(np.mean([min(p["dim"] / max(l1["dim"], 1e-6), 1.2) for p in e[1:]]))
    else:
        input_kept = l1["input_r2"]
        dim_kept = min(l1["dim"] / 15.0, 1.0)
    fq = 0.0
    if f:
        fq = 0.5 * (np.mean([p["input_r2"] for p in f]) / max(l1["input_r2"], 1e-6)
                    + np.mean([p["dim"] for p in f]) / max(l1["dim"], 1e-6))
    pen = 0.0
    for p in result["per_pop"]:
        if not 2.0 <= p["hz"] <= 80.0:
            pen += 0.2
        if p.get("psi_dead", 0.0) > 0.5:
            pen += 0.1
    return float(0.4 * input_kept + 0.3 * dim_kept + 0.3 * fq - pen), {
        "input_kept": input_kept, "dim_kept": dim_kept, "fields_quality": fq, "penalty": pen}


def evaluate_genome(args):
    """Runs in its own process (released afterwards): build, calibrate, evaluate one genome."""
    idx, genome, layers, size, backend, frames_path, out_dir = args
    import reservoir_search as rs
    name = f"c{idx:04d}"
    work = Path(out_dir) / "work" / name
    work.mkdir(parents=True, exist_ok=True)
    d = np.load(frames_path)
    rates, ys = d["rates"], d["ys"]
    hp = genome_to_hp(genome, layers, size)
    t0 = time.time()
    built = rs.build(hp, size, layers)
    with rs.Sim(built, name, work, backend=backend) as sim:
        rs.calibrate(sim, rates[:1500], genome["target_hz"], log=lambda s: None)
        per_pop = rs.evaluate(sim, rates[1500:], ys[1500:])
        factors = dict(sim.factors)
    result = {"idx": idx, "genome": genome, "hp": hp, "per_pop": per_pop,
              "calibration_factors": {str(k): v for k, v in factors.items()},
              "snake_config": rs.snake_config(hp, factors, layers, size), "seconds": round(time.time() - t0, 1)}
    result["score"], result["parts"] = score(result, layers)
    import shutil
    shutil.rmtree(work, ignore_errors=True)                 # the build folder (code, binaries): ~MBs per candidate
    return result


def load_results(out):
    path = out / "results.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def report(out, top=10, export=3):
    res = sorted(load_results(out), key=lambda r: -r["score"])
    if not res:
        print("no results yet"); return
    print(f"{len(res)} candidates; best {top}:")
    print(f"  {'idx':>5s} {'score':>6s} {'input':>6s} {'dim':>5s} {'fields':>6s} {'pen':>4s}  genome")
    for r in res[:top]:
        p = r["parts"]
        g = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in r["genome"].items())
        print(f"  {r['idx']:5d} {r['score']:6.3f} {p['input_kept']:6.3f} {p['dim_kept']:5.2f} "
              f"{p['fields_quality']:6.3f} {p['penalty']:4.1f}  {g}")
    best = out / "best_configs"
    best.mkdir(exist_ok=True)
    for rank, r in enumerate(res[:export], 1):
        cfg = {**r["snake_config"], "_meta": {"architecture_search": {"idx": r["idx"], "score": r["score"],
                                                                      "genome": r["genome"]}}}
        (best / f"rank{rank}_c{r['idx']:04d}.json").write_text(json.dumps(cfg, indent=1))
    print(f"exported the best {export} as Snake configs (network keys only; add a hidden_rule) to {best}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--size", default="full", choices=["small", "full"])
    ap.add_argument("--backend", default="cuda", choices=["cuda", "single_threaded_cpu"])
    ap.add_argument("--hours", type=float, default=24.0, help="stop after this long (resumable)")
    ap.add_argument("--max", type=int, default=100000, help="stop after this many candidates in total")
    ap.add_argument("--random", type=int, default=20, help="random candidates before mutating the best")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true", help="only print the ranking and export the best configs")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.report:
        report(out); return
    frames_path = out / "frames.npz"
    if not frames_path.exists():
        import reservoir_search as rs
        r, y = rs.record_frames(3500)
        np.savez(frames_path, rates=r, ys=y)
    done = load_results(out)
    rng = random.Random(args.seed + len(done))
    t_end = time.time() + args.hours * 3600
    ctx = mp.get_context("spawn")
    while time.time() < t_end and len(done) < args.max:
        idx = len(done)
        if idx < args.random or rng.random() < 0.3:
            genome = random_genome(rng, args.layers)
        else:
            top = sorted(done, key=lambda r: -r["score"])[:5]
            genome = mutate(rng, rng.choice(top)["genome"], args.layers)
        with ctx.Pool(1, maxtasksperchild=1) as pool:       # own process: GPU memory freed afterwards
            try:
                res = pool.apply(evaluate_genome, ((idx, genome, args.layers, args.size, args.backend,
                                                    str(frames_path), str(out)),))
            except Exception as e:                           # a broken candidate must not stop the search
                res = {"idx": idx, "genome": genome, "error": repr(e), "score": -1e9,
                       "parts": {"input_kept": 0, "dim_kept": 0, "fields_quality": 0, "penalty": 0}}
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
        done.append(res)
        best = max(done, key=lambda r: r["score"])
        print(f"[{time.strftime('%H:%M:%S')}] c{idx:04d} score {res['score']:.3f} ({res.get('seconds', 0):.0f} s)"
              f" | best c{best['idx']:04d} {best['score']:.3f} | {len(done)} done", flush=True)
    report(out)


if __name__ == "__main__":
    main()
