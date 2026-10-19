"""Unattended architecture search ("evolution") for Snake networks, meant to run for hours on one GPU.

    python architecture_evolution.py OUT_DIR --layers 1 --size full --backend cuda --hours 24
    python architecture_evolution.py OUT_DIR --report            # ranking so far, and export the best configs

Every candidate is an untrained network built from a "genome" of connectivity settings (local toroidal mode). Each
one is compiled, calibrated layer by layer to the target rate (non-local pre-processing; reservoir_search.py), and
driven with recorded Snake frames (all learning off). The measures are task-agnostic:
- per layer: effective dimensionality, and how much of the input frame a linear readout recovers;
- for the fields: the same;
- rates, and psi (how often neurons sit near threshold).

Score (higher is better, at most 1; every metric is stored, so --report re-ranks old results with the current score):
    0.4 * l1 + 0.4 * fields + 0.2 * memory - penalties
- l1: layer 1 in absolute terms, (input reconstruction + min(dimensionality / DIM_REF, 1)) / 2;
- fields: how much of layer 1 the fields keep, (min(input_r2 / L1's, 1) + min(dim / L1's, 1)) / 2, capped at 1 so a
  weak layer 1 cannot inflate it. The layers in between only matter through what reaches the fields;
- memory: how much of the input 1-3 frames ago the fields' state (V and adaptation A) recovers beyond what the current frame predicts
  (reservoir_search.memory_r2), mean over the lags, / MEM_FULL, capped at 1;
- reported, not scored: kept per layer, and mixing (per stage, min((1 - R2) / MIX_FULL, 1), R2 = how much of its
  spike counts a linear map from the previous layer's E counts explains; a relay that copies has R2 ~ 1). Copies are
  ruled out by the wiring penalty instead;
- penalties: any population below 2 Hz or above 80 Hz after calibration, or more than half its neurons almost never
  near threshold (psi ~ 0); and wiring: for every feed-forward projection (input -> L1, Lk -> Lk+1, last layer ->
  fields), the fraction of target neurons with fewer than --min-fan-in synapses (default 10) or from fewer than
  --min-cells source grid cells (default 3) (copies of one cell, or silent neurons); the mean of these fractions is subtracted.

Disk: each candidate's GeNN build folder is deleted once its results are saved.
GPU: needs a CUDA-enabled GeNN (CUDA_PATH set). One candidate at a time, in its own process.

Prior (--prior relay): what training showed for deep networks. The layer-to-layer projections stay a near copy
(ff_sigma 0.3-0.6, no surround or shortcuts) and mixing is left to the input projection, the recurrence and the
fields; the score's middle term becomes the weakest stage (min over L2..Ln and the fields of what they keep of L1),
so every step must preserve the information. Use with --min-fan-in 0 --min-cells 0 (a relay is sparse on purpose)
and --seed-genomes to start from a known network.

Feed-forward kernel (--ff-kernel): "gaussian" (default) or "surround", a narrow centre (ff_sigma) plus a sparse wide
surround (genes ff_surround_p, ff_surround_ratio) on every layer-to-layer projection and the projection into the
fields: the centre relays, the surround mixes. The surround counts toward the wiring minimum (it is local); --min-fan-in 0 --min-cells 0 turns the minimum off.

Families (--shortcuts): "off" (default) searches purely local networks; "on" adds the gene global_fan_in, random
long-range inputs per neuron (p_global = fan-in / E population size). The wiring minimum counts local synapses only
(it is checked on the same network without the shortcuts), and every result stores long_range_fraction, the share
of hidden-to-hidden synapses that are long-range. Run the two families as separate searches (separate folders).

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
    "inhibition_scale": ("lin", 2.0, 8.0),      # inhibitory weight factor (E/I balance)
    "ff_sigma": ("log", 0.3, 2.0),              # layer k -> k+1 neighbourhood (only with >1 layer)
    "field_sigma": ("log", 0.3, 2.5),           # last layer -> fields neighbourhood
    "field_shape": ("choice", ["inhibitory", "excitatory"]),   # fields sized like the I or the E grid
    "target_hz": ("choice", [10.0, 20.0, 30.0]),               # calibration target
    "ff_inhibition": ("choice", [False, True]),     # I also projects forward (to the next layer / the fields)
    "ff_targets": ("choice", ["both", "e"]),        # E feed-forward reaches the next layer's E and I, or E only
    "global_fan_in": ("int", 1, 40, 4.0),           # random long-range inputs per neuron (--shortcuts on only):
                                                    # first samples 1-40, mutations +-4 (sd), no upper bound
    "inh_sigma_ratio": ("log", 0.5, 3.0),           # recurrent I -> E / I -> I spread relative to sigma_cells
    "ff_surround_p": ("log", 0.005, 0.3),           # --ff-kernel surround: peak probability of the wide surround
    "ff_surround_ratio": ("log", 2.0, 8.0),         # ... its spread relative to the feed-forward centre (ff_sigma)
}
SURROUND_GENES = ("ff_surround_p", "ff_surround_ratio")
RELAY_FF = ("log", 0.3, 0.6)            # --prior relay: the layer-to-layer projection stays a near copy
SURROUND_CENTRE = ("log", 0.3, 0.8)     # with a surround the feed-forward centre stays narrow (a relay): the surround
                                        # has to provide the mixing, instead of a wide single Gaussian


def sample_gene(rng, spec):
    kind = spec[0]
    if kind == "choice":
        return rng.choice(spec[1])
    lo, hi = spec[1], spec[2]
    if kind == "int":
        return int(rng.randint(lo, hi))
    if kind == "log":
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    return float(rng.uniform(lo, hi))


GENE_DEFAULTS = {"ff_inhibition": False, "ff_targets": "both", "inh_sigma_ratio": 1.0}     # genomes from before these genes existed


def random_genome(rng, layers, shortcuts=False, surround=False, relay=False):
    g = {k: sample_gene(rng, s) for k, s in SPACE.items()}
    if relay and layers > 1:
        g["ff_sigma"] = sample_gene(rng, RELAY_FF)
    if not shortcuts:
        g.pop("global_fan_in")                    # the purely local family
    if not surround or layers == 1:
        for k in SURROUND_GENES:
            g.pop(k)                              # single-Gaussian feed-forward kernel
    else:
        g["ff_sigma"] = sample_gene(rng, SURROUND_CENTRE)
    if layers == 1:
        g.pop("ff_sigma")
        g.pop("ff_targets")                       # no layer-to-layer projection (ff_inhibition still acts on the fields)
    return g


def mutate(rng, genome, layers, relay=False):
    g = {**GENE_DEFAULTS, **genome}
    if layers == 1:
        g.pop("ff_targets", None)
    keys = [k for k in SPACE if (layers > 1 or k not in ("ff_sigma", "ff_targets"))
            and (k not in ("global_fan_in",) + SURROUND_GENES or k in g)]   # a genome stays in its family
    for k in rng.sample(keys, rng.randint(1, 3)):
        spec = SPACE[k]
        if spec[0] == "choice":
            g[k] = rng.choice(spec[1])
        elif spec[0] == "int":                  # additive steps, at least 1
            g[k] = max(1, int(round(g[k] + rng.gauss(0.0, spec[3]))))
        else:                                   # multiplicative jitter within the bounds
            lo, hi = spec[1], spec[2]
            if k == "ff_sigma" and "ff_surround_p" in g:
                lo, hi = SURROUND_CENTRE[1], SURROUND_CENTRE[2]
            if k == "ff_sigma" and relay:
                lo, hi = RELAY_FF[1], RELAY_FF[2]
            g[k] = float(min(hi, max(lo, g[k] * math.exp(rng.gauss(0.0, 0.25)))))
    return g


def genome_to_hp(g, layers, size):
    hid_e, hid_i = (20, 15) if size == "full" else (6, 4)
    g = {**GENE_DEFAULTS, **g}
    local = {"sigma_cells": round(g["sigma_cells"], 3), "p_max": round(g["p_max"], 3),
             "input_sigma_cells": round(g["input_sigma_cells"], 3), "inhibition_scale": round(g["inhibition_scale"], 3),
             "feedforward_inhibition": bool(g["ff_inhibition"]),
             # long-range inputs per neuron from an E population -> probability per pair (I sources, with fewer
             # neurons, give proportionally fewer; the input projection has none)
             "p_global": round(g.get("global_fan_in", 0) / float(hid_e * hid_e * 3), 6),
             "inhibition_sigma_ratio": round(g["inh_sigma_ratio"], 3)}
    if size != "full":
        local["input_sigma_ref"] = "source"
    net = {"fields": {"conn": {"sigma_cells": round(g["field_sigma"], 3)},
                      "shape": hid_i if g["field_shape"] == "inhibitory" else hid_e}}
    if "ff_surround_p" in g:                      # the fields get the same centre-surround kernel (around field_sigma)
        net["fields"]["conn"].update(p_surround=round(g["ff_surround_p"], 4),
                                     surround_ratio=round(g["ff_surround_ratio"], 3))
    if layers > 1:
        to_next = {"sigma_cells": round(g["ff_sigma"], 3)}
        if "ff_surround_p" in g:                  # centre-surround feed-forward kernel
            to_next.update(p_surround=round(g["ff_surround_p"], 4), surround_ratio=round(g["ff_surround_ratio"], 3))
        net["layers"] = [{"to_next": dict(to_next)} for _ in range(layers - 1)] + [{}]
    hp = {"toroidal": "local", "local": local, "network": net}
    if layers > 1:
        hp["feedforward_targets"] = g["ff_targets"]
    return hp


DIM_REF = {"full": 15.0, "small": 20.0}    # layer-1 E dimensionality (participation ratio) that earns full marks
MEM_FULL = 0.3          # fields' state (V, A) recovering 30% of the past frames' new information (lags 1-3): full marks (full size reaches ~0.15)
MIX_FULL = 0.3          # a stage whose counts are at most 70% linearly predictable from the layer below: full marks
MIN_FAN_IN, MIN_CELLS = 10, 3                # feed-forward wiring every target neuron should have


def wiring(built, min_fan_in=MIN_FAN_IN, min_cells=MIN_CELLS):
    """Per feed-forward projection: fan-in statistics and the fraction of target neurons below the minimum wiring."""
    labels = built.labels
    depth = lambda pop: (0 if pop is built.input else int(labels[pop][1:].split("_")[0])
                         if labels.get(pop, "").startswith("L") else 10 ** 6)
    out = []
    for c in built.network.connections:
        con = c.connectivity
        if c.is_feedback or getattr(con, "pre_ind", None) is None:
            continue
        src, tgt = c.source(), c.target()
        if tgt not in labels or depth(tgt) <= depth(src):
            continue                                    # recurrent / within-layer, or into the heads
        pre, post = np.asarray(con.pre_ind), np.asarray(con.post_ind)
        nt = int(np.prod(tgt.shape))
        ch = src.shape[-1] if len(src.shape) == 3 else 1
        fan = np.bincount(post, minlength=nt)
        order = np.argsort(post, kind="stable")
        split = np.split(pre[order] // ch, np.cumsum(fan)[:-1])          # source grid cells, per target neuron
        cells = np.array([len(np.unique(g)) for g in split])
        bad = (fan < min_fan_in) | (cells < min_cells)
        out.append({"proj": f"{labels.get(src, 'input')}->{labels[tgt]}", "fan_in": float(fan.mean()),
                    "fan_in_min": int(fan.min()), "cells": float(cells.mean()), "silent": float(np.mean(fan == 0)),
                    "below_min": float(bad.mean())})
    return out


def synapse_count(built):
    """Synapses between hidden populations (recurrent and feed-forward; not the input, heads or feedback)."""
    n = 0
    for c in built.network.connections:
        con = c.connectivity
        if (not c.is_feedback and c.source() in built.labels and c.target() in built.labels
                and getattr(con, "pre_ind", None) is not None):
            n += len(con.pre_ind)
    return n


def score(result, layers):
    """0.4 * layer 1 (absolute) + 0.4 * what the fields keep of it + 0.2 * the fields' memory - penalties.
    Under the relay prior (result["prior"] == "relay") the middle term is the WEAKEST stage instead of the fields:
    every layer and the fields must preserve layer 1, a loss at one step cannot be made up elsewhere."""
    e = [p for p in result["per_pop"] if p["pop"].endswith("_E")]
    f = [p for p in result["per_pop"] if "field" in p["pop"]]
    l1 = e[0]
    dim_ref = DIM_REF["full" if l1["n"] >= 400 else "small"]
    l1_abs = 0.5 * (max(l1["input_r2"], 0.0) + min(l1["dim"] / dim_ref, 1.0))

    def kept(input_r2, dim):
        return 0.5 * (min(max(input_r2, 0.0) / max(l1["input_r2"], 1e-6), 1.0)
                      + min(dim / max(l1["dim"], 1e-6), 1.0))

    def mixing(pred):
        pred = [x for x in pred if x is not None and np.isfinite(x)]
        return float(np.clip((1.0 - np.mean(pred)) / MIX_FULL, 0.0, 1.0)) if pred else None

    stages = [kept(p["input_r2"], p["dim"]) for p in e[1:]]
    mixes = [mixing([p.get("pred_from_prev")]) for p in e[1:]]
    if f:
        stages.append(kept(np.mean([p["input_r2"] for p in f]), np.mean([p["dim"] for p in f])))
        mixes.append(mixing([p.get("pred_from_prev") for p in f]))
    fields = stages[-1] if f else 0.0
    later = float(np.mean(stages)) if stages else 0.0
    mixes = [m for m in mixes if m is not None]
    mix = float(np.mean(mixes)) if mixes else 0.0
    pen = 0.0
    for p in result["per_pop"]:
        if not 2.0 <= p["hz"] <= 80.0:
            pen += 0.2
        if p.get("psi_dead", 0.0) > 0.5:
            pen += 0.1
    wire = float(np.mean([w["below_min"] for w in result.get("wiring", [])])) if result.get("wiring") else 0.0
    mem = [p["memory"] for p in f if p.get("memory")]
    memory = float(min(np.mean([np.mean(m) for m in mem]) / MEM_FULL, 1.0)) if mem else 0.0
    weakest = float(min(stages)) if stages else 0.0
    keep = weakest if result.get("prior") == "relay" else fields
    return float(0.4 * l1_abs + 0.4 * keep + 0.2 * memory - pen - wire), {
        "l1": l1_abs, "fields": fields, "weakest": weakest, "memory": memory, "later_kept": later, "mixing": mix, "stages": [round(x, 3) for x in stages],
        "penalty": pen, "wiring": wire}


def rescore(results, layers):
    """Recompute stored results with the current score() (results from an older score keep their raw metrics)."""
    for r in results:
        if "per_pop" in r:
            r["score"], r["parts"] = score(r, layers)
    return results


def evaluate_genome(args):
    """Runs in its own process (released afterwards): build, calibrate, evaluate one genome."""
    idx, genome, layers, size, backend, frames_path, out_dir, min_fan_in, min_cells = args
    import reservoir_search as rs
    name = f"c{idx:04d}"
    work = Path(out_dir) / "work" / name
    work.mkdir(parents=True, exist_ok=True)
    d = np.load(frames_path)
    rates, ys = d["rates"], d["ys"]
    hp = genome_to_hp(genome, layers, size)
    t0 = time.time()
    built = rs.build(hp, size, layers)
    # the wiring minimum counts local synapses only: check it on the same network without long-range shortcuts
    local_built = built
    if hp["local"].get("p_global", 0.0) > 0:
        import copy
        hp_local = copy.deepcopy(hp)
        hp_local["local"]["p_global"] = 0.0
        local_built = rs.build(hp_local, size, layers)
    wires = wiring(local_built, min_fan_in, min_cells)
    long_range = 1.0 - synapse_count(local_built) / max(synapse_count(built), 1)
    with rs.Sim(built, name, work, backend=backend) as sim:
        rs.calibrate(sim, rates[:1500], genome["target_hz"], log=lambda s: None)
        per_pop = rs.evaluate(sim, rates[1500:], ys[1500:])
        factors = dict(sim.factors)
    result = {"idx": idx, "genome": genome, "hp": hp, "per_pop": per_pop, "wiring": wires,
              "long_range_fraction": round(long_range, 4),
              "calibration_factors": {str(k): v for k, v in factors.items()},
              "snake_config": rs.snake_config(hp, factors, layers, size), "seconds": round(time.time() - t0, 1)}
    result["score"], result["parts"] = score(result, layers)
    import shutil
    shutil.rmtree(work, ignore_errors=True)                 # the build folder (code, binaries): ~MBs per candidate
    return result


def _child(conn, task):
    try:
        conn.send(evaluate_genome(task))
    except Exception as e:
        conn.send({"error": repr(e)})
    conn.close()


def run_isolated(ctx, task, timeout):
    """evaluate_genome in its own process (GPU memory is released when it exits). A candidate whose process dies
    without an answer (e.g. a C++ abort in code generation) or runs past timeout seconds becomes an error entry
    instead of blocking the search."""
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, task))
    proc.start()
    child.close()
    try:
        if parent.poll(timeout):
            return parent.recv()
        return {"error": f"timeout after {timeout} s"}
    except EOFError:
        return {"error": "the candidate's process died without a result"}
    finally:
        proc.join(5)
        if proc.is_alive():
            proc.kill(); proc.join()


def seed_genomes(args, done):
    """The best --seed-top genomes of other searches (--seed-from folders), ranked by the current score, converted to
    this search's family and depth: genes missing in them get GENE_DEFAULTS, genes this search does not use are
    dropped, and (shortcuts off) old p_global / global_fan_in values are removed. Genomes already evaluated here are
    skipped, so a resumed search does not evaluate its seeds twice."""
    found = []
    if getattr(args, "seed_genomes", None):
        found += [{"genome": g, "score": float("inf")} for g in json.load(open(args.seed_genomes))]
    if not args.seed_from and not found:
        return []
    for folder in (args.seed_from or []):
        res = [r for r in rescore(load_results(Path(folder).resolve()), args.layers) if "per_pop" in r]
        found += sorted(res, key=lambda r: -r["score"])[:args.seed_top]
    seen = [r["genome"] for r in done]
    out = []
    for r in sorted(found, key=lambda r: -r["score"]):
        g = {**GENE_DEFAULTS, **r["genome"]}
        g.pop("p_global", None)                          # the old probability gene (now global_fan_in)
        if args.shortcuts == "off":
            g.pop("global_fan_in", None)
        elif "global_fan_in" not in g:
            g["global_fan_in"] = 1
        surround = getattr(args, "ff_kernel", "gaussian") == "surround" and args.layers > 1
        if not surround:
            for k in SURROUND_GENES:
                g.pop(k, None)
        elif "ff_surround_p" not in g:                   # start from a faint surround: close to the seed's own kernel
            g.update(ff_surround_p=0.01, ff_surround_ratio=4.0)
        if args.layers == 1:
            g.pop("ff_sigma", None); g.pop("ff_targets", None)
        elif "ff_sigma" not in g:
            continue                                     # a 1-layer genome has no feed-forward spread to start from
        g = {k: v for k, v in g.items() if k in SPACE}
        if g not in seen and g not in out:
            out.append(g)
    return out


def load_results(out):
    path = out / "results.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def infer_layers(results):
    for r in results:
        if "per_pop" in r:
            return sum(p["pop"].endswith("_E") for p in r["per_pop"])
    return 1


def report(out, top=10, export=3):
    res = load_results(out)
    res = sorted(rescore(res, infer_layers(res)), key=lambda r: -r["score"])
    if not res:
        print("no results yet"); return
    print(f"{len(res)} candidates; best {top}:")
    print(f"  {'idx':>5s} {'score':>6s} {'L1':>5s} {'field':>5s} {'weak':>5s} {'mem':>5s} {'mix':>5s} {'pen':>4s} {'wire':>5s}  "
          f"{'kept per stage (L2.., fields)':30s} genome")
    for r in res[:top]:
        p = r["parts"]
        g = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in r["genome"].items())
        if "per_pop" not in r:
            print(f"  {r['idx']:5d}  failed: {r.get('error', '?')[:80]}"); continue
        st = " ".join(f"{x:.2f}" for x in p["stages"])
        if "long_range_fraction" in r:
            st = f"LR {r['long_range_fraction']:.2f} | " + st
        print(f"  {r['idx']:5d} {r['score']:6.3f} {p['l1']:5.2f} {p.get('fields', 0):5.2f} {p.get('weakest', 0):5.2f} "
              f"{p.get('memory', 0):5.2f} "
              f"{p.get('mixing', 0):5.2f} "
              f"{p['penalty']:4.1f} {p.get('wiring', 0):5.2f}  "
              f"{st:30s} {g}")
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
    ap.add_argument("--prior", default="none", choices=["none", "relay"],
                    help="relay: layer-to-layer projections stay a near copy (ff_sigma 0.3-0.6) and the score asks every "
                         "stage to preserve layer 1 (weakest stage instead of the fields); use with --min-fan-in 0")
    ap.add_argument("--seed-genomes", help="JSON file with a list of genomes to evaluate first (e.g. a known network)")
    ap.add_argument("--ff-kernel", default="gaussian", choices=["gaussian", "surround"],
                    help="feed-forward kernel between layers: one Gaussian, or a narrow centre plus a sparse wide surround")
    ap.add_argument("--shortcuts", default="off", choices=["off", "on"],
                    help="network family: purely local, or with random long-range inputs (gene global_fan_in)")
    ap.add_argument("--seed-from", nargs="+", default=[], metavar="DIR",
                    help="start by evaluating the best genomes of these earlier searches (then search as usual)")
    ap.add_argument("--seed-top", type=int, default=5, help="how many genomes to take from each --seed-from folder")
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds before a candidate counts as failed")
    ap.add_argument("--min-fan-in", type=int, default=MIN_FAN_IN, help="feed-forward synapses every target needs")
    ap.add_argument("--min-cells", type=int, default=MIN_CELLS, help="source grid cells every target needs")
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
    done = rescore(load_results(out), args.layers)     # older results re-ranked with the current score
    rng = random.Random(args.seed + len(done))
    t_end = time.time() + args.hours * 3600
    ctx = mp.get_context("spawn")
    queue = seed_genomes(args, done)                    # --seed-from: evaluated first, then the search goes on
    if queue:
        print(f"seeding with {len(queue)} genomes from {', '.join(args.seed_from)}", flush=True)
    while time.time() < t_end and len(done) < args.max:
        idx = len(done)
        if queue:
            genome = queue.pop(0)
        elif idx < args.random or rng.random() < 0.3:
            genome = random_genome(rng, args.layers, args.shortcuts == "on", args.ff_kernel == "surround",
                                   args.prior == "relay")
        else:
            top = sorted(done, key=lambda r: -r["score"])[:5]
            genome = mutate(rng, rng.choice(top)["genome"], args.layers, args.prior == "relay")
        task = (idx, genome, args.layers, args.size, args.backend, str(frames_path), str(out), args.min_fan_in,
                args.min_cells)
        res = run_isolated(ctx, task, args.timeout)
        if args.prior != "none" and "per_pop" in res:        # scored under the search's prior
            res["prior"] = args.prior
            res["score"], res["parts"] = score(res, args.layers)
        if "error" in res:                                   # a broken candidate must not stop the search
            res.update(idx=idx, genome=genome, score=-1e9,
                       parts={"l1": 0, "fields": 0, "memory": 0, "later_kept": 0, "mixing": 0, "stages": [],
                              "penalty": 0, "wiring": 0})
            import shutil
            shutil.rmtree(out / "work" / f"c{idx:04d}", ignore_errors=True)
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
        done.append(res)
        best = max(done, key=lambda r: r["score"])
        print(f"[{time.strftime('%H:%M:%S')}] c{idx:04d} score {res['score']:.3f} ({res.get('seconds', 0):.0f} s)"
              f" | best c{best['idx']:04d} {best['score']:.3f} | {len(done)} done", flush=True)
    report(out)


if __name__ == "__main__":
    main()
