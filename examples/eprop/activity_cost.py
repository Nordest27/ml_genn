"""Activity cost of Snake runs: spikes and synaptic events per decision, from the neuron monitor.

    python activity_cost.py RUN_DIR [RUN_DIR ...]      (each RUN_DIR has config.json and outputs/*_neurons.csv)

Per run, from the last monitor report (and the first, to show the change):
- the mean rate (Hz) of every hidden population;
- spikes per decision = sum over hidden populations of rate x size x decision time;
- synaptic events per decision = sum over hidden populations of rate x outgoing synapses x decision time.
  The outgoing synapses are counted from the network the run's config builds (connectivity is resampled with the
  run's seed, so the counts match exactly when the config sets one).
- the final score: mean reward per episode over the last 200 episodes.

A decision = one environment move (WAIT_INC simulation steps of 1 ms; 30 in snake.py).
Energy on neuromorphic hardware is dominated by synaptic events, so a sparse E population held there by a busy I
population can cost more than its E rate suggests: compare the totals, not the E rate.
"""
import contextlib
import csv
import glob
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snake_hparams                                    # noqa: E402
import snake_network                                    # noqa: E402

MS_PER_DECISION = 30.0                                  # WAIT_INC steps x dt (1 ms)
INPUT_SHAPE, NUM_ACTIONS = (20, 20, 3), 4


def build(cfg):
    hp = {**snake_hparams.DEFAULTS, **{k: v for k, v in cfg.items() if k != "_meta"}}
    from ml_genn.compilers.eprop_compiler import default_params
    if hp.get("seed") is not None:
        np.random.seed(hp["seed"])
    with contextlib.redirect_stdout(io.StringIO()):
        return snake_network.build_network(snake_network.spec_from_hparams(hp, INPUT_SHAPE, NUM_ACTIONS),
                                           default_params)


def outgoing_synapses(built):
    """Outgoing hidden-to-hidden and hidden-to-head synapses per hidden population (feedback paths excluded)."""
    out = {p: 0 for p in built.hidden}
    for c in built.network.connections:
        if c.is_feedback or c.source() not in out:
            continue
        con = c.connectivity
        if getattr(con, "pre_ind", None) is not None:
            n = len(con.pre_ind)
        elif hasattr(con, "p"):
            n = con.p * np.prod(c.source().shape) * np.prod(c.target().shape)
        else:
            n = np.prod(c.source().shape) * np.prod(c.target().shape)
        out[c.source()] += float(n)
    return out


def final_score(run):
    f = [x for x in glob.glob(os.path.join(run, "outputs", "*.csv"))
         if not x.endswith(("_neurons.csv", "_policy.csv"))]
    if not f:
        return float("nan")
    rows = list(csv.DictReader(open(f[0])))
    return float(np.mean([float(r["score"]) for r in rows[-200:]])) if rows else float("nan")


def report(run):
    cfg = json.load(open(os.path.join(run, "config.json")))
    neurons = glob.glob(os.path.join(run, "outputs", "*_neurons.csv"))
    if not neurons:
        print(f"{run}: no neuron monitor output (set \"monitor\": true)")
        return
    rows = list(csv.DictReader(open(neurons[0])))
    built = build(cfg)
    sizes = {built.labels[p]: int(np.prod(p.shape)) for p in built.hidden}
    syn = {built.labels[p]: n for p, n in outgoing_synapses(built).items()}
    name = cfg.get("_meta", {}).get("variant", os.path.basename(os.path.normpath(run)))
    print(f"\n{name}  (score, last 200 episodes: {final_score(run):+.2f})")
    for label, r in (("first", rows[0]), ("last", rows[-1])):
        hz = {p: float(r[f"{p}_hz"]) for p in sizes if f"{p}_hz" in r}
        spikes = sum(hz[p] * sizes[p] for p in hz) * MS_PER_DECISION / 1000.0
        events = sum(hz[p] * syn[p] for p in hz) * MS_PER_DECISION / 1000.0
        rates = " ".join(f"{p}={hz[p]:.1f}" for p in hz)
        print(f"  {label:5s} moves {int(r['moves']):>8d} | {rates} Hz | spikes/decision {spikes:8.0f} | "
              f"synaptic events/decision {events:10.0f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for run in sys.argv[1:]:
        report(run)
