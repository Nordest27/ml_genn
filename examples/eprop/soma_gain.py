"""Expected number of presynaptic spikes per step onto each hidden population: the soma_noise_gain at which
noise_site "soma" (sigma * Noise1 per step, uniform) matches the current synapse noise (SynSig * Noise1 per
transmitting synapse) on average for that population.

    python soma_gain.py CONFIG.json [--neurons OUT_neurons.csv] [--hidden-hz 20] [--input-hz 5]

Fan-ins come from the network the config builds (same path as snake.py); rates from the last row of a run's
_neurons.csv (columns <label>_hz) when given, else --hidden-hz for every hidden population. The input's mean rate is
--input-hz (the Poisson rate per input neuron, averaged over the frame: most pixels are off). Prints, per hidden
population, the spikes per step from each source and in total.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def fan_in(conn, src, tgt):
    from ml_genn.connectivity import Explicit, FixedProbability
    c = conn.connectivity
    name = type(c).__name__
    if isinstance(c, Explicit):
        return len(np.asarray(c.pre_ind)) / max(1, int(np.prod(tgt.shape)))
    if isinstance(c, FixedProbability):
        return float(c.p) * int(np.prod(src.shape))
    if name == "ToroidalGaussian2D":
        if getattr(c, "p_max", None):
            return float(c.expected_fan_in(src.shape, tgt.shape))
        return float(c.fan_in)
    if name == "LocalPermutation":
        return 1.0
    raise ValueError(f"unknown connectivity {name}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--neurons", default=None)
    ap.add_argument("--hidden-hz", type=float, default=20.0); ap.add_argument("--input-hz", type=float, default=5.0)
    a = ap.parse_args()
    import snake_hparams, snake_network
    from ml_genn.compilers.eprop_compiler import default_params
    hp = dict(snake_hparams.DEFAULTS); hp.update(json.load(open(a.config)))
    headless_vector = hp.get("headless") is not None
    n_out = 2 if (hp.get("selection") or headless_vector or hp.get("reach")) else 4
    channels = 4 if hp.get("cued_foraging") else 3
    spec = snake_network.spec_from_hparams(hp, (20, 20, channels), n_out, channels=channels)
    built = snake_network.build_network(spec, default_params)
    labels = {p: built.labels.get(p, p.name) for p in built.network.populations}
    rates = {}
    if a.neurons:
        rows = list(csv.DictReader(open(a.neurons)))
        last = rows[-1]
        rates = {k[:-3]: float(v) for k, v in last.items() if k.endswith("_hz") and v not in ("", "nan")}
    hidden = set(built.hidden)
    print(f"{'target':16s} {'source':16s} {'fan-in':>8s} {'rate Hz':>8s} {'spikes/step':>12s}")
    totals = {}
    for c in built.network.connections:
        tgt, src = c.target(), c.source()
        if tgt not in hidden:
            continue
        lt, ls = labels[tgt], labels[src]
        if src is built.input:
            hz = a.input_hz
        else:
            hz = rates.get(ls, a.hidden_hz)
        k = fan_in(c, src, tgt)
        n = k * hz / 1000.0
        totals[lt] = totals.get(lt, 0.0) + n
        print(f"{lt:16s} {ls:16s} {k:8.1f} {hz:8.1f} {n:12.3f}")
    print("\nper hidden population, spikes per step (= matching soma_noise_gain):")
    for lt, n in totals.items():
        print(f"  {lt:16s} {n:.2f}")
    print(f"  mean over populations: {np.mean(list(totals.values())):.2f}")


if __name__ == "__main__":
    main()
