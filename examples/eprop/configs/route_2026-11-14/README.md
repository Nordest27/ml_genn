# Routed credit (14 November 2026)

Every hidden neuron learns, without reward, how its activity moves the input of other neurons ("route links"), and
uses those learned sensitivities to receive credit back from them, on top of the drift and contrast terms. Each
neuron weighs the routed credit by its own trust: it starts at the scale of its own perturbation credit and follows
the measured agreement between the two as evidence accumulates. Everything is local: a neuron's own noise, activity,
pseudo-derivative and age, the two ends of each link, and the broadcast TD error.

Merge an overlay onto a headless config, keeping that config's network:

    python ../gpu_tests_2026-11-03/merge.py narrow_5l_headless.json route_spike.json -o narrow_5l_headless_route.json

| Overlay | |
|---|---|
| `route_spike.json` | route links back along every forward connection, learned from the receiver's spikes |
| `route_spike_local.json` | route links between nearby neurons (16 per neuron, Gaussian, sigma 1.5 sheet cells), wired or not |

Status: in a 30-neuron spiking check, spike-learned links track the forward weights they mirror (correlation 0.92 in
50k steps; links learned from noise only reached 0.2-0.5 in 200k). In closed loop it is not yet shown to help: the
first CPU runs (v031 volume, headless selection task) are level with the baseline at mid-run. Compare against the
same config without the overlay, with the same seed.

What to look at: the reward and value curves against the baseline (an earlier start is what the rate toy showed),
and the firing rates (the links do not enter the membrane, so rates should look like the baseline's).
