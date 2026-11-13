# Somatic noise (13 November 2026)

`hidden_rule.noise_site: "soma"` moves the exploration noise from the synapses to the soma. Today every neuron draws
one sample Noise1 per step and every synapse that transmits in that step delivers SynSig * Noise1, so the injected
noise is SynSig * Noise1 * (number of inputs that fired). With "soma" the neuron adds sigma * Noise1 every step
(the same sample, the same step), whatever its input does, and the synapses only keep the bookkeeping: a synapse
still adds sigma * Noise1 to its own noise trace when it transmits, so each synapse is still credited for the
steps at which it fired and the scores keep their per-synapse form (sigma = exp(log_syn_sigma - 5) *
soma_noise_gain; the synapses' SynSig is set to the same value, so the score's scale stays exact).

Why: exploration becomes uniform with activity (cleaner), and the noise-free membrane becomes cheap to compute (one
extra variable), which a shadow baseline for the local rewards, or a noise-free psibar, can use later.

| overlay | |
|---|---|
| `soma_g2.json` | gain 2: matches the current noise of the narrow 5-layer architecture at ~20 Hz (see below) |
| `soma_g1.json` | gain 1: half of it (if 2 explores too much) |

Merge onto your run's config (nested dicts merge key by key, your hidden_rule keeps its other keys):

    python configs/gpu_tests_2026-11-03/merge.py YOUR_5L.json configs/soma_noise_2026-11-13/soma_g2.json -o 5L_soma_g2.json

Choosing the gain: `python soma_gain.py YOUR_5L.json --neurons outputs/RUN(0)_neurons.csv --input-hz H` prints the
expected input spikes per step onto every hidden population (fan-ins from the config, rates from the run's last
monitor row): that number is the gain that matches the current noise of that population. For
`arch_gpu/d5_narrow_proposed_decorr1.json` at 20 Hz hidden / 5 Hz input it gives ~2.4 (L1) and ~1.9 (L2-L5), but only
~0.14 for the policy / value fields: one gain for every population gives the fields ~14x their current noise.

Checks done (CPU): the open-loop replay of a recorded volume network gives the same estimator quality (drift score vs
finite-difference derivative 0.69 with soma noise, 0.73 with synapse noise); in that volume, synapse noise gave the
input slab ~4 units per step and the deep slabs 0.01-0.09 (they barely explored). A closed-loop CPU comparison on the
selection task (v031 volume, gains 0.56 and 4.2) was running when this was written.

What to compare: the same 5-layer run with and without the overlay (score, value curves, per-layer rates and dead
fractions in _neurons.csv).
