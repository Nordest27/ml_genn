# Big 3D volume on Snake (13 November 2026)

One hidden "layer" that is a 3D volume (`network.layers[0].volume`, see `VolumeSpec` in `snake_network.py`): a
16 x 16 x D sheet of E neurons (8 x 8 x D inhibitory), connected locally in 3D (Gaussian kernel, sigma 1 cell, p_max
0.8) with a forward bias (`z_shift` 1.5: an E neuron's kernel is centred 1.5 slabs deeper), plus a fixed number of
random long-range connections per neuron (`k_global` 3: every neuron receives exactly 3 from E sources anywhere in
the volume, and proportionally from I sources). The input reaches only the front slab; the policy fields read only
the last 3 slabs, so the signal must cross the volume (locally about 8-10 synapses, or about 2 through long-range
links). Same rule as the CPU volume runs: drift + contrast (lambda-trace credit), decorr 0.001, uncentred, lr 1e-6,
headless default decoder, 30M steps.

| config | what | question |
|---|---|---|
| `bigvol16.json` | depth 16 (4096 E + 1024 I) | main run: does a deep volume learn Snake? |
| `bigvol12.json` | depth 12 (3072 E + 768 I) | shallower: more task information reaches the fields at init |
| `bigvol16_ip.json` | depth 16 + intrinsic plasticity | rate control: CPU volumes learned, then rates ran away (37 -> 116 Hz) and the fields died |
| `bigvol16_local.json` | depth 16, `k_global` 0 | no long-range connections: are the shortcuts needed? |

Suggested order, one at a time: `bigvol16`, then `bigvol16_ip`, then `bigvol12` / `bigvol16_local`.

What to look at: `_neurons.csv` (L1_E / L1_I rates and dead fractions over time: the CPU volumes' failure was a rate
climb that killed the policy fields), the score curve, and `policy_field` dead fraction.

Untrained CPU checks (selection-task frames, frozen): with the input gain x2.5 (a 16-wide sheet gets ~6x fewer input
synapses per neuron than the 6-wide CPU volume) activity crosses all 16 slabs at ~0.55-0.6 spikes per move, task
decoding 1.00 at the input slab -> 0.19-0.24 at the back -> 0.16 in the fields (the CPU winner had 0.16); depth 12
gives 0.25 in the fields. Stronger recurrence (internal 0.8) saturates the volume (6-7 spikes per move, decoding at
chance), so don't raise it.

New option (`VolumeSpec`): `k_global` (count instead of the probability `p_global`, which is ignored when `k_global`
is set), `k_global_mode` ("in": every neuron receives k; "out": every neuron sends k), `k_global_i` (I sources;
default keeps the proportion p_global would give). A fractional k is exact on average.
