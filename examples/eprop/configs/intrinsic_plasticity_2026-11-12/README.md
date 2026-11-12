# Intrinsic plasticity overlays (12 November 2026)

Add to your working config (for example the drift + contrast reward run with the clean action pattern):

    python ../gpu_tests_2026-11-03/merge.py your_config.json intrinsic_plasticity.json -o ip_run.json

**Before using it, add the key to your `snake_hparams.py` `DEFAULTS`** (that file is not in the zips, and the config
loader rejects unknown keys):

    "intrinsic_plasticity": None,

## What it does

Every hidden ALIF neuron (L1, the fields; plain ALIF only) gets a slow bias current in its membrane update,
`V = Alpha * V + IPBias + Isyn`, homeostatic on its own time near threshold:

    IPPsiBar = decay * IPPsiBar + (1 - decay) * psi            (psi: the rule's pseudo-derivative)
    IPVBar   = decay * IPVBar   + (1 - decay) * (V - threshold) (outside refractoriness)
    IPBias  += eta * (psi_target - IPPsiBar) * (+1 if IPVBar < 0 else -1),  clipped to +-max_bias

A neuron that spends too little time near threshold is moved towards it: up if it sits below (silent), down if it
sits above (saturated). No synapse changes, so the learned weights and their ratios are untouched; psi (and so the
plasticity) follows the moved voltage. It cannot revive a neuron that receives no input.

## Why

Your run with drift + contrast reward (10.7k episodes) lost a growing share of neurons to silence or saturation:
L1 36 % dead, value field 44 %, policy field 20 % (saturated), psi down 3-5x, the voltage spreading (10th percentile
-1.9). The deaths grew steadily without levelling off.

On the CPU (small 1-layer Snake, your reference setup, 1.3-1.5M steps):
- the contrast leg alone, which drained the network to silence (L1 51 % dead, both fields 100 % dead), kept every
  neuron near threshold with intrinsic plasticity (psi on target, 0 % dead), although the leg still pushed the
  rates to zero;
- drift + contrast reward: lower rates with it in all three seeds (L1 4-10 Hz lower, fields mostly lower), learning
  ahead in two seeds and behind in one (not conclusive). The small CPU network develops almost no dead neurons, so
  the main test (preventing the deaths) needs your GPU run.

## Overlays

| Overlay | |
|---|---|
| `intrinsic_plasticity.json` | eta 2e-5 (the CPU tests) |
| `intrinsic_plasticity_slow.json` | eta 1e-5 (the default; slower homeostasis, less interference with learning) |
| `intrinsic_plasticity_long.json` | eta 5e-6, decay 0.9999 (psi averaged over ~10 s): only counters losses of plasticity that build up over ~1M steps or more; suggested for long GPU runs |

`_neurons.csv` gets new columns per population: `ip|<pop>_mean`, `_p10`, `_p90` (the bias across neurons) and
`_at_max`, `_at_min` (fractions within 1 % of the limits). Many neurons at a limit would mean the homeostat is
fighting the learning.

## Update (12 November, later): at depth, and the floor mode

On the CPU with 2-3 hidden layers (mexhat stack, selection task), the target mode (psi target 0.07) removed the deep
layers' dead neurons (L2 / L3 about 40 % -> 0-6 %) but made learning worse: the deep layers became dense (L3 spikes per
move 1.3 -> 5, policy field 1.5 -> 6.9) and lost state information (L3 decoding 0.58 -> 0.29). The "dead" deep neurons
were partly a useful sparsification. New `"mode": "floor"`: `psi_target` is then a floor and the bias moves only while
a neuron's psibar is below it, so silenced-but-responsive neurons keep their sparseness (`intrinsic_plasticity_floor.json`,
floor 0.01; CPU test running). In your 28M-step run (lr 1e-6) the deaths grew slowly (fields 12-13 % at 29M, psi down
~1/3): the floor or the long overlay is what I would try there.
