# 5 layers: can decorrelation raise the compression ceiling?

At 5 layers the homeostat gives the best score so far: compression helps, but it caps how much information reaches the top.

**Decorrelation used here: covariance plastic inhibition with local synaptic scaling.** Each neuron keeps its received inhibition near its own warm-up level, so the rule only redistributes inhibition, towards the I cells that co-vary with the neuron. On the small CPU network, at η 1e-3 with scaling 1e-2, it cut redundancy (16-neuron saturation 0.74 → 0.50–0.55, |r| 0.26 → 0.19, dimension ~×2), with inhibition bounded and only a small loss of decoding. It is fully local.

**Caution from the earlier 5-layer runs.** Targeted inhibition subtracts what is shared and predictable, and a signal propagating from the layer below is exactly that. The normalised version redistributes rather than grows, so it should cancel far less than the rate rule did. But placement matters, hence the variants.

| config | what it tests |
|---|---|
| d5_homeo | reference (proposed + homeostat 1, centred; your current best) |
| d5_homeo_scale | decorrelation in every layer |
| d5_homeo_scale_L1 | only in layer 1: a less redundant code enters the stack |
| d5_homeo_scale_deep | only in layers 2–5: keep information from collapsing on the way up |
| d5_homeo_scale_ffE | every layer, with `feedforward_targets: "e"` (the next layer's I cells see only their own layer, so they can't pre-emptively cancel the incoming signal) |
| d5_homeo_scale_slow | slow variant (η 1e-4, tau_mean 10 s): targets persistent correlations only |

**The key measure is information surviving the layers.** `python probe_analyse.py runs/` reports decoding per population (L1_E … L5_E, fields).
- Compared with d5_homeo, the drop from L1 to L5 should be smaller, with the score at least as good.
- Also look at the redundancy table (mean |r|, saturation) and `dW|L{k}_I->L{k}_E` (should stay bounded, ~1–2).

**Caveat from the small network:** the homeostat at strength 1 combined with decorrelation compressed hard there (dimension 7 → 3). If d5_homeo_scale loses to d5_homeo for that reason, try the scaling with the homeostat at 0.3.

Suggested order: d5_homeo_scale, d5_homeo_scale_deep, d5_homeo_scale_ffE, then the rest. Legacy connectivity, no Dale (as in your homeostat runs). Add `"dale": {"every_episodes": 1, "local": true}` to test with real E/I.
