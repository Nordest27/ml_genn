# Headless at depth (12 November 2026)

The recipe of your best headless run (reward mean 3, length 15): drift + contrast reward (lambda-trace credit),
decorr 0.001, uncentred TD error, learning rate 1e-6, no other local rewards, no `c_reg`, shared optimiser.
Merge it onto a deep architecture (for example your narrow 5-layer config), keeping that config's network:

    python ../gpu_tests_2026-11-03/merge.py narrow_5l.json recipe.json -o narrow_5l_headless.json
    python ../gpu_tests_2026-11-03/merge.py narrow_5l.json recipe.json ../intrinsic_plasticity_2026-11-12/intrinsic_plasticity.json -o narrow_5l_headless_ip.json
    python ../gpu_tests_2026-11-03/merge.py narrow_5l.json recipe.json no_contrast.json -o narrow_5l_headless_drift.json

Check the merged file: nested dicts are merged key by key, so keys of your base config's `hidden_rule` that the
recipe does not set (for example `homeostat`, `center_eligibility`, `split_optimiser`, an optimiser block) stay.
`lr` sets the main optimiser's rate; if your base config has a `readout_optimiser`, the value head keeps its own.

| Overlay | |
|---|---|
| `recipe.json` | the recipe above, `headless` with the default decoder |
| `no_contrast.json` | (after `recipe.json`) the same without the contrast reward: drift + decorr only, for attribution |

Suggested order, one at a time: 2-3 layers with the recipe (how fast the result degrades with depth); the 5-layer
architecture with the recipe, with and without `intrinsic_plasticity.json` (deep layers had 70-77 % of neurons in the
dead zone in your earlier narrow 5-layer run, which is where intrinsic plasticity should matter most); then
`no_contrast.json` for attribution.

What to look at: `_neurons.csv` per layer (dead fractions, psi, rates over time), the value curve against the reward,
and whether the deep layers' rates and psi stay in range. A CPU test of the same recipe on the selection task with
1-3 layers (action information per layer after learning) is running.
