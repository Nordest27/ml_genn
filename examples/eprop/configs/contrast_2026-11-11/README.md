# Contrast reward overlays (11 November 2026)

Overlays for your working headless Snake config (for example the no-fields config with G that reached length 7).
Merge with `../gpu_tests_2026-11-03/merge.py`:

    python ../gpu_tests_2026-11-03/merge.py your_headless.json contrast.json -o contrast_run.json
    SNAKE_CONFIG=contrast_run.json python run_headless.py

## What the contrast reward is

A TD-gated local reward, credited through each neuron's own noise like the other local rewards (the gradient score
trace): neuron i's reward is (delta - mean delta) x trace[(z_i - mean z_i)^2], z_i its spike trace and mean z_i a slow
running average, minus the reward's own running mean. After a better-than-expected outcome a neuron is rewarded for
having been far from its usual activity, in either direction; after a worse one, for having stayed close to it.
In expectation the update is delta x (z - mean z) x psi x pre: active neurons are strengthened, near-threshold
neurons that fired less than usual are weakened, and nothing changes where psi = 0, so it limits itself.

## Why (CPU, small mexhat, L1 as the motor population, log-sum-exp decoder)

- Contingency test (commit rewarded on half the candidate types, skip on the other half; selectivity = commit rate
  where commit is rewarded minus where skip is): drift only +0.14 (3 of 4 runs right), TD-gated Hebbian term +0.37
  (4 of 4), contrast +0.68 (4 of 4). Firing rates and psi stayed at the drift-only values (the Hebbian term doubled
  the rates and killed up to 15 % of the E neurons).
- Selection task (commit only on candidates with features 0 and 1): contrast stays selective in all 5 runs (the
  Hebbian term collapsed to always / never commit in 4 of 5), with half the drift-only commit rate on bad candidates
  (0.17-0.29 vs 0.37-0.47). It learns fast and then plateaus; best score -0.04 to +0.18 vs -0.18 / -0.32 drift only.
- Drift plus contrast: no measurable difference from contrast alone on the contingency test; on selection one of two
  seeds was the best run so far. A test with a rule switch (where the drift's unfreezing should matter) is running.

## Overlays

| Overlay | What it changes |
|---|---|
| `contrast.json` | adds `contrast` 1.0 (and `center_drift`) to your rule as it is |
| `drift_contrast.json` | drift + contrast, no G (`gradient` 0) |
| `contrast_only.json` | contrast alone: no drift, no G, no homeostat (decorr and the other local rewards stay) |
| `split_optimiser.json` | G and the local objectives (contrast included) get their own optimiser copy, separate from the drift |

Suggested order, one at a time: `contrast.json` on your best headless config, then `drift_contrast.json`.
The contrast weight (1.0) is not tuned; with SGD + factor_norm it is normalised like a local objective (post, pre,
noise) plus the TD factor.

## New options (headers of `variants.py`)

- `hidden_rule.contrast`, the reward above (with `hebb_decay` for the running mean of the spike trace).
- `hidden_rule.hebb`: the TD-gated Hebbian term (not recommended: it ran away; not available with factor_norm).
- `hidden_rule.split_optimiser`: separate optimiser copy for G and the local objectives (update group
  `GradientLearnG`, called by `snake.py` automatically).
- `hidden_rule.backprop_signal`: `"noise"` (default) or `"activity"` for the existing `backprop` option (each neuron
  sends back its spike-trace deviation instead of its noise score). Not helpful in a short 2-layer test.
- `selection`: `commit_types` / `skip_types` / `table_reward` (the contingency test).
- `move_log`: two new columns, `value` (the critic's V at each decision) and `value_mean` (its mean over the move).
  This reads the value head every step while `move_log` is on (one small device read per step on the GPU).
