# Contrast reward overlays (11 November 2026)

Overlays for your working headless Snake config (for example the no-fields config with G that reached length 7).
Merge with `../gpu_tests_2026-11-03/merge.py`:

    python ../gpu_tests_2026-11-03/merge.py your_headless.json contrast.json -o contrast_run.json
    SNAKE_CONFIG=contrast_run.json python run_headless.py

## What the contrast reward is

A TD-gated local reward (`hidden_rule.local.contrast`), credited through each neuron's own noise like the other
local rewards: neuron i's reward is (delta - mean delta) x trace[(z_i - mean z_i)^2], z_i its spike trace and mean z_i
a slow running average. It is added to the neuron's local reward (with decorr and the others: one reward, one
baseline) and credited by the local `route`; `contrast_route` credits it on its own route instead. After a better-than-expected outcome a neuron is rewarded for
having been far from its usual activity, in either direction; after a worse one, for having stayed close to it.
In expectation the update is delta x (z - mean z) x psi x pre: active neurons are strengthened, near-threshold
neurons that fired less than usual are weakened, and nothing changes where psi = 0, so it limits itself. It also has
a gradient part (the TD error's dependence on the neuron's noise) weighted by the squared deviation.

## The contrast leg (`hidden_rule.contrast_drift`)

The same idea in the drift's form: a third leg next to the drift and G. The trace is
trace[(dev^2 - kappa * mean dev^2) * pre * xi / sigma^2] (dev = z - mean z, the drift's trace with the squared
deviation as its gate), and the update is the TD error (centred with `center_drift`) times that trace, with the sign
that raises the squared deviation after a positive TD error. By Stein's lemma its expectation is the TD error times
the weight derivative of the squared deviation through the true dynamics: one lambda-trace of the product, each
moment weighted by lambda^age as in TD(lambda), instead of the local reward's product of two traces.
`contrast_drift_centre` (kappa, default 0) sets how much of the gate's mean is removed: 0 keeps a REINFORCE part
weighted by the neuron's mean squared deviation, with the same sign as G (both ascend the return); 1 removes it.
It stays with the drift under `split_optimiser`.

`contrast_drift_gate`: `"covariance"` (default) uses the gate dev^2. Its derivative has a positive mean (firing more
means more variance), so with an uncentred TD error the mean TD error pushes every rate one way (on the GPU: the leg
alone drained to 0 Hz; with the drift, the rates rose to 60 Hz), and with a centred one the gate's link to the rate
still pushes the rates up. `"rate_neutral"` uses dev^2 - F * z, F = mean dev^2 / mean z (the trace's Fano-like
ratio): a pure rate change leaves it unchanged, so neither the mean TD error nor the rate can push the rates. Its
expected update is delta x (2 dev - F) x sensitivity x pre, a BCM-like threshold at mean z + F / 2. In a simulated
spike train (10 ms trace) the old gate grows 4x from 10 to 40 Hz, the new one stays within +-0.005 (Poisson) or
-0.02 at 40 Hz with a 5 ms refractory period. Untested in learning when this zip was made (CPU tests running).

Contingency test (selectivity, 4 runs each): local-reward contrast +0.68; contrast leg centred (kappa 1) +0.54;
contrast leg kappa 0 +0.63; drift + local-reward contrast +0.70;
drift + centred leg +0.56. Not yet run on the selection task.

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
| `contrast.json` | adds `local.contrast` 1.0 (and `center_drift`) to your rule as it is; credited by your local `route` |
| `contrast_as_tested.json` | the same with `contrast_route` `GRADIENT_TRACE`: the form all the CPU results above used (decorr on `GRADIENT_INSTANT`, contrast on the trace) |
| `drift_contrast.json` | drift + contrast, no G (`gradient` 0) |
| `contrast_only.json` | contrast alone: no drift, no G, no homeostat (decorr and the other local rewards stay) |
| `contrast_leg.json` | adds the contrast leg (`contrast_drift` 1.0, kappa 0) to your rule as it is |
| `drift_contrast_leg.json` | drift + contrast leg, no G |
| `contrast_leg_rate_neutral.json` | the contrast leg with the rate-neutral gate |
| `drift_contrast_leg_rate_neutral.json` | drift + contrast leg with the rate-neutral gate, no G (works with your uncentred TD error) |
| `split_optimiser.json` | G and the local objectives (contrast included) get their own optimiser copy, separate from the drift |

Suggested order, one at a time: `contrast_as_tested.json` on your best headless config, then `drift_contrast.json`
(add `"contrast_route": "GRADIENT_TRACE"` to its `local` to match the tested form). With `GRADIENT_INSTANT` (credited
by the instantaneous score, like decorr) it was much weaker on the CPU (selectivity +0.27 against +0.68). The contrast weight (1.0) is not
tuned. With SGD + factor_norm it is normalised like the other local objectives (post, pre, noise); on its own
`contrast_route` also by the TD factor.

## New options (headers of `variants.py`)

- `hidden_rule.contrast_drift`, `hidden_rule.contrast_drift_centre`, `hidden_rule.contrast_drift_gate`: the contrast
  leg above.
- `move_log.activity_at`: a list of timesteps; records the activity window (`activity` moves) starting at each of
  them, saved to `<prefix>(<rep>)_activity_<t>.npz` (for activity after learning; join with `_moves.csv` on the
  timestep for the actions).
- `hidden_rule.local.contrast`, `hidden_rule.local.contrast_route`: the reward above (the rule's `hebb_decay` sets the
  running mean of the spike trace).
- `hidden_rule.hebb`: the TD-gated Hebbian term (not recommended: it ran away; not available with factor_norm).
- `hidden_rule.split_optimiser`: separate optimiser copy for G and the local objectives (update group
  `GradientLearnG`, called by `snake.py` automatically).
- `hidden_rule.backprop_signal`: `"noise"` (default) or `"activity"` for the existing `backprop` option (each neuron
  sends back its spike-trace deviation instead of its noise score). Not helpful in a short 2-layer test.
- `selection`: `commit_types` / `skip_types` / `table_reward` (the contingency test).
- `move_log`: two new columns, `value` (the critic's V at each decision) and `value_mean` (its mean over the move).
  This reads the value head every step while `move_log` is on (one small device read per step on the GPU).
