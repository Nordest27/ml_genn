# Surprise-modulated drift

`hidden_rule.surprise_drift = c_S` adds c_S * (|delta| - mean|delta|) to the drift's modulator; `drift_td` is the
weight of the TD error in it (0: the drift loses the outcome sign; the gradient part always keeps delta).
- c_S > 0: responses around surprising moments (good or bad) are frozen, routine ones stay plastic.
- c_S < 0 (inverse): surprising moments are sensitised (plasticity where the agent is wrong), routine ones frozen.

All configs = `arch_gpu/d5_narrow_proposed_decorr1.json` + the drift modulator below. Compare with your
d5_narrow_proposed_decorr1 runs.

| config | drift modulator |
|---|---|
| d5_decorr1_surprise_only | |delta| - mean|delta| |
| d5_decorr1_td_surprise | delta + (|delta| - mean|delta|) |
| d5_decorr1_inverse_surprise | -(|delta| - mean|delta|) |

Small CPU network (1 layer, plain rule without decorrelation), seed 0, last 500 games: current drift -0.63,
delta + surprise -0.30, surprise only +0.04, reservoir -0.55. More seeds and the inverse: analysis/surprise_seeds.
The interaction with the decorrelation reward is untested.
