# GPU test overlays (3 November 2026)

Small JSON overlays to add to your working 5-layer config (e.g. the narrow fast-readout decorrelation run):

    python merge.py your_config.json boolean_head.json -o boolean.json
    SNAKE_CONFIG=boolean.json python run_headless.py

`merge.py` merges nested dicts key by key. A `hidden_rule` overlay therefore keeps the rest of your rule settings, so `reservoir.json` changes only the preset.

| Overlay | What it tests | Suggested runs |
|---|---|---|
| `boolean_head.json` | fixed random logic units (BooleanUnit, k = 2) between the fields and the heads: no gradient path from the heads into the network | your rule, and your e-prop config, each with and without it |
| `boolean_hold.json` | the same logic units with held bits instead of the 10-step window: an input spike sets its bit for 10 steps and the table is applied every step, so the output follows the inputs without the window's clock and delay | same as `boolean_head.json`; compare the two |
| `boolean_parity.json` | negative control: 3-input parity (XOR / XNOR) tables, hold 30 so the bits sit near 50 % on. A linear readout can't read parity, so on the CPU (1 layer) no rule learned, the reservoir included | optional; check `parity_linear` in `<prefix>_boolean.csv` |
| `headless_noise.json` | no policy head: the policy field is split into 4 motor pools, the action is the most active pool (no softmax, no sampling), the policy error is 0; the motor pools get 4.5x the default weight noise (`motor_log_syn_sigma` 1.5), the noise the rule credits | your rule. On the CPU (1 layer) it reached -0.23 vs -0.71 at default noise and -0.06 for the softmax head; x20 noise was too much |
| `headless_noise_big.json` | the same with 9.5x noise (2.25) | optional, to bracket the noise level |
| `selection_headless.json` | instead of Snake, candidate selection: one candidate per move, commit or skip, good iff features 0 and 1 are both present (+1 / -1; 0 if all are skipped; oracle 0.91, random about -0.35), headless with motor noise | your rule; the cleanest test of learning a choice by selection alone. Logged score = the episode's reward |
| `cnn_head.json` | a host-side convolutional actor (NumPy, online) reads the fields' spike counts each move and acts; the critic is the network's own value head, trained every timestep, whose TD error drives the network (episodic) | your rule; `+ reservoir.json` as the frozen-feature reference |
| `noise_head.json` | no policy distribution: argmax over noisy logits, readout trained by node perturbation | your rule (possibility test; σ = 3 was tuned on the input-only agent and may need adjusting to your logit scale) |
| `reservoir.json` | hidden weights frozen (only the head learns) | as a reference with `cnn_head.json` |

## Notes

- **Boolean head:** while its output is 1, a Boolean unit fires at the refractory frequency, so the heads' input (and the initial policy's sharpness) is larger than with the fields. If the Boolean runs learn clearly worse, rerun with weaker readout weights to separate that scale effect from the logic itself.
- **Boolean logging:** with any Boolean head, `outputs/<prefix>(<rep>)_boolean.csv` logs per episode each logic population's input-bit on-fraction (`bit_on`), output on-fraction (`out_on`) and `parity_linear` (mean of |1 − 2p|^(k−1): an input's first-order effect through a parity gate; 0 balanced, 1 OR-like). Further options in `boolean_head`: `"tables"` (`random` / `unate` / `parity`), `"heads"` (`both` / `policy`), `"position"` (`after_fields` / `before_fields`).
- **CNN head:**
  - it needs field layers and replaces the network's policy readout (frozen). The value head is the usual per-timestep critic; the CNN's advantage of a move is the discounted sum of its TD errors over the move;
  - it can't be combined with `boolean_head` or `policy_head`;
  - it costs about 2–5 ms per move on the CPU, plus one device-to-host transfer of the fields' spike counts per move;
  - watch `entropy` in `<prefix>_policy.csv`: on the CPU, both e-prop runs fell into a deterministic suicidal policy (entropy 0, 3 decisions per episode) by about 0.4M steps and never recovered, since at a saturated softmax the actor gradient is zero. Our rule's runs recovered from near-saturation.
- **Noise head:** set `entropy_bonus` to 0 (the overlay does), since it defines no distribution.
- **Policy plots:** with the noise head, the logged policy (plots and `_policy.csv`) is now the policy the head actually follows, P(argmax(logits + σξ) = a), estimated from 256 fresh noise samples per decision (the action and the PG still use the one sample drawn). With `headless`, which defines no policy, it is each motor pool's share of the pools' activity, a display proxy.
- **Headless:** `"surrogate_tau": τ` in `headless` gives e-prop a softmax surrogate (softmax(pool / τ) - onehot through the exact pool membership); without it e-prop gets no policy signal and does not learn. τ ≈ 0.05 fitted the pool-activity scale on the CPU.
