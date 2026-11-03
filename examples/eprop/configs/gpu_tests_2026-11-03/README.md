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
