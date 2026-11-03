# GPU test overlays (3 November 2026)

Small JSON overlays to add to your working 5-layer config (e.g. the narrow fast-readout decorrelation run):

    python merge.py your_config.json boolean_head.json -o boolean.json
    SNAKE_CONFIG=boolean.json python run_headless.py

`merge.py` merges nested dicts key by key. A `hidden_rule` overlay therefore keeps the rest of your rule settings, so `reservoir.json` changes only the preset.

| Overlay | What it tests | Suggested runs |
|---|---|---|
| `boolean_head.json` | fixed random logic units (BooleanUnit, k = 2) between the fields and the heads: no gradient path from the heads into the network | your rule, and your e-prop config, each with and without it |
| `cnn_head.json` | a host-side convolutional actor-critic (NumPy, online) reads the fields' spike counts each move and acts; its TD error drives the network (episodic) | your rule; `+ reservoir.json` as the frozen-feature reference |
| `noise_head.json` | no policy distribution: argmax over noisy logits, readout trained by node perturbation | your rule (possibility test; σ = 3 was tuned on the input-only agent and may need adjusting to your logit scale) |
| `reservoir.json` | hidden weights frozen (only the head learns) | as a reference with `cnn_head.json` |

## Notes

- **Boolean head:** while its output is 1, a Boolean unit fires at the refractory frequency, so the heads' input (and the initial policy's sharpness) is larger than with the fields. If the Boolean runs learn clearly worse, rerun with weaker readout weights to separate that scale effect from the logic itself.
- **CNN head:**
  - it needs field layers and replaces the network's readout (whose weights are frozen);
  - it can't be combined with `boolean_head` or `policy_head`;
  - it costs about 2–5 ms per move on the CPU, plus one device-to-host transfer of the fields' spike counts per move.
- **Noise head:** set `entropy_bonus` to 0 (the overlay does), since it defines no distribution.
