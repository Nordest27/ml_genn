# Network configurations (plain Snake, "toroidal": "local")

Run one at a time:

    cd examples/eprop && PYTHON=/path/to/python configs/run_configs.sh "0 1" configs/network_gpu/<names>.json

All configs use the local-mode defaults (`snake_network.LOCAL_DEFAULTS`):
- connectivity: σ = 1.75 cells, p_max 1, measured in the coarser grid;
- feedforward between layers and into the fields is **excitatory only**: E projects to both the E and I of the next layer, and inhibition stays local;
- gains: `ff_gain` 0.5 between layers, `field_gain` 0.25 into the fields.

Rules: `prop_homeo_c` is proposed + homeostat 1 + centred drift; `random` is random-feedback e-prop.
Every config sets `monitor` and `probe`, so check `outputs/*_neurons.csv` for per-layer Hz, dead fractions and dW onset.

## Starting activity (CPU, first 100 moves, proposed + homeostat)

| network | L1 E/I Hz | L2 E/I | L3 E/I | fields |
|---|---|---|---|---|
| legacy, 1 layer | 30 / 32 | | | **silent** |
| legacy, 3 layers | 32 / 35 | **silent** | **silent** | **silent** |
| n1_base | 13 / 28 | | | 13–21 |
| n1_wide (30×30), measured with gains 0.1 | 15–17 / 19–22 | | | 10–15 (higher at the default gains) |
| n3_const | 10–15 / 27–29 | 8–10 / 17–21 | 9–15 / 16–23 (20% of E dead) | 12–24 |
| n3_lowgain (gain 0.1) | 10–14 / 28 | 0.1–0.4 / 0.3–1 | **silent** | **silent** |
| n3_funnel (20→14→8) | 11–13 / 24–28 | 14–18 / 30–35 | 30–34 / 44–53 | 36–52 |
| n3_expand (8→14→20) | 38–40 / 58–63 | 36–45 / 51–56 | 50–54 / 61–65 | ~105 |

Every legacy network, and the old 1-layer local network with inhibitory feedforward, starts with its fields (and its layers beyond the first) silent. Those layers must be revived by learning, so earlier depth results mix two things: the speed of revival and the speed of learning.

## Configs and predictions

| config | question | prediction |
|---|---|---|
| `n3_const_prop_homeo_c` | Is depth still fast once every layer is active from the start? | Yes, close to n1_base, because the rule is local. Learning onset (dW) should be nearly simultaneous across layers, instead of starting only after revival. |
| `n3_const_random` | Was e-prop's depth penalty actually revival? | e-prop at 3 layers improves a lot over the earlier GPU runs (which had silent L2/L3/fields), but stays slower than at 1 layer. The proposed rule's depth advantage shrinks; that is the key test. |
| `ref_legacy_n3_prop_homeo_c` | Reproduces the earlier GPU depth run (legacy, silent deep layers) | Same as before. A reference for the two configs above. |
| `n3_lowgain_prop_homeo_c` / `n3_ffinh_prop_homeo_c` | Controls: deep layers silent at the start, from weak feedforward or from inhibitory feedforward | Onset delayed relative to n3_const by the revival time. Final performance similar if revival succeeds. |
| `n1_base_prop_homeo_c` / `n1_base_random` | The local network vs the legacy 1-layer runs (connectivity_gpu/legacy_*) | Similar final score, earlier onset (the fields are active from the start). e-prop still faster than proposed on this fixed task. |
| `n1_wide_prop_homeo_c` | Does size help the proposed rule? | Proposed ≥ n1_base (more candidate features for the drift to select from). Moderate confidence. |
| `n3_desc_prop_homeo_c` | Descending fan-in (p_max ×0.6 per depth: 57 → 34 → 21 inputs) | About n3_const, maybe slightly slower. Low confidence: fewer inputs make deep units more selective but noisier. |
| `n3_funnel_prop_homeo_c` / `n3_funnel_random` | Wide → narrow, pooling | Narrow layers start hot (30–50 Hz), so expect a short settling phase. After that, the best 3-layer variant for e-prop (a compact last layer for the readout). Proposed is about n3_const. Low confidence. |
| `n3_expand_prop_homeo_c` | Narrow → wide (8×8 first layer) | Worst: an input bottleneck of 192 neurons for 1200 inputs, plus a hot start. Supports "wide where features are formed". |
| `n1_sw` / `n3_sw` (`p_global` 0.01, about 12 random long-range inputs per neuron) | Do long-range connections within a layer pair help? | About the same as without them. Snake's view is local and the readout already integrates globally. A small value-learning gain is possible. |

Suggested order (one GPU): n3_const_prop_homeo_c, n3_const_random, ref_legacy_n3, n3_lowgain → n1_base ×2 → n1_wide, n3_funnel ×2, n3_expand → the rest.
