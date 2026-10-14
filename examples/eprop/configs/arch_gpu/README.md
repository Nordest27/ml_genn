# 5 layers on the searched architecture

Found by `reservoir_search.py` at full size (untrained, layer-wise calibrated to ~20 Hz). Changes from the default local network:
- **feed-forward neighbourhood σ 0.45 cells** (instead of 1.75): each layer projects almost one-to-one, so it doesn't smooth away the previous layer's detail;
- **a narrow projection into the fields** (σ 0.45);
- **calibration folded into the weight scales**: input × 0.79, layers 2–5 × 1.5–1.8, fields × 1.9.

Untrained, against the default local network:

| | default | this architecture |
|---|---|---|
| input recoverable at layer 5 | 68% of layer 1's | 90% |
| spike-count dimension L1 -> L5 | shrinks (10.7 -> 8.8) | grows (11.1 -> 17.7) |
| fields (input reconstruction / dim) | 0.35 / 6 | 0.62 / 17 |
| perturbation reach from each layer to the fields | 0.08–0.16 | 0.56–0.73 |

Cost: more deep neurons rarely near threshold (ψ ≈ 0 for up to 29% at L4–L5).

| config | rule |
|---|---|
| d5_narrow_prop_homeo_c | proposed + homeostat (your best at 5 layers) |
| d5_narrow_proposed | plain proposed |
| d5_narrow_prop_homeo_c_decorr1 | proposed + homeostat + the local decorrelation reward |
| d5_legacy_prop_homeo_c | reference: legacy connectivity, your current 5-layer setup |

    cd examples/eprop && PYTHONPATH=../../ml_genn SNAKE_CONFIG=configs/arch_gpu/d5_narrow_prop_homeo_c.json python run_headless.py
    # or several, one at a time:
    PYTHON=python configs/run_configs.sh "0 1" configs/arch_gpu/*.json

Per layer after training: `python layer_dims.py runs/<run>`.
