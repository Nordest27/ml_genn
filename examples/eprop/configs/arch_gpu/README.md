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

## Decorrelation reward, correlation-normalised (`"decorr_norm": "correlation"`)

The covariance form, −c (z − z̄)(I − Ī)/Ī, also shrinks when a neuron simply fires less, so going quiet lowers the penalty. With the homeostat at 5 layers the run drifted to low rates.

The correlation form divides by the neuron's spiking spread √(z̄(1 − z̄)) and by its inhibition's s.d.: only how aligned the firing is with the inhibition counts, not how much it fires.

| config | rule |
|---|---|
| d5_narrow_proposed_decorr1_corr | plain proposed + correlation reward |
| d5_narrow_prop_homeo_c_decorr1_corr | homeostat + correlation reward |

Both set `"c_reg": 1e-4` explicitly (the usual strength).

**Note:** if `snake_hparams.py` DEFAULTS on your machine has `"c_reg": 1e-2`, every config that does not set `c_reg` uses it.

## The strong run, and e-prop on the same architecture

GPU, 5 layers, searched architecture: plain proposed + decorrelation reward (covariance form, no homeostat) reached a mean of 7 in under 1,000 episodes at ~55 Hz. No rule had come close at 5 layers before.

| config | question |
|---|---|
| d5_narrow_proposed_decorr1 | the strong run, as a config: rerun with other seeds |
| d5_legacy_proposed_decorr1 | the same reward on the legacy network: is it the reward, or reward + architecture? |
| d5_narrow_eprop_random, d5_narrow_eprop_adaptive | e-prop on the searched architecture |
| d5_legacy_eprop_random, d5_legacy_eprop_adaptive | e-prop on legacy (the reference it should improve on) |

All set `"c_reg": 1e-4` explicitly.

The e-prop runs use no decorrelation reward. The local-objective route correlates the reward with the rule's perturbation noise, which e-prop doesn't have, so "e-prop + decorrelation" would need a separate design.

Suggested order:
1. d5_narrow_proposed_decorr1 (seed 1);
2. d5_narrow_eprop_adaptive;
3. d5_legacy_proposed_decorr1;
4. the rest.

Compare episodes to mean 5 / 7 / 10, and the final mean, and run `python layer_dims.py runs/<run>` on each.

## The key control for both rules: the reservoir

| config | what it is |
|---|---|
| d5_narrow_reservoir | the searched architecture with no hidden learning (only the readout learns) |
| d5_legacy_reservoir | the same on legacy |

How to read them:
- **reservoir ≪ proposed + decorr** → the rule builds features that matter.
- **reservoir ≈ e-prop** → e-prop's random feedback adds little at depth.
- **reservoir close to proposed + decorr** → most of the gain is the architecture.
