# arch_v4_gpu: one-layer architectures from the full-size search (v4), proposed + decorrelation reward

Two purely local one-layer networks of equal quality as untrained representations, found by
`architecture_evolution.py` (score 0.4 L1 + 0.4 fields kept + 0.2 memory, wiring minimum on local synapses).
Both: full size (20x20 E, 15x15 I, 15x15 fields), feed-forward inhibition into the fields on, calibrated to 20 Hz,
no long-range shortcuts. They differ in the shape of the recurrent connectivity:

| config | E spread (cells) | I spread / E spread | inhibition scale | search score |
|---|---|---|---|---|
| d1_mexhat_proposed_decorr1 | 1.0 | 3.0 (Mexican hat: inhibition reaches 3x further) | 3.5 | 0.837 |
| d1_wide_proposed_decorr1   | 2.2 | 1.18 (about the same reach)                         | 4.3 | 0.824 |

Question: does a broad inhibitory surround help the decorrelation reward (competition over a wider neighbourhood)
or hurt it (inhibition approximating the whole layer, so neurons decorrelate from global fluctuations)?

Reservoir controls (`d1_*_reservoir`): the same networks with the hidden layer and fields frozen (only the readout
learns). On the small CPU network the readout alone stayed flat at -0.67 from 0.5M to 1.4M timesteps while
proposed + decorr climbed from -0.66 to +0.30 over 2M.

Five layers (`d5_local_*`): the best purely local 5-layer network so far (search score 0.787: fields keep 77% of
L1, kept per layer 0.95 -> 0.85, no wiring violations). Wide sparse recurrence (sigma 3.0, p_max 0.44), strong
inhibition (8.0), feed-forward sigma 1.47, fields sigma 1.69 (20x20), calibrated to 30 Hz, no feed-forward
inhibition. The search is still running; this may be replaced by a better network later.

Rule: proposed + local decorrelation reward 1.0 (covariance form), c_reg 1e-4, monitor and probe on.
Compare at matched episodes: mean score, E and I rates (monitor CSV), flip|L1_E->L1_I (Dale), voltage loss.

Needs the updated `snake_network.py` (keys `local.inhibition_sigma_ratio` and `local.feedforward_inhibition` into the
fields via `network.fields.conn_i`).

    SNAKE_CONFIG=configs/arch_v4_gpu/d1_mexhat_proposed_decorr1.json python run_headless.py
    SNAKE_CONFIG=configs/arch_v4_gpu/d1_wide_proposed_decorr1.json python run_headless.py
    SNAKE_CONFIG=configs/arch_v4_gpu/d1_mexhat_reservoir.json python run_headless.py        # controls
    SNAKE_CONFIG=configs/arch_v4_gpu/d1_wide_reservoir.json python run_headless.py
    SNAKE_CONFIG=configs/arch_v4_gpu/d5_local_proposed_decorr1.json python run_headless.py  # five layers
    SNAKE_CONFIG=configs/arch_v4_gpu/d5_local_reservoir.json python run_headless.py
