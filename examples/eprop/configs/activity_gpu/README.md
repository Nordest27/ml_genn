# Learning at low activity: proposed vs e-prop

The claim to test: the perturbation rule (no homeostat) with plastic inhibition learns well at low firing rates (~6 Hz on the GPU), and e-prop does not. If it holds, it is an energy result: few spikes per decision.

All configs use legacy connectivity and plastic inhibition (rate mode, eta 1e-5) at the given target. e-prop's own firing-rate regulariser (`f_target`) is set to the same target, so the two don't fight. The proposed rule + homeostat did fight it, and its weights ran away.

## 1. Reward against activity (1 layer)

| target | proposed | random e-prop |
|---|---|---|
| 3 Hz | d1_proposed_pi3 | d1_eprop_pi3 |
| 6 Hz | d1_proposed_pi6 | d1_eprop_pi6 |
| 10 Hz | d1_proposed_pi10 | d1_eprop_pi10 |
| 20 Hz | d1_proposed_pi20 | d1_eprop_pi20 |

- Plot the final reward per episode against the achieved mean rate (`L1_E_hz`, `L1_I_hz` and the field rates in `outputs/*_neurons.csv`), one curve per rule.
- The positive result: the proposed curve stays high towards low rates, while e-prop's drops.

## 2. Depth (3 layers)

| config | what it is |
|---|---|
| d3_proposed_pi10 | proposed + plastic inhibition |
| d3_eprop_pi10 | random e-prop + plastic inhibition |
| d3_adaptive_pi10 | adaptive e-prop + plastic inhibition (learned feedback: if feedback alignment is e-prop's bottleneck at depth, this should be less slow) |
| d3_proposed, d3_eprop | without plastic inhibition (references) |

Prediction:
- e-prop stays slow at depth even with plastic inhibition, because its bottleneck is credit assignment (one random feedback projection to every layer), not activity. The 100 Hz e-prop runs were slow too.
- The proposed rule keeps its depth advantage.
- Check `dW|...` per layer for when each layer starts to learn: e-prop's layer 1 should start late.

Run one at a time, at least 2 seeds per config for the sweep. Order: d1 at 6 and 10 Hz (both rules) → 3 and 20 Hz → depth.
