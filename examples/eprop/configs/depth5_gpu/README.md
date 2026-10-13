# 5 layers: does plastic inhibition stop the signal propagating?

GPU observation: 5 layers, proposed (no homeostat) + plastic inhibition at 50 Hz. Every layer and the readouts are active, but it doesn't learn. Before, the proposed rule learned quickly at depth.

**Hypothesis.** The rate rule strengthens the I cells whose firing co-varies with their E target. In a deep layer, that co-varying part is the signal arriving from the layer below, so each layer's inhibition learns to cancel it (detailed balance). What each layer passes on is then increasingly its own fluctuations. Over 5 layers, the input is explained away step by step.

**A competing factor.** At 50 Hz, neurons sit near threshold most of the time, so ψ ≈ ψ̄ and the drift's (ψ − ψ̄) factor carries little.

| config | separates |
|---|---|
| d5_proposed_pi50 | your run (reference) |
| d5_proposed_pi50_L1 | PI in layer 1 only (`"plastic_inhibition_layers": [1]`): learns → deep PI cancels the propagating signal |
| d5_proposed_pi10, d5_proposed_pi10_L1 | the same at 10 Hz: the 10 Hz pair learns but 50 doesn't → the high rate is the problem |
| d5_proposed_pi_cov | covariance mode in every layer (cancels only excess co-activity, not the mean drive) |
| d5_proposed | no PI: is plain proposed still quick at 5 layers? |
| d5_prop_homeo_c | the earlier quick setup (homeostat, no PI) at 5 layers |

**Direct evidence of the cancelling:**
- `python probe_analyse.py runs/` prints decoding per population (L1_E … L5_E). If the hypothesis holds, decoding falls off with depth much faster with PI in every layer than with PI in layer 1 only.
- `dW|L{k}_I->L{k}_E` shows how much each layer's inhibition grew.

Suggested order: d5_proposed_pi50_L1, d5_proposed, d5_proposed_pi10_L1, then the rest.

## Update: is it the feedforward inhibition that cancels the signal?

Layer n projects to both populations of layer n+1. So layer n+1's inhibition receives a copy of the same incoming signal as its E cells, through two paths:
- E_n -> I_n+1 -> E_n+1 (disynaptic);
- I_n -> E_n+1 (direct, legacy only, 3× weight).

Plastic inhibition on I_n+1 -> E_n+1 can then learn to subtract the incoming signal as it arrives. If the next layer's I cells are driven only by their own layer (feedback inhibition), they can still control rates and compete within the layer, but they can't pre-emptively cancel the input from below.

| config | feedforward from layer n into layer n+1 |
|---|---|
| d5_proposed_pi50 | E_n -> E, I; I_n -> E, I (legacy, your run) |
| d5_pi50_ffE | E_n -> E; I_n -> E (`"feedforward_targets": "e"`) |
| d5_pi50_ffE_noIff | E_n -> E only |
| d5_pi50_ffE_noIff_gain | E_n -> E only, gain 0.5 (every layer active from the start) |
| d5_proposed_ffE_noIff_gain | as above without plastic inhibition (control) |

Starting activity (CPU, full size, first 50 moves, L1 … L5 E rates):
- d5_proposed_pi50 and d5_pi50_ffE: 28 / 0 / 0 / 0 / 0 Hz (the deep layers are revived by learning later).
- d5_pi50_ffE_noIff: 28 / 3.4 / 0.2 / 0 / 0 Hz.

Prediction: the noIff variants learn with plastic inhibition in every layer, while d5_proposed_pi50 stalls.
- If only the gain variant learns, slow revival was the issue rather than cancelling.
- Per-layer decoding (`probe_analyse.py`) should stay level across layers in the noIff variants and fall off in d5_proposed_pi50.

New network option: `"feedforward_targets": "both" | "e"`.
