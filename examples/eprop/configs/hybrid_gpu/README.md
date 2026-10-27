# Expand (our rule) / compress (e-prop) hybrid, and its controls

Architecture `hyb_chan` (untrained tests in analysis/hybrid_arch): the 20x20 sheet throughout, channels
3 -> 4 -> 6 (L1-L3, expanding) -> 3 -> 2 (L4-L5, compressing) -> fields 20x20x2; shift-3 narrow relays as narrow_5l;
weights from the search's calibration (20 Hz). Untrained it keeps as much information as narrow_5l or more (fields
input reconstruction 0.70 vs 0.68, dimensionality 24 vs 22, same memory and responsiveness).

Every config: adaptive feedback, c_reg 1e-2, f_target 10, Dale on inhibitory, L4-L5 policy-only e-prop
(eprop_value 0), fields e-prop with eprop_value 0.1. The fields only receive feedback from their own head, so the
critic part reaches only the value field (which has no recurrence). Learning rate: your default.

| config | L1-L3 | question |
|---|---|---|
| hyb_chan_ours_eprop | our rule + local objectives at 1.0 (as narrow_5l(5)) | the hybrid |
| hyb_chan_frozen_eprop | reservoir preset (frozen) | does shaping by our rule beat a fixed reservoir under the same readout? |
| hyb_chan_eprop | policy-only e-prop | reference: e-prop everywhere on the same architecture |
| narrow_eprop_vfield | (narrow_5l, e-prop everywhere) | your best run + a learning signal for the value field |

Note: c_reg is one compiler value, so 1e-2 also sets the firing-rate gradient in our rule's layers (their rate is
mostly set by the local rate reward at 1.0).
