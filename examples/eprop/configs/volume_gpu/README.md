# 3D volume: the 5-layer narrow stack with backward connections

`to_prev` (per layer, in `network.layers[k]`) adds layer k -> layer k-1 projections; with `to_next` and the
recurrence the stack becomes a 3D volume whose depth axis is the layer index.

All configs here = `arch_gpu/d5_narrow_proposed_decorr1.json` with 20% of every forward projection's strength moved
to a backward projection (same narrow spread, E sources only; total drive per layer unchanged).

| config | question |
|---|---|
| vol20_proposed_decorr1 | does the volume learn better than `arch_gpu/d5_narrow_proposed_decorr1` (the feedforward stack)? |
| vol20_proposed_decorr1_dale_inh | same with Dale on the inhibitory connections (your "inhibitory" Dale run) |
| vol20_reservoir | the untrained volume (reservoir) for the gap |

Untrained CPU checks (`analysis/volume_untrained`): at 20% the volume keeps the stack's task information through all
five layers (decoding 0.33-0.36) and transforms the input more (linear R2 from the frame at L5 0.49 vs 0.80). At 50%
backward - excitatory, balanced by backward inhibition, net inhibitory, or onto I only - input information is gone
after layer 2 or 3, so stronger backward coupling is not worth GPU time with these gains.
