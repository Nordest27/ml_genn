# Value-head options

| config | change |
|---|---|
| d5_decorr1_value_sqrt_head | `value_transform: "sqrt_head"`: the head learns h(V) as below, but the network receives the TD error in the original units (same scale as the normal rule) |
| d5_decorr1_value_sqrt | `value_transform: "sqrt"` on arch_gpu/d5_narrow_proposed_decorr1: the value head learns h(V),
h(x) = sign(x)(sqrt(|x|+1)-1) + 0.001 x, with the transformed TD target h(r + gamma h^-1(V_h)) (Pohlen et al. 2018) |

Compare with your d5_narrow_proposed_decorr1 runs. Logged values (and the V-vs-G analysis) are then in transformed
units: apply h^-1 (ml_genn.compilers.eprop.neuron_logic.value_transform_h_inv) before comparing with returns.
The broadcast TD error is compressed too (large surprises damped), which also affects the hidden rule.
Other new options (snake.py, off by default): reward_decay_normalise, value_afterglow
{"lr", "max"}, network.reward_relay.
