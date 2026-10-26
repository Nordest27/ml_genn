# e-prop with a centred critic part

`hidden_rule.eprop_center_value: true`: the critic (value) part of e-prop's hidden update uses delta minus its running
mean; the policy part keeps delta. e-prop's eligibility is never negative, so with a biased critic (E[delta] != 0) the
value part pushes all of a neuron's incoming weights the same way (by the sign of its value feedback weight). Upward
drift reinforces itself through the recurrence (more firing -> larger presynaptic traces), downward drift stops itself
(a silent neuron has no eligibility): the runaway seen in the 5-layer Snake run (30 -> 130 Hz, saturated deep layers,
policy losing certainty after ~5k episodes, at lr 1e-6 too).

d5_narrow_eprop_random_center = arch_gpu/d5_narrow_eprop_random + eprop_center_value (critic part only).
d5_narrow_eprop_random_center_td = the same with eprop_center_td: the whole update uses delta - its mean (for the
policy part a baseline: no change in expectation, lower variance). Use the same settings as your
narrow_5l_eprop run otherwise (lr, Dale), so the only difference is the centring.
What to watch: L*_E_hz and L*_vloss should plateau instead of climbing; score past ~5k episodes.
(The modular compiler's value feedback already carries only the feedback weight, value_feedback_ret_e = 0, so the
TD error is not accumulated in the trace; the old monolithic compiler still does that.)

CPU check (1 layer, small, where e-prop worked before; 2 seeds, final 500 games): plain e-prop -0.37 / +6.45 (one seed
never took off), critic part centred +6.46 / +7.79 (both took off, ahead at every checkpoint). Two seeds only.
