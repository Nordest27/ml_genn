# e-prop with the gradient voltage regulariser

`hidden_rule.vreg_gradient = c`: the derivative of the squared hinge 1/2 [max(0, V - th)^2 + max(0, -V - th)^2]
(th = Vthresh + Beta A, the adaptive threshold), with e-prop's dV/dW = ZFilter and dA/dW = epsA, added to every hidden
synapse's update: saturated neurons' input weights are pulled down, strongly hyperpolarised ones' up, nothing inside
the band. Checked against finite differences and against the simulated state (tests/.../test_eprop_rules.py).
This is the e-prop-style voltage regulariser (the original code had it as `0.0 * CReg * VReg * (...)`, i.e. off, and
with the penalty's magnitude instead of its signed slope). Its target is the collapse failure mode: the critic part's
drift saturating the hidden layers (L5 membranes up to +2.3 against a threshold of 0.61).

Both configs = arch_gpu/d5_narrow_eprop_random with the critic part ON (the default eprop_value 0.1), so the question
is whether the regulariser alone holds the collapse back. Use the same settings as your narrow_5l_eprop run otherwise
(lr, Dale). Two strengths: 0.01 and 0.1 (AdaBelief normalises the total update per weight, so what matters is the
size relative to the e-prop terms; the regulariser is zero for every neuron inside the band).
What to watch: L*_E_hz, L*_vloss and dead_firing should stay flat; the score past ~5k episodes.
