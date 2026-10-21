# Reviving neurons under the decorrelation reward: the one-sided ("floor") homeostat

`hidden_rule.homeostat_floor: true` makes the homeostat one-sided: c_H * mean|delta| * min(0, psibar - psi_target).
It only acts on neurons whose average sensitivity psibar is below psi_target and is exactly zero for every other
neuron, so it does not compete with the decorrelation reward over the healthy range. It pushes psibar up whichever
way the neuron is dead (silent or saturated): the drift trace gives the direction.

psi_target 0.01 = the monitor's dead threshold (mean psi < 0.01), so only dead and dying neurons are pushed.
The deviation is at most 0.01 (vs up to 0.1 for the two-sided homeostat at target 0.1), hence the larger c_H.

| config | compare with | question |
|---|---|---|
| d5_decorr1_dale_inh_floor3 | your d5 decorr run with Dale "inhibitory" | fewer dead neurons (L4/L5 had 20-24%) without hurting learning? |
| d5_decorr1_dale_inh_floor10 | same | stronger push |
| d5_decorr1_floor10 | arch_gpu/d5_narrow_proposed_decorr1 (no Dale; L4/L5 13-16% dead) | does it help without Dale too? |

What to look at in the monitor CSV: `L*_dead`, `L*_dead_silent` / `L*_dead_firing`, `L*_recovered`, and the score.
Small CPU networks (1 or 5 layers, 6x6) have no dead neurons, so this can only be tested at full size.
