# Reward delivery: accumulation fix and the reward decay

**Bug fix (always on, no option):** the value head's reward variable used to be *overwritten* at every reward, so
rewards in quick succession erased each other's remaining trace (the logged/visualised reward_trace did accumulate).
Rewards now add to it. Runs before this fix are not strictly comparable.

`reward_decay_normalise: true` scales every delivered reward by (1 - d) / (1 - d0) (d the per-step decay, d0 that of
reward_decay_env = 0.1), so one reward delivers the same total to the value head whatever the decay. Without it a
slower decay also means a larger reward (0.5 per move: 3.2x, 0.9: 21x), which shifts the balance against c_reg, the
entropy bonus and local rewards.

| config | question |
|---|---|
| arch_gpu/d5_narrow_proposed_decorr1 (unchanged) | control: the accumulation fix alone |
| d5_decorr1_decay05_norm | decay 0.5 per move, normalised (best on the small CPU network) |
| d5_decorr1_decay07_norm | does the best timescale move with network size? |

Small CPU network (5 layers, 6x6), 2 seeds, last 500 games: original +1.60 / +0.08; accumulation fix (decay 0.1)
+1.90 / +0.78; decay 0.5 +1.82 / +1.88; decay 0.9 +0.44 / -0.44 (a learned afterglow weight in the head did not
rescue 0.9). The small network can tell an apple happened only ~1-2 moves back, which fits 0.5 working and 0.9 not.
