# Arena (coins + enemies in the dark), 2026-11-17

`snake_switch.ArenaEnv`, selected with `"arm": {"task": "arena", ...}` (any other `arm` keys are ArenaEnv options).
A body on a bounded square world collects yellow coins and shoots red enemies; only lit places are visible: the player
light around the body and the gaze light around the gaze point. The gaze is a joystick (the stick sets its offset,
fixed reach 0.6; released, it returns to the body) and also the aim: past half the stick, it fires toward the gaze.
4 action axes (body x/y, gaze x/y) -> 8 motor pools (the "joints" decoder's flexor/extensor pairs) or 8 readout units.

View: an RGB frame centred on the body (never rotated): lit floor grey 0.3, dark floor 0.01, outside black; coins
yellow, enemies red (glowing at 0.05 in the dark), body and shots blue; objects as Gaussian bumps (sigma 0.8 cells,
sub-pixel positions; `"render": "pixels"` for single cells). Needs 3 input channels.

Curriculum (`"curriculum": {}`): a staircase on coins - hits per 3000-step segment (up at >= 15, down at <= 3)
moves a level from 0 to 1: coins 16 -> 2, enemies 0 -> 3, enemy speed 0.004 -> 0.012, both lights 0.6 -> 0.2,
decoys 0 -> 4 of each kind. Only coins pay (kills 0, hits -1): shooting is defence. Decoys glow like coins / enemies
in the dark and are nothing when lit, so the lights are needed.

PPO (MLP 256, persistent exploration noise) on this curriculum: ~40 coins per segment at level ~0.5 after 1.15M
steps; turning the glow off costs it little (32.5 vs 29.5 coins per 3000 steps), so it forages with the lights.
Earlier versions failed instructively: with a kill reward it farmed respawning enemies from a corner (0 coins);
with 8 coins, 1 enemy and 0.45 lights at level 0 it learned to avoid hits instead of foraging.

Configs here are fragments to merge into a full network config (your 1L / 5L ones):
- `arena_rule_motor_pools.json`: the rule, motor pools trained by the learned joint surrogate
- `arena_eprop_motor_pools.json`: centred e-prop, same pools
- `arena_readout.json`: a learned Gaussian readout instead of motor pools (no headless), for either rule

Play it yourself: `python arena_play.py` (Tkinter; WASD move, arrows look/shoot, +/- level, space pause, r reset).
