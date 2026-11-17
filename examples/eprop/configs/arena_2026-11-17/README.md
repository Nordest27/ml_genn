# Arena (coins + enemies in the dark), 2026-11-17

`snake_switch.ArenaEnv`, selected with `"arm": {"task": "arena", ...}` (any other `arm` keys are ArenaEnv options).
A body on a bounded square world collects yellow coins and shoots red enemies; only lit places are visible: the player
light around the body and the gaze light around the gaze point. The gaze is a joystick (the stick sets its offset,
fixed reach 0.6; released, it returns to the body) and also the aim: past half the stick, it fires toward the gaze.
4 action axes (body x/y, gaze x/y) -> 8 motor pools (the "joints" decoder's flexor/extensor pairs) or 8 readout units.

View: an RGB frame centred on the body (never rotated): lit floor grey 0.3, dark floor 0.01, outside black; coins
yellow, enemies red (glowing at 0.05 in the dark), body and shots blue; objects as Gaussian bumps (sigma 0.8 cells,
sub-pixel positions; `"render": "pixels"` for single cells). Needs 3 input channels.

Curriculum (`"curriculum": {}`): a staircase on coins + kills - hits per 3000-step segment (up at >= 15, down at
<= 3) moves a level from 0 to 1: coins 8 -> 2, enemies 1 -> 3, enemy speed 0.004 -> 0.012, both lights 0.45 -> 0.2.
Logged as `level` in the move log; "Snake len" = coins + kills of the segment.

References per 3000-step segment at level 0: standing still -14, random (persistent noise) +14, an oracle with full
knowledge (walk to the nearest coin, shoot the nearest enemy) +314.

Configs here are fragments to merge into a full network config (your 1L / 5L ones):
- `arena_rule_motor_pools.json`: the rule, motor pools trained by the learned joint surrogate
- `arena_eprop_motor_pools.json`: centred e-prop, same pools
- `arena_readout.json`: a learned Gaussian readout instead of motor pools (no headless), for either rule

Play it yourself: `python arena_play.py` (Tkinter; WASD move, arrows look/shoot, +/- level, space pause, r reset).
