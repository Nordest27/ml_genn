# Two-link arm with two joint motors (16 November 2026)

A planar two-link arm reaches for targets, driven every simulation step by two joint motors. There is no decision
window: the joint speed sets how much time the network has per unit of movement.

## The task (`snake_switch.ArmEnv`, config key `arm`)

- **Arm:** shoulder at the centre of the frame, links 0.45 + 0.45. Each step the action is the two joint velocities in
  [−1, 1], times `joint_speed` radians (0.02 here). Targets are reachable by construction.
- **Continuous (`continuous: true`, Gym Reacher style):** the arm is never reset.
  - A reached target is replaced by a new one from wherever the hand is, and so is a target not reached within
    `target_steps` (300).
  - `done` comes only every `segment_steps` (3000), for logging. The pose and the target carry over, and
    "Snake len" − 1 is the number of reaches in the segment.
  - Use `episodic_traces: false`, so segment ends are not terminal.
- **Difficulty staircase (`curriculum`):** a level in [0, 1] sets the reach radius (0.25 → 0.1) and how far targets
  start (close → anywhere). It goes up by 0.05 after each reach and down by 0.05 after each timeout, so the task tracks
  what the learner can just manage. The level reached is a direct measure of skill.
- **Reward:** progress, 10 × (decrease of the hand-to-target distance) per step, plus 1 on a reach. `reward_mode:
  "sign"` (±0.05 per step closer or farther, as in Chadderdon et al. 2012) learned more slowly with PPO.
- **View:** top-down, allocentric: channel 0 the arm (3 cells wide), 1 the target (5 × 5), 2 the hand (3 × 3).
- **Motor side, headless decoder `joints`:** four motor pools, the flexor and extensor of each joint. Each joint's
  velocity is `gain` × (flexor − extensor) / (flexor + extensor) of a short trace of the pools' spikes
  (`joints_decay` 0.9 per step).
- **`surrogate_sigma`:** e-prop's Gaussian surrogate per joint, with the score sent back through the pool membership.
- **`surrogate_rho`:** Ornstein–Uhlenbeck exploration noise with the exact innovation score. It was slower than white
  noise for PPO in every setting tried.

## Results so far (CPU)

- **PPO** (backprop, 1 × 256 ReLU, γ 0.95, white noise), continuous with the staircase: level 0.91 at 600k steps,
  about 8 reaches per 3000-step segment. It learns the full task.
- **Without the continuous staircase,** every learner was much slower. On the slow per-step arm (0.005 rad per step)
  nothing learned direction (hand moving toward the target at chance).
- **Spiking candidates on the continuous arm:** running.

## Overlays (merge onto a headless config)

| Overlay | |
|---|---|
| `arm_rule.json` | the continuous arm with your rule (no policy) |
| `arm_rule_surrogate_motor.json` | your rule in the hidden layers; the motor pools are pure readouts trained by the centred surrogate (σ 0.1) |
| `arm_eprop_surrogate.json` | surrogate e-prop, centred, `c_reg` 1e-4 |
| `arm_v057_base.json` | the full CPU base used here: the v057 volume, input × 3, field gain 2.0 (motor pools spike in ~95 % of 31-step windows) |

## Tools

- `arm.record_every` saves trajectories.
- `arm_replay.py` turns them into GIFs; in continuous mode the targets and their radius change over the replay.
- `move_log.activity` / `activity_at` record per-move spike counts.
- `volume_activity.py` draws the volume's sheets.
