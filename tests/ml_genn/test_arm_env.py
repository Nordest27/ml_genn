"""The two-link arm task (snake_switch.ArmEnv), its agonist-antagonist decoder and its Gaussian surrogate."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "eprop"))
import snake_switch  # noqa: E402

SHAPE = (20, 20, 3)


def _env(**kw):
    e = snake_switch.ArmEnv(0, SHAPE, seed=0, **kw)
    e.reset()
    return e


def test_forward_kinematics_and_limits():
    e = _env()
    assert np.allclose(e.hand(np.array([0.0, 0.0])), [0.9, 0.0])
    assert np.allclose(e.hand(np.array([np.pi / 2, 0.0])), [0.0, 0.9])
    assert np.allclose(e.hand(np.array([0.0, np.pi / 2])), [0.45, 0.45])
    t = e._clip(np.array([np.pi + 0.5, 3.0]))
    assert np.isclose(t[0], -np.pi + 0.5) and np.isclose(t[1], e.elbow_limit)


def test_targets_are_reachable_and_away_from_the_hand():
    e = snake_switch.ArmEnv(0, SHAPE, seed=1)
    for _ in range(200):
        e.reset()
        r = np.linalg.norm(e.target)
        assert e.min_reach <= r <= e.l1 + e.l2 + 1e-9
        assert np.linalg.norm(e.offset) > 2 * e.radius


def test_reward_is_progress_and_a_reach_ends_the_episode_with_the_bonus():
    e = _env(max_steps=1000)
    d0 = np.linalg.norm(e.offset)
    _, r, done = e.step([0.3, -0.2])
    assert np.isclose(r, e.progress * (d0 - np.linalg.norm(e.offset))) and not done
    e.target = e.hand(e._clip(e.theta + e.joint_speed * np.array([1.0, 0.0])))   # one move away
    _, r, done = e.step([1.0, 0.0])
    assert done and len(e.snake) == 2 and r > e.reach_reward - 1e-6


def test_episode_ends_after_max_steps_and_actions_are_clipped():
    e = _env(max_steps=3)
    t0 = e.theta.copy()
    e.target = np.array([10.0, 10.0])                    # never reached
    e.step([5.0, -5.0])
    assert np.allclose(e._clip(t0 + e.joint_speed * np.array([1.0, -1.0])), e.theta)
    e.step([0.0, 0.0]); _, _, done = e.step([0.0, 0.0])
    assert done


def test_view_channels():
    e = _env()
    obs = e.get_local_img_observation()
    assert obs.shape == SHAPE
    assert obs[..., 0].sum() >= 9                         # the arm: a line of cells from the centre
    assert obs[e._cell(np.zeros(2)) + (0,)] == 1.0        # the shoulder at the centre
    assert obs[e._cell(e.hand()) + (2,)] == 1.0
    assert obs[e._cell(e.target) + (1,)] == 1.0


def test_the_arm_moves_every_step_and_the_bookkeeping_window_cycles():
    e = snake_switch.ArmEnv(2, SHAPE, seed=0, max_steps=1000)
    e.reset()
    e.target = np.array([10.0, 10.0])                     # never reached
    counts = []
    for k in range(6):
        t0 = e.theta.copy()
        counts.append(e.wait_count)
        e.step([1.0, 0.0])
        assert not np.allclose(e.theta, t0)               # every step moves the shoulder
    assert counts == [2, 1, 0, 2, 1, 0]                   # wait_count only cycles (the loop's per-move cadence)
    # the move log's cosine: the hand's displacement over the last window
    e2 = snake_switch.ArmEnv(0, SHAPE, seed=0, max_steps=1000)
    e2.reset(); h0, o0 = e2.hand(), e2.offset
    e2.step([0.5, -0.5])
    dh = e2.hand() - h0
    assert np.isclose(e2.last_cos, dh @ o0 / (np.linalg.norm(dh) * np.linalg.norm(o0)))


def test_joint_velocities_decoder():
    w = snake_switch.joint_velocities([3.0, 1.0, 0.0, 0.0], gain=1.0)
    assert np.allclose(w, [0.5, 0.0])                     # silent joint: 0
    w = snake_switch.joint_velocities([1.0, 3.0, 2.0, 2.0], gain=3.0)
    assert np.allclose(w, [-1.0, 0.0])                    # clipped
    with pytest.raises(ValueError):
        snake_switch.joint_velocities([1.0, 2.0, 3.0], gain=1.0)


def test_joint_surrogate_score():
    rng = np.random.default_rng(0)
    omega = np.array([0.2, -0.4])
    acts, pgs = [], []
    for _ in range(4000):
        a, pg, _ = snake_switch.joint_surrogate(omega, 0.3, rng)
        acts.append(a); pgs.append(pg)
    acts, pgs = np.array(acts), np.array(pgs)
    assert np.allclose(pgs[:, 0], -pgs[:, 1]) and np.allclose(pgs[:, 2], -pgs[:, 3])   # flexor / extensor
    assert np.all(np.abs(acts) <= 1.0)
    assert np.allclose(pgs[:, [0, 2]].mean(0), 0.0, atol=0.06)          # the score has mean zero
    assert np.allclose(pgs[:, [0, 2]].std(0), 1.0, atol=0.06)           # unit scale
    # PG = -d log pi / d omega, exactly -noise: a sample above omega gives the flexor a negative error (after a good
    # outcome the update then pushes omega up), the extensor the opposite; the action is the clipped sample

    class Fixed:
        def standard_normal(self, shape):
            return np.array([1.0, -2.0])
    a, pg, _ = snake_switch.joint_surrogate(np.array([0.2, -0.9]), 0.3, Fixed())
    assert np.allclose(pg, [-1.0, 1.0, 2.0, -2.0])
    assert np.allclose(a, [0.5, -1.0])                     # 0.2 + 0.3, and -0.9 - 0.6 clipped to -1


def test_bad_arm_parameters():
    with pytest.raises(ValueError):
        snake_switch.ArmEnv(0, SHAPE, l1=0.6, l2=0.6)
    with pytest.raises(ValueError):
        snake_switch.ArmEnv(0, SHAPE, min_reach=0.9)


def test_joint_surrogate_ornstein_uhlenbeck_noise():
    """rho > 0: the exploration noise is an OU process with unit variance and lag-1 correlation rho; the error sent
    back is minus the innovation (the exact score given the previous noise); rho = 0 is the white case."""
    rng = np.random.default_rng(1)
    rho, sig = 0.95, 0.2
    omega = np.zeros(2)
    state, ns, pgs, xs = None, [], [], []

    class Rec:
        def standard_normal(self, shape):
            x = rng.standard_normal(shape); xs.append(x.copy()); return x
    r = Rec()
    for _ in range(40000):
        a, pg, state = snake_switch.joint_surrogate(omega, sig, r, rho=rho, state=state)
        ns.append(state.copy()); pgs.append(pg)
    ns, pgs, xs = np.array(ns), np.array(pgs), np.array(xs)
    assert np.allclose(ns[1000:].var(0), 1.0, atol=0.15)
    lag1 = [np.corrcoef(ns[1000:-1, j], ns[1001:, j])[0, 1] for j in range(2)]
    assert np.allclose(lag1, rho, atol=0.02)
    assert np.allclose(pgs[:, [0, 2]], -xs) and np.allclose(pgs[:, [1, 3]], xs)
    a0, pg0, n0 = snake_switch.joint_surrogate(np.array([0.1, 0.2]), 0.3, np.random.default_rng(5))
    a1, pg1, n1 = snake_switch.joint_surrogate(np.array([0.1, 0.2]), 0.3, np.random.default_rng(5), rho=0.0,
                                               state=np.array([3.0, 3.0]))
    assert np.allclose(a0, a1) and np.allclose(pg0, pg1)       # rho = 0 ignores the state


def test_curriculum_raises_the_level_with_the_reach_rate():
    """The level starts at `start`; while it is low, targets start close and the radius is large; it rises by `step`
    once the reach rate over `window` episodes reaches `threshold`, up to 1 (the task's own radius and full reach)."""
    e = snake_switch.ArmEnv(0, SHAPE, seed=0, joint_speed=0.02, max_steps=300,
                            curriculum={"window": 10, "threshold": 0.5, "step": 0.5})
    e.reset()
    assert e.level == 0.0 and np.isclose(e.cur_radius, 0.25)
    d0 = [np.linalg.norm((e.reset(), e.offset)[1]) for _ in range(5)]
    assert max(d0) <= 1.2 * 0.25 + 0.2 + 1e-9 and min(d0) > 1.2 * 0.25
    for _ in range(12):                                   # misses: the level stays
        e.reset()
    assert e.level == 0.0
    for _ in range(12):                                   # reaches: the level rises
        e.snake.append(None); e.reset()
    assert e.level == 0.5
    for _ in range(12):
        e.snake.append(None); e.reset()
    assert e.level == 1.0 and np.isclose(e.cur_radius, e.radius)
    assert "level" in e.move_info()


def test_continuous_mode_staircase_and_sign_reward():
    """continuous: no reset on a reach or a timeout (a new target appears, the level steps up / down), done only every
    segment_steps, and reset() keeps the pose and the target; reward_mode "sign": +-sign_reward per step."""
    e = snake_switch.ArmEnv(0, SHAPE, seed=0, joint_speed=0.02, continuous=True, target_steps=20, segment_steps=50,
                            reward_mode="sign", sign_reward=0.1, curriculum={"up": 0.2, "down": 0.1, "start": 0.5})
    e.reset()
    t0 = e.target.copy()
    for _ in range(20):                                    # never moves: a timeout, level down, a new target
        _, r, d = e.step([0.0, 0.0])
        assert r == 0.0 and not d
    assert e.timeouts == 1 and np.isclose(e.level, 0.4) and not np.allclose(e.target, t0)
    e.target = e.hand(e._clip(e.theta + e.joint_speed * np.array([1.0, 0.0])))     # one step away: a reach
    _, r, d = e.step([1.0, 0.0])
    assert e.reaches == 1 and np.isclose(e.level, 0.6) and np.isclose(r, 0.1 + e.reach_reward) and not d
    for k in range(28):
        _, r, d = e.step([0.0, 0.0])
    assert not d
    _, r, d = e.step([0.0, 0.0])                           # step 50: the segment ends
    assert d and len(e.snake) == 2
    pose, tgt = e.theta.copy(), e.target.copy()
    e.reset()
    assert np.allclose(e.theta, pose) and np.allclose(e.target, tgt) and len(e.snake) == 1 and not e.done


def test_episode_per_target_with_reach_only_reward():
    """continuous + episode_per_target: a reach (reward = reach_reward only) or a timeout ends the episode; reset() keeps
    the pose and draws the next target; reward_mode "reach" pays nothing on the way."""
    e = snake_switch.ArmEnv(0, SHAPE, seed=0, joint_speed=0.02, continuous=True, episode_per_target=True,
                            target_steps=10, reward_mode="reach", curriculum={"start": 0.5})
    e.reset()
    for _ in range(9):
        _, r, d = e.step([0.3, 0.2])
        assert r == 0.0 and not d
    _, r, d = e.step([0.3, 0.2])                            # the 10th step: a timeout ends the episode
    assert d and r == 0.0 and e.timeouts == 1 and np.isclose(e.level, 0.45)
    pose, old = e.theta.copy(), e.target.copy()
    e.reset()
    assert np.allclose(e.theta, pose) and not np.allclose(e.target, old) and not e.done
    e.target = e.hand(e._clip(e.theta + e.joint_speed * np.array([1.0, 0.0])))
    _, r, d = e.step([1.0, 0.0])                            # a reach: the reach reward only, and done
    assert d and r == e.reach_reward and e.reaches == 1 and np.isclose(e.level, 0.5)
    pose = e.theta.copy(); e.reset()
    assert np.allclose(e.theta, pose)
    with pytest.raises(ValueError):
        snake_switch.ArmEnv(0, SHAPE, episode_per_target=True)


def test_progress_reward_fades_with_the_level():
    e = snake_switch.ArmEnv(0, SHAPE, seed=0, joint_speed=0.02, continuous=True, episode_per_target=True,
                            curriculum={"start": 0.25}, progress_fade=0.5)
    e.reset()
    d0 = np.linalg.norm(e.offset); _, r, _ = e.step([0.4, -0.2]); d1 = np.linalg.norm(e.offset)
    assert np.isclose(r, 0.5 * e.progress * (d0 - d1))           # level 0.25 of 0.5: half the progress reward
    e.level = 0.6
    _, r, d = e.step([0.4, -0.2])
    assert r == 0.0 or d                                          # faded out: only a reach pays
    with pytest.raises(ValueError):
        snake_switch.ArmEnv(0, SHAPE, progress_fade=0.5)          # needs a curriculum


def test_point_mode_egocentric():
    """kinematics "point" + view "ego": the action is the hand's velocity on an open plane; the hand sits at the centre
    of the view and the target at its offset; targets start within the band; an ideal controller climbs to level 1."""
    e = snake_switch.ArmEnv(0, SHAPE, seed=0, kinematics="point", view="ego", joint_speed=0.02, continuous=True,
                            episode_per_target=True, curriculum={}, blob=3)
    e.reset()
    obs = e.get_local_img_observation()
    assert obs[e._cell(np.zeros(2)) + (0,)] == 1.0                       # the hand at the centre
    assert obs[e._cell(np.clip(e.offset, -0.999, 0.999)) + (1,)] == 1.0  # the target at its offset
    p0 = e.hand().copy(); e.step([1.0, -0.5])
    assert np.allclose(e.hand(), p0 + 0.02 * np.array([1.0, -0.5]))
    d = np.linalg.norm(e.offset)
    assert d <= 1.2 * e.cur_radius + e.curriculum["dist0"] + 1e-9
    for _ in range(4000):
        _, r, dn = e.step(np.clip(10 * e.offset, -1, 1))
        if dn:
            e.reset()
    assert e.level == 1.0 and e.timeouts == 0
    with pytest.raises(ValueError):
        snake_switch.ArmEnv(0, SHAPE, view="ego")                        # ego with the arm: not supported


def test_polar_mode():
    """kinematics "polar": a telescoping arm (angle, length); joint 0 turns it by joint_speed radians, joint 1 extends
    it by joint_speed x mid-reach; length clipped to [min_reach / 2, l1 + l2]; a controller that turns toward the
    target's angle and extends toward its distance climbs to level 1."""
    e = snake_switch.ArmEnv(0, SHAPE, seed=0, kinematics="polar", joint_speed=0.02, continuous=True,
                            episode_per_target=True, curriculum={})
    e.reset()
    th = e.theta.copy()
    assert np.allclose(e.hand(), th[1] * np.array([np.cos(th[0]), np.sin(th[0])]))
    assert np.allclose(e.elbow(), e.hand())
    assert 0.2 <= th[1] <= 0.9
    e.step([1.0, -1.0])
    assert np.allclose(e.theta, [th[0] + 0.02, th[1] - 0.02 * 0.45])
    for _ in range(200):
        if e.step([0.0, 1.0])[2]:
            e.reset()
    assert abs(e.theta[1] - 0.9) < 1e-12
    obs = e.get_local_img_observation()
    assert obs[e._cell(e.hand()) + (2,)] == 1.0 and obs[e._cell(e.hand() / 2) + (0,)] == 1.0   # hand and link drawn
    for _ in range(20000):
        h, t = e.hand(), e.target
        da = (np.arctan2(t[1], t[0]) - np.arctan2(h[1], h[0]) + np.pi) % (2 * np.pi) - np.pi
        dr = np.linalg.norm(t) - np.linalg.norm(h)
        _, r, dn = e.step(np.clip([20 * da, 20 * dr], -1, 1))
        if dn:
            e.reset()
    assert e.level == 1.0


@pytest.mark.parametrize("kin", ["arm", "polar", "point"])
def test_target_time_tracks_ideal_movement(kin):
    """target_time: the limit is slack x ideal_steps + min_steps (slack from `slack` at level 0 to `slack_hard` at
    level 1); ideal_steps is attainable: driving every motor at full speed toward the target's pose reaches it within
    ideal_steps + 1 steps."""
    e = snake_switch.ArmEnv(0, SHAPE, seed=1, kinematics=kin, joint_speed=0.02, continuous=True, episode_per_target=True,
                            target_time={"slack": 3.0, "slack_hard": 1.5, "min_steps": 10}, radius=0.02)
    for _ in range(20):
        e.reset()
        ideal = e.ideal_steps()
        assert e._tlimit == int(np.ceil(1.5 * ideal + 10))            # no curriculum: level 1, slack_hard
        if kin == "point":
            goal = e.target
        elif kin == "polar":
            goal = np.array([np.arctan2(e.target[1], e.target[0]), np.linalg.norm(e.target)])
        else:
            goal = None
            for el in (1, -1):                                         # the elbow solution ideal_steps used
                c2 = (e.target @ e.target - 0.45 ** 2 * 2) / (2 * 0.45 ** 2)
                t2 = el * np.arccos(np.clip(c2, -1, 1))
                t1 = np.arctan2(e.target[1], e.target[0]) - np.arctan2(0.45 * np.sin(t2), 0.45 + 0.45 * np.cos(t2))
                d = (np.array([t1, t2]) - e.theta + np.pi) % (2 * np.pi) - np.pi
                d[1] = t2 - e.theta[1]
                if abs(t2) <= e.elbow_limit and np.isclose(np.max(np.abs(d)) / 0.02, ideal):
                    goal = d + e.theta
        scale = np.array([0.02, 0.02 * (0.45 if kin == "polar" else 1.0)])
        for k in range(int(ideal) + 2):
            cur = e.hand() if kin == "point" else e.theta
            diff = goal - cur
            if kin != "point":
                diff[0] = (diff[0] + np.pi) % (2 * np.pi) - np.pi
            if e.step(np.clip(diff / scale, -1, 1))[2]:
                break
        assert len(e.snake) > 1, (kin, ideal, k)
    with pytest.raises(ValueError):
        snake_switch.ArmEnv(0, SHAPE, target_time={})                  # needs continuous


def test_target_time_slack_follows_level():
    e = snake_switch.ArmEnv(0, SHAPE, seed=0, continuous=True, curriculum={}, target_time={"slack": 4, "slack_hard": 2,
                                                                                            "min_steps": 0})
    e.reset()
    assert e._tlimit == int(np.ceil(4 * e.ideal_steps()))
    e.level = 0.5; e._new_target()
    assert e._tlimit == int(np.ceil(3 * e.ideal_steps()))


@pytest.mark.parametrize("kin", ["arm", "polar"])
def test_joint_view(kin):
    """view "joint": channel 0 a bump at the current pose, channel 1 at the target's pose, channel 2 at the pose error
    (centre = none); the error bump moves right / up when the target's angle / joint 1 is ahead; the pose bumps wrap
    around in the angle; target_pose puts the hand on the target."""
    e = snake_switch.ArmEnv(0, SHAPE, seed=3, kinematics=kin, view="joint", joint_speed=0.02, continuous=True,
                            episode_per_target=True)
    e.reset()
    assert np.allclose(e.hand(e.target_pose()), e.target)
    obs = e.get_local_img_observation()
    assert obs.shape == SHAPE and obs.max() <= 1.0 and 10 < obs[..., 0].sum() < 40      # one bump per channel
    peak = lambda ch: np.unravel_index(np.argmax(obs[..., ch]), obs.shape[:2])
    err = e.pose_error()
    r, c = peak(2)
    assert (c >= 10) == (err[0] > 0) and (r < 10) == (err[1] > 0)
    e.theta = e.target_pose().copy(); e._render(); obs = e.get_local_img_observation()
    assert set(peak(2)) <= {9, 10}                                                       # no error: the centre
    assert peak(0) == peak(1)
    e.theta[0] = np.pi - 1e-3; e._render(); obs = e.get_local_img_observation()
    assert obs[peak(0)[0], 0, 0] > 0.3                                                   # the angle wraps
    with pytest.raises(ValueError):
        snake_switch.ArmEnv(0, SHAPE, kinematics="point", view="joint")


@pytest.mark.parametrize("kin", ["arm", "polar"])
def test_body_view(kin):
    """view "body": channel 0 the joint-space pose bump, channel 1 the hand (and, for the arm, the elbow at elbow_amp)
    in the workspace, channel 2 the target in the workspace; at the target the hand and target bumps coincide."""
    e = snake_switch.ArmEnv(0, SHAPE, seed=3, kinematics=kin, view="body", joint_speed=0.02, continuous=True,
                            episode_per_target=True)
    e.reset()
    obs = e.get_local_img_observation()
    peak = lambda ch: np.unravel_index(np.argmax(obs[..., ch]), obs.shape[:2])
    near = lambda a, b: max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1
    assert near(peak(1), e._cell(e.hand())) and near(peak(2), e._cell(e.target))
    if kin == "arm":
        y, x = e._cell(e.elbow())
        assert 0.3 < obs[y, x, 1] <= 0.5 + 1e-9                         # the elbow, weaker than the hand
    e.theta = e.target_pose().copy(); e._render(); obs = e.get_local_img_observation()
    assert near(peak(1), peak(2))                                        # reached: the bumps meet


def test_surrogate_gain_learns_the_decoder_jacobian():
    """SurrogateGain: from the step-to-step changes of the pools' traces and the decoded command it recovers the
    decoder's local Jacobian (shape: normalised to the mean |J|), and a saturated joint's gain decays toward 0;
    joint_surrogate with gains scales each pool's score by its gain (gains [1, -1] = the fixed surrogate)."""
    rng = np.random.default_rng(0)
    sg = snake_switch.SurrogateGain(2, rate=0.1)
    base = np.array([0.30, 0.10, 2.0, 0.05])        # joint 0 in range (gain 1), joint 1 saturated (gain 3)
    for _ in range(3000):
        tr = base + 0.01 * rng.standard_normal(4)
        sg.update(tr, np.concatenate([snake_switch.joint_velocities(tr[:2], 1.0),     # in range (0.5)
                                      snake_switch.joint_velocities(tr[2:], 3.0)]))   # saturated (> 1)
    J = sg.J
    f, e = 0.30, 0.10
    true0 = np.array([2 * e, -2 * f]) / (f + e) ** 2
    assert np.allclose(J[0], true0, rtol=0.1)
    assert np.abs(J[1]).max() < 0.05 * np.abs(J[0]).max()      # clipped at +1: no effect, no credit
    g = sg.gains()
    assert np.isclose(np.abs(g[0]).mean() * 2, np.abs(g[0]).sum()) and np.abs(g).mean() > 0
    _, pg_fixed, _ = snake_switch.joint_surrogate(np.zeros(2), 0.1, np.random.default_rng(1))
    _, pg_unit, _ = snake_switch.joint_surrogate(np.zeros(2), 0.1, np.random.default_rng(1),
                                                 gains=np.array([[1.0, -1.0], [1.0, -1.0]]))
    assert np.allclose(pg_fixed, pg_unit)
    _, pg, _ = snake_switch.joint_surrogate(np.zeros(2), 0.1, np.random.default_rng(1), gains=g)
    assert np.allclose(pg.reshape(2, 2), pg_fixed.reshape(2, 2)[:, :1] * g)


@pytest.mark.parametrize("kin", ["arm", "polar"])
def test_hand_view(kin):
    """view "hand": the camera on the hand, the shoulder straight below; turning the whole scene about the shoulder
    leaves the image unchanged; for the polar arm extending moves the target down the frame and turning moves it
    sideways."""
    e = snake_switch.ArmEnv(0, SHAPE, seed=4, kinematics=kin, view="hand", joint_speed=0.02, continuous=True,
                            episode_per_target=True)
    e.reset()
    obs = e.get_local_img_observation().copy()
    sh = e.hand_frame(np.zeros(2))
    assert abs(sh[0]) < 1e-9 and sh[1] < 0                              # the shoulder straight below
    assert np.allclose(e.hand_frame(e.hand()), 0)
    a = 0.7                                                              # turn the scene about the shoulder
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    e.theta = e.theta + np.array([a, 0.0]); e.target = R @ e.target; e._render()
    assert np.allclose(e.get_local_img_observation(), obs)
    if kin == "polar":
        e.target = e.hand() + 0.3 * np.array([np.cos(e.theta[0] + 1.0), np.sin(e.theta[0] + 1.0)])
        t0 = e.hand_frame(e.target)
        e.step([0.0, 1.0]); t1 = e.hand_frame(e.target)                 # extend: the target moves down
        assert t1[1] < t0[1] and abs(t1[0] - t0[0]) < 1e-9
        e.step([1.0, 0.0]); t2 = e.hand_frame(e.target)                 # turn (counter-clockwise): sideways
        assert abs(t2[0] - t1[0]) > 1e-3


def test_coin_lights():
    """CoinGazeEnv: a body-centred frame of fixed size; only lit places (the player light around the body, the gaze
    light around the joystick's gaze point) show coins and floor; the gaze returns to the body when released and its
    reach is fixed; a touched coin pays and respawns; segments end every segment_steps; img() shows world and view."""
    e = snake_switch.CoinGazeEnv(0, SHAPE, seed=0, n_coins=2, reach=0.6, light=0.3, segment_steps=500)
    obs = e.reset()
    assert e.half == pytest.approx(0.9)
    assert obs[..., 0][9:11, 9:11].min() == 1.0 and obs[..., 0].sum() == 9          # the body at the centre
    lit = obs[..., 2] > 0
    assert lit[10, 10] and not lit[0, 0] and lit.sum() < 0.3 * lit.size            # only the player light
    e.coins = np.array([[0.5, 0.0], [-0.5, 0.0]]); e._render()
    assert e.get_local_img_observation()[..., 1].sum() == 0                         # both coins in the dark
    for _ in range(30):
        e.step([0, 0, 1, 0])                                                         # look right
    assert np.allclose(e.gaze_off, [0.6, 0.0], atol=1e-3)
    o = e.get_local_img_observation()
    y, x = e._cell(e.coins[0])
    assert o[y, x, 1] == 1.0 and e.get_local_img_observation()[..., 1].sum() == 9    # the right coin lit, alone
    for _ in range(40):
        e.step([0, 0, 0, 0])
    assert np.linalg.norm(e.gaze_off) < 1e-4
    e.coins[1] = e.body + [0.03, 0.0]
    _, r, _ = e.step([0, 0, 0, 0])
    assert r == 1.0 and len(e.snake) == 2
    e.body = np.array([0.95, 0.0]); e._render()
    o = e.get_local_img_observation()
    assert o[10, 9, 2] == 1.0 and o[10, 13, 2] == 0.0                                # the floor ends at the wall
    b = snake_switch.CoinGazeEnv(0, SHAPE, seed=0, render="bumps"); b.reset()
    assert 0.8 < b.get_local_img_observation()[..., 0].max() <= 1.0              # a bump between pixels
    frame = e.img(scale=4)
    assert frame.shape == (80, 80 + 4 + 80, 3) and frame.dtype == np.uint8
    while not e.step([0, 0, 0, 0])[2]:
        pass
    assert e.steps == 500


def test_coin_curriculum():
    """The staircase: a rich segment raises the level (fewer coins, both lights smaller, the gaze's reach and the
    frame unchanged), a poor one lowers it; "time" follows the step count."""
    e = snake_switch.CoinGazeEnv(0, SHAPE, seed=0, segment_steps=10, reach=0.6,
                                 curriculum={"step": 0.5, "up": 1, "down": 0})
    e.reset()
    assert len(e.coins) == 8 and e.light == 0.3 and e.half == pytest.approx(0.9)
    e.snake.append(None)                                   # one coin in the segment
    e.reset()
    assert e.level == 0.5 and len(e.coins) == 5 and e.light == pytest.approx(0.2) and e.reach == 0.6
    assert e.half == pytest.approx(0.9)
    e.reset()                                              # an empty segment: back down
    assert e.level == 0.0 and len(e.coins) == 8 and e.light == 0.3
    t = snake_switch.CoinGazeEnv(0, SHAPE, seed=0, segment_steps=10 ** 6, curriculum={"mode": "time", "steps": 1000})
    t.reset()
    for _ in range(1000):
        t.step([0, 0, 0, 0])
    assert t.level == 1.0 and len(t.coins) == 2 and t.light == pytest.approx(0.1)



def test_shooter():
    """ShooterEnv: the aim past the threshold fires a projectile (cooldown respected) that kills the enemy it hits
    (+1, respawn); an enemy reaching the body costs -1; the view shows body, enemies and floor; the curriculum adds
    enemies and speed after a good segment; a scripted aim-and-shoot policy scores well, a still one is hit."""
    e = snake_switch.ShooterEnv(0, SHAPE, seed=0, segment_steps=400)
    e.reset()
    e.enemies = np.array([[0.5, 0.0]])
    o = e.get_local_img_observation()
    assert o[..., 0][9:12, 9:12].sum() > 0 and o[..., 2].sum() > 0
    _, r, _ = e.step([0, 0, 1, 0])
    assert len(e.shots) == 1 and e.n_shots == 1
    e.step([0, 0, 1, 0]); assert e.n_shots == 1                               # cooldown
    total = 0.0
    for _ in range(20):
        total += e.step([0, 0, 0, 0])[1]
    assert total == 1.0 and len(e.snake) == 2                                   # the shot killed it
    e.enemies = np.array([[0.05, 0.0]])
    _, r, _ = e.step([0, 0, 0, 0])
    assert r == -1.0 and e.hits == 1
    def play(policy, seed):
        env = snake_switch.ShooterEnv(0, SHAPE, seed=seed, segment_steps=3000); env.reset(); tot = 0.0
        for _ in range(3000):
            tot += env.step(policy(env))[1]
        return tot
    aim = lambda env: np.r_[0, 0, env.nearest() / (np.linalg.norm(env.nearest()) + 1e-9)]
    assert play(aim, 1) > 20 and play(lambda env: np.zeros(4), 1) < 0
    c = snake_switch.ShooterEnv(0, SHAPE, seed=0, curriculum={"step": 0.5, "up": 1, "down": 0})
    c.reset(); c.snake.append(None); c.reset()
    assert c.level == 0.5 and len(c.enemies) == 2 and c.enemy_speed == pytest.approx(0.01)
    assert c.img(scale=4).shape == (80, 164, 3)


def test_arena():
    """ArenaEnv: an RGB picture centred on the body: lit floor grey, dark floor at `dark`, outside black; coins yellow
    and enemies red, lit at full colour and glowing at `glow` in the dark; the gaze stick past the threshold fires
    toward the gaze and kills an enemy; coins pay; the curriculum thins coins, adds enemies and shrinks the lights."""
    e = snake_switch.ArenaEnv(0, SHAPE, seed=0, n_coins=1, n_enemies=1, segment_steps=500, render="pixels")
    e.reset()
    e.coins = np.array([[0.0, 0.8]]); e.enemies = np.array([[0.8, 0.0]]); e._render()
    o = e.get_local_img_observation()
    assert np.allclose(o[10, 10], [0, 0, 1])                                       # the body, blue
    y, x = e._cell(e.coins[0]); assert np.allclose(o[y, x], [0.05, 0.05, 0.0])     # an unlit coin glows yellow
    y, x = e._cell(e.enemies[0]); assert np.allclose(o[y, x], [0.05, 0.0, 0.0])    # an unlit enemy glows red
    assert np.allclose(o[10, 7], [0.3] * 3) and np.allclose(o[1, 1], [0.01] * 3)   # lit and dark floor
    for _ in range(25):
        e.step([0, 0, 0, 1])                                                       # look (and shoot) up
    y, x = e._cell(e.coins[0]); o = e.get_local_img_observation()
    assert np.allclose(o[y, x], [1, 1, 0]) and e.n_shots > 0                       # the coin lit by the gaze
    e.enemies = np.array([[0.5, 0.0]]); e.shots = np.zeros((0, 4)); e.cool = 0; k = e.kills
    for _ in range(15):
        e.step([0, 0, 1, 0])                                                       # look and shoot right
    assert e.kills == k + 1
    e.body = np.array([0.95, 0.0]); e._render(); o = e.get_local_img_observation()
    assert np.allclose(o[10, 19], 0.0)                                             # outside the world: black
    c = snake_switch.ArenaEnv(0, SHAPE, seed=0, curriculum={"step": 1.0, "up": 1, "down": -1})
    c.reset(); c.n_got = 1; c.reset()
    assert c.level == 1.0 and len(c.coins) == 2 and len(c.enemies) == 3 and c.light == pytest.approx(0.2)
    assert c.img(scale=4).shape == (80, 164, 3)


def test_arena_bumps():
    """render "bumps" (default): an object is a Gaussian bump at its exact position, so moving it by a fraction of a
    cell shifts the brightness between neighbouring cells (sub-pixel information); colours as with pixels; the floor
    stays as cells."""
    e = snake_switch.ArenaEnv(0, SHAPE, seed=0, n_coins=1, n_enemies=0)
    e.reset()
    cell = 2 * e.half / SHAPE[1]
    e.coins = np.array([[0.2, 0.0]]); e._render(); a = e.get_local_img_observation().copy()
    e.coins = np.array([[0.2 + 0.3 * cell, 0.0]]); e._render(); b = e.get_local_img_observation().copy()
    y, x = e._cell(np.array([0.2, 0.0]))
    assert a[y, x, 1] > b[y, x, 1] and a[y, x + 1, 1] < b[y, x + 1, 1]          # the coin moved within its cell
    assert np.allclose(a[..., 0][a[..., 2] < 0.29], a[..., 1][a[..., 2] < 0.29])  # yellow: red = green off the floor
    assert np.allclose(a[10, 7], [0.3] * 3)                                       # the lit floor, unchanged



def test_arena_records_segments(tmp_path):
    """record_every: every that many segments the arena saves the segment's trajectory for replays."""
    e = snake_switch.ArenaEnv(0, SHAPE, seed=0, n_enemies=2, segment_steps=50, record_every=2, record_dir=str(tmp_path))
    for _ in range(3):
        e.reset()
        while not e.step([0.5, 0, 1, 0])[2]:
            pass
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["seg000000.npz", "seg000002.npz"]
    z = np.load(tmp_path / "seg000000.npz")
    assert z["body"].shape == (50, 2) and z["enemies"].shape[1:] == (2, 2) and z["coins"].shape == (50, 8, 2)
    snake_switch.CoinGazeEnv(0, SHAPE, record_every=5)          # accepted (ignored)
    snake_switch.ShooterEnv(0, SHAPE, record_every=5)


def test_beta_policy_score():
    """beta_policy: actions in [-1, 1]; the mean follows alpha / (alpha + beta); PG is minus the exact gradient of
    log pi with respect to the readout (checked by finite differences), and 0 where the concentration cap binds."""
    from scipy.stats import beta as B
    r = np.array([0.3, -0.8, 2.0, 0.5])
    a, pg, mean = snake_switch.beta_policy(r, np.random.default_rng(0), scale=1.5)
    assert np.all(np.abs(a) <= 1) and mean.shape == (2,)
    x = (a + 1) / 2
    def logp(rr):
        sp = 1 + np.logaddexp(0, 1.5 * rr.reshape(-1, 2))
        return sum(B.logpdf(x[j], sp[j, 0], sp[j, 1]) for j in range(2))
    num = np.array([(logp(r + 1e-6 * np.eye(4)[k]) - logp(r - 1e-6 * np.eye(4)[k])) / 2e-6 for k in range(4)])
    assert np.allclose(-pg, num, rtol=1e-4, atol=1e-6)
    _, pg2, _ = snake_switch.beta_policy(np.array([100.0, 0.0]), np.random.default_rng(0), max_conc=50)
    assert pg2[0] == 0.0 and pg2[1] != 0.0
    means = np.mean([snake_switch.beta_policy(np.array([2.0, -1.0]), np.random.default_rng(s))[0] for s in range(3000)], 0)
    al, be = 1 + np.logaddexp(0, 2.0), 1 + np.logaddexp(0, -1.0)
    assert abs(means[0] - (2 * al / (al + be) - 1)) < 0.03


def test_joint_beta_surrogate():
    """joint_beta: alpha / beta from the flexor / extensor traces; the mean is the ratio decoder's command in
    probabilistic form (alpha / (alpha + beta)); PG is minus the exact gradient of log pi w.r.t. each pool's trace."""
    from scipy.stats import beta as B
    tr = np.array([0.10, 0.02, 0.0, 0.05])
    a, pg, mean = snake_switch.joint_beta(tr, 50.0, np.random.default_rng(1))
    x = (a + 1) / 2
    def logp(t):
        ab = 1 + 50.0 * t.reshape(-1, 2)
        return sum(B.logpdf(x[j], ab[j, 0], ab[j, 1]) for j in range(2))
    num = np.array([(logp(tr + 1e-7 * np.eye(4)[k]) - logp(tr - 1e-7 * np.eye(4)[k])) / 2e-7 for k in range(4)])
    assert np.allclose(-pg, num, rtol=1e-4, atol=1e-5)
    assert np.isclose(mean[0], 2 * 6.0 / (6.0 + 2.0) - 1) and mean[1] < 0          # flexor ahead: positive
    assert np.all(np.abs(a) <= 1)


def test_arena_decoys():
    """Decoys glow like coins / enemies in the dark, vanish in the light, and touching them does nothing."""
    e = snake_switch.ArenaEnv(0, SHAPE, seed=0, n_coins=1, n_enemies=0, n_coin_decoys=1, n_enemy_decoys=1,
                              render="pixels")
    e.reset()
    e.coins = np.array([[0.9, -0.9]]); e.coin_decoys = np.array([[0.0, 0.8]]); e.enemy_decoys = np.array([[0.8, 0.0]])
    e._render(); o = e.get_local_img_observation()
    y, x = e._cell(e.coin_decoys[0]); assert np.allclose(o[y, x], [0.05, 0.05, 0.0])   # glows like a coin
    y, x = e._cell(e.enemy_decoys[0]); assert np.allclose(o[y, x], [0.05, 0.0, 0.0])   # glows like an enemy
    e.gaze_off = np.array([0.0, 0.6]); e._render(); o = e.get_local_img_observation()
    y, x = e._cell(e.coin_decoys[0]); assert np.allclose(o[y, x], [0.3] * 3)           # lit: just floor
    e.coin_decoys = np.array([[0.02, 0.0]]); e.enemy_decoys = np.array([[0.03, 0.0]])
    _, r, _ = e.step([0, 0, 0, 0])
    assert r == 0.0 and e.hits == 0 and e.n_got == 0
    c = snake_switch.ArenaEnv(0, SHAPE, seed=0, curriculum={"d0": 0, "d1": 4, "step": 1.0, "up": 1, "down": -1})
    c.reset(); assert len(c.coin_decoys) == 0
    c.n_got = 1; c.reset(); assert len(c.coin_decoys) == 4 and len(c.enemy_decoys) == 4


@pytest.mark.parametrize("kin", ["arm", "polar", "point"])
def test_arm_img_shows_world_and_view(kin):
    """ArmEnv.img: the arm from above next to the network's input (as the arena's), uint8 RGB."""
    e = snake_switch.ArmEnv(0, SHAPE, seed=0, kinematics=kin, view="ego" if kin == "point" else "hand",
                            joint_speed=0.02, continuous=True, episode_per_target=True)
    e.reset()
    for _ in range(20):
        e.step([0.5, -0.5])
    im = e.img(scale=6)
    assert im.dtype == np.uint8 and im.shape == (120, 120 + 4 + 120, 3)
    assert (im[:, :120] == [60, 120, 255]).all(-1).any()                     # the hand is drawn
