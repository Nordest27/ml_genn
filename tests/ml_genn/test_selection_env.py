import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "eprop"))
import snake_switch  # noqa: E402


def _play(env, policy, episodes):
    total = 0.0
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            action = policy(env) if env.wait_count == 0 else 1
            _, reward, done = env.step(action)
        total += reward
    return total / episodes


def test_selection_env_rewards_and_episode_structure():
    env = snake_switch.SelectionEnv(30, (10, 10, 3), n_candidates=8, seed=0)
    obs = env.reset()
    assert obs.shape == (10, 10, 3) and env.wait_count == 30
    # the oracle commits exactly to good candidates: 1 - P(no good among 8) = 1 - 0.75^8
    assert abs(_play(env, lambda e: 0 if e.good else 1, 3000) - (1 - 0.75 ** 8)) < 0.03
    # always skipping ends every episode with none_reward after 8 decisions
    assert _play(env, lambda e: 1, 50) == 0.0
    # committing at once: E = P(good) - P(bad) = 0.25 - 0.75
    assert abs(_play(env, lambda e: 0, 3000) + 0.5) < 0.05


def test_motor_pools_are_a_balanced_partition():
    import snake_network
    pools = snake_network.motor_pools(48, 4, seed=3)
    assert sorted(np.bincount(pools).tolist()) == [12, 12, 12, 12]


def test_selection_env_switch_changes_the_required_features():
    env = snake_switch.SelectionEnv(0, (10, 10, 3), n_candidates=8, seed=0, switch_moves=50, required_after=[(2, 3)])
    seen = []                              # (switches, ctype, good) at each decision, judged before the step
    for _ in range(200):
        env.reset()
        done = False
        while not done:
            seen.append((env.switches, env.ctype, env.good))
            _, _, done = env.step(1)
    assert env.switches == len(seen) // 50
    for sw, t, good in seen:
        req = (0, 1) if sw % 2 == 0 else (2, 3)
        assert good == all((t >> j) & 1 for j in req)
