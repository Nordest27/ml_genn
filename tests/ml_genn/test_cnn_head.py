import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "eprop"))
import cnn_head  # noqa: E402


def test_cnn_head_gradients_match_finite_differences():
    for seed in (0, 1):
        assert cnn_head.gradient_check(seed) < 1e-5


def test_cnn_head_actor_critic_step_moves_the_policy_toward_a_rewarded_action():
    head = cnn_head.CNNHead((6, 6, 2), 3, channels=(4, 4), lr=1e-2, entropy=0.0, seed=0)
    x = np.random.default_rng(0).random((6, 6, 2))
    p0 = head.policy(head.forward(x)[0])[1]
    for _ in range(20):
        logits, v, cache = head.forward(x)
        head.learn(cache, logits, 1, 1.0)            # action 1 was better than expected
    assert head.policy(head.forward(x)[0])[1] > p0
