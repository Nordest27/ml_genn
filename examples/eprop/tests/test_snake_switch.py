"""snake_switch without GeNN: python -m pytest -q tests"""
import csv, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snake_switch as S  # noqa: E402


class FakeSnake:
    """Moves every `wait_inc` steps; a move earns `good` reward if the env-level action equals `target`."""
    inp_shape = (4, 4, 3)

    def __init__(self, wait_inc=2, target=1):
        self.wait_inc, self.target, self.wait_count, self.received = wait_inc, target, 0, []
        self.img_calls = 0

    def reset(self):
        self.wait_count = self.wait_inc
        o = np.zeros(self.inp_shape); o[0, 0, 0] = 1.0; o[3, 2, 1] = 0.5
        return o

    def step(self, action):
        if self.wait_count > 0:
            self.wait_count -= 1
            return self.reset_obs(), 0.0, False
        self.received.append(action)
        self.wait_count = self.wait_inc
        return self.reset_obs(), (1.0 if action == self.target else -1.0), False

    def reset_obs(self):
        o = np.zeros(self.inp_shape); o[0, 0, 0] = 1.0; o[3, 2, 1] = 0.5
        return o

    def img(self, scale=1):
        self.img_calls += 1
        return np.full((4 * scale, 4 * scale, 3), 100, np.uint8)

    def get_local_img_observation(self):
        return self.reset_obs()


def drive(env, policy, steps):
    obs = env.reset()
    for _ in range(steps):
        obs, r, d = env.step(policy(obs))
    return obs


def test_switch_triggers_on_criterion_and_remaps_actions(tmp_path):
    base = FakeSnake()
    env = S.SwitchingEnv(base, criterion=0.9, window=20, mangles=("actions",), seed=0, log_path=str(tmp_path / "s.csv"))
    drive(env, lambda o: 1, 3 * 25)                  # agent index 1 == env action 1 -> perfect until a switch
    assert env.switches == 1                         # after the switch, index 1 maps elsewhere: rate collapses
    assert int(env.act[1]) != 1
    assert base.received[-1] == int(env.act[1])
    rows = list(csv.reader(open(tmp_path / "s.csv")))
    assert rows[1][0] == "1" and int(rows[1][2]) == 20


def test_no_switch_below_criterion_or_before_window_fills():
    env = S.SwitchingEnv(FakeSnake(), criterion=0.9, window=50, mangles=("channels",), seed=0)
    drive(env, lambda o: 0, 3 * 200)                 # always wrong: rate -1
    assert env.switches == 0
    env = S.SwitchingEnv(FakeSnake(), criterion=0.9, window=50, mangles=("channels",), seed=0)
    drive(env, lambda o: 1, 3 * 49)                  # perfect but window not full
    assert env.switches == 0


def test_channel_permutation_and_flip_are_bijections():
    env = S.SwitchingEnv(FakeSnake(), criterion=-2, window=1, mangles=("channels", "flip"), seed=3)
    o0 = FakeSnake().reset_obs()
    for _ in range(5):
        env._switch()
        o = env._mangle(o0)
        assert sorted(o.ravel()) == sorted(o0.ravel())
        assert o[..., list(env.chan).index(0)].sum() == 1.0      # channel 0 moved to its new place


def test_same_seed_same_sequence_and_delegation():
    a = S.SwitchingEnv(FakeSnake(), mangles=("channels", "actions"), seed=7)
    b = S.SwitchingEnv(FakeSnake(), mangles=("channels", "actions"), seed=7)
    for _ in range(4):
        a._switch(); b._switch()
        assert np.array_equal(a.chan, b.chan) and np.array_equal(a.act, b.act)
    a.wait_count = 5
    assert a.env.wait_count == 5 and a.img(scale=2).shape == (34 + 8, 8 + 4 + 8, 3)
    with pytest.raises(ValueError):
        S.SwitchingEnv(FakeSnake(), mangles=("rotate",))


def test_monitor_classifies_dead_neurons(tmp_path):
    vth = 0.61
    # neuron 0: near threshold (plastic); 1: silent far below; 2: far above and firing (refractory half the time)
    v = np.array([vth, 0.0, 5.0]); refrac_seq = [np.array([0, 0, 2.0]), np.array([0, 0, 0.0])]
    class Pop:
        shape, name = (3,), "h"
        neuron = SimpleNamespace(beta=0.0, v_thresh=vth, tau_refrac=3.0)
    pop = Pop()
    state = {"i": 0}

    class Var:
        def __init__(self, f): self.f = f
        def pull_from_device(self): pass
        @property
        def view(self): return self.f()
    vars_ = {"V": Var(lambda: v), "A": Var(lambda: np.zeros(3)), "RefracTime": Var(lambda: refrac_seq[state["i"] % 2])}
    net = SimpleNamespace(neuron_populations={pop: SimpleNamespace(vars=vars_)}, connection_populations={})
    m = S.NeuronMonitor(net, [pop], ["h"], str(tmp_path / "n.csv"), report_every=10)
    for k in range(10):
        state["i"] = k
        m.sample(k, 0)
    rows = list(csv.DictReader(open(tmp_path / "n.csv")))
    assert float(rows[0]["h_dead"]) == pytest.approx(2 / 3, abs=1e-3)
    assert float(rows[0]["h_dead_silent"]) == pytest.approx(1 / 3, abs=1e-3)
    assert float(rows[0]["h_dead_firing"]) == pytest.approx(1 / 3, abs=1e-3)
    # voltage loss: neuron 2 sits at 5.0, i.e. 5.0 - 0.61 above threshold; the others inside the band
    assert float(rows[0]["h_vloss"]) == pytest.approx((5.0 - vth) / 3, abs=1e-3)


def test_frame_shows_agent_view_and_probs_use_real_directions():
    env = S.SwitchingEnv(FakeSnake(), mangles=("channels", "actions"), seed=1)
    env._switch()
    frame = env.img(scale=10)                                   # header 34 px, board 40 px, sep 4, view 40
    view = frame[34:, 44:]
    expected = np.kron(env.agent_view(), np.ones((10, 10, 1), np.uint8))
    assert np.array_equal(view, expected)
    assert env.agent_view()[0, 0, list(env.chan).index(0)] == 255  # the mangled view, not the raw one
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    real = env.env_probs(probs)
    for k in range(4):
        assert real[env.act[k]] == probs[k]                      # index k moves in direction act[k]


def test_mappings_never_repeat_until_exhausted():
    env = S.SwitchingEnv(FakeSnake(), mangles=("channels", "actions"), seed=0)
    keys = [env._key(env.chan, env.act, env.flip)]
    for _ in range(143):                                   # 6 x 24 = 144 mappings in total
        env._switch()
        keys.append(env._key(env.chan, env.act, env.flip))
    assert len(set(keys)) == 144
    env._switch()                                          # all used: still changes
    assert env._key(env.chan, env.act, env.flip) != keys[-1]


def test_entropy_error_raises_entropy_when_descended():
    rng = np.random.default_rng(0)
    H = lambda z: -(np.exp(z) / np.exp(z).sum() * np.log(np.exp(z) / np.exp(z).sum())).sum()
    for _ in range(20):
        z = rng.normal(0, 2, 4)
        p = np.exp(z - z.max()); p /= p.sum()
        e = S.entropy_error(p, 1.0)
        # matches the numerical gradient of -H
        num = np.array([-(H(z + 1e-5 * np.eye(4)[k]) - H(z - 1e-5 * np.eye(4)[k])) / 2e-5 for k in range(4)])
        np.testing.assert_allclose(e, num, atol=1e-4)
        assert H(z - 0.1 * e) > H(z)                     # a descent step increases the entropy
    assert np.allclose(S.entropy_error(np.full(4, 0.25), 1.0), 0)   # uniform policy: nothing to do


def test_policy_monitor_reports_entropy_and_relative_strength(tmp_path):
    m = S.PolicyMonitor(str(tmp_path / "p.csv"))
    p = np.array([0.7, 0.1, 0.1, 0.1]); pg = p - np.eye(4)[0]
    e = S.entropy_error(p, 0.05)
    for _ in range(3):
        m.decision(p, pg, 0.05, e, abs_td=0.2)
    row = m.end_episode(1, 300, 0)
    h = -(p * np.log(p)).sum()
    assert row[4] == pytest.approx(h) and row[5] == pytest.approx(0.7)
    assert row[8] == pytest.approx(np.linalg.norm(e) / (0.2 * np.linalg.norm(pg)))
    assert m.end_episode(2, 400) is None                      # nothing logged without decisions
    m.decision(np.full(4, 0.25), np.zeros(4))                  # bonus off: ratio undefined, entropy logged
    assert np.isnan(m.end_episode(3, 500)[8])


def test_old_entropy_key_is_rejected_for_snake(tmp_path, monkeypatch):
    import json
    import snake_hparams
    path = tmp_path / "c.json"; path.write_text(json.dumps({"entropy_coeff": 100.0}))
    monkeypatch.setenv("SNAKE_CONFIG", str(path))
    with pytest.raises(KeyError, match="entropy_bonus"):
        snake_hparams.load()
    path.write_text(json.dumps({"entropy_bonus": 0.3}))
    assert snake_hparams.load()["entropy_bonus"] == 0.3


class AppleSnake:
    """Fake board: the apple is drawn in channel 2 of the observation; action 1 eats it (new apple appears)."""
    inp_shape = (4, 4, 3)

    def __init__(self, wait_inc=1):
        self.wait_inc, self.wait_count, self.apples, self.n = wait_inc, 0, [(1, 1)], 0

    def get_local_img_observation(self):
        o = np.zeros(self.inp_shape)
        for (y, x) in self.apples:
            o[y, x, 2] = 1.0
        return o

    def reset(self):
        self.apples = [(1, 1)]; self.wait_count = self.wait_inc
        return self.get_local_img_observation()

    def step(self, action):
        if self.wait_count > 0:
            self.wait_count -= 1
            return self.get_local_img_observation(), 0.0, False
        self.wait_count = self.wait_inc
        if action == 1:
            self.n += 1; self.apples = [((self.n * 2) % 4, 3)]
            return self.get_local_img_observation(), 1.0, False
        return self.get_local_img_observation(), 0.0, False

    def img(self, scale=1):
        return np.full((4 * scale, 4 * scale, 3), 100, np.uint8)


def test_apple_disappears_after_visible_moves_and_reappears_when_respawned():
    env = S.MemoryEnv(AppleSnake(), visible_moves=2)
    o = env.reset()
    assert o[..., 2].sum() == 1                               # visible at spawn
    seen = []
    for _ in range(4):                                       # 2 env steps per move (wait_inc = 1)
        env.step(0); o, r, d = env.step(0)
        seen.append(o[..., 2].sum())
    assert seen == [1, 0, 0, 0]                              # visible for 2 moves (spawn + 1), then hidden
    assert env.env.apples                                    # still on the board
    env.step(1); o, r, d = env.step(1)
    assert r == 1.0 and o[..., 2].sum() == 1                 # eaten while hidden; the new apple is visible
    frame = env.img(scale=2)
    assert frame.shape == (20 + 8, 8 + 4 + 8, 3)


def test_memory_and_switching_compose():
    env = S.SwitchingEnv(S.MemoryEnv(AppleSnake(), visible_moves=1), mangles=("channels",), seed=0)
    env.reset()
    env._switch()
    for _ in range(2):
        env.step(0)
    view = env.agent_view()
    assert view.sum() == 0                                   # the apple is hidden, whatever the channel order
    assert env.img(scale=2).shape[1] == 8 + 4 + 8            # one board, one agent view (no double panel)


def test_monitor_tracks_weight_change_per_connection(tmp_path):
    class Pop:
        shape, name = (2,), "h"
        neuron = SimpleNamespace(beta=0.0, v_thresh=0.61, tau_refrac=3.0)
    class Src:
        name = "inp"
    pop, src = Pop(), Src()
    w = {"g": np.array([1.0, 2.0, 2.0])}

    class G:
        def pull_from_device(self): pass
        @property
        def values(self): return w["g"]
    class Conn:
        is_feedback = False
        def target(self): return pop
        def source(self): return src
    conn = Conn()
    sg = SimpleNamespace(vars={"g": G()})
    state = SimpleNamespace(view=np.zeros(2), pull_from_device=lambda: None)
    vars_ = {k: state for k in ("V", "A", "RefracTime")}
    net = SimpleNamespace(neuron_populations={pop: SimpleNamespace(vars=vars_)}, connection_populations={conn: sg})
    m = S.NeuronMonitor(net, [pop], ["h"], str(tmp_path / "n.csv"), report_every=1)
    w["g"] = np.array([1.0, 2.0, 0.0])                      # one weight moved by 2; ||W0|| = 3
    m.sample(0, 0)
    rows = list(csv.DictReader(open(tmp_path / "n.csv")))
    assert float(rows[0]["dW|inp->h"]) == pytest.approx(2 / 3, abs=1e-3)
