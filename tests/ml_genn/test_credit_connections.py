"""network.credit (snake_network.add_credit_connections): credit connections for the learned learning signal."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "eprop"))


def _built(credit, layers=1, reward_pool=None, reward_links=None):
    import snake_hparams
    import snake_network
    from ml_genn.compilers.eprop_compiler import default_params
    np.random.seed(0)
    hp = dict(snake_hparams.DEFAULTS)
    hp.update({"toroidal": "local", "ei_layers": layers, "hid_e": 6, "hid_i": 4, "fan_in": 50,
               "network": {"fields": {"shape": 4}, "credit": credit, "reward_pool": reward_pool,
                                       "reward_links": reward_links}})
    spec = snake_network.spec_from_hparams(hp, (20, 20, 3), 2, channels=3)
    return snake_network.build_network(spec, default_params)


def test_credit_connections_wiring():
    b = _built({"sources": ["policy_field", "value_field"], "k": 5, "self_k": 3, "seed": 1})
    credit = [c for c in b.network.connections if getattr(c, "is_credit", False)]
    targets = [p for p in b.hidden]
    # every hidden ALIF population gets one connection per source plus one from itself
    assert len(credit) == 3 * len(targets)
    for c in credit:
        conn, tgt, src = c.connectivity, c.target(), c.source()
        n_t = int(np.prod(tgt.shape))
        k = 3 if c.name.startswith("credit_self_") else 5   # a field listed as a source also feeds itself (k)
        assert np.all(np.bincount(np.asarray(conn.post_ind), minlength=n_t) == k)        # fixed in-degree
        assert np.all(np.asarray(conn.weight) == 0.0) or float(np.asarray(conn.weight)) == 0.0
        if src is tgt:
            assert not np.any(np.asarray(conn.pre_ind) == np.asarray(conn.post_ind))   # no self-connection
        pairs = np.asarray(conn.post_ind) * 10 ** 6 + np.asarray(conn.pre_ind)
        assert len(np.unique(pairs)) == len(pairs)                                     # no duplicates
    assert all(c.name.startswith("credit_") for c in credit)


def test_credit_connections_checks():
    with pytest.raises(KeyError):
        _built({"sources": ["nope"]})
    with pytest.raises(KeyError):
        _built({"sources": ["policy_field"], "kk": 3})
    b = _built(None)
    assert not any(getattr(c, "is_credit", False) for c in b.network.connections)


def test_reward_pool_wiring():
    b = _built(None, reward_pool={"n": 10, "in_from": ["L1_E"], "k_in": 4, "k_out": 3, "out_scale": 2.0, "seed": 2})
    pool = next(p for p, lab in b.labels.items() if lab == "reward_pool")
    assert pool in b.hidden and int(np.prod(pool.shape)) == 10
    inputs = [c for c in b.network.connections if c.target() is pool]
    assert len(inputs) == 1 and np.all(np.bincount(np.asarray(inputs[0].connectivity.post_ind), minlength=10) == 4)
    rewards = [c for c in b.network.connections if getattr(c, "is_reward", False)]
    targets = [p for p in b.hidden if p is not pool]
    assert {c.target() for c in rewards} == set(targets) and all(c.source() is pool for c in rewards)
    for c in rewards:
        w = np.asarray(c.connectivity.weight)
        assert np.all(np.bincount(np.asarray(c.connectivity.post_ind)) == 3) and (w > 0).any() and (w < 0).any()
    # the TD error is transported to the reward neurons like to every other hidden population
    assert any(c.source() is pool and c.is_feedback and c.name.endswith("tde_transport")
               for c in b.network.connections)
    with pytest.raises(KeyError):
        _built(None, reward_pool={"in_from": ["nope"]})


def test_reward_pool_own_rule():
    rule = {"preset": "proposed", "trace_lambda": 0.9999}
    b = _built(None, reward_pool={"n": 6, "in_from": ["L1_E"], "rule": rule})
    pool = next(p for p, lab in b.labels.items() if lab == "reward_pool")
    assert b.population_rules[pool] == rule
    from ml_genn.compilers.eprop.variants import get_hidden_rule
    assert get_hidden_rule(rule).trace_lambda == 0.9999
    with pytest.raises(ValueError):
        get_hidden_rule({"preset": "proposed", "trace_lambda": 1.0})


def test_reward_pool_signal_channel():
    b = _built(None, reward_pool={"n": 6, "in_from": ["L1_E"], "channel": "signal"})
    out = [c for c in b.network.connections if c.name.startswith("signal_to_")]
    assert out and all(getattr(c, "is_signal", False) and not getattr(c, "is_reward", False) for c in out)
    with pytest.raises(ValueError):
        _built(None, reward_pool={"n": 6, "in_from": ["L1_E"], "channel": "loss"})


@pytest.mark.parametrize("channel", ["reward", "signal"])
def test_reward_links_between_the_network_neurons(channel):
    b = _built(None, reward_links={"channel": channel, "k": 5, "scale": 2.0, "seed": 3})
    links = [c for c in b.network.connections if getattr(c, "is_reward" if channel == "reward" else "is_signal", False)]
    assert links and all(c.name.startswith(f"{channel}link_") for c in links)
    for tgt in b.hidden:
        into = [c for c in links if c.target() is tgt]
        n_t = int(np.prod(tgt.shape))
        indeg = sum(np.bincount(np.asarray(c.connectivity.post_ind), minlength=n_t) for c in into)
        assert np.all(indeg == 5)                                              # k links per target neuron
        for c in into:
            if c.source() is tgt:
                assert not np.any(np.asarray(c.connectivity.pre_ind) == np.asarray(c.connectivity.post_ind))
    w = np.concatenate([np.asarray(c.connectivity.weight) for c in links])
    assert (w > 0).any() and (w < 0).any()
    with pytest.raises(ValueError):
        _built(None, reward_links={"channel": "loss"})
    with pytest.raises(KeyError):
        _built(None, reward_links={"sources": ["nope"]})


def test_reward_links_mirror_the_network_wiring():
    b = _built(None, reward_links={"channel": "signal", "topology": "mirror", "scale": 1.0, "seed": 0})
    hidden = set(b.hidden)
    normal = [c for c in b.network.connections if not c.is_feedback and c.source() in hidden and c.target() in hidden
              and not getattr(c, "is_signal", False)]
    links = [c for c in b.network.connections if getattr(c, "is_signal", False)]
    pairs = lambda cs: sorted((c.source().name, c.target().name, int(i), int(j)) for c in cs
                              for i, j in zip(np.asarray(c.connectivity.pre_ind), np.asarray(c.connectivity.post_ind)))
    assert len(links) == len(normal) and pairs(links) == pairs(normal)       # one link per existing synapse
    with pytest.raises(ValueError):
        _built(None, reward_links={"topology": "grid"})


def test_reward_links_zero_init():
    b = _built(None, reward_links={"channel": "reward", "topology": "mirror", "init": "zero"})
    links = [c for c in b.network.connections if getattr(c, "is_reward", False)]
    assert links and all(np.all(np.asarray(c.connectivity.weight) == 0.0) for c in links)
    with pytest.raises(ValueError):
        _built(None, reward_links={"init": "ones"})


def test_reward_links_reverse_the_network_wiring():
    b = _built(None, reward_links={"channel": "signal", "topology": "reverse", "init": "zero"})
    hidden = set(b.hidden)
    normal = [c for c in b.network.connections if not c.is_feedback and c.source() in hidden and c.target() in hidden
              and not getattr(c, "is_signal", False)]
    links = [c for c in b.network.connections if getattr(c, "is_signal", False)]
    fwd = sorted((c.source().name, c.target().name, int(i), int(j)) for c in normal
                 for i, j in zip(np.asarray(c.connectivity.pre_ind), np.asarray(c.connectivity.post_ind)))
    back = sorted((c.target().name, c.source().name, int(j), int(i)) for c in links
                  for i, j in zip(np.asarray(c.connectivity.pre_ind), np.asarray(c.connectivity.post_ind)))
    assert len(links) == len(normal) and back == fwd          # every synapse i -> j has a link j -> i


def test_route_links_reverse_the_network_wiring():
    b = _built(None, reward_links={"channel": "route", "topology": "reverse", "init": "zero"})
    hidden = set(b.hidden)
    links = [c for c in b.network.connections if getattr(c, "is_route", False)]
    normal = [c for c in b.network.connections if not c.is_feedback and c.source() in hidden and c.target() in hidden
              and not getattr(c, "is_route", False)]
    assert len(links) == len(normal) > 0
    assert all(np.all(np.asarray(c.connectivity.weight) == 0.0) for c in links)


def test_route_links_local_topology_prefers_near_neurons():
    import snake_network
    b = _built(None, reward_links={"channel": "route", "topology": "local", "k": 6, "sigma": 1.0, "init": "zero"})
    links = [c for c in b.network.connections if getattr(c, "is_route", False)]
    assert links
    side = b.layers[0][0].shape[0]
    dists, rnd = [], []
    rng = np.random.default_rng(0)
    for c in links:
        sp = snake_network.volume_positions(tuple(c.source().shape) + (1,) * (3 - len(c.source().shape)), side, False)
        tp = snake_network.volume_positions(tuple(c.target().shape) + (1,) * (3 - len(c.target().shape)), side, False)
        pre, post = np.asarray(c.connectivity.pre_ind), np.asarray(c.connectivity.post_ind)
        assert not np.any((c.source() is c.target()) & (pre == post))          # no self links
        d = np.abs(sp[pre, :2] - tp[post, :2]); d = np.minimum(d, side - d); dists.append(np.hypot(*d.T))
        r = sp[rng.integers(0, len(sp), len(post)), :2] - tp[post, :2]; r = np.minimum(np.abs(r), side - np.abs(r))
        rnd.append(np.hypot(*r.T))
    assert np.mean(np.concatenate(dists)) < 0.6 * np.mean(np.concatenate(rnd))
    per_target = {}
    for c in links:
        for j in np.asarray(c.connectivity.post_ind):
            per_target[(c.target().name, int(j))] = per_target.get((c.target().name, int(j)), 0) + 1
    assert set(per_target.values()) == {6}
