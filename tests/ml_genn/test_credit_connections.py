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


def test_motor_pools_contiguous_and_pool_relay():
    import snake_network
    p = snake_network.motor_pools(4 * 4 * 2, 4, 0, "contiguous", (4, 4, 2))
    assert np.array_equal(np.bincount(p), [8, 8, 8, 8])
    q = p.reshape(4, 4, 2)
    assert np.all(q[:2, :2] == 0) and np.all(q[:2, 2:] == 1) and np.all(q[2:, :2] == 2) and np.all(q[2:, 2:] == 3)
    s = snake_network.motor_pools(6 * 4, 2, 0, "contiguous", (6, 4))
    assert np.array_equal(np.bincount(s), [12, 12])
    r = snake_network.motor_pools(40, 4, 3)
    assert np.array_equal(np.bincount(r), [10, 10, 10, 10])
    with pytest.raises(ValueError):
        snake_network.motor_pools(40, 4, 3, "stripes", (4, 10))


def test_headless_pool_credit_builds_relays_and_motor_rule():
    import snake_hparams
    import snake_network
    from ml_genn.compilers.eprop_compiler import default_params
    np.random.seed(0)
    hp = dict(snake_hparams.DEFAULTS)
    hp.update({"toroidal": "local", "ei_layers": 1, "hid_e": 6, "hid_i": 4, "fan_in": 50,
               "network": {"fields": {"shape": 4}},
               "headless": {"seed": 0, "pools": "contiguous", "pool_credit": True, "pool_homeo": 1e-4}})
    spec = snake_network.spec_from_hparams(hp, (20, 20, 3), 4, channels=3)
    assert spec.motor_pool_layout == "contiguous" and spec.motor_pool_credit and spec.motor_pool_bias
    assert spec.motor_rule["pool_noise"] is True
    b = snake_network.build_network(spec, default_params)
    relays = [c for c in b.network.connections if getattr(c, "is_pool", False)]
    assert len(relays) == 1
    fld = b.fields[0]
    assert relays[0].source() is fld and relays[0].target() is fld and getattr(fld, "pool_bias", False)
    n = int(np.prod(fld.shape))
    pools = snake_network.motor_pools(n, 4, 0, "contiguous", tuple(fld.shape))
    pre, post = np.asarray(relays[0].connectivity.pre_ind), np.asarray(relays[0].connectivity.post_ind)
    assert np.all(pools[pre] == pools[post]) and len(pre) == sum(np.bincount(pools) ** 2)


@pytest.mark.parametrize("mode", ["credit", "inject"])
def test_local_noise_relays(mode):
    b = _built(None)
    import snake_network
    spec_ln = {"sigma": 1.0, "mode": mode}
    snake_network.add_local_noise(b, spec_ln, None)
    relays = [c for c in b.network.connections if getattr(c, "is_pool", False)]
    alif = [p for p in b.hidden if isinstance(p.neuron, snake_network.AdaptiveLeakyIntegrateFire)]
    assert len(relays) == len(alif) and all(c.source() is c.target() for c in relays)
    for c in relays:
        n = int(np.prod(c.target().shape))
        post, w = np.asarray(c.connectivity.post_ind), np.asarray(c.connectivity.weight)
        tot = np.bincount(post, weights=w if mode == "credit" else w * w, minlength=n)
        assert np.allclose(tot, 1.0)                          # mean (credit) or unit variance (inject)
        assert np.bincount(post, minlength=n).min() > 1       # every neuron hears its neighbours, not only itself


def test_pool_noise_inject_rule():
    from dataclasses import replace
    from ml_genn.compilers.eprop import PRESETS, build_td_hidden_model
    kw = dict(c_reg=1e-4, alpha=0.9, rho=0.99, f_target=0.01, alpha_fav=0.998, v_thresh=0.61, td_lambda=0.99)
    code = build_td_hidden_model(replace(PRESETS["proposed"], pool_noise=True, pool_noise_inject=True),
                                 **kw)["model"]["pre_spike_syn_code"]
    assert "const scalar eps = SynSig * PoolNoise_post;" in code and "addToPost(g + eps);" in code
    assert "NoiseTrace += eps;" in code and "epsT" not in code
    with pytest.raises(ValueError):
        replace(PRESETS["proposed"], pool_noise_inject=True)


def test_headless_route_pool_overlay_without_fields():
    """The generalised e-prop overlay on a headless network without fields: the motor pools are the last layer's E
    neurons, which get the motor rule (pool_noise + overrides), pool relays, and route links back along the wiring."""
    import json, pathlib
    import snake_hparams
    import snake_network
    from ml_genn.compilers.eprop_compiler import default_params
    np.random.seed(0)
    over = json.loads((pathlib.Path(snake_network.__file__).parent / "configs" / "route_pool_2026-11-16" /
                       "route_pool.json").read_text())
    hp = dict(snake_hparams.DEFAULTS)
    hp.update({"toroidal": "local", "ei_layers": 2, "hid_e": 6, "hid_i": 4, "fan_in": 50})
    for k, v in over.items():
        hp[k] = {**(hp.get(k) or {}), **v} if isinstance(v, dict) and isinstance(hp.get(k), dict) else v
    hp["headless"] = {"seed": 0, **over["headless"]}
    hp["network"] = {**hp["network"], "fields": None}         # no fields: the last layer is the motor population
    spec = snake_network.spec_from_hparams(hp, (20, 20, 3), 4, channels=3)
    assert spec.fields is None
    assert spec.motor_rule["pool_noise"] is True and spec.motor_rule["gradient"] == 1.0
    assert spec.motor_rule["route_msg_own"] is False          # inherited from the hidden rule (unused: pool noise)
    b = snake_network.build_network(spec, default_params)
    last_e = b.layers[-1][0]
    assert b.population_rules[last_e] is spec.motor_rule
    relays = [c for c in b.network.connections if getattr(c, "is_pool", False)]
    assert len(relays) == 1 and relays[0].source() is last_e
    routes = [c for c in b.network.connections if getattr(c, "is_route", False)]
    assert any(c.source() is last_e for c in routes)          # credit leaves the motor pools backwards


def test_direct_route_links_fill_the_influence_cone():
    """topology 'direct': feedback links only from the motor population, into every hidden population that can reach
    it, at most k per neuron, all within cutoff * sigma_h (sheet distance), and wider cones for deeper populations."""
    import snake_hparams
    import snake_network
    from ml_genn.compilers.eprop_compiler import default_params
    np.random.seed(0)
    hp = dict(snake_hparams.DEFAULTS)
    hp.update({"toroidal": "local", "ei_layers": 3, "hid_e": 8, "hid_i": 4, "fan_in": 50,
               "network": {"fields": None, "reward_links": {"channel": "route", "topology": "direct", "init": "random",
                                                           "k": 6, "sigma": 1.0, "sigma_per_hop": 1.5, "cutoff": 2.0}},
               "hidden_rule": {"preset": "proposed", "route": 1.0, "route_eta": 0.0, "route_msg_own": False,
                               "route_gate": "activity", "route_trust": "scale"},
               "headless": {"seed": 0, "pool_credit": True}})
    spec = snake_network.spec_from_hparams(hp, (20, 20, 3), 4, channels=3)
    b = snake_network.build_network(spec, default_params)
    motor = b.motor_population
    assert motor is b.layers[-1][0]
    links = [c for c in b.network.connections if getattr(c, "is_route", False)]
    assert links and all(c.source() is motor for c in links)
    hops = snake_network._forward_hops(b, motor)
    pos, side = snake_network._hidden_positions(b, spec, {c.target() for c in links} | {motor})
    mean_d = {}
    for c in links:
        tgt = c.target(); h = hops[tgt]; sg = 1.0 + 1.5 * (h - 1)
        pre, post = np.asarray(c.connectivity.pre_ind), np.asarray(c.connectivity.post_ind)
        assert np.bincount(post).max() <= 6
        d = np.abs(pos[motor][pre, :2] - pos[tgt][post, :2]); d = np.minimum(d, side - d)
        dist = np.sqrt((d * d).sum(1))
        assert np.all(dist <= 2.0 * sg + 1e-9)
        assert np.all(np.asarray(c.connectivity.weight) != 0.0)          # init "random": fixed random feedback
        mean_d.setdefault(h, []).append(dist.mean())
    deep, shallow = max(mean_d), min(mean_d)
    assert deep > shallow and np.mean(mean_d[deep]) > np.mean(mean_d[shallow])
    l1 = b.layers[0][0]
    assert hops[l1] == len(b.layers) - 1                       # L1 E -> L2 E -> L3 E (the motor population)
    assert hops[b.layers[0][1]] == hops[l1] + 1                 # its inhibitory population acts through it


@pytest.mark.parametrize("headless", [{"pool_credit": True}, {"motor_rule": {"gradient": 1.0}}])
def test_headless_pool_options_with_a_noise_free_rule_explain_themselves(headless):
    """pool_credit / noise terms in motor_rule on top of the noise-free e-prop rule (surrogate e-prop): a clear error
    that says why; pool_homeo alone is fine."""
    import snake_hparams
    import snake_network
    hp = dict(snake_hparams.DEFAULTS)
    hp.update({"toroidal": "local", "ei_layers": 1, "hid_e": 6, "hid_i": 4, "fan_in": 50, "hidden_rule": "eprop",
               "headless": {"seed": 0, "surrogate_tau": "auto", **headless}})
    with pytest.raises(ValueError, match="surrogate already feeds back"):
        snake_network.spec_from_hparams(hp, (20, 20, 3), 4, channels=3)
    hp["headless"] = {"seed": 0, "surrogate_tau": "auto", "pool_homeo": 1e-4}
    spec = snake_network.spec_from_hparams(hp, (20, 20, 3), 4, channels=3)
    assert spec.motor_pool_bias and spec.motor_rule is None
