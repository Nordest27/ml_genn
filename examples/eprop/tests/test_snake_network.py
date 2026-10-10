"""snake_network: specs from the hyperparameters and from "network" overrides build the intended networks."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snake_network as SN  # noqa: E402
from ml_genn.compilers.eprop_compiler import default_params  # noqa: E402

HP = dict(hid_e=6, hid_i=4, fan_in=50, ei_layers=1, explicit_feedback=True, node_sigma=1e-2, toroidal="exact")


def build(**over):
    hp = {**HP, **over}
    return SN.build_network(SN.spec_from_hparams(hp, (20, 20, 3), 4), default_params)


def shapes(b):
    return [tuple(p.shape) for layer in b.layers for p in layer]


def test_default_and_depth():
    b = build()
    assert shapes(b) == [(6, 6, 3), (4, 4, 3)] and b.fields is not None
    assert [b.labels[p] for p in b.hidden] == ["L1_E", "L1_I", "policy_field", "value_field"]
    b3 = build(ei_layers=3)
    assert len(b3.layers) == 3 and b3.labels[b3.layers[2][0]] == "L3_E"


def test_expanding_and_reduced_layers_with_their_own_rules():
    b = build(network={"layers": [{"e": 8}, {"e": 6, "i": 4}, {"e": 4, "rule": {"preset": "eprop"}}],
                       "fields": {"shape": 3}})
    assert shapes(b) == [(8, 8, 3), (6, 6, 3), (6, 6, 3), (4, 4, 3), (4, 4, 3), (3, 3, 3)]
    assert tuple(b.fields[0].shape) == (3, 3, 3)
    assert set(b.population_rules) == set(b.layers[2]) and b.population_rules[b.layers[2][0]] == {"preset": "eprop"}


def test_no_fields_reads_the_last_layer():
    b = build(network={"fields": None})
    assert b.fields is None
    heads = {b.policy, b.value}
    forward = [c for c in b.network.connections if not c.is_feedback and c.target() in heads]
    assert {c.source() for c in forward} == set(b.layers[-1])


def test_explicit_feedback_switch_and_unknown_keys():
    with_fb = build()
    no_fb = build(explicit_feedback=False)
    count = lambda b: sum(1 for c in b.network.connections if c.is_feedback and "feedback" in c.name
                          and "tde" not in c.name)
    assert count(with_fb) == 6 and count(no_fb) == 0       # (E, I) x (policy, value) + one per field
    with pytest.raises(KeyError):
        build(network={"layers": [{"e": 6, "fan_in": 3}]})   # LayerSpec has fan_in_e / fan_in_i, not fan_in


def test_inhibitory_feedforward_override():
    b = build(ei_layers=2, network={"layers": [{"to_next_i": {"mean_scale": 0.033}}, {}]})
    i1, e2 = b.layers[0][1], b.layers[1][0]
    conn = next(c for c in b.network.connections if c.source() is i1 and c.target() is e2)
    assert conn.connectivity.weight.mean == pytest.approx(-3 * 0.033 / np.sqrt(50))
    assert conn.connectivity.exact is True                     # inherited from the layer's to_next settings


def test_matched_mode_reproduces_legacy_density_and_weights():
    np.random.seed(0)
    legacy = build(toroidal="legacy", hid_e=8, hid_i=6)
    np.random.seed(0)
    matched = build(toroidal="matched", hid_e=8, hid_i=6)
    def fan(b, src_pop, tgt_pop):
        c = next(c for c in b.network.connections if c.source() is src_pop and c.target() is tgt_pop)
        return np.bincount(c.connectivity.post_ind).mean(), c.connectivity.weight.mean, c.connectivity.exact
    for k in (0, 1):                                    # E->E and I->E of the first layer
        lf, lw, lex = fan(legacy, legacy.layers[0][k], legacy.layers[0][0])
        mf, mw, mex = fan(matched, matched.layers[0][k], matched.layers[0][0])
        assert abs(mf - lf) <= 1.0 and mw == pytest.approx(lw) and mex is True and lex is False
    with pytest.raises(ValueError):
        build(toroidal="uneven")


def test_local_mode_descending_fan_in_and_expected_fan_in_weights():
    np.random.seed(0)
    b = build(toroidal="local", hid_e=10, hid_i=8, ei_layers=2, local={"p_max_decay": 0.5})
    def conn(src, tgt):
        return next(c for c in b.network.connections if c.source() is src and c.target() is tgt).connectivity
    l1, l2 = conn(b.layers[0][0], b.layers[0][0]), conn(b.layers[1][0], b.layers[1][0])
    f1, f2 = np.bincount(l1.post_ind).mean(), np.bincount(l2.post_ind).mean()
    assert f2 == pytest.approx(f1 / 2, rel=0.15)                         # p_max halves at depth 1
    assert l1.weight.mean == pytest.approx(0.1 / np.sqrt(l1.expected_fan_in((10, 10, 3), (10, 10, 3))))
    assert conn(b.layers[0][0], b.layers[1][0]).p_max == pytest.approx(0.5)   # feedforward into depth 1
    with pytest.raises(KeyError):
        build(toroidal="local", local={"pmax": 1.0})


def test_local_mode_expanding_and_contracting_stacks_use_every_source():
    for sizes in ((6, 12, 20), (20, 10, 4)):
        np.random.seed(0)
        b = build(toroidal="local", ei_layers=3, network={"layers": [{"e": n} for n in sizes]})
        for k in range(2):
            src = b.layers[k][0]                                 # E: the only feedforward source by default
            for tgt in b.layers[k + 1]:
                c = next(c for c in b.network.connections if c.source() is src and c.target() is tgt).connectivity
                out = np.bincount(c.pre_ind, minlength=int(np.prod(src.shape)))
                assert np.all(out > 0), (sizes, k)
            assert not any(c.source() is b.layers[k][1] and c.target() in b.layers[k + 1] for c in b.network.connections)
    np.random.seed(0)                                            # inhibitory feedforward on request
    b = build(toroidal="local", ei_layers=2, local={"feedforward_inhibition": True})
    assert any(c.source() is b.layers[0][1] and c.target() is b.layers[1][0] for c in b.network.connections)
    b = build(toroidal="local", ei_layers=2, network={"layers": [{"to_next_i": {"mean_scale": 0.03}}, {}]})
    assert any(c.source() is b.layers[0][1] and c.target() is b.layers[1][0] for c in b.network.connections)


def test_sheet_sigma_keeps_e_and_i_of_a_layer_on_the_same_neighbourhood():
    np.random.seed(0)
    b = build(toroidal="local", hid_e=20, hid_i=15)
    e, i = b.layers[0]
    def fan(src, tgt):
        c = next(c for c in b.network.connections if c.source() is src and c.target() is tgt).connectivity
        return np.bincount(c.post_ind).mean()
    assert fan(e, i) == pytest.approx(fan(e, e), rel=0.1)                 # I does not pool its own layer
    assert fan(i, e) == pytest.approx(fan(e, e) * (15 / 20) ** 2, rel=0.15)  # sparser I: fewer inputs, same area
    assert fan(b.input, i) == pytest.approx(fan(b.input, e), rel=0.1)
