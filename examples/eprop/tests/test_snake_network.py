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
