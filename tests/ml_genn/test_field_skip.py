"""Skip connections into the fields (snake_network FieldSpec.skip_from)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "eprop"))


def _built(layers, fields_extra=None):
    import snake_hparams
    import snake_network
    from ml_genn.compilers.eprop_compiler import default_params
    hp = dict(snake_hparams.DEFAULTS)
    hp.update({"toroidal": "local", "ei_layers": layers, "hid_e": 6, "hid_i": 4, "fan_in": 50,
               "local": {"sigma_cells": 0.75, "p_max": 0.791, "input_sigma_cells": 1.857, "inhibition_scale": 7.963,
                         "feedforward_inhibition": True, "p_global": 0.0, "inhibition_sigma_ratio": 1.891,
                         "input_sigma_ref": "source"},
               "network": {"fields": {"shape": 4, "conn": {"sigma_cells": 1.124, "mean_scale": 1.03},
                                      **(fields_extra or {})}}})
    spec = snake_network.spec_from_hparams(hp, (20, 20, 3), 2, channels=3)
    return spec, snake_network.build_network(spec, default_params)


def _into_policy_field(built):
    pf = built.fields[0]
    return sorted(built.labels.get(c.source(), "?") for c in built.network.connections if c.target() is pf)


def test_skip_adds_the_layer_1_sources_and_keeps_the_last_layer():
    _, plain = _built(3)
    assert _into_policy_field(plain) == ["L3_E", "L3_I"]
    spec, skip = _built(3, {"skip_from": [1]})
    assert _into_policy_field(skip) == ["L1_E", "L1_I", "L3_E", "L3_I"]


def test_skip_conn_defaults_to_the_field_connection_of_the_shallow_network():
    spec1, _ = _built(1)
    spec3, _ = _built(3, {"skip_from": [1]})
    assert spec3.fields.skip_conn is None and spec3.fields.conn == spec1.fields.conn   # same L1 -> field connection
    spec3b, _ = _built(3, {"skip_from": [1], "skip_conn": {"mean_scale": 0.5}})
    assert spec3b.fields.skip_conn.mean_scale == 0.5 and spec3b.fields.skip_conn.sigma_cells == 1.124


def test_skip_from_the_last_layer_is_rejected():
    with pytest.raises(ValueError, match="skip_from"):
        _built(3, {"skip_from": [3]})
