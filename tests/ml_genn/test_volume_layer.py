"""The 3D volume layer (snake_network VolumeSpec): geometry of the input, internal and field connections."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "eprop"))


def _built(volume, layers=1, seed=0):
    import snake_hparams
    import snake_network
    from ml_genn.compilers.eprop_compiler import default_params
    np.random.seed(seed)
    hp = dict(snake_hparams.DEFAULTS)
    hp.update({"toroidal": "local", "ei_layers": layers, "hid_e": 6, "hid_i": 4, "fan_in": 50,
               "local": {"feedforward_inhibition": True},
               "network": {"fields": {"shape": 4}, "layers": [{"volume": volume}] + [{}] * (layers - 1)}})
    spec = snake_network.spec_from_hparams(hp, (20, 20, 3), 2, channels=3)
    return snake_network.build_network(spec, default_params)


def _conn(built, src_label, tgt_label):
    inv = {v: k for k, v in built.labels.items()}
    src = built.input if src_label == "input" else inv[src_label]
    tgt = inv[tgt_label]
    c = [c for c in built.network.connections if c.source() is src and c.target() is tgt]
    assert len(c) == 1, (src_label, tgt_label, len(c))
    return c[0].connectivity


def test_volume_shapes_and_depth_routing():
    b = _built({"depth": 6, "input_depth": 2, "output_depth": 1, "p_global": 0.0})
    e, i = b.layers[0]
    assert e.shape == (6, 6, 6) and i.shape == (4, 4, 6)
    z_e = np.arange(np.prod(e.shape)) % 6                     # flat C order: depth is the last axis
    c = _conn(b, "input", "L1_E")
    assert len(c.post_ind) > 0 and set(np.unique(z_e[c.post_ind])) <= {0, 1}       # input reaches the front slabs only
    f = _conn(b, "L1_E", "policy_field")
    assert len(f.pre_ind) > 0 and set(np.unique(z_e[f.pre_ind])) == {5}            # fields read the back slab only
    z_i = np.arange(np.prod(i.shape)) % 6
    fi = _conn(b, "L1_I", "policy_field")
    assert set(np.unique(z_i[fi.pre_ind])) == {5}


def test_volume_local_kernel_and_global_connections():
    local = _conn(_built({"depth": 8, "sigma_cells": 0.7, "p_global": 0.0}), "L1_E", "L1_E")
    z = np.arange(6 * 6 * 8) % 8
    dz_local = np.abs(z[local.pre_ind] - z[local.post_ind])
    assert dz_local.max() <= 4 and np.mean(dz_local <= 1) > 0.9               # local: mostly neighbouring slabs
    assert not np.any(local.pre_ind == local.post_ind)                         # no self-connections
    glob = _conn(_built({"depth": 8, "sigma_cells": 0.7, "p_global": 0.02}), "L1_E", "L1_E")
    assert np.sum(np.abs(z[glob.pre_ind] - z[glob.post_ind]) >= 5) > 20        # global: long jumps across the depth
    w = np.asarray(glob.weight)
    assert np.all(w > -1e-9) or np.mean(w > 0) > 0.95                         # excitatory source


def test_volume_needs_a_single_layer():
    with pytest.raises(ValueError, match="volume"):
        _built({"depth": 4}, layers=2)


def test_volume_z_shift_makes_excitation_flow_forward():
    z = np.arange(6 * 6 * 8) % 8
    iso = _conn(_built({"depth": 8, "sigma_cells": 0.7, "p_global": 0.0}), "L1_E", "L1_E")
    fwd = _conn(_built({"depth": 8, "sigma_cells": 0.7, "p_global": 0.0, "z_shift": 1.0}), "L1_E", "L1_E")
    dz_iso, dz_fwd = z[iso.post_ind] - z[iso.pre_ind], z[fwd.post_ind] - z[fwd.pre_ind]
    assert abs(np.mean(dz_iso)) < 0.1 and np.mean(dz_fwd) > 0.5           # targets lie deeper than their sources
    inh = _conn(_built({"depth": 8, "sigma_cells": 0.7, "p_global": 0.0, "z_shift": 1.0}), "L1_I", "L1_E")
    zi = np.arange(4 * 4 * 8) % 8
    assert abs(np.mean(z[inh.post_ind] - zi[inh.pre_ind])) < 0.2           # inhibition stays isotropic


def test_volume_k_global_counts():
    """k_global: an exact long-range count on top of the local connections, in- or out-degree, no duplicates/self."""
    import snake_network as sn
    rng = np.random.RandomState(0)
    pos = sn.volume_positions((8, 8, 6), 8, True, 1.0)
    n = len(pos)

    def pairs(**kw):
        c = sn.volume_connectivity(pos, pos, 8, 1.0, 0.8, 0.0, +1, 0.6, 0.1, 1.0, same=True, rng=rng, **kw)
        return np.asarray(c.pre_ind), np.asarray(c.post_ind)

    pre0, post0 = pairs()                                   # local only (same rng stream is fine: counts differ)
    for mode in ("in", "out"):
        pre, post = pairs(k_global=5, k_global_mode=mode)
        keys = post.astype(np.int64) * n + pre
        assert len(np.unique(keys)) == len(keys) and not np.any(pre == post)
        # the long-range part: pairs beyond the local kernel's reach are all global
        d = np.abs(pos[post] - pos[pre]); d[:, :2] = np.minimum(d[:, :2], 8 - d[:, :2])
        far = (d ** 2).sum(1) > 6.0 ** 2                      # p_local ~ 0.8 exp(-18) there: never local
        if mode == "in":
            per = np.bincount(post, minlength=n)
        else:
            per = np.bincount(pre, minlength=n)
        assert per.min() >= 5                                 # every neuron got at least its 5 globals
        assert far.sum() > 0
    # exact totals: local count from a fresh stream + n * k
    r1, r2 = np.random.RandomState(3), np.random.RandomState(3)
    a = sn.volume_connectivity(pos, pos, 8, 1.0, 0.8, 0.0, +1, 0.6, 0.1, 1.0, same=True, rng=r1)
    b = sn.volume_connectivity(pos, pos, 8, 1.0, 0.8, 0.0, +1, 0.6, 0.1, 1.0, same=True, rng=r2, k_global=5)
    assert len(b.pre_ind) == len(a.pre_ind) + 5 * n
    c = sn.volume_connectivity(pos, pos, 8, 1.0, 0.8, 0.0, +1, 0.6, 0.1, 1.0, same=True,
                               rng=np.random.RandomState(3), k_global=2.5)
    assert abs((len(c.pre_ind) - len(a.pre_ind)) / n - 2.5) < 0.15     # fractional: Bernoulli remainder
    with pytest.raises(ValueError):
        sn.volume_connectivity(pos, pos, 8, 1.0, 0.8, 0.01, +1, 0.6, 0.1, 1.0, k_global=5)


def test_volume_k_global_in_network():
    b = _built({"depth": 6, "input_depth": 1, "output_depth": 1, "k_global": 4})
    e, i = b.layers[0]
    n_e = int(np.prod(e.shape))
    ee = _conn(b, "L1_E", "L1_E")
    loc = _built({"depth": 6, "input_depth": 1, "output_depth": 1, "p_global": 0.0})
    assert len(ee.pre_ind) - len(_conn(loc, "L1_E", "L1_E").pre_ind) == pytest.approx(4 * n_e, rel=0.05)
