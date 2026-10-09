"""ToroidalGaussian2D: the legacy sampler is unchanged; exact=True gives the requested fan-in on a proper torus."""
import contextlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from ml_genn.connectivity.toroidal_gaussian_2d import ToroidalGaussian2D


def _connect(conn, src_shape, tgt_shape=None, same=False, seed=0):
    np.random.seed(seed)
    src = SimpleNamespace(shape=src_shape)
    tgt = src if same else SimpleNamespace(shape=tgt_shape)
    with contextlib.redirect_stdout(io.StringIO()):
        conn.connect(src, tgt)
    return conn.pre_ind, conn.post_ind


def test_exact_fan_in_and_no_self_connections():
    pre, post = _connect(ToroidalGaussian2D(sigma=0.05, fan_in=300, weight=0.0, exact=True), (20, 20, 3), same=True)
    fan = np.bincount(post, minlength=1200)
    assert np.all(fan == 300)
    assert not np.any(pre == post)
    assert len(set(zip(pre.tolist(), post.tolist()))) == len(pre)            # no duplicate synapses


def test_exact_is_uniform_on_the_torus():
    pre, post = _connect(ToroidalGaussian2D(sigma=0.1, fan_in=300, weight=0.0, exact=True), (20, 20, 3), (20, 20, 3))
    cols = np.bincount((pre // 3) % 20, minlength=20)
    rows = np.bincount((pre // 3) // 20, minlength=20)
    for counts in (cols, rows):
        assert counts.min() > 0.93 * counts.mean() and counts.max() < 1.07 * counts.mean()


def test_exact_allows_matching_indices_between_populations_and_caps_fan_in():
    pre, post = _connect(ToroidalGaussian2D(sigma=2.0, fan_in=500, weight=0.0, exact=True), (4, 4, 2), (4, 4, 2))
    assert np.all(np.bincount(post, minlength=32) == 32)                     # capped at all 32 sources
    assert np.any(pre == post)                                              # different populations: allowed


def test_exact_is_local():
    pre, post = _connect(ToroidalGaussian2D(sigma=0.05, fan_in=50, weight=0.0, exact=True), (20, 20, 1), (20, 20, 1))
    pr, pc = pre // 20, pre % 20
    qr, qc = post // 20, post % 20
    d = np.minimum(np.abs(pr - qr), 20 - np.abs(pr - qr)) ** 2 + np.minimum(np.abs(pc - qc), 20 - np.abs(pc - qc)) ** 2
    assert np.sqrt(d).mean() < 5                                            # concentrated around the target position


def test_legacy_unchanged(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "legacy_toroidal", "/home/user/repos/personal/backups/toroidal_gaussian_2d_before_fix.py",
        submodule_search_locations=None)
    import os
    if spec is None or not os.path.exists(spec.origin):
        pytest.skip("original implementation not available on this machine")
    src = open(spec.origin).read().replace("from .sparse_base import SparseBase",
                                           "from ml_genn.connectivity.sparse_base import SparseBase") \
                                  .replace("from ..utils.value import", "from ml_genn.utils.value import")
    ns = {"__name__": "legacy_toroidal"}
    exec(compile(src, "legacy", "exec"), ns)
    for shapes, same in ((((20, 20, 3), (15, 15, 3)), False), (((15, 15, 3), None), True)):
        new = _connect(ToroidalGaussian2D(sigma=0.05, fan_in=300, weight=0.0), *shapes, same=same)
        old = _connect(ns["ToroidalGaussian2D"](sigma=0.05, fan_in=300, weight=0.0), *shapes, same=same)
        np.testing.assert_array_equal(new[0], old[0]); np.testing.assert_array_equal(new[1], old[1])
