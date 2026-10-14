import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inhibition_normaliser import normalise_columns  # noqa: E402


def test_normalisation_restores_each_neurons_total_and_keeps_its_ratios():
    rng = np.random.default_rng(0)
    post = rng.integers(0, 20, 400)
    g0 = -np.abs(rng.normal(0.05, 0.01, 400))
    target = np.bincount(post, weights=np.abs(g0), minlength=20)
    g = g0 * rng.uniform(0.5, 3.0, 400)                    # the plastic rule grew and redistributed inhibition
    out = normalise_columns(g, post, target)
    assert np.allclose(np.bincount(post, weights=np.abs(out), minlength=20), target)   # totals restored
    assert np.all(out <= 0)                                                              # still inhibitory
    for j in range(20):                                     # redistribution within each neuron preserved
        m = post == j
        if m.sum() > 1:
            assert np.allclose(out[m] / out[m].sum(), g[m] / g[m].sum())


def test_neurons_without_inhibition_are_left_alone():
    post = np.array([0, 0, 1])
    g = np.array([0.0, 0.0, -0.2])
    out = normalise_columns(g, post, np.array([0.1, 0.1]))
    assert np.allclose(out, [0.0, 0.0, -0.1])
