import numpy as np
import pytest

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import SpikeRecorder
from ml_genn.compilers import InferenceCompiler
from ml_genn.connectivity import Dense, Explicit
from ml_genn.neurons import BooleanUnit, LeakyIntegrate, SpikeInput
from ml_genn.utils.data import preprocess_spikes

WINDOW, PERIOD, N_UNITS = 10, 3, 8


@pytest.mark.parametrize("seed", [0, 1])
def test_boolean_units_apply_their_truth_table_to_the_input_bits_of_each_window(seed):
    """Two inputs, k = 2: window w presents input pattern w (bit 0 = input 0, bit 1 = input 1), with the two inputs
    spiking at different timesteps of the window (the unit ORs them). During the next window a unit fires (at the
    refractory frequency) iff its truth table maps that pattern to 1."""
    times, ids = [], []
    for w in range(4):                                  # patterns 00, 01, 10, 11
        if w & 1:
            times.append(w * WINDOW + 2); ids.append(0)
        if w & 2:
            times.append(w * WINDOW + 6); ids.append(1)
    spikes = preprocess_spikes(np.array(times), np.array(ids), num_neurons=2)
    unit = BooleanUnit(k=2, window=WINDOW, period=PERIOD, seed=seed)
    network = Network()
    with network:
        inp = Population(SpikeInput(max_spikes=len(times)), 2)
        bools = Population(unit, N_UNITS, record_spikes=True)
        out = Population(LeakyIntegrate(readout="var"), 1)
        # every unit reads input 0 with weight 1 and input 1 with weight 2
        Connection(inp, bools, Explicit(pre_ind=np.tile([0, 1], N_UNITS), post_ind=np.repeat(np.arange(N_UNITS), 2),
                                        weight=np.tile([1.0, 2.0], N_UNITS)))
        Connection(bools, out, Dense(weight=0.0))
    compiled = InferenceCompiler(evaluate_timesteps=5 * WINDOW, dt=1.0).compile(
        network, f"test_boolean_units_{seed}")
    with compiled:
        _, data = compiled.evaluate({inp: [spikes]}, {out: [np.zeros(1)]}, "mean_square_error",
                                    callbacks=[SpikeRecorder(bools, key="b")])
    t, i = data["b"][0][0], data["b"][1][0]
    tables = unit.truth_tables(N_UNITS)
    for w in range(4):
        # the pattern of window w is evaluated on its last step; the output spans the following window
        lo, hi = w * WINDOW + WINDOW - 1, (w + 1) * WINDOW + WINDOW - 1
        for u in range(N_UNITS):
            n = np.sum((i == u) & (t >= lo) & (t < hi))
            expected = (int(tables[u]) >> w) & 1
            if expected:   # one spike per period; the phase carries over when the output stays 1 between windows
                assert WINDOW // PERIOD <= n <= -(-WINDOW // PERIOD), (w, u, n)
            else:
                assert n == 0, (w, u, n)


def test_boolean_truth_tables_are_non_constant():
    tables = BooleanUnit(k=2, seed=3).truth_tables(1000)
    assert tables.min() >= 1 and tables.max() <= 14
