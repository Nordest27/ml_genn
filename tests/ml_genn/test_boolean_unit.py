import os

import numpy as np
import pytest

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import SpikeRecorder
from ml_genn.compilers import InferenceCompiler
from ml_genn.connectivity import Dense, Explicit
from ml_genn.neurons import BooleanUnit, LeakyIntegrate, SpikeInput
from ml_genn.utils.data import preprocess_spikes

WINDOW, PERIOD, N_UNITS = 10, 3, 8
BACKEND = os.environ.get("MLGENN_TEST_BACKEND", "single_threaded_cpu")


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
    compiled = InferenceCompiler(evaluate_timesteps=5 * WINDOW, dt=1.0, backend=BACKEND).compile(
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


@pytest.mark.parametrize("seed", [0, 1])
def test_held_bits_apply_the_truth_table_every_timestep(seed):
    """hold: an input spike sets its bit for HOLD steps and the table is applied every step. Input 0 spikes at t = 5,
    input 1 at t = 9: bits (0, 0) before t = 5, (1, 0) for 5..8, (1, 1) for 9..12 (input 0 held until 5 + HOLD - 1),
    (0, 1) for 13..16 and (0, 0) after; while a unit's output is 1 it fires every PERIOD steps."""
    HOLD = 8
    spikes = preprocess_spikes(np.array([5, 9]), np.array([0, 1]), num_neurons=2)
    unit = BooleanUnit(k=2, period=PERIOD, hold=HOLD, seed=seed)
    network = Network()
    with network:
        inp = Population(SpikeInput(max_spikes=2), 2)
        bools = Population(unit, N_UNITS, record_spikes=True)
        out = Population(LeakyIntegrate(readout="var"), 1)
        Connection(inp, bools, Explicit(pre_ind=np.tile([0, 1], N_UNITS), post_ind=np.repeat(np.arange(N_UNITS), 2),
                                        weight=np.tile([1.0, 2.0], N_UNITS)))
        Connection(bools, out, Dense(weight=0.0))
    compiled = InferenceCompiler(evaluate_timesteps=30, dt=1.0, backend=BACKEND).compile(network, f"test_held_bits_{seed}")
    with compiled:
        _, data = compiled.evaluate({inp: [spikes]}, {out: [np.zeros(1)]}, "mean_square_error",
                                    callbacks=[SpikeRecorder(bools, key="b")])
    t, i = np.asarray(data["b"][0][0]), np.asarray(data["b"][1][0])
    tables = unit.truth_tables(N_UNITS)
    pattern = np.zeros(30, int)                       # the held pattern seen at each step (spike at t acts at t + 1)
    for step in range(30):
        b0 = 6 <= step <= 5 + HOLD
        b1 = 10 <= step <= 9 + HOLD
        pattern[step] = int(b0) + 2 * int(b1)
    for u in range(N_UNITS):
        on = np.array([(int(tables[u]) >> pattern[s]) & 1 for s in range(30)], bool)
        fired = np.zeros(30, bool); fired[t[i == u].astype(int)] = True
        assert not np.any(fired & ~on), (u, np.where(fired & ~on))       # never fires while its output is 0
        # fires every PERIOD steps (from the first step) through each stretch where its output is 1
        cnt = 0
        for s in range(30):
            expect = on[s] and cnt <= 0
            assert fired[s] == expect, (u, s, pattern[s])
            cnt = PERIOD if expect else cnt
            cnt -= 1


@pytest.mark.parametrize("k", [2, 3])
def test_unate_and_parity_table_families(k):
    """unate: every input's effect has one sign in all contexts (k = 2: all but XOR / XNOR); parity: XOR / XNOR."""
    n_pat = 2 ** k
    unate = BooleanUnit(k=k, tables="unate").table_family()
    assert len(unate) == {2: 12, 3: 102}[k]                 # unate functions minus the two constants
    parity = BooleanUnit(k=k, tables="parity").table_family()
    for t in parity:
        out = [(int(t) >> b) & 1 for b in range(n_pat)]
        for j in range(k):                                   # flipping any input always flips the output
            assert all(out[b] != out[b ^ (1 << j)] for b in range(n_pat))
    assert not set(parity.tolist()) & set(unate.tolist())
    drawn = BooleanUnit(k=k, tables="parity", seed=5).truth_tables(200)
    assert set(drawn.tolist()) == set(parity.tolist())


def test_held_bits_monitor_counts():
    """BitSum / Samples is the mean number of held input bits per step; OutSum / Samples the output's on-fraction."""
    HOLD = 8
    spikes = preprocess_spikes(np.array([5, 9]), np.array([0, 1]), num_neurons=2)
    unit = BooleanUnit(k=2, period=PERIOD, hold=HOLD, tables="parity", seed=0)
    network = Network()
    with network:
        inp = Population(SpikeInput(max_spikes=2), 2)
        bools = Population(unit, 1)
        out = Population(LeakyIntegrate(readout="var"), 1)
        Connection(inp, bools, Explicit(pre_ind=np.array([0, 1]), post_ind=np.array([0, 0]),
                                        weight=np.array([1.0, 2.0])))
        Connection(bools, out, Dense(weight=0.0))
    compiled = InferenceCompiler(evaluate_timesteps=30, dt=1.0, backend=BACKEND).compile(
        network, "test_held_bits_monitor")
    with compiled:
        compiled.evaluate({inp: [spikes]}, {out: [np.zeros(1)]}, "mean_square_error")
        npop = compiled.neuron_populations[bools]
        got = {}
        for v in ("BitSum", "OutSum", "Samples"):
            npop.vars[v].pull_from_device()
            got[v] = float(np.asarray(npop.vars[v].view).ravel()[0])
    # bits held: input 0 for steps 6..13, input 1 for 10..17 -> 16 bit-steps; XOR/XNOR output 1 for 8 steps
    # (only one bit held: 6..9 and 14..17) or 22 (XNOR: the other 30 - 8)
    assert got["Samples"] == 30.0
    assert got["BitSum"] == 2 * HOLD
    assert got["OutSum"] in (8.0, 22.0)
