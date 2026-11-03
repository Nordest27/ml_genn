from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING
from pygenn import VarAccess
from .neuron import Neuron
from ..utils.model import NeuronModel

if TYPE_CHECKING:
    from .. import Population


MONITOR_VARS = [("BitSum", "scalar"), ("OutSum", "scalar"), ("Samples", "scalar")]


class BooleanUnit(Neuron):
    """A deterministic logic unit: a fixed random Boolean function of k binary inputs, with no voltage, threshold or
    pseudo-derivative (nothing a gradient could use).

    Its k input synapses carry the fixed weights 1, 2, ..., 2^(k-1) (Delta synapses), so within a timestep the summed
    input is the bit pattern of which inputs spiked. Over a window of ``window`` timesteps these patterns are ORed:
    input j's bit is 1 if it spiked at least once. At the end of the window the unit looks the pattern up in its truth
    table (2^k entries, a non-constant function drawn at random per unit) and, during the next window, fires once
    every ``period`` timesteps (the refractory frequency) if the result is 1, and stays silent if it is 0. The truth
    tables are fixed; the unit does not learn, and the connections into it are static.

    With ``hold`` set there is no window (no clock): an input spike sets that input's bit for ``hold`` timesteps
    (re-armed by every further spike), and the truth table is applied every timestep to the bits currently held, so
    the output changes as soon as an input's bit does.

    ``tables`` picks the family the truth tables are drawn from: "random" (any non-constant function), "unate" (monotone
    in each input, increasing or decreasing: each input's effect on the output has a fixed sign) or "parity" (parity of
    the k inputs or its complement, i.e. XOR / XNOR for k = 2: the effect of each input flips with the others and,
    for balanced inputs, has no linear part).

    For monitoring, BitSum and OutSum accumulate the input bits and the output over Samples evaluations (every step
    with ``hold``, every window end otherwise); the training loop reads and zeroes them.

    Args:
        k:          Number of inputs (1 to 4)
        window:     Length of the input window [timesteps]
        period:     Interval between output spikes while the output is 1 [timesteps]
        seed:       Seed of the random truth tables
        hold:       None: windowed inputs; else held bits: how long an input spike keeps its bit set [timesteps]
        tables:     Family of the truth tables: "random", "unate" or "parity"
    """

    def __init__(self, k: int = 2, window: int = 10, period: int = 3, seed: int = 0, hold: int = None,
                 tables: str = "random", readout=None, **kwargs):
        super(BooleanUnit, self).__init__(readout, **kwargs)
        if not 1 <= int(k) <= 4:
            raise ValueError("BooleanUnit: k must be 1 to 4")
        if hold is not None and int(hold) < 1:
            raise ValueError("BooleanUnit: hold must be at least 1 timestep")
        self.k, self.window, self.period, self.seed = int(k), int(window), int(period), int(seed)
        self.hold = None if hold is None else int(hold)
        if tables not in ("random", "unate", "parity"):
            raise ValueError(f"BooleanUnit: tables must be 'random', 'unate' or 'parity', not {tables!r}")
        self.tables = tables

    def table_family(self) -> np.ndarray:
        """All truth tables of the chosen family, as bitmasks (bit b = output for input pattern b)."""
        k, n_pat = self.k, 2 ** self.k
        if self.tables == "parity":
            par = sum(1 << b for b in range(n_pat) if bin(b).count("1") % 2)
            return np.array([par, (2 ** n_pat - 1) ^ par], np.uint32)
        out = []
        for t in range(1, 2 ** n_pat - 1):
            if self.tables == "unate":
                ok = True
                for j in range(k):
                    diffs = {((t >> (b | 1 << j)) & 1) - ((t >> b) & 1) for b in range(n_pat) if not b >> j & 1}
                    if 1 in diffs and -1 in diffs:      # input j raises the output in one context, lowers it in another
                        ok = False
                        break
                if not ok:
                    continue
            out.append(t)
        return np.array(out, np.uint32)

    def truth_tables(self, n: int) -> np.ndarray:
        """n random non-constant truth tables of k inputs, as bitmasks (bit b = output for input pattern b)."""
        rng = np.random.default_rng(self.seed)
        if self.tables == "random":                  # (kept as before, so that existing seeds give the same tables)
            return rng.integers(1, 2 ** (2 ** self.k) - 1, size=n, endpoint=False).astype(np.uint32)
        return rng.choice(self.table_family(), size=n).astype(np.uint32)

    def get_model(self, population: Population, dt: float, batch_size: int) -> NeuronModel:
        n = int(np.prod(population.shape))
        k = self.k
        # one flag per input (GeNN's transpiler has no bitwise operators): S_j = 1 once input j spiked in the window.
        # The summed input is the bit pattern of this step's spikes; bit j = floor(I / 2^j) mod 2
        decode = "\n".join(f"S{j} = fmax(S{j}, fmod(floor(inBits / {float(2 ** j)}), 2.0));" for j in range(k))
        pattern = " + ".join(f"{float(2 ** j)} * S{j}" for j in range(k))
        clear = " ".join(f"S{j} = 0.0;" for j in range(k))
        if self.hold is not None:
            # held bits: H_j counts down from Hold after each spike of input j; bit j = H_j > 0
            arm = "\n".join(f"H{j} = (fmod(floor(inBits / {float(2 ** j)}), 2.0) > 0.5) ? Hold : fmax(H{j} - 1.0, 0.0);"
                             for j in range(k))
            held = " + ".join(f"{float(2 ** j)} * ((H{j} > 0.5) ? 1.0 : 0.0)" for j in range(k))
            held_count = " + ".join(f"((H{j} > 0.5) ? 1.0 : 0.0)" for j in range(k))
            genn_model = {
                "vars": [*[(f"H{j}", "scalar") for j in range(k)], ("Out", "scalar"), ("Cnt", "int"),
                         *MONITOR_VARS, ("Table", "scalar", VarAccess.READ_ONLY)],
                "params": [("Hold", "scalar"), ("Period", "int")],
                "sim_code": f"""
                    const scalar inBits = floor(Isyn + 0.5);
                    {arm}
                    Out = fmod(floor(Table / pow(2.0, {held})), 2.0);
                    BitSum += {held_count};
                    OutSum += Out;
                    Samples += 1.0;
                    if (Cnt > 0) {{
                        Cnt -= 1;
                    }}
                    """,
                "threshold_condition_code": "Out > 0.5 && Cnt <= 0",
                "reset_code": "Cnt = Period;"}
            var_vals = {**{f"H{j}": 0.0 for j in range(k)}, "Out": 0.0, "Cnt": 0, **{v[0]: 0.0 for v in MONITOR_VARS},
                        "Table": self.truth_tables(n).astype(np.float64)}
            return NeuronModel(genn_model, "Out", {"Hold": float(self.hold), "Period": self.period}, var_vals)
        genn_model = {
            "vars": [*[(f"S{j}", "scalar") for j in range(k)], ("Out", "scalar"), ("Phase", "int"), ("Cnt", "int"),
                     *MONITOR_VARS, ("Table", "scalar", VarAccess.READ_ONLY)],            # fixed: never reset
            "params": [("Window", "int"), ("Period", "int")],
            "sim_code": f"""
                const scalar inBits = floor(Isyn + 0.5);
                {decode}
                Phase += 1;
                if (Phase >= Window) {{
                    const scalar pattern = {pattern};
                    Out = fmod(floor(Table / pow(2.0, pattern)), 2.0);
                    BitSum += {" + ".join(f"S{j}" for j in range(k))};
                    OutSum += Out;
                    Samples += 1.0;
                    {clear}
                    Phase = 0;
                }}
                if (Cnt > 0) {{
                    Cnt -= 1;
                }}
                """,
            "threshold_condition_code": "Out > 0.5 && Cnt <= 0",
            "reset_code": "Cnt = Period;"}
        var_vals = {**{f"S{j}": 0.0 for j in range(k)}, "Out": 0.0, "Phase": 0, "Cnt": 0, **{v[0]: 0.0 for v in MONITOR_VARS},
                    "Table": self.truth_tables(n).astype(np.float64)}
        return NeuronModel(genn_model, "Out", {"Window": self.window, "Period": self.period}, var_vals)
