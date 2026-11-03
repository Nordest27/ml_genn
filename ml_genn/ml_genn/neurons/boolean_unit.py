from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING
from pygenn import VarAccess
from .neuron import Neuron
from ..utils.model import NeuronModel

if TYPE_CHECKING:
    from .. import Population


class BooleanUnit(Neuron):
    """A deterministic logic unit: a fixed random Boolean function of k binary inputs, with no voltage, threshold or
    pseudo-derivative (nothing a gradient could use).

    Its k input synapses carry the fixed weights 1, 2, ..., 2^(k-1) (Delta synapses), so within a timestep the summed
    input is the bit pattern of which inputs spiked. Over a window of ``window`` timesteps these patterns are ORed:
    input j's bit is 1 if it spiked at least once. At the end of the window the unit looks the pattern up in its truth
    table (2^k entries, a non-constant function drawn at random per unit) and, during the next window, fires once
    every ``period`` timesteps (the refractory frequency) if the result is 1, and stays silent if it is 0. The truth
    tables are fixed; the unit does not learn, and the connections into it are static.

    Args:
        k:          Number of inputs (1 to 4)
        window:     Length of the input window [timesteps]
        period:     Interval between output spikes while the output is 1 [timesteps]
        seed:       Seed of the random truth tables
    """

    def __init__(self, k: int = 2, window: int = 10, period: int = 3, seed: int = 0, readout=None, **kwargs):
        super(BooleanUnit, self).__init__(readout, **kwargs)
        if not 1 <= int(k) <= 4:
            raise ValueError("BooleanUnit: k must be 1 to 4")
        self.k, self.window, self.period, self.seed = int(k), int(window), int(period), int(seed)

    def truth_tables(self, n: int) -> np.ndarray:
        """n random non-constant truth tables of k inputs, as bitmasks (bit b = output for input pattern b)."""
        n_funcs = 2 ** (2 ** self.k)
        rng = np.random.default_rng(self.seed)
        return rng.integers(1, n_funcs - 1, size=n, endpoint=False).astype(np.uint32)

    def get_model(self, population: Population, dt: float, batch_size: int) -> NeuronModel:
        n = int(np.prod(population.shape))
        k = self.k
        # one flag per input (GeNN's transpiler has no bitwise operators): S_j = 1 once input j spiked in the window.
        # The summed input is the bit pattern of this step's spikes; bit j = floor(I / 2^j) mod 2
        decode = "\n".join(f"S{j} = fmax(S{j}, fmod(floor(inBits / {float(2 ** j)}), 2.0));" for j in range(k))
        pattern = " + ".join(f"{float(2 ** j)} * S{j}" for j in range(k))
        clear = " ".join(f"S{j} = 0.0;" for j in range(k))
        genn_model = {
            "vars": [*[(f"S{j}", "scalar") for j in range(k)], ("Out", "scalar"), ("Phase", "int"), ("Cnt", "int"),
                     ("Table", "scalar", VarAccess.READ_ONLY)],            # fixed: never reset
            "params": [("Window", "int"), ("Period", "int")],
            "sim_code": f"""
                const scalar inBits = floor(Isyn + 0.5);
                {decode}
                Phase += 1;
                if (Phase >= Window) {{
                    const scalar pattern = {pattern};
                    Out = fmod(floor(Table / pow(2.0, pattern)), 2.0);
                    {clear}
                    Phase = 0;
                }}
                if (Cnt > 0) {{
                    Cnt -= 1;
                }}
                """,
            "threshold_condition_code": "Out > 0.5 && Cnt <= 0",
            "reset_code": "Cnt = Period;"}
        var_vals = {**{f"S{j}": 0.0 for j in range(k)}, "Out": 0.0, "Phase": 0, "Cnt": 0,
                    "Table": self.truth_tables(n).astype(np.float64)}
        return NeuronModel(genn_model, "Out", {"Window": self.window, "Period": self.period}, var_vals)
