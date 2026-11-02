from __future__ import annotations

import numpy as np

from .adaptive_leaky_integrate_fire import AdaptiveLeakyIntegrateFire


class ResonatorALIF(AdaptiveLeakyIntegrateFire):
    """ALIF neuron with a resonate-and-fire membrane: the voltage V and a second variable W form a damped oscillator,

        (V, W) <- Alpha R(2 pi dt / period) (V, W) + (gain I, 0),

    with the usual threshold, relative reset (on V) and adaptation. The learning rules are not told about W: their
    presynaptic trace assumes the standard monotonic decay (the "unknown model" condition). ``period=None`` is the
    standard neuron exactly. The code is applied by the e-prop compiler (neuron_logic.apply_resonator) after its own
    modifications of the ALIF model."""

    def __init__(self, period: float = None, gain: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.period = None if period is None else float(period)
        self.gain = float(gain)

    def rotation(self, dt: float = 1.0):
        theta = 0.0 if self.period is None else 2.0 * np.pi * dt / self.period
        return float(np.cos(theta)), float(np.sin(theta)), self.gain
