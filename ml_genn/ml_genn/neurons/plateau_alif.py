from __future__ import annotations

from .adaptive_leaky_integrate_fire import AdaptiveLeakyIntegrateFire


class PlateauALIF(AdaptiveLeakyIntegrateFire):
    """ALIF neuron with a bistable persistent ("plateau") current P, a single-neuron memory:

        P <- P + (sigmoid((V - theta_p) / s) - P) / tau,    V <- Alpha V + g P + gain I,

    with the usual threshold, relative reset and adaptation. With enough feedback g, a depolarisation switches P on
    and P sustains the depolarisation (and firing) long after the input; it decays when the neuron is pushed down.
    The learning rules are not told about P (their presynaptic trace assumes the standard membrane). ``g=0`` and
    ``gain=1`` give the standard neuron's voltage exactly. The code is applied by the e-prop compiler
    (neuron_logic.apply_plateau) after its own modifications of the ALIF model."""

    def __init__(self, g: float = 0.08, theta_p: float = 0.3, s: float = 0.05, tau_p: float = 30.0,
                 gain: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.plateau = dict(g=float(g), theta_p=float(theta_p), s=float(s), tau_p=float(tau_p), gain=float(gain))
