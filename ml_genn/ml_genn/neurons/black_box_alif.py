from __future__ import annotations

import zlib

import numpy as np

from .adaptive_leaky_integrate_fire import AdaptiveLeakyIntegrateFire


class BlackBoxALIF(AdaptiveLeakyIntegrateFire):
    """ALIF neuron whose dynamics differ per neuron in ways a learning rule is not told about ("black box").

    With heterogeneity ``h`` (0 = the standard ALIF neuron, exactly), every neuron j draws its own:

    - nonlinear leak  V <- LeakA_j V + LeakB_j tanh(LeakC_j V)  (slope at rest AlphaEff_j = LeakA_j + LeakB_j LeakC_j);
    - saturating input  InG_j InS_j tanh(I / InS_j)  (gain InG_j at small inputs);
    - threshold  Thr_j  and partial relative reset  V -= ResetF_j (Thr_j - Vreset);
    - adaptation time constant (RhoJ_j) and strength (Beta_j);
    - whether it integrates its input while refractory (IntRefr_j).

    Only the per-neuron parameters are local information (the e-prop compiler gives both learning rules Thr_j,
    AlphaEff_j, RhoJ_j and Beta_j); the functional form is not. The code is applied by the e-prop compiler
    (compilers/eprop/neuron_logic.apply_black_box) after its own modifications of the ALIF model.
    """

    def __init__(self, heterogeneity: float = 0.0, black_box_seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.heterogeneity = float(heterogeneity)
        self.black_box_seed = int(black_box_seed)

    def coefficients(self, population, dt: float = 1.0) -> dict:
        """Per-neuron coefficients for this population (deterministic in seed and population name)."""
        n = int(np.prod(population.shape))
        h = self.heterogeneity
        rng = np.random.default_rng([self.black_box_seed, zlib.crc32(population.name.encode())])
        g = lambda: rng.standard_normal(n)
        vth = float(self.v_thresh.value if hasattr(self.v_thresh, "value") else self.v_thresh)
        tau_mem = float(self.tau_mem.value if hasattr(self.tau_mem, "value") else self.tau_mem)
        tau_adapt = float(self.tau_adapt.value if hasattr(self.tau_adapt, "value") else self.tau_adapt)
        beta = float(self.beta.value if hasattr(self.beta, "value") else self.beta)
        alpha = np.exp(-dt / tau_mem)
        # leak: stable (LeakA <= 0.98), slope at rest kept in [0.7, 0.97]
        leak_b = h * 0.1 * g()
        leak_c = np.full(n, 1.0 / vth)
        alpha_eff = np.clip(alpha + h * 0.06 * g(), 0.7, 0.97) if h > 0 else np.full(n, alpha)
        leak_a = np.clip(alpha_eff - leak_b * leak_c, 0.5, 0.98) if h > 0 else np.full(n, alpha)
        alpha_eff = leak_a + leak_b * leak_c
        in_g = np.exp(h * 0.3 * g())
        in_s = (2.0 * vth / h) * np.exp(0.5 * g()) if h > 0 else np.full(n, 1e6)
        thr = np.clip(vth * np.exp(h * 0.2 * g()), 0.4, 1.0) if h > 0 else np.full(n, vth)
        reset_f = np.clip(1.0 - h * 0.3 * np.abs(g()), 0.4, 1.0)
        rho_j = np.exp(-dt / (tau_adapt * np.exp(h * 0.5 * g())))
        beta_j = beta * np.exp(h * 0.5 * g())
        int_refr = (rng.random(n) >= 0.5 * h).astype(float)
        return {"LeakA": leak_a, "LeakB": leak_b, "LeakC": leak_c, "InG": in_g, "InS": in_s, "Thr": thr,
                "ResetF": reset_f, "RhoJ": rho_j, "AlphaEff": alpha_eff, "IntRefr": int_refr, "Beta": beta_j}
