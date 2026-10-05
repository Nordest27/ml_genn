from typing import Optional, Tuple
from pygenn import VarAccessMode
from .optimiser import Optimiser
from ..utils.model import CustomUpdateModel
from ..utils.snippet import ConstantValueDescriptor

from copy import deepcopy


# ---------------------------------------------------------------------------
# Asynchronous, per-synapse threshold-triggered Adam.
#
# Unlike the original model (which relies on a single host-side step counter
# shared by every synapse, with MomentScale1/2 broadcast once per update via
# set_step), this version:
#
#   1. Accumulates "Gradient" locally exactly as before (e.g. the
#      distributed-e-prop credit trace p_ji).
#   2. Fires an update ONLY when |Gradient| crosses a per-synapse threshold
#      "Theta" -- so different synapses update at different, unsynchronised
#      times, with no global reduction or coordination required.
#   3. On firing, increments its OWN step counter "Step" (a per-synapse
#      state variable, not a host-broadcast parameter) and computes the
#      Adam bias-correction terms (MomentScale1/2) locally from that
#      counter, so a rarely-firing synapse and a frequently-firing synapse
#      each get correctly-indexed bias correction rather than sharing one
#      global t.
#   4. Resets its own accumulator to 0 after firing, exactly as the
#      zero_gradient option already did in the synchronous version.
#
# The host no longer needs to call set_step() every update -- Alpha/Beta1/
# Beta2/Epsilon/Theta stay as (optionally dynamic) constants, and Step/M/V
# are genuine per-synapse state that persists and evolves independently.
# ---------------------------------------------------------------------------
genn_model = {
    "vars": [("M", "scalar"), ("V", "scalar"), ("Step", "scalar"), ("UpdateRate", "scalar")],
    "params": [("Beta1", "scalar"), ("Beta2", "scalar"),
               ("Epsilon", "scalar"), ("Alpha", "scalar"),
               ("Theta", "scalar"), ("AlphaFireRate", "scalar")],
    "var_refs": [("Gradient", "scalar"),
                 ("Variable", "scalar")],
    "update_code":
        """

        // Decay the running estimate every timestep, exactly like FAvg
        // does for spike frequency -- so old firing history fades out
        // and this tracks *current* firing rate, not a cumulative one.
        UpdateRate *= AlphaFireRate;

        if (fabs(Gradient) >= Theta) {
            Step += 1.0;
            M = (Beta1 * M);
            V = (Beta2 * V);

            UpdateRate += (1.0 - AlphaFireRate);

            M += (1.0 - Beta1) * Gradient;
            V += (1.0 - Beta2) * Gradient * Gradient;

            scalar momentScale1 = 1.0 / (1.0 - pow(Beta1, Step));
            scalar momentScale2 = 1.0 / (1.0 - pow(Beta2, Step));

            Variable -= (Alpha * M * momentScale1) / (sqrt(V * momentScale2) + Epsilon);

            Gradient = 0;
        }
        // Gradient *= 0.999;
        """}


class AsyncLocalAdam(Optimiser):
    """Adam variant with fully local, per-synapse asynchronous updates.

    Each synapse accumulates its own gradient/credit trace and fires its
    own Adam update independently, whenever that trace's magnitude crosses
    ``theta``. Bias correction is computed from a per-synapse step counter
    rather than a shared, host-broadcast step, so update frequency can vary
    arbitrarily across synapses without corrupting Adam's moment estimates.

    Args:
        alpha:      Learning rate
        beta1:      The exponential decay rate for the 1st moment estimates.
        beta2:      The exponential decay rate for the 2nd moment estimates.
        epsilon:    A small constant for numerical stability.
        theta:      Per-synapse firing threshold on |Gradient|. A synapse
                    only applies an update (and only then advances its own
                    step counter) once its accumulated gradient magnitude
                    reaches this value.
    """
    alpha = ConstantValueDescriptor()
    beta1 = ConstantValueDescriptor()
    beta2 = ConstantValueDescriptor()
    epsilon = ConstantValueDescriptor()
    theta = ConstantValueDescriptor()
    def __init__(self, alpha: float = 0.001, beta1: float = 0.9,
                beta2: float = 0.999, epsilon: float = 1e-8,
                theta: float = 1e-1, alpha_fire_rate: float = 0.999,
                clamp_var: Optional[Tuple[float, float]] = None,
                clamp_grad: Optional[Tuple[float, float]] = None,
                soft_grad_clip: Optional[float] = None):

        self.alpha_fire_rate = alpha_fire_rate      
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.theta = theta
        self.clamp_var = clamp_var
        self.clamp_grad = clamp_grad
        self.soft_grad_clip = soft_grad_clip

    def set_step(self, genn_cu, step):
        # No-op by design: there is no longer a single global step to push
        # to the device. Each synapse tracks and uses its own Step var.
        # Kept only so existing training-loop code that unconditionally
        # calls o.set_step(c, ...) doesn't need to branch on optimiser type.
        pass

    def get_model(self, gradient_ref, var_ref, zero_gradient: bool,
                  clamp_var: Optional[Tuple[float, float]] = None,
                  clamp_grad: Optional[Tuple[float, float]] = None,
                  soft_grad_clip: Optional[float] = None) -> CustomUpdateModel:

        if clamp_var is None:
            clamp_var = self.clamp_var
        if clamp_grad is None:
            clamp_grad = self.clamp_grad
        if soft_grad_clip is None:
            soft_grad_clip = self.soft_grad_clip

        _genn_model = deepcopy(genn_model)

        if clamp_grad is not None:
            _genn_model["update_code"] = f"""
            Gradient = fmax({clamp_grad[0]}, fmin({clamp_grad[1]}, Gradient));
            {_genn_model["update_code"]}
            """

        if soft_grad_clip is not None:
            _genn_model["update_code"] = f"""
            Gradient = Gradient / (1.0 + fabs(Gradient) / {soft_grad_clip});
            {_genn_model["update_code"]}
            """

        model = CustomUpdateModel(
            _genn_model,
            {"Beta1": self.beta1, "Beta2": self.beta2,
            "Epsilon": self.epsilon, "Alpha": self.alpha,
            "Theta": self.theta, "AlphaFireRate": self.alpha_fire_rate},
            {"M": 0.0, "V": 0.0, "Step": 0.0, "UpdateRate": 0.0},
            {"Gradient": gradient_ref, "Variable": var_ref})

        # Alpha/Theta can still be tuned online (e.g. annealed) without
        # recompiling, same as the original model's dynamic Alpha.
        model.set_param_dynamic("Alpha")
        model.set_param_dynamic("Theta")

        # If a gradient accumulator should be zeroed once consumed --
        # NOTE: for the threshold-triggered model this must only happen
        # INSIDE the fired branch, not unconditionally, or a synapse that
        # hasn't crossed threshold yet would have its accumulating trace
        # wiped every update. We therefore inline this into update_code
        # above the existing structure rather than appending it
        # unconditionally at model-build time.
        # if zero_gradient:
        #     model.set_var_ref_access_mode("Gradient",
        #                                   VarAccessMode.READ_WRITE)
        #     # Insert the zeroing *inside* the "if (fabs(Gradient) >= Theta)"
        #     # block, right after the variable update, by rewriting the
        #     # already-built update_code string.
        #     fired_update_code = _genn_model["update_code"]
        #     fired_update_code = fired_update_code.replace(
        #         "Variable -= (Alpha * M * momentScale1) / (sqrt(V * momentScale2) + Epsilon);",
        #         """Variable -= (Alpha * M * momentScale1) / (sqrt(V * momentScale2) + Epsilon);

        #     // Reset this synapse's accumulator now that it has fired
        #     Gradient = 0.0;"""
        #     )
        #     model.update_code = fired_update_code

        if clamp_var is not None:
            model.add_param("VariableMin", "scalar", clamp_var[0])
            model.add_param("VariableMax", "scalar", clamp_var[1])

            model.append_update_code(
                """
                // Clamp variable
                Variable = fmax(VariableMin, fmin(VariableMax, Variable));
                """)

        return model