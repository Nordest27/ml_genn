from typing import Optional, Tuple
from pygenn import VarAccessMode
from .optimiser import Optimiser
from ..utils.model import CustomUpdateModel
from ..utils.snippet import ConstantValueDescriptor

from copy import deepcopy


# ---------------------------------------------------------------------------
# AdaBelief (Zhuang et al., 2020, "AdaBelief Optimizer: Adapting Stepsizes
# by the Belief in Observed Gradients"), adapted to fully local, per-synapse,
# every-timestep updates, matching the structure of this codebase's CAdam.
#
# The only change relative to vanilla Adam: the second moment tracks the
# squared *deviation of the sample from momentum*, S = EMA[(g - m)^2],
# rather than the squared sample itself, V = EMA[g^2]. Intuition: when the
# current sample matches what momentum predicted (small deviation), S is
# small, so the effective step size (1/sqrt(S_hat)) is large -- the
# optimiser "believes" the momentum direction and commits to it. When the
# sample surprises momentum (large deviation), S is large, so the step
# shrinks -- the optimiser hedges rather than over-reacting to a possible
# outlier or a genuine regime change it hasn't confirmed yet.
#
# This is a *soft*, continuously-scaled version of the same comparison
# CAdam makes with its hard binary Aligned mask: both ask "does this
# sample agree with momentum," but AdaBelief answers with a magnitude
# rescale, CAdam answers with an on/off gate.
# ---------------------------------------------------------------------------
genn_model = {
    "vars": [("M", "scalar"), ("S", "scalar"), ("Step", "scalar"),
             ("Surprise", "scalar")],
    "params": [("Beta1", "scalar"), ("Beta2", "scalar"),
               ("Epsilon", "scalar"), ("Alpha", "scalar")],
    "var_refs": [("Gradient", "scalar"),
                 ("Variable", "scalar")],
    "update_code":
        """

        // "Gradient" is fully consumed and reset every timestep (see
        // bottom of this block), so it IS this step's local sample.
        scalar deltaSample = Gradient;

        Step += 1.0;
        M = (Beta1 * M) + (1.0 - Beta1) * deltaSample;

        // AdaBelief's core change: track squared deviation from momentum,
        // not squared raw sample. Epsilon added inside the EMA (as in the
        // paper) so S can't collapse to exactly zero even if the sample
        // exactly matches momentum every step.
        scalar deviation = deltaSample - M;
        S = (Beta2 * S) + (1.0 - Beta2) * deviation * deviation + Epsilon;

        scalar momentScale1 = 1.0 / (1.0 - pow(Beta1, Step));
        scalar momentScale2 = 1.0 / (1.0 - pow(Beta2, Step));

        scalar mHat = M * momentScale1;
        scalar sHat = S * momentScale2;

        // Diagnostic only (does not affect the update): how large the
        // deviation from momentum was this step, directly analogous in
        // spirit to CAdam's AlignRate -- a per-synapse trace of how much
        // this step's sample "surprised" the running trend.
        Surprise = deviation * deviation;

        Variable -= (Alpha * mHat) / (sqrt(sHat) + Epsilon);

        // Always fully consume the external accumulator.
        Gradient = 0;
        """}


class AdaBelief(Optimiser):
    """Faithful, per-synapse-local, every-timestep implementation of
    AdaBelief (Zhuang et al., 2020).

    Identical to Adam except the second moment tracks squared deviation
    of the sample from momentum (S = EMA[(g - m)^2]) rather than squared
    raw sample (V = EMA[g^2]). This makes the effective step size shrink
    when the current sample surprises the running momentum trend, and
    grow (relatively) when the sample confirms it -- a continuous,
    magnitude-based analogue of CAdam's hard sign-agreement gate.

    Args:
        alpha:      Learning rate
        beta1:      Decay rate for the momentum (1st moment) estimate,
                    specified in units of one task step (see task_steps).
        beta2:      Decay rate for the belief (2nd moment) estimate,
                    specified in units of one task step (see task_steps).
        epsilon:    Small constant for numerical stability, and also
                    added inside the S update per the paper (prevents S
                    collapsing to exactly zero).
        task_steps: Number of local integration timesteps that make up
                    one step of the task as the agent/environment
                    experiences it (e.g. actions in an RL env). beta1 and
                    beta2 are given at this coarser, task-level timescale
                    and converted internally to the equivalent
                    per-timestep decay rate via
                        beta_timestep = beta_task_step ** (1/task_steps)
                    so the effective memory span, measured in task steps,
                    is invariant to how finely each task step is
                    integrated. task_steps=1 (default) recovers the
                    literal per-timestep beta values.
        l2_init_strength: Optional L2 regularisation strength that pulls
                    each variable back toward its own *initial* value
                    instead of toward zero (as plain weight decay would).
                    Applied as a decoupled term, added directly to the
                    variable after the main AdaBelief step rather than
                    folded into the moment estimates (the same style as
                    AdamW's decoupled weight decay):
                        Variable -= Alpha * l2_init_strength * (Variable - VarInit)
                    VarInit is captured automatically and lazily: the
                    very first time this custom update runs for a given
                    synapse, whatever value Variable holds at that point
                    -- its true pre-training initial value, since nothing
                    else modifies it before this optimiser's first step
                    -- is snapshotted into VarInit once and never
                    overwritten again. No external wiring or separate
                    initialisation pass is required. None (the default)
                    disables the regularisation and adds no extra
                    per-synapse state.
    """
    alpha = ConstantValueDescriptor()
    beta1 = ConstantValueDescriptor()
    beta2 = ConstantValueDescriptor()
    epsilon = ConstantValueDescriptor()

    def __init__(self, alpha: float = 0.001, beta1: float = 0.8,
                 beta2: float = 0.99, epsilon: float = 1e-8,
                 task_steps: int = 1,
                 clamp_var: Optional[Tuple[float, float]] = None,
                 clamp_grad: Optional[Tuple[float, float]] = None,
                 soft_grad_clip: Optional[float] = None,
                 l2_init_strength: Optional[float] = None):

        self.alpha = alpha
        self.task_steps = task_steps

        # Keep user-facing, per-task-step decay rates for introspection.
        self.beta1_per_task_step = beta1
        self.beta2_per_task_step = beta2

        # Convert to per-timestep rates for the model, same rationale as
        # in this codebase's CAdam.
        self.beta1 = beta1 ** (1.0 / task_steps)
        self.beta2 = beta2 ** (1.0 / task_steps)

        self.epsilon = epsilon
        self.clamp_var = clamp_var
        self.clamp_grad = clamp_grad
        self.soft_grad_clip = soft_grad_clip
        self.l2_init_strength = l2_init_strength

    def set_step(self, genn_cu, step):
        # No-op by design: no host-broadcast global step; each synapse
        # tracks and uses its own Step var.
        pass

    def get_model(self, gradient_ref, var_ref, zero_gradient: bool,
                  clamp_var: Optional[Tuple[float, float]] = None,
                  clamp_grad: Optional[Tuple[float, float]] = None,
                  soft_grad_clip: Optional[float] = None,
                  l2_init_strength: Optional[float] = None) -> CustomUpdateModel:

        if clamp_var is None:
            clamp_var = self.clamp_var
        if clamp_grad is None:
            clamp_grad = self.clamp_grad
        if soft_grad_clip is None:
            soft_grad_clip = self.soft_grad_clip
        if l2_init_strength is None:
            l2_init_strength = self.l2_init_strength

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

        if l2_init_strength is not None:
            # VarInit holds each synapse's own pre-training value, snapshotted
            # once (see InitDone below); the regularisation term then pulls
            # Variable back toward VarInit rather than toward zero, applied
            # decoupled from -- i.e. after, and outside of -- the AdaBelief
            # moment-based step, matching AdamW's decoupled weight decay.
            _genn_model["vars"] = _genn_model["vars"] + [
                ("VarInit", "scalar"), ("InitDone", "scalar")]
            _genn_model["update_code"] = f"""
            if (InitDone < 0.5) {{
                VarInit = Variable;
                InitDone = 1.0;
            }}
            {_genn_model["update_code"]}
            Variable -= Alpha * {l2_init_strength} * (Variable - VarInit);
            """

        model = CustomUpdateModel(
            _genn_model,
            {"Beta1": self.beta1, "Beta2": self.beta2,
             "Epsilon": self.epsilon, "Alpha": self.alpha},
            {"M": 0.0, "S": 0.0, "Step": 0.0, "Surprise": 0.0,
             **({"VarInit": 0.0, "InitDone": 0.0}
                if l2_init_strength is not None else {})},
            {"Gradient": gradient_ref, "Variable": var_ref})

        model.set_var_ref_access_mode("Gradient", VarAccessMode.READ_WRITE)
        model.set_param_dynamic("Alpha")

        if clamp_var is not None:
            model.add_param("VariableMin", "scalar", clamp_var[0])
            model.add_param("VariableMax", "scalar", clamp_var[1])

            model.append_update_code(
                """
                // Clamp variable
                Variable = fmax(VariableMin, fmin(VariableMax, Variable));
                """)

        return model