from typing import Optional, Tuple
from pygenn import VarAccessMode
from .optimiser import Optimiser
from ..utils.model import CustomUpdateModel
from ..utils.snippet import ConstantValueDescriptor

from copy import deepcopy


# ---------------------------------------------------------------------------
# CAdam (Wang et al., 2024, "CAdam: Confidence-Based Optimization for
# Online Learning"), implemented as faithfully as possible to Algorithm 1
# of the paper, adapted to fully local, per-synapse, every-timestep updates.
#
# Unlike the previous continuous-confidence version, this is NOT a soft
# gate with tunable shape parameters -- it is the paper's exact mechanism:
# a hard binary mask on the (bias-corrected) momentum, based purely on
# whether the sign of the momentum agrees with the sign of the current
# sample:
#
#       Xi_t = { i : m_t,i * g_t,i >= 0 }
#       m_hat_masked = m_hat  if in Xi_t, else 0
#       Variable -= Alpha * m_hat_masked / (sqrt(v_hat) + Epsilon)
#
# If momentum and the current sample disagree in direction, the update is
# skipped for that synapse this step (letting momentum decay naturally);
# if they agree, a normal Adam step is taken. No additional hyperparameter
# is introduced by this mechanism -- the paper's explicit selling point is
# that it requires no extra tuning over vanilla Adam.
#
# "Gradient" is treated the same way as in the confidence-gated version:
# always fully consumed and reset to 0 every timestep, so it always
# represents just this step's local sample.
# ---------------------------------------------------------------------------
genn_model = {
    "vars": [("M", "scalar"), ("V", "scalar"), ("Step", "scalar"),
             ("Aligned", "scalar"), ("AlignRate", "scalar")],
    "params": [("Beta1", "scalar"), ("Beta2", "scalar"),
               ("Epsilon", "scalar"), ("Alpha", "scalar"),
               ("AlphaAlignRate", "scalar")],
    "var_refs": [("Gradient", "scalar"),
                 ("Variable", "scalar")],
    "update_code":
        """
        Step += 1.0;
        
        M = (Beta1 * M) + (1.0 - Beta1) * Gradient;
        V = (Beta2 * V) + (1.0 - Beta2) * Gradient * Gradient;

        scalar momentScale1 = 1.0 / (1.0 - pow(Beta1, Step));
        scalar momentScale2 = 1.0 / (1.0 - pow(Beta2, Step));

        scalar mHat = M * momentScale1;
        scalar vHat = V * momentScale2;

        // CAdam's confidence mechanism: a hard, tuning-free mask based
        // only on whether momentum and the current sample agree in sign
        // (Eq. 5 of Wang et al., 2024). No threshold parameter -- the
        // comparison point is fixed at zero by construction.
        Aligned = (M * Gradient >= 0.0) ? 1.0 : 0.0;
         
        // Diagnostic only (does not affect the update): a decaying
        // estimate of how often this synapse is currently in agreement,
        // directly analogous to the "alignment ratio" the CAdam paper
        // uses to visualise behaviour under distribution shift.
        AlignRate = (AlphaAlignRate * AlignRate) + (1.0 - AlphaAlignRate) * Aligned;

        Variable -= Aligned * (Alpha * mHat) / (sqrt(vHat) + Epsilon);
        
        // Always fully consume the external accumulator.
        Gradient = 0;
        """}


class CAdam(Optimiser):
    """Faithful, per-synapse-local, every-timestep implementation of CAdam
    (Wang et al., 2024).

    Every synapse takes a normal Adam step every timestep, EXCEPT that the
    step is masked to zero whenever the (bias-corrected) momentum and the
    current local sample disagree in sign -- i.e. whenever recent history
    and the newest evidence are pointing in opposite directions. Agreement
    lets the update through unchanged; disagreement skips the update for
    that step and lets momentum decay naturally, rather than reinforcing
    what may be a stale trend (under distribution shift) or a one-off
    noisy sample.

    Unlike the threshold/t-statistic-based variants tried previously, this
    introduces no new tunable hyperparameter: the agreement test is a
    fixed sign comparison, not a magnitude threshold. This is also the
    paper's explicit selling point -- CAdam is meant to be a drop-in
    Adam replacement requiring no additional tuning.

    Args:
        alpha:      Learning rate
        beta1:      The exponential decay rate for the 1st moment estimates.
        beta2:      The exponential decay rate for the 2nd moment estimates.
        epsilon:    A small constant for numerical stability.
        alpha_align_rate: Decay rate for the diagnostic AlignRate trace
                    (does not affect the update itself, purely for
                    monitoring how often synapses are in agreement).
        l2_init_strength: Optional L2 regularisation strength that pulls
                    each variable back toward its own *initial* value
                    instead of toward zero (as plain weight decay would).
                    Applied as a decoupled term, added directly to the
                    variable after the main CAdam step (independent of,
                    and unaffected by, the alignment mask) rather than
                    folded into the moment estimates -- the same style as
                    AdamW's decoupled weight decay:
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
             alpha_align_rate: float = 0.999,
             task_steps: int = 1,
             clamp_var: Optional[Tuple[float, float]] = None,
             clamp_grad: Optional[Tuple[float, float]] = None,
             soft_grad_clip: Optional[float] = None,
             l2_init_strength: Optional[float] = None):
        """
        Args (additions):
            task_steps: Number of local integration timesteps that make up
                one step of the task as the agent/environment experiences it
                (e.g. your WAIT_INC -- the number of simulation substeps per
                action/decision). Note this is a task-level unit, distinct
                from an Adam "update", since every timestep triggers an Adam
                update regardless of task_steps. beta1, beta2, and
                alpha_align_rate are specified at this coarser task-step
                timescale and are internally converted to the equivalent
                per-timestep decay rate via
                    beta_timestep = beta_task_step ** (1 / task_steps)
                so the *effective memory span, measured in task steps*, stays
                fixed regardless of how finely each task step is discretised
                into timesteps. Passing task_steps=1 (the default) recovers
                the original per-timestep behaviour unchanged.
        """
        self.alpha = alpha
        self.task_steps = task_steps

        # Keep the user-facing, per-task-step decay rates around for
        # introspection/logging -- these are the numbers you actually
        # reasoned about (e.g. "retain 90% of momentum per action").
        self.beta1_per_task_step = beta1
        self.beta2_per_task_step = beta2
        self.alpha_align_rate_per_task_step = alpha_align_rate

        # Convert to per-timestep rates, since that's what the custom
        # update model consumes every single timestep.
        self.beta1 = beta1 ** (1.0 / task_steps)
        self.beta2 = beta2 ** (1.0 / task_steps)
        self.alpha_align_rate = alpha_align_rate ** (1.0 / task_steps)

        self.epsilon = epsilon
        self.clamp_var = clamp_var
        self.clamp_grad = clamp_grad
        self.soft_grad_clip = soft_grad_clip
        self.l2_init_strength = l2_init_strength

    def set_step(self, genn_cu, step):
        # No-op by design: there is no host-broadcast global step. Each
        # synapse tracks and uses its own Step var.
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
            # decoupled from -- i.e. after, and outside of -- the CAdam
            # alignment-masked step, matching AdamW's decoupled weight decay.
            # Deliberately NOT gated by Aligned: it should pull toward the
            # init value every step, independent of whether this step's
            # momentum/sample happened to agree in sign.
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

        # Every synapse consumes and resets "Gradient" every timestep
        # (see update_code), regardless of zero_gradient -- kept in the
        # signature only for interface compatibility.
        model = CustomUpdateModel(
            _genn_model,
            {"Beta1": self.beta1, "Beta2": self.beta2,
             "Epsilon": self.epsilon, "Alpha": self.alpha,
             "AlphaAlignRate": self.alpha_align_rate},
            {"M": 0.0, "V": 0.0, "Step": 0.0, "Aligned": 0.0,
             "AlignRate": 0.0,
             **({"VarInit": 0.0, "InitDone": 0.0}
                if l2_init_strength is not None else {})},
            {"Gradient": gradient_ref, "Variable": var_ref})

        model.set_var_ref_access_mode("Gradient", VarAccessMode.READ_WRITE)

        # Alpha can still be tuned online (e.g. annealed) without
        # recompiling, same as the original model's dynamic Alpha.
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