from typing import Optional, Tuple
from pygenn import VarAccessMode
from .optimiser import Optimiser
from ..utils.model import CustomUpdateModel
from ..utils.snippet import ConstantValueDescriptor

from copy import deepcopy


genn_model = {
    "vars": [("M", "scalar"), ("V", "scalar"), ("Step", "scalar")],
    "params": [("Beta1", "scalar"), ("Beta2", "scalar"),
               ("Epsilon", "scalar"), ("Alpha", "scalar")],
    "var_refs": [("Gradient", "scalar"),
                 ("Variable", "scalar")],
    "update_code":
        """
        Step += 1.0;

        // Update biased first moment estimate
        M = (Beta1 * M) + ((1.0 - Beta1) * Gradient);

        // Update biased second moment estimate
        V = (Beta2 * V) + ((1.0 - Beta2) * Gradient * Gradient);

        // Bias-correction computed locally on the GPU from each
        // synapse's own Step, rather than being pushed in from the
        // host every timestep via set_dynamic_param_value.
        scalar momentScale1 = 1.0 / (1.0 - pow(Beta1, Step));
        scalar momentScale2 = 1.0 / (1.0 - pow(Beta2, Step));

        // Add gradient to variable, scaled by learning rate
        Variable -= (Alpha * M * momentScale1) / (sqrt(V * momentScale2) + Epsilon);

        Gradient = 0.0;
        """}


class Adam(Optimiser):
    """Optimizer that implements the Adam algorithm [Kingma2014]_.
    Adam optimization is a stochastic gradient descent method that 
    is based on adaptive estimation of first-order and second-order moments.

    Bias-correction terms (the ``MomentScale1``/``MomentScale2`` factors)
    are computed on-device from a per-synapse ``Step`` counter rather than
    being pushed in from the host each timestep, matching the fully local,
    every-timestep style used by this codebase's ``AdaBelief`` optimiser.
    As a result ``set_step`` is a no-op here.

    Args:
        alpha:      Learning rate
        beta1:      The exponential decay rate for the 1st moment estimates.
        beta2:      The exponential decay rate for the 2nd moment estimates.
        epsilon:    A small constant for numerical stability. This is
                    the epsilon in Algorithm 1 of the [Kingma2014]_
        l2_init_strength: Optional L2 regularisation strength that pulls
                    each variable back toward its own *initial* value
                    instead of toward zero (as plain weight decay would).
                    Applied as a decoupled term, added directly to the
                    variable after the main Adam step rather than folded
                    into the moment estimates (the same style as AdamW's
                    decoupled weight decay):
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

    def __init__(self, alpha: float = 0.001, beta1 : float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8, 
                 clamp_var: Optional[Tuple[float, float]] = None,
                 clamp_grad: Optional[Tuple[float, float]] = None, 
                 soft_grad_clip: Optional[float] = None,
                 l2_init_strength: Optional[float] = None):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clamp_var = clamp_var
        self.clamp_grad = clamp_grad
        self.soft_grad_clip = soft_grad_clip
        self.l2_init_strength = l2_init_strength

    def set_step(self, genn_cu, step):
        # No-op by design: bias-correction is now computed on-device from
        # each synapse's own local Step variable (see genn_model above),
        # so there is no host-broadcast global step to push in anymore.
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
            # decoupled from -- i.e. after, and outside of -- the Adam
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
            {"M": 0.0, "V": 0.0, "Step": 0.0,
             **({"VarInit": 0.0, "InitDone": 0.0}
                if l2_init_strength is not None else {})},
            {"Gradient": gradient_ref, "Variable": var_ref})

        # Make learning rate dynamic (MomentScale1/2 no longer exist as
        # params -- they're computed locally from Step each timestep).
        model.set_param_dynamic("Alpha")

        # If variable should be clamped
        # **THINK** this is generic across all optimisers like readout-adding
        if clamp_var is not None:
            # Add minimum and maximum parameters
            model.add_param("VariableMin", "scalar", clamp_var[0])
            model.add_param("VariableMax", "scalar", clamp_var[1])
            
            # Add update code to clamp variable
            model.append_update_code(
                """
                // Clamp variable
                Variable = fmax(VariableMin, fmin(VariableMax, Variable));
                """)

        # Return model
        return model
