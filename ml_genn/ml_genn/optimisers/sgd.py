from typing import Optional, Tuple
from .optimiser import Optimiser
from ..utils.model import CustomUpdateModel
from ..utils.snippet import ConstantValueDescriptor

from copy import deepcopy


genn_model = {
    "vars": [("M", "scalar"), ("Step", "scalar")],
    "params": [("Beta1", "scalar"), ("Alpha", "scalar")],
    "var_refs": [("Gradient", "scalar"),
                 ("Variable", "scalar")],
    "update_code":
        """
        Step += 1.0;

        // Momentum: running mean of the gradient (bias-corrected like Adam's first moment)
        M = (Beta1 * M) + ((1.0 - Beta1) * Gradient);
        Variable -= Alpha * M / (1.0 - pow(Beta1, Step));

        Gradient = 0.0;
        """}


class SGD(Optimiser):
    """Stochastic gradient descent with momentum, written as a running mean: the step is
    ``Alpha * M`` with ``M = Beta1 * M + (1 - Beta1) * Gradient`` (bias-corrected), so ``Alpha``
    keeps the meaning of a per-step learning rate whatever ``Beta1`` is. ``Beta1 = 0`` is plain SGD.

    Unlike Adam-type optimisers there is no per-variable normalisation: every variable moves in
    proportion to its own gradient, so the relative sizes of the updates (and, for a non-gradient
    update, the direction of each neuron's weight change) are kept.

    Args:
        alpha:      Learning rate
        beta1:      Momentum (decay rate of the running mean of the gradient)
        l2_init_strength: Optional decoupled pull towards each variable's initial value,
                    ``Variable -= Alpha * l2_init_strength * (Variable - VarInit)``, as in Adam
    """
    alpha = ConstantValueDescriptor()
    beta1 = ConstantValueDescriptor()

    def __init__(self, alpha: float = 0.001, beta1: float = 0.9,
                 clamp_var: Optional[Tuple[float, float]] = None,
                 l2_init_strength: Optional[float] = None):
        self.alpha = alpha
        self.beta1 = beta1
        self.clamp_var = clamp_var
        self.l2_init_strength = l2_init_strength

    def set_step(self, genn_cu, step):
        # bias-correction is computed on-device from each variable's own Step
        pass

    def get_model(self, gradient_ref, var_ref, zero_gradient: bool,
                  clamp_var: Optional[Tuple[float, float]] = None,
                  l2_init_strength: Optional[float] = None) -> CustomUpdateModel:
        if clamp_var is None:
            clamp_var = self.clamp_var
        if l2_init_strength is None:
            l2_init_strength = self.l2_init_strength

        _genn_model = deepcopy(genn_model)
        if l2_init_strength is not None:
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
            {"Beta1": self.beta1, "Alpha": self.alpha},
            {"M": 0.0, "Step": 0.0,
             **({"VarInit": 0.0, "InitDone": 0.0}
                if l2_init_strength is not None else {})},
            {"Gradient": gradient_ref, "Variable": var_ref})
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
