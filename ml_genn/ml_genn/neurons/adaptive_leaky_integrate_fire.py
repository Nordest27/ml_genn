from __future__ import annotations

import numpy as np

from typing import Optional, TYPE_CHECKING
from .neuron import Neuron
from ..utils.model import NeuronModel
from ..utils.value import InitValue, ValueDescriptor

if TYPE_CHECKING:
    from .. import Population

from ..utils.decorators import network_default_params

class AdaptiveLeakyIntegrateFire(Neuron):
    
    v_thresh = ValueDescriptor("Vthresh")
    v_reset = ValueDescriptor("Vreset")
    v = ValueDescriptor("V")
    a = ValueDescriptor("A")
    beta = ValueDescriptor("Beta")
    tau_mem = ValueDescriptor(("Alpha", lambda val, dt: np.exp(-dt / val)))
    tau_refrac = ValueDescriptor("TauRefrac")
    tau_adapt = ValueDescriptor(("Rho", lambda val, dt: np.exp(-dt / val)))
    perturbation_eps = ValueDescriptor("PertEps")
    perturbation_eps_trace = ValueDescriptor("PertEpsTrace")
    perturbation_eps_trace_div_sigma = ValueDescriptor("PertEpsTraceDivSigma")
    sigma = ValueDescriptor("Sigma")
    log_sigma = ValueDescriptor("LogSigma")
    sigma_lr = ValueDescriptor("SigmaLR")

    @network_default_params
    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, a: InitValue = 0.0, beta: InitValue = 0.0174,
                 tau_mem: InitValue = 20.0, tau_refrac: InitValue = None,
                 tau_adapt: InitValue = 2000.0, relative_reset: bool = True,
                 integrate_during_refrac: bool = True, perturbation_eps_std: float = 0.01,
                 sigma_lr: float = 1e-4,
                 readout=None):
        super(AdaptiveLeakyIntegrateFire, self).__init__(readout)

        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v
        self.a = a
        self.beta = beta
        self.tau_mem = tau_mem
        self.tau_refrac = tau_refrac
        self.tau_adapt = tau_adapt
        self.relative_reset = relative_reset
        self.integrate_during_refrac = integrate_during_refrac
        self.perturbation_eps_std = perturbation_eps_std
        self.sigma = 0.0
        self.perturbation_eps = 0.0
        self.perturbation_eps_trace = 0.0
        self.perturbation_eps_trace_div_sigma = 0.0
        self.log_sigma = np.log(perturbation_eps_std)
        self.sigma_lr = sigma_lr

    def get_model(self, population: Population,
                  dt: float, batch_size: int) -> NeuronModel:
        genn_model = {
            "vars": [("V", "scalar"), ("A", "scalar"), ("Beta", "scalar"),
                     ("PertEpsTrace", "scalar"), ("PertEpsTraceDivSigma", "scalar"), 
                     ("PertEps", "scalar"), ("Sigma", "scalar"), ("LogSigma", "scalar"),
                     ("TdE", "scalar")],
            "params": [("Vthresh", "scalar"), ("Vreset", "scalar"),
                       ("Alpha", "scalar"), ("Rho", "scalar"), ("SigmaLR", "scalar")],
            "threshold_condition_code": "V >= (Vthresh + (Beta * A))"}

        if self.relative_reset:
            genn_model["reset_code"] = """
                V -= (Vthresh - Vreset);
                A += 1.0;
                """
        else:
            genn_model["reset_code"] = """
                V = Vreset;
                A += 1.0;
                """

        sim_code_perturbation = f"""
                // State-independent learned sigma
                Sigma = exp(LogSigma);
                PertEps = sqrt(1.0 - Alpha*Alpha) * Sigma * gennrand_normal();
                PertEpsTrace = PertEpsTrace * Alpha + PertEps;
                // PertEpsTraceDivSigma = PertEpsTraceDivSigma * Alpha + (PertEps*PertEps / (Sigma*Sigma) - 1.0);

                // Score function update + entropy regularization
                LogSigma = fmax(fmin(
                    LogSigma + SigmaLR * TdE * (PertEpsTrace * PertEpsTrace / (Sigma * Sigma) - 1.0)
                    , -2.0), -15.0);

                V = Alpha * V + Isyn + PertEps;
                A *= Rho;
                """

        if self.tau_refrac is not None:
            genn_model["vars"].append(("RefracTime", "scalar"))
            genn_model["params"].append(("TauRefrac", "scalar"))

            if self.integrate_during_refrac:
                genn_model["sim_code"] = f"""
                {sim_code_perturbation}
                if (RefracTime > 0.0) {{
                    RefracTime -= dt;
                }}
                """
            else:
                genn_model["sim_code"] = f"""
                A *= Rho;
                if (RefracTime > 0.0) {{
                    RefracTime -= dt;
                }}
                else {{
                    {sim_code_perturbation}
                }}
                """

            genn_model["reset_code"] += """
                RefracTime = TauRefrac;
                """
            genn_model["threshold_condition_code"] += " && RefracTime <= 0.0"
        else:
            genn_model["sim_code"] = f"""
                {sim_code_perturbation}
                """

        var_vals = {} if self.tau_refrac is None else {"RefracTime": 0.0}
        var_vals["PertEpsTrace"] = 0.0
        var_vals["PertEpsTraceDivSigma"] = 0.0
        var_vals["PertEps"] = 0.0
        var_vals["Sigma"] = self.perturbation_eps_std
        var_vals["LogSigma"] = np.log(self.perturbation_eps_std)
        var_vals["TdE"] = 0.0
        var_vals["Beta"] = 0.0
        return NeuronModel.from_val_descriptors(genn_model, "V", self, dt,
                                                 var_vals=var_vals)


class AdaptiveLeakyIntegrateFireOrig(Neuron):
    """A leaky-integrate and fire neuron with an adaptive firing threshold
    as described by [Bellec2018]_.
    
    Args:
        v_thresh:                   Membrane voltage firing threshold
        v_reset:                    After a spike is emitted, this value is
                                    *subtracted* from the membrane voltage
                                    ``v`` if ``relative_reset`` is ``True``.
                                    Otherwise, if ``relative_reset`` is 
                                    ``False``, the membrane voltage is set to
                                    this value.
        v:                          Initial value of membrane voltage
        a:                          Initial value of adaptation
        beta:                       Strength of adaptation
        tau_mem:                    Time constant of membrane voltage [ms]
        tau_refrac:                 Duration of refractory period [ms]
        tau_adapt:                  Time constant of adaptation [ms]
        relative_reset:             How is ``v`` reset after a spike?
        integrate_during_refrac:    Should ``v`` continue to integrate inputs
                                    during refractory period?
        readout:                    Type of readout to attach to this
                                    neuron's output variable
    """
    
    v_thresh = ValueDescriptor("Vthresh")
    v_reset = ValueDescriptor("Vreset")
    v = ValueDescriptor("V")
    a = ValueDescriptor("A")
    beta = ValueDescriptor("Beta")
    tau_mem = ValueDescriptor(("Alpha", lambda val, dt: np.exp(-dt / val)))
    tau_refrac = ValueDescriptor("TauRefrac")
    tau_adapt = ValueDescriptor(("Rho", lambda val, dt: np.exp(-dt / val)))
    perturbation_eps = ValueDescriptor("PertEps")
    perturbation_eps_trace = ValueDescriptor("PertEpsTrace")
    sigma = ValueDescriptor("Sigma")
    sigma_of_sigma = ValueDescriptor("SigmaOfSigma")
    sigma_pert_eps = ValueDescriptor("SigmaPertEps")

    @network_default_params
    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, a: InitValue = 0.0, beta: InitValue = 0.0174,
                 tau_mem: InitValue = 20.0, tau_refrac: InitValue = None,
                 tau_adapt: InitValue = 2000.0, relative_reset: bool = True,
                 integrate_during_refrac: bool = True, perturbation_eps_std: float = 1.0,
                 readout=None):
        super(AdaptiveLeakyIntegrateFire, self).__init__(readout)

        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v
        self.a = a
        self.beta = beta
        self.tau_mem = tau_mem
        self.tau_refrac = tau_refrac
        self.tau_adapt = tau_adapt
        self.relative_reset = relative_reset
        self.integrate_during_refrac = integrate_during_refrac
        self.perturbation_eps_std = perturbation_eps_std
        self.sigma = 0.0
        self.perturbation_eps = 0.0
        self.perturbation_eps_trace = 0.0
        self.sigma_of_sigma = 0.0
        self.sigma_pert_eps = 0.0

    def get_model(self, population: Population,
                  dt: float, batch_size: int) -> NeuronModel:
        # Build basic model
        genn_model = {
            "vars": [("V", "scalar"), ("A", "scalar"), 
                     ("PertEpsTrace", "scalar"), ("PertEps", "scalar"), 
                     ("Sigma", "scalar"), ("SigmaOfSigma", "scalar"), 
                     ("SigmaPertEps", "scalar")],
            "params": [("Vthresh", "scalar"), ("Vreset", "scalar"),
                       ("Alpha", "scalar"), ("Beta", "scalar"), 
                       ("Rho", "scalar")],
            "threshold_condition_code": "V >= (Vthresh + (Beta * A))"}

        # Build reset code depending on whether
        # reset should be relative or not
        if self.relative_reset:
            genn_model["reset_code"] =\
                """
                V -= (Vthresh - Vreset);
                A += 1.0;
                """
        else:
            genn_model["reset_code"] =\
                """
                V = Vreset;
                A += 1.0;
                """

        # If neuron has refractory period
        if self.tau_refrac is not None:
            # Add state variable and parameter to control refractoryness
            genn_model["vars"].append(("RefracTime", "scalar"))
            genn_model["params"].append(("TauRefrac", "scalar"))

            # Build correct sim code depending on whether
            # we should integrate during refractory period
            if self.integrate_during_refrac:
                genn_model["sim_code"] =\
                    f"""
                    // SigmaPertEps = SigmaOfSigma * gennrand_normal();
                    // Sigma = 0.1 * Vthresh * exp(ISynSigmaEps + SigmaPertEps);
                    // PertEps = sqrt(1 - Alpha*Alpha) * Sigma * gennrand_normal();
                    // PertEpsTrace = PertEpsTrace * Alpha + PertEps;
                    V = Alpha * V + Isyn + PertEps;
                    A *= Rho;
                    if (RefracTime > 0.0) {{
                        RefracTime -= dt;
                    }}
                    """
            else:
                genn_model["sim_code"] =\
                    f"""
                    A *= Rho;
                    if (RefracTime > 0.0) {{
                        RefracTime -= dt;
                    }}
                    else {{
                        // SigmaPertEps = SigmaOfSigma * gennrand_normal();
                        // Sigma = 0.1 * Vthresh * exp(ISynSigmaEps + SigmaPertEps);
                        // PertEps = sqrt(1 - Alpha*Alpha) * Sigma * gennrand_normal();
                        // PertEpsTrace = PertEpsTrace * Alpha + PertEps;
                        V = Alpha * V + Isyn + PertEps;
                    }}
                    """

            # Add refractory period initialisation to reset code
            genn_model["reset_code"] +=\
                """
                RefracTime = TauRefrac;
                """

            # Add refractory check to threshold condition
            genn_model["threshold_condition_code"] +=\
                " && RefracTime <= 0.0"
        # Otherwise, build non-refractory sim-code
        else:
            genn_model["sim_code"] =\
                f"""
                // SigmaPertEps = SigmaOfSigma * gennrand_normal();
                // Sigma = 0.1 * Vthresh * exp(ISynSigmaEps + SigmaPertEps);
                // PertEps = sqrt(1 - Alpha*Alpha) * Sigma * gennrand_normal();
                // PertEpsTrace = PertEpsTrace * Alpha + PertEps;
                V = Alpha * V + Isyn + PertEps;
                A *= Rho;
                """

        # Return model
        var_vals = {} if self.tau_refrac is None else {"RefracTime": 0.0}
        var_vals["PertEpsTrace"] = 0.0
        var_vals["PertEps"] = 0.0
        var_vals["Sigma"] = 0.0
        var_vals["SigmaOfSigma"] = self.perturbation_eps_std
        var_vals["SigmaPertEps"] = 0.0
        return NeuronModel.from_val_descriptors(genn_model, "V", self, dt,
                                                var_vals=var_vals)
