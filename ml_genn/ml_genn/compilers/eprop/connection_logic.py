"""Helpers for choosing and building the weight-update model for a
connection, given its role (feedback / hidden / output) and the
compiler's configuration (RL vs supervised, feedback type, etc.).

This replaces the large if/elif ladder that used to live inline in
`EPropCompiler.build_weight_update_model`.
"""

import numpy as np

from .variants import FeedbackType, FeedbackRule, HiddenRule, OutputRule, rule_for_population
from .models import get_hidden_model, get_output_model, get_feedback_model
from .hidden_rule import build_td_hidden_model
from ...neurons import AdaptiveLeakyIntegrateFire, LeakyIntegrateFire
from ...utils.model import WeightUpdateModel


def _build_rl_feedback_wum(conn, connect_snippet, compiler, compile_state,
                           target_pop, alpha):
    """Feedback connections (conn.is_feedback) under the RL / TD(lambda)
    configuration. Dispatches on the connection's name suffix and whether
    its target is a policy head.
    """
    name = conn.name

    if name.endswith("tde_transport"):
        compile_state.tde_transport_connections.append(conn)
        return WeightUpdateModel(
            model=get_feedback_model(FeedbackRule.RANDOM_ERROR),
            var_vals={"g": connect_snippet.weight},
            post_neuron_var_refs={"E_post": "E"})

    if name.endswith("policy_reward"):
        compile_state.policy_reward_connections.append(conn)
        return WeightUpdateModel(
            model=get_feedback_model(FeedbackRule.RANDOM_ERROR),
            var_vals={"g": connect_snippet.weight},
            post_neuron_var_refs={"E_post": "PRew"})

    if name.endswith("pert_eps_feedback"):
        compile_state.pert_eps_transport_connections.append(conn)
        return WeightUpdateModel(
            model=get_feedback_model(FeedbackRule.RANDOM_ERROR),
            var_vals={"g": connect_snippet.weight},
            post_neuron_var_refs={"E_post": "PertEpsTrace"})

    if target_pop in compiler.policy_heads:
        if name.endswith("policy_feedback"):
            compile_state.policy_feedback_connections.append(conn)
            post_var_name = "PG"
        elif name.endswith("policy_regularisation"):
            compile_state.policy_regularisation_connections.append(conn)
            post_var_name = "E"
        else:
            raise ValueError(f"Unrecognised policy feedback connection name: {name}")

        return WeightUpdateModel(
            model=get_feedback_model(FeedbackRule.POLICY_ADAPTIVE),
            param_vals={"Alpha": alpha, "Lambda": compiler.td_lambda},
            var_vals={"g": connect_snippet.weight, "DeltaG": 0.0, "RLTrace": 0.0},
            pre_var_vals={"ZFilter": 0.0},
            post_neuron_var_refs={"E_post": "E", "PG_post": "PG",
                                 "TdE_post": "TdE", "F_post": post_var_name})

    # Value-head feedback
    if name.endswith("value_feedback"):
        compile_state.value_feedback_connections.append(conn)
        post_var_name = "E"
    elif name.endswith("value_regularisation"):
        compile_state.value_regularisation_connections.append(conn)
        post_var_name = "ValReg"
    else:
        raise ValueError(f"Unrecognised value feedback connection name: {name}")

    return WeightUpdateModel(
        model=get_feedback_model(FeedbackRule.VALUE_ADAPTIVE),
        param_vals={"RetE": compiler.value_feedback_ret_e, "Alpha": alpha,
                    "GammaLambda": compiler.gamma_lambda},
        var_vals={"g": connect_snippet.weight, "DeltaG": 0.0, "RLTrace": 0.0},
        pre_var_vals={"ZFilter": 0.0},
        post_neuron_var_refs={"E_post": "E", "ValReg_post": "ValReg", "F_post": post_var_name})


def _build_supervised_feedback_wum(conn, connect_snippet, compiler, compile_state, alpha):
    """Feedback connection under plain (non-RL) adaptive e-prop."""
    compile_state.feedback_connections.append(conn)
    return WeightUpdateModel(
        model=get_output_model(OutputRule.SUPERVISED_ADAPTIVE_FEEDBACK),
        param_vals={"Alpha": alpha},
        var_vals={"g": connect_snippet.weight, "DeltaG": 0.0},
        pre_var_vals={"ZFilter": 0.0},
        post_neuron_var_refs={"E_post": "E"})


def build_feedback_wum(conn, connect_snippet, compiler, compile_state, target_pop, alpha):
    """Entry point for any connection with conn.is_feedback == True."""
    if compiler.gamma_lambda is not None:
        return _build_rl_feedback_wum(conn, connect_snippet, compiler,
                                      compile_state, target_pop, alpha)
    return _build_supervised_feedback_wum(conn, connect_snippet, compiler,
                                          compile_state, alpha)


def build_plastic_inhibition_wum(cfg, weight, dt, target_neuron=None):
    """I -> E synapses with anti-Hebbian inhibitory plasticity (PlasticInhibitionConfig): |g| += eta * x_pre *
    (y_post - y0), applied directly every timestep, g kept <= 0. No DeltaG, so the optimiser leaves it alone."""
    decay = float(np.exp(-dt / cfg.tau))
    y0 = cfg.rate * (dt / 1000.0) / (1.0 - decay)        # mean post trace at the target rate
    model = {
        "params": [("Eta", "scalar"), ("Decay", "scalar"), ("Y0", "scalar")],
        "vars": [("g", "scalar")],
        "pre_vars": [("XTrace", "scalar")],
        "post_vars": [("YTrace", "scalar")],
        "pre_spike_code": "XTrace += 1.0;",
        "pre_dynamics_code": "XTrace *= Decay;",
        "post_spike_code": "YTrace += 1.0;",
        "post_dynamics_code": "YTrace *= Decay;",
        "pre_spike_syn_code": "addToPost(g);",
        "synapse_dynamics_code": "g = fmin(0.0, g - Eta * XTrace * (YTrace - Y0));",
    }
    w_max = cfg.w_max
    if cfg.w_max_rel is not None:
        mean = getattr(weight, "mean", None)
        if mean is None:
            mean = float(np.mean(np.abs(weight)))
        w_max = cfg.w_max_rel * abs(float(mean))
    floor = "" if w_max is None else f"g = fmax(g, {-float(w_max)!r});"
    param_vals = {"Eta": cfg.eta, "Decay": decay, "Y0": y0}
    pre_var_vals, post_var_vals = {"XTrace": 0.0}, {"YTrace": 0.0}
    if cfg.mode == "covariance":
        # running means start at the trace of a ~10 Hz neuron, so the first seconds do not see a large covariance
        start = 10.0 * (dt / 1000.0) / (1.0 - decay)
        model["params"].append(("MeanDecay", "scalar"))
        model["pre_vars"].append(("XMean", "scalar"))
        model["post_vars"].append(("YMean", "scalar"))
        model["pre_dynamics_code"] += "\nXMean = MeanDecay * XMean + (1.0 - MeanDecay) * XTrace;"
        model["post_dynamics_code"] += "\nYMean = MeanDecay * YMean + (1.0 - MeanDecay) * YTrace;"
        model["synapse_dynamics_code"] = "g = fmin(0.0, g - Eta * (XTrace - XMean) * (YTrace - YMean));"
        param_vals["MeanDecay"] = float(np.exp(-dt / cfg.tau_mean))
        pre_var_vals["XMean"], post_var_vals["YMean"] = start, start
    post_refs = {}
    if cfg.mode in ("psi", "psi_fast"):
        if not isinstance(target_neuron, AdaptiveLeakyIntegrateFire):
            raise ValueError(f"plastic inhibition '{cfg.mode}' mode needs an ALIF target")
        # the hidden rule's pseudo-derivative and its running mean, computed here from the postsynaptic state
        model["params"] += [("Vthresh_post", "scalar"), ("PsiTarget", "scalar"), ("PsiBarDecay", "scalar")]
        model["post_vars"] += [("Psi", "scalar"), ("PsiBar", "scalar")]
        post_refs = {"V_post": "V", "A_post": "A", "Beta_post": "Beta", "RefracTime_post": "RefracTime"}
        model["post_neuron_var_refs"] = [(k, "scalar") for k in post_refs]
        model["post_dynamics_code"] += """
if (RefracTime_post > 0.0) {
    Psi = 0.0;
}
else {
    Psi = (1.0 / Vthresh_post) * 0.3 * fmax(0.0, 1.0 - fabs((V_post - (Vthresh_post + (Beta_post * A_post))) / Vthresh_post));
}
PsiBar = PsiBarDecay * PsiBar + (1.0 - PsiBarDecay) * Psi;"""
        psi = "Psi" if cfg.mode == "psi_fast" else "PsiBar"
        model["synapse_dynamics_code"] = f"g = fmin(0.0, g - Eta * XTrace * ({psi} - PsiTarget));"
        param_vals.update({"Vthresh_post": target_neuron.v_thresh, "PsiTarget": cfg.psi_target,
                           "PsiBarDecay": cfg.psibar_decay})
        post_var_vals.update({"Psi": 0.0, "PsiBar": cfg.psi_target})
    if floor:
        model["synapse_dynamics_code"] += "\n" + floor
    return WeightUpdateModel(model=model, param_vals=param_vals, var_vals={"g": weight},
                             pre_var_vals=pre_var_vals, post_var_vals=post_var_vals,
                             post_neuron_var_refs=post_refs)


def build_hidden_wum(conn, connect_snippet, compiler, compile_state, target_neuron, alpha):
    """Weight-update model for a connection targeting a hidden LIF/ALIF neuron."""
    if conn in compiler.plastic_inhibition:
        return build_plastic_inhibition_wum(compiler.plastic_inhibition[conn], connect_snippet.weight, compiler.dt,
                                            target_neuron)
    if isinstance(target_neuron, LeakyIntegrateFire):
        return WeightUpdateModel(
            model=get_hidden_model(HiddenRule.LIF),
            param_vals={"CReg": compiler.c_reg, "Alpha": alpha,
                        "FTarget": (compiler.f_target * compiler.dt) / 1000.0,
                        "AlphaFAv": np.exp(-compiler.dt / compiler.tau_reg),
                        "Vthresh_post": target_neuron.v_thresh},
            var_vals={"g": connect_snippet.weight, "eFiltered": 0.0,
                      "DeltaG": 0.0, "NoiseTrace": 0.0, "RLNoiseTrace": 0.0},
            pre_var_vals={"ZFilter": 0.0},
            post_var_vals={"Psi": 0.0, "FAvg": 0.0},
            post_neuron_var_refs={"RefracTime_post": "RefracTime", "V_post": "V",
                                 "E_post": "E", "Ebase_post": "Ebase", "Noise": "Noise"})

    # AdaptiveLeakyIntegrateFire
    # A synapse group has one presynaptic target. Hidden -> hidden synapses that send the backward gradient
    # signal target ISynBack; all others keep ISynPertEps (the hidden rule sends nothing there).
    rule = compiler.rule_for(conn.target()) if compiler.gamma_lambda is not None else compiler.hidden_rule
    send_back = (compiler.gamma_lambda is not None and rule.backprop != 0
                 and isinstance(conn.source().neuron, AdaptiveLeakyIntegrateFire))
    if send_back:
        compile_state.backprop_connections.append(conn)
    elif compiler.gamma_lambda is not None:
        compile_state.pert_eps_transport_connections.append(conn)
    rho = np.exp(-compiler.dt / compile_state.tau_adapt)
    base_params = {"CReg": compiler.c_reg, "Alpha": alpha, "Rho": rho,
                  "FTarget": (compiler.f_target * compiler.dt) / 1000.0,
                  "AlphaFAv": np.exp(-compiler.dt / compiler.tau_reg),
                  "Vthresh_post": target_neuron.v_thresh}

    if compiler.gamma_lambda is None:
        return WeightUpdateModel(
            model=get_hidden_model(HiddenRule.ALIF),
            param_vals={**base_params, "Beta_post": target_neuron.beta},
            var_vals={"g": connect_snippet.weight, "eFiltered": 0.0,
                      "DeltaG": 0.0, "epsilonA": 0.0},
            pre_var_vals={"ZFilter": 0.0},
            post_var_vals={"Psi": 0.0, "FAvg": 0.0},
            post_neuron_var_refs={"RefracTime_post": "RefracTime", "V_post": "V",
                                 "A_post": "A", "E_post": "E"})

    kw = build_td_hidden_model(
        rule, c_reg=compiler.c_reg, alpha=alpha, rho=rho,
        f_target=(compiler.f_target * compiler.dt) / 1000.0,
        alpha_fav=np.exp(-compiler.dt / compiler.tau_reg),
        v_thresh=target_neuron.v_thresh, td_lambda=compiler.td_lambda, send_back=send_back)
    kw["var_vals"]["g"] = connect_snippet.weight
    return WeightUpdateModel(**kw)


def _register_symmetric_feedback(conn, compiler, compile_state, target_pop):
    """When feedback_type == symmetric, the forward output connection
    itself doubles as the feedback path, so register it as such.
    """
    if conn.is_feedback:
        return
    if compiler.gamma_lambda is None:
        compile_state.feedback_connections.append(conn)
    elif target_pop in compiler.policy_heads:
        compile_state.policy_feedback_connections.append(conn)
    else:
        compile_state.value_feedback_connections.append(conn)


def build_output_wum(conn, connect_snippet, compiler, compile_state, target_pop, alpha):
    """Weight-update model for a connection targeting a readout population."""
    symmetric = (compiler.feedback_type == FeedbackType.SYMMETRIC.value)

    if compiler.gamma_lambda is not None:
        if target_pop in compiler.policy_heads:
            rule = OutputRule.POLICY_SYMMETRIC if symmetric else OutputRule.POLICY_RANDOM
            wum = WeightUpdateModel(
                model=get_output_model(rule),
                param_vals={"Alpha": alpha, "Lambda": compiler.td_lambda},
                var_vals={"g": connect_snippet.weight, "DeltaG": 0.0, "RLTrace": 0.0},
                pre_var_vals={"ZFilter": 0.0},
                post_neuron_var_refs={"E_post": "E", "PG_post": "PG", "TdE_post": "TdE"})
        else:
            rule = OutputRule.VALUE_SYMMETRIC if symmetric else OutputRule.VALUE_RANDOM
            wum = WeightUpdateModel(
                model=get_output_model(rule),
                param_vals={"RetE": 0.0, "Alpha": alpha, "GammaLambda": compiler.gamma},
                var_vals={"g": connect_snippet.weight, "DeltaG": 0.0, "RLTrace": 0.0},
                pre_var_vals={"ZFilter": 0.0},
                post_neuron_var_refs={"E_post": "E", "ValReg_post": "ValReg"})
    else:
        rule = OutputRule.SUPERVISED_SYMMETRIC if symmetric else OutputRule.SUPERVISED_RANDOM
        wum = WeightUpdateModel(
            model=get_output_model(rule),
            param_vals={"Alpha": alpha},
            var_vals={"g": connect_snippet.weight, "DeltaG": 0.0},
            pre_var_vals={"ZFilter": 0.0},
            post_neuron_var_refs={"E_post": "E"})

    if symmetric:
        _register_symmetric_feedback(conn, compiler, compile_state, target_pop)

    return wum
