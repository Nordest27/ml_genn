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


def build_hidden_wum(conn, connect_snippet, compiler, compile_state, target_neuron, alpha):
    """Weight-update model for a connection targeting a hidden LIF/ALIF neuron."""
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
