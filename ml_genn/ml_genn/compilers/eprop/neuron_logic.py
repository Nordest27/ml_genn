"""Helpers for building per-neuron GeNN model modifications for e-prop.

These were previously all inlined in one long `build_neuron_model` method.
Splitting them up means each policy-head flavour, output flavour, and hidden
flavour is independently readable and testable.
"""

import numpy as np

from .variants import PolicyType, as_policy_type


def add_softmax_output_var(model_copy, compile_state, pop):
    """If this output uses cross-entropy loss, add a softmax'd output var
    and register the population for later softmax custom-update creation.
    """
    output_var = model_copy.output_var
    softmax_var_name = output_var[0] + "Softmax"
    model_copy.add_var(softmax_var_name, output_var[1], 0)
    model_copy.output_var_name = softmax_var_name
    compile_state.softmax_populations.append((pop, output_var[0], softmax_var_name))


def add_categorical_policy_code(model_copy, entropy_coeff, entropy_coeff_decay,
                                entropy_coeff_min):
    """Reward-based policy gradient + entropy regularisation for a
    categorical (softmax) policy head.
    """
    model_copy.add_additional_input_var("ISynTdE", "scalar", 0.0)
    model_copy.add_var("PG", "scalar", 0.0)
    model_copy.add_var("pre_PG", "scalar", 0.0)
    model_copy.add_var("TdE", "scalar", 0.0)
    model_copy.add_var("entropyCoeff", "scalar", entropy_coeff)

    model_copy.append_sim_code(
        f"""
        TdE = ISynTdE;

        const scalar p = {model_copy.output_var_name};
        const scalar logp = log(fmax(p, (scalar)1e-8));
        const scalar entropyGrad = p * (logp + 1.0);

        E = entropyCoeff * entropyGrad;
        entropyCoeff = fmax(
            {entropy_coeff_decay} * entropyCoeff,
            {entropy_coeff_min}
        );

        if (actionTaken != 0) {{
            PG = (p - yTrue);
        }}
        else {{
            PG = 0;
        }}
        actionTaken = 0;
        tdError = 0;
        """
    )


def add_gaussian_trace_policy_code(model_copy):
    """Continuous Gaussian policy head with a learned log-sigma and an
    exploration-noise eligibility trace (tanh-squashed action).
    """
    model_copy.add_additional_input_var("ISynTdE", "scalar", 0.0)
    model_copy.add_var("TanhOut", "scalar", 0.0)
    model_copy.add_var("TdE", "scalar", 0.0)
    model_copy.add_var("LogSigma", "scalar", np.log(0.01))
    model_copy.add_var("Action", "scalar", 0.0)
    model_copy.add_var("SigmaTrace", "scalar", 0.0)
    model_copy.add_var("SigmaLR", "scalar", 1e-4)
    model_copy.add_var("EpsTrace", "scalar", 0.0)
    model_copy.add_var("PG", "scalar", 0.0)

    model_copy.append_sim_code(
        f"""
        TdE = ISynTdE;

        const scalar mu = {model_copy.output_var_name};
        const scalar sigma = exp(LogSigma);

        const scalar eps = sigma * gennrand_normal();
        EpsTrace = EpsTrace * Alpha + eps;

        {model_copy.output_var_name} = {model_copy.output_var_name} + eps;
        TanhOut = tanh({model_copy.output_var_name});

        const scalar tanh_sq = TanhOut * TanhOut;
        PG = (EpsTrace / (sigma * sigma)) * (1.0 - tanh_sq);
        """
    )


def add_generic_policy_code(model_copy):
    """Generic policy head: consumes CPU-staged PG/E/PRew values for exactly
    one timestep at a time (used when the policy gradient is computed off
    of GPU, e.g. by an external environment/agent loop).
    """
    model_copy.add_additional_input_var("ISynTdE", "scalar", 0.0)
    model_copy.add_var("PG", "scalar", 0.0)
    model_copy.add_var("pre_PG", "scalar", 0.0)
    model_copy.add_var("TdE", "scalar", 0.0)
    model_copy.add_var("pre_E", "scalar", 0.0)
    model_copy.add_var("pre_PRew", "scalar", 0.0)

    model_copy.append_sim_code(
        """
        TdE = ISynTdE;

        // Consume CPU-written values for exactly one timestep,
        // then zero the staging vars so they don't accumulate.
        PG = pre_PG;
        pre_PG = 0.0;

        E = pre_E;
        pre_E = 0.0;

        PRew = pre_PRew;
        pre_PRew = 0;
        """
    )


def add_value_head_code(model_copy, gamma, reward_decay, value_reg=0.0):
    """TD error of the value (critic) head: E = gamma V_t + r_t - V_{t-1}, with the reward decaying by
    reward_decay per step. value_reg > 0 adds a smoothness regulariser value_reg * (V_t - V_{t-1})
    (ValReg, used by the value readout's update); 0 reproduces the original implementation."""
    model_copy.add_var("ValReg", "scalar", 0.0)
    model_copy.add_var("PrevVal", "scalar", 0.0)
    out = model_copy.output_var_name
    reg = f"ValReg = {value_reg!r} * ({out} - PrevVal);\n" if value_reg != 0 else ""
    model_copy.append_sim_code(
        f"""
        E = {out} * {gamma} + reward - PrevVal; // TD error
        {reg}PrevVal = {out};
        reward *= {reward_decay};
        tdError = 0;
        """
    )


_POLICY_HEAD_BUILDERS = {
    PolicyType.CATEGORICAL: add_categorical_policy_code,
    PolicyType.GAUSSIAN_TRACE: lambda model_copy, **_: add_gaussian_trace_policy_code(model_copy),
    PolicyType.GENERIC: lambda model_copy, **_: add_generic_policy_code(model_copy),
}


def add_rl_output_head_code(model_copy, pop, compiler):
    """Dispatch to the correct RL output-head code builder (policy or value)
    based on whether `pop` is registered as a policy head or the value head.
    """
    model_copy.add_var("PRew", "scalar", 0.0)

    policy_type = compiler.policy_heads.get(pop)
    if policy_type is not None:
        policy_type = as_policy_type(policy_type)
        builder = _POLICY_HEAD_BUILDERS[policy_type]
        builder(model_copy,
               entropy_coeff=compiler.entropy_coeff,
               entropy_coeff_decay=compiler.entropy_coeff_decay,
               entropy_coeff_min=compiler.entropy_coeff_min)
    else:
        add_value_head_code(model_copy, compiler.gamma, compiler.reward_decay, compiler.value_reg)


def add_supervised_error_code(model_copy):
    """Plain supervised E = output - target error code (non-RL path)."""
    model_copy.append_sim_code(f"E = {model_copy.output_var_name} - yTrue;")


def add_output_bias_training_code(model_copy, compile_state, pop):
    """Make Bias trainable and accumulate its gradient (DeltaBias)."""
    model_copy.make_param_var("Bias")
    model_copy.add_var("DeltaBias", "scalar", 0.0)
    model_copy.append_sim_code("DeltaBias += E;")
    compile_state.bias_optimiser_populations.append(pop)
    compile_state.checkpoint_population_vars.append((pop, "Bias"))


def add_hidden_feedback_code(model_copy):
    """Feedback-receiving code common to every hidden (non-output,
    non-input) neuron: consume ISynFeedback into E, track a running
    baseline, and draw the per-timestep noise samples used by the
    perturbation-based (distributed) rules.
    """
    model_copy.add_additional_input_var("ISynFeedback", "scalar", 0.0)
    model_copy.add_var("E", "scalar", 0.0)
    model_copy.add_var("Ebase", "scalar", 0.0)
    model_copy.add_var("Noise1", "scalar", 0.0)
    model_copy.add_var("Noise2", "scalar", 0.0)
    model_copy.add_var("Noise3", "scalar", 0.0)
    model_copy.add_var("Noise4", "scalar", 0.0)

    model_copy.append_sim_code(
        """
        E = ISynFeedback;
        Ebase = Ebase * 0.999 + E * 0.001;
        Noise1 = gennrand_normal();
        Noise2 = gennrand_uniform();
        Noise3 = gennrand_normal();
        Noise4 = gennrand_normal();
        """
    )


def add_hidden_rl_input_refs(model_copy, backprop=False):
    """RL-specific additional input vars/state for a hidden neuron, plus
    the sim-code that unpacks each ISyn* channel into its named variable.
    """
    for isyn_name in ("ISynSigmaEps", "ISynPolicyGradient", "ISynValueError",
                      "ISynPolicyRegularisation", "ISynValueRegularisation",
                      "ISynTdE", "ISynPolicyReward", "ISynPertEps"):
        model_copy.add_additional_input_var(isyn_name, "scalar", 0.0)

    model_copy.add_var("PG", "scalar", 0.0)
    model_copy.add_var("PRew", "scalar", 0.0)
    model_copy.add_var("VE", "scalar", 0.0)
    model_copy.add_var("PR", "scalar", 0.0)
    model_copy.add_var("VR", "scalar", 0.0)
    model_copy.add_var("PGEps", "scalar", 0.0)
    if not model_copy.has_var("TdE"):          # the ALIF neuron model already declares TdE
        model_copy.add_var("TdE", "scalar", 0.0)

    model_copy.append_sim_code(
        """
        PG = ISynPolicyGradient;
        VE = ISynValueError;
        PR = ISynPolicyRegularisation;
        VR = ISynValueRegularisation;
        TdE = ISynTdE;
        PRew = ISynPolicyReward;
        PGEps = ISynPertEps;
        """
    )
    if backprop:
        # backward gradient signal: sum over this neuron's postsynaptic hidden neurons of g * score_post
        model_copy.add_additional_input_var("ISynBack", "scalar", 0.0)
        model_copy.add_var("Back", "scalar", 0.0)
        model_copy.append_sim_code("Back = ISynBack;")


def add_input_noise_code(model_copy):
    """Per-timestep noise samples for an Input-neuron population (needed
    so that downstream distributed/perturbation rules have noise to draw
    on even at the input layer).
    """
    model_copy.add_additional_input_var("ISynPertEps", "scalar", 0.0)
    model_copy.add_var("Noise1", "scalar", 0.0)
    model_copy.add_var("Noise2", "scalar", 0.0)
    model_copy.add_var("Noise3", "scalar", 0.0)
    model_copy.add_var("Noise4", "scalar", 0.0)

    model_copy.append_sim_code(
        """
        Noise1 = gennrand_normal();
        Noise2 = gennrand_normal();
        Noise3 = gennrand_normal();
        Noise4 = gennrand_normal();
        """
    )


# Switches in the ALIF neuron model (ml_genn.neurons.AdaptiveLeakyIntegrateFire) that keep membrane
# noise off by default.
_NODE_NOISE_SWITCHES = (("Sigma = 0.0 * exp(LogSigma);", "Sigma = exp(LogSigma);"),
                        ("V = Alpha * V + Isyn + 0.0 * PertEps;", "V = Alpha * V + Isyn + PertEps;"))


def enable_node_noise(model_copy):
    """Turn on membrane ("node") noise in an ALIF hidden neuron: PertEps = Sigma * N(0, 1) is added to
    the membrane and filtered into PertEpsTrace by the neuron model itself."""
    sim = model_copy.model.get("sim_code", "")
    for source, target in _NODE_NOISE_SWITCHES:
        if source not in sim:
            raise RuntimeError(f"node noise: '{source}' not found in the ALIF sim code; the neuron "
                               "model changed, update _NODE_NOISE_SWITCHES")
        model_copy.replace_sim_code(source, target)
