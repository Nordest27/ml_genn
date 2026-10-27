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


# Compressive value transform (Pohlen et al. 2018): h(x) = sign(x) (sqrt(|x| + 1) - 1) + eps x and its exact inverse.
# The head learns V_h ~ h(V) with the transformed TD target h(r + gamma h^-1(V_h)): returns are unchanged (exact for
# deterministic transitions), large values are compressed, and the broadcast TD error is compressed with them.
VALUE_TRANSFORM_EPS = 1e-3


def value_transform_h(x, eps=VALUE_TRANSFORM_EPS):
    return np.sign(x) * (np.sqrt(np.abs(x) + 1.0) - 1.0) + eps * x


def value_transform_h_inv(y, eps=VALUE_TRANSFORM_EPS):
    return np.sign(y) * (((np.sqrt(1.0 + 4.0 * eps * (np.abs(y) + 1.0 + eps)) - 1.0) / (2.0 * eps)) ** 2 - 1.0)


def _h_code(x, eps):
    return f"(copysign(sqrt(fabs({x}) + 1.0) - 1.0, {x}) + {eps!r} * ({x}))"


def _h_inv_code(y, eps):
    root = f"((sqrt(1.0 + {4.0 * eps!r} * (fabs({y}) + {1.0 + eps!r})) - 1.0) / {2.0 * eps!r})"
    return f"copysign({root} * {root} - 1.0, {y})"


def add_value_head_code(model_copy, gamma, reward_decay, value_reg=0.0, afterglow=None, transform=None,
                        centring=None):
    """TD error of the value (critic) head: E = gamma V_t + r_t - V_{t-1}, with the reward decaying by
    reward_decay per step. value_reg > 0 adds a smoothness regulariser value_reg * (V_t - V_{t-1})
    (ValReg, used by the value readout's update); 0 reproduces the original implementation.

    afterglow = {"lr": eta, "max": w_max}: the head's value is V = output + AgW * reward, where the decaying reward
    variable is the reward still to come from past rewards (its exact value is gamma d / (1 - gamma d) * reward). AgW
    starts at 0 and is learned by TD on that one feature, clipped to [0, w_max]: while AgW is small the reward trace
    spreads credit over the following steps, as it approaches the exact value the TD error approaches that of an
    instantaneous reward (eta sets how fast credit goes from coarse to fine)."""
    model_copy.add_var("ValReg", "scalar", 0.0)
    model_copy.add_var("PrevVal", "scalar", 0.0)
    out = model_copy.output_var_name
    reg = f"ValReg = {value_reg!r} * ({out} - PrevVal);\n" if value_reg != 0 else ""
    if centring:
        # reward centring (continuing tasks): the TD error uses r - RhoBar, RhoBar a running estimate of the reward per
        # step, so the value function only has to learn values relative to the average (no r / (1 - gamma) offset).
        # mode "reward": RhoBar = running mean of the reward; "td": RhoBar += rate * TD error (it also absorbs the
        # critic's mean error, so the broadcast TD error becomes ~zero-mean)
        mode, rate = centring.get("mode", "reward"), float(centring.get("rate", 1e-4))
        if mode not in ("reward", "td"):
            raise ValueError(f"value centring mode must be 'reward' or 'td', not {mode!r}")
        if transform is not None or afterglow:
            raise ValueError("value centring cannot be combined with value_transform or value_afterglow")
        model_copy.add_var("RhoBar", "scalar", 0.0)
        pre = f"RhoBar = {1.0 - rate!r} * RhoBar + {rate!r} * reward;\n" if mode == "reward" else ""
        post = f"RhoBar += {rate!r} * E;\n" if mode == "td" else ""
        model_copy.append_sim_code(
            f"""
            {pre}E = {out} * {gamma} + (reward - RhoBar) - PrevVal; // TD error with the centred reward
            {post}{reg}PrevVal = {out};
            reward *= {reward_decay};
            tdError = 0;
            """
        )
        return
    if transform is not None:
        # "sqrt": the TD error is in transformed units everywhere (the broadcast is compressed too);
        # "sqrt_head": the readout learns from the transformed error, the network receives the TD error in the
        # original units, r + gamma h^-1(V_h) - h^-1(V_h previous) (TdBroadcast, read by the TD-error transport)
        if transform not in ("sqrt", "sqrt_head"):
            raise ValueError(f"value transform must be None, 'sqrt' or 'sqrt_head', not {transform!r}")
        if afterglow:
            raise ValueError("value_afterglow and value_transform cannot be combined")
        eps = VALUE_TRANSFORM_EPS
        target = _h_code(f"(reward + {gamma} * {_h_inv_code(out, eps)})", eps)
        broadcast = ""
        if transform == "sqrt_head":
            model_copy.add_var("TdBroadcast", "scalar", 0.0)
            broadcast = (f"TdBroadcast = reward + {gamma} * {_h_inv_code(out, eps)} - "
                         f"{_h_inv_code('PrevVal', eps)};\n")
        model_copy.append_sim_code(
            f"""
            E = {target} - PrevVal; // TD error in transformed units
            {broadcast}{reg}PrevVal = {out};
            reward *= {reward_decay};
            tdError = 0;
            """
        )
        return
    if afterglow:
        model_copy.add_var("AgW", "scalar", 0.0)
        model_copy.add_var("PrevRew", "scalar", 0.0)
        lr, w_max = float(afterglow.get("lr", 1e-4)), float(afterglow.get("max", 1e6))
        model_copy.append_sim_code(
            f"""
            const scalar valNow = {out} + AgW * reward;
            E = valNow * {gamma} + reward - PrevVal; // TD error
            {reg.replace(out, "valNow")}PrevVal = valNow;
            AgW = fmin(fmax(AgW + {lr!r} * E * PrevRew, 0.0), {w_max!r});
            PrevRew = reward;
            reward *= {reward_decay};
            tdError = 0;
            """
        )
        return
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
        add_value_head_code(model_copy, compiler.gamma, compiler.reward_decay, compiler.value_reg,
                            getattr(compiler, "value_afterglow", None), getattr(compiler, "value_transform", None),
                            getattr(compiler, "value_centring", None))


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


def add_inhibitory_reversal(model_copy, e_inh):
    """Conductance-like inhibition: inhibitory synapses (routed to ISynInh) act through the driving force
    (V - E_inh) / (0 - E_inh), unchanged at rest (V = 0) for small inputs, vanishing at the reversal potential
    E_inh (< 0), so inhibition cannot push V below E_inh (shunting instead of unlimited subtraction)."""
    if e_inh >= 0:
        raise ValueError(f"inhibitory reversal potential must be below rest (0), not {e_inh}")
    model_copy.add_additional_input_var("ISynInh", "scalar", 0.0)
    sim = model_copy.model.get("sim_code", "")
    source = "V = Alpha * V + Isyn"
    if sim.count(source) != 1:
        raise RuntimeError("inhibitory reversal: the ALIF membrane update was not found exactly once")
    # exact step of a conductance G = ISynInh / E_inh (>= 0): V relaxes towards E_inh by (E_inh - V)(1 - exp(-G)),
    # = ISynInh (V - E_inh) / -E_inh for small inputs, and never past E_inh however many inhibitory spikes arrive
    model_copy.replace_sim_code(source, f"V = Alpha * V + Isyn + ({e_inh!r} - V) * (1.0 - exp(-fmax(0.0, ISynInh / ({e_inh!r}))))")


def add_inhibitory_channel(model_copy):
    """Route inhibitory synapses to their own input ISynInh, added to the membrane like Isyn (current-based, as
    before); used when the neuron needs to know the inhibition it receives (decorrelation reward)."""
    model_copy.add_additional_input_var("ISynInh", "scalar", 0.0)
    sim = model_copy.model.get("sim_code", "")
    source = "V = Alpha * V + Isyn"
    if sim.count(source) != 1:
        raise RuntimeError("inhibitory channel: the ALIF membrane update was not found exactly once")
    model_copy.replace_sim_code(source, "V = Alpha * V + Isyn + ISynInh")


def add_cancel_tracking(model_copy, tau, dt):
    """Neuron state for the cancellation reward: DExc, the change of the excitatory drive this step (with the
    inhibitory channel, Isyn holds the excitatory input only), and DExcMean, its running average (time constant tau)."""
    model_copy.add_var("PrevExc", "scalar", 0.0)
    model_copy.add_var("DExc", "scalar", 0.0)
    model_copy.add_var("DExcMean", "scalar", 0.0)
    decay = float(np.exp(-dt / tau))
    model_copy.append_sim_code(f"""
        DExc = fabs(Isyn - PrevExc);
        DExcMean = {decay!r} * DExcMean + {1.0 - decay!r} * DExc;
        PrevExc = Isyn;
        """)


def add_decorr_tracking(model_copy, tau, dt):
    """Neuron state for the decorrelation reward: InhNow, the magnitude of the inhibitory input this step, and
    InhMean, its running average (time constant tau)."""
    model_copy.add_var("InhNow", "scalar", 0.0)
    model_copy.add_var("InhMean", "scalar", 0.0)
    model_copy.add_var("InhVar", "scalar", 0.0)
    decay = float(np.exp(-dt / tau))
    model_copy.append_sim_code(f"""
        InhNow = fabs(ISynInh);
        InhMean = {decay!r} * InhMean + {1.0 - decay!r} * InhNow;
        InhVar = {decay!r} * InhVar + {1.0 - decay!r} * (InhNow - InhMean) * (InhNow - InhMean);
        """)


def add_inhibition_tracking(model_copy, tau, warmup, dt, reversal):
    """Neuron state for local synaptic scaling of plastic inhibition: InhAvg, a running average (time constant tau)
    of the magnitude of the inhibitory current received on the plastic-inhibition channel, and InhTarget, that
    average captured once at time `warmup`. Without an inhibitory reversal the plastic inhibitory synapses get their
    own input channel ISynPI, added to the membrane like Isyn; with it they share ISynInh."""
    channel = "ISynInh" if reversal else "ISynPI"
    if not reversal:
        model_copy.add_additional_input_var("ISynPI", "scalar", 0.0)
        sim = model_copy.model.get("sim_code", "")
        source = "V = Alpha * V + Isyn"
        if sim.count(source) != 1:
            raise RuntimeError("inhibition tracking: the ALIF membrane update was not found exactly once")
        model_copy.replace_sim_code(source, "V = Alpha * V + Isyn + ISynPI")
    model_copy.add_var("InhAvg", "scalar", 0.0)
    model_copy.add_var("InhTarget", "scalar", 0.0)
    decay = float(np.exp(-dt / tau))
    model_copy.append_sim_code(f"""
        InhAvg = {decay!r} * InhAvg + {1.0 - decay!r} * fabs({channel});
        if (InhTarget == 0.0 && t >= {float(warmup)!r}) {{
            InhTarget = fmax(InhAvg, 1e-6);
        }}
        """)


def enable_node_noise(model_copy):
    """Turn on membrane ("node") noise in an ALIF hidden neuron: PertEps = Sigma * N(0, 1) is added to
    the membrane and filtered into PertEpsTrace by the neuron model itself."""
    sim = model_copy.model.get("sim_code", "")
    for source, target in _NODE_NOISE_SWITCHES:
        if source not in sim:
            raise RuntimeError(f"node noise: '{source}' not found in the ALIF sim code; the neuron "
                               "model changed, update _NODE_NOISE_SWITCHES")
        model_copy.replace_sim_code(source, target)
