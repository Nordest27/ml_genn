"""Raw GeNN weight-update model definitions for e-prop and its variants.

Each model here is a single, self-contained GeNN WUM dict - no commented-out
alternatives living inside the same string. To pick one at compile time, use
`get_hidden_model`, `get_output_model`, or `get_feedback_model` below with the
enums from `.variants`.
"""

from copy import deepcopy
from pygenn import VarAccess, CustomUpdateVarAccess

from .variants import HiddenRule, OutputRule, FeedbackRule, NoiseSource, PseudoDerivativeGate, RoutingMode 


# ============================================================================
# Hidden-layer (recurrent) eligibility-trace models
# ============================================================================

_EPROP_LIF_PERT = {
    "params": [("CReg", "scalar"), ("Alpha", "scalar"),
               ("FTarget", "scalar"), ("AlphaFAv", "scalar"),
               ("Vthresh_post", "scalar")],
    "vars": [("g", "scalar", VarAccess.READ_ONLY),
             ("eFiltered", "scalar"), ("DeltaG", "scalar"),
             ("NoiseTrace", "scalar"), ("RLNoiseTrace", "scalar")],
    "pre_vars": [("ZFilter", "scalar")],
    "post_vars": [("Psi", "scalar"), ("FAvg", "scalar")],
    "post_neuron_var_refs": [("RefracTime_post", "scalar"), ("V_post", "scalar"),
                             ("E_post", "scalar"), ("Ebase_post", "scalar"),
                             ("Noise", "scalar")],

    "pre_spike_code": "ZFilter += 1.0;",
    "pre_dynamics_code": "ZFilter *= Alpha;",

    "post_spike_code": "FAvg += (1.0 - AlphaFAv);",
    "post_dynamics_code": """
    FAvg *= AlphaFAv;
    if (RefracTime_post > 0.0) {
      Psi = 0.0;
    }
    else {
      Psi = (1.0 / Vthresh_post) * 0.3 * fmax(0.0, 1.0 - fabs((V_post - Vthresh_post) / Vthresh_post));
    }
    """,

    "pre_spike_syn_code": """
    scalar eps = 0.01 * Noise;
    addToPost(g + eps);
    NoiseTrace += eps;
    """,

    "synapse_dynamics_code": """
    const scalar e = ZFilter * Psi;
    scalar eF = eFiltered;
    eF = (eF * Alpha) + e;

    DeltaG += 0.0 * (eF * E_post) + ((FAvg - FTarget) * CReg * e) + e * NoiseTrace * fabs(E_post);

    NoiseTrace *= Alpha;
    eFiltered = eF;
    """}

_EPROP_ALIF = {
    "params": [("CReg", "scalar"), ("Alpha", "scalar"), ("Rho", "scalar"),
               ("FTarget", "scalar"), ("AlphaFAv", "scalar"),
               ("Vthresh_post", "scalar"), ("Beta_post", "scalar")],
    "vars": [("g", "scalar", VarAccess.READ_ONLY),
             ("eFiltered", "scalar"), ("epsilonA", "scalar"),
             ("DeltaG", "scalar")],
    "pre_vars": [("ZFilter", "scalar")],
    "post_vars": [("Psi", "scalar"), ("FAvg", "scalar")],
    "post_neuron_var_refs": [("RefracTime_post", "scalar"), ("V_post", "scalar"),
                             ("A_post", "scalar"), ("E_post", "scalar")],

    "pre_spike_code": "ZFilter += 1.0;",
    "pre_dynamics_code": "ZFilter *= Alpha;",

    "post_spike_code": "FAvg += (1.0 - AlphaFAv);",
    "post_dynamics_code": """
    FAvg *= AlphaFAv;
    if (RefracTime_post > 0.0) {
      Psi = 0.0;
    }
    else {
      Psi = (1.0 / Vthresh_post) * 0.3 * fmax(0.0, 1.0 - fabs((V_post - (Vthresh_post + (Beta_post * A_post))) / Vthresh_post));
    }
    """,

    "pre_spike_syn_code": "addToPost(g);",
    "synapse_dynamics_code": """
    scalar epsA = epsilonA;
    const scalar psiZFilter = Psi * ZFilter;
    const scalar psiBetaEpsilonA = Psi * Beta_post * epsA;

    const scalar e = psiZFilter - psiBetaEpsilonA;
    epsilonA = psiZFilter + ((Rho * epsA) - psiBetaEpsilonA);

    scalar eF = eFiltered;
    eF = (eF * Alpha) + e;

    DeltaG += (eF * E_post) + ((FAvg - FTarget) * CReg * e);
    eFiltered = eF;
    """}

_ALIF_TD_BASE_PARAMS = [("CReg", "scalar"), ("Alpha", "scalar"), ("Rho", "scalar"),
                        ("FTarget", "scalar"), ("AlphaFAv", "scalar"),
                        ("Vthresh_post", "scalar"), ("Lambda", "scalar")]

_ALIF_TD_BASE_VARS = [("g", "scalar", VarAccess.READ_ONLY),
                      ("eFiltered", "scalar"), ("epsilonA", "scalar"),
                      ("DeltaG", "scalar"), ("RLTrace", "scalar"),
                      ("RLEpsTrace", "scalar"), ("NoiseTrace", "scalar"),
                      ("RLNoiseTrace", "scalar"), ("LogSynSig", "scalar"),
                      ("SynSig", "scalar"), ("SynSigTrace", "scalar"),
                      ("UpdateCount", "scalar")]

_ALIF_TD_POST_VARS = [("Psi", "scalar"), ("FAvg", "scalar"), ("FAvgTrace", "scalar"),
                      ("Z", "scalar"), ("NeuronNoiseTrace", "scalar")]

_ALIF_TD_POST_NEURON_VAR_REFS = [
    ("RefracTime_post", "scalar"), ("V_post", "scalar"), ("Beta_post", "scalar"),
    ("A_post", "scalar"), ("E_post", "scalar"), ("PG_post", "scalar"),
    ("VE_post", "scalar"), ("PR_post", "scalar"), ("VR_post", "scalar"),
    ("TdE_post", "scalar"),
    ("PGEps_post", "scalar"), ("PRew_post", "scalar"),
    ("Noise1", "scalar"), ("Noise2", "scalar"),
    ("Noise3", "scalar"), ("Noise4", "scalar")]

# Only consumed by _ROUTING_ADDITIVE_EXTRA / _ROUTING_MULTIPLICATIVE_EXTRA below.
# Kept separate from the base refs above because the hidden neuron doesn't
# declare Sigma/PertEpsTrace (see add_hidden_rl_input_refs / the
# _UNWIRED_ROUTING_MODES guard on get_hidden_model) - including them
# unconditionally would break RoutingMode.NONE, which doesn't need them.
_ALIF_TD_ROUTING_POST_NEURON_VAR_REFS = [
    ("PertEpsTrace_post", "scalar"), ("Sigma_post", "scalar")]

_ALIF_TD_POST_DYNAMICS_CODE = """
Z = 0.0;
FAvg *= AlphaFAv;

scalar fireReg = fabs(FAvg - FTarget);
FAvgTrace = Lambda * FAvgTrace + (1.0 - Lambda) * fabs(fireReg);

if ( RefracTime_post > 0.0 ) {
  Psi = 0.0;
}
else {
  Psi = (1.0 / Vthresh_post) * 0.3 * fmax(0.0, 1.0 - fabs((V_post - (Vthresh_post + (Beta_post * A_post))) / Vthresh_post));
}
"""

# --- pre_spike_syn_code variants: where the perturbation noise comes from ---

_PRE_SPIKE_SYN_NODE_NOISE = """
scalar Noise = Noise1;
SynSig = 0.0 * exp(LogSynSig - 5);

scalar eps = SynSig * Noise;
addToPost(g + eps);
NoiseTrace += eps;
"""

_PRE_SPIKE_SYN_SYNAPTIC_ROTATING_NOISE = """
scalar Noise;
const scalar selector = ((int)(1e7*eFiltered)) % 2;
if (selector == 0) Noise = Noise1;
else if (selector == 1) Noise = Noise2;
else if (selector == 2) Noise = Noise3;
else Noise = Noise4;

SynSig = 0.0 * exp(LogSynSig - 5);

scalar eps = SynSig * Noise;
addToPost(g + eps);
NoiseTrace += eps;
"""

# --- synapse_dynamics_code variants: pseudo-derivative gate on/off ---

_SYN_DYNAMICS_GATED = """
UpdateCount *= AlphaFAv;
if (DeltaG == 0.0) UpdateCount += (1.0 - AlphaFAv);
scalar reward = TdE_post + PRew_post;
DeltaG += reward * (RLTrace + RLEpsTrace + RLNoiseTrace)
    + 0.0 * CReg * (PR_post + VR_post) * eFiltered;

DeltaG += CReg * Z * (eFiltered - (RLEpsTrace + RLNoiseTrace));

scalar epsA = epsilonA;
const scalar psiZFilter = Psi * ZFilter;
const scalar psiBetaEpsilonA = Psi * Beta_post * epsA;

const scalar e = psiZFilter - psiBetaEpsilonA;
epsilonA = psiZFilter + ((Rho * epsA) - psiBetaEpsilonA);

eFiltered = (eFiltered * Alpha) + e;

RLTrace = Lambda * RLTrace + eFiltered * ( 1.0 * PG_post - 0.1 * VE_post + 0.00 * PGEps_post );

NoiseTrace *= Alpha;
"""

# "weight dist no heuristic" ablation: e/epsilonA computed WITHOUT the Psi
# pseudo-derivative gate multiplying the perturbation-driven terms. This is
# the failure-mode variant from Fig. snake-comparison — kept only for
# reproducing that ablation, not recommended for actual training.
_SYN_DYNAMICS_UNGATED = """
UpdateCount *= AlphaFAv;
if (DeltaG == 0.0) UpdateCount += (1.0 - AlphaFAv);
scalar reward = TdE_post + PRew_post;
DeltaG += reward * (RLTrace + RLEpsTrace + RLNoiseTrace)
    + 0.0 * CReg * (PR_post + VR_post) * eFiltered;

DeltaG += CReg * Z * eFiltered;

scalar epsA = epsilonA;
const scalar psiZFilter = ZFilter;
const scalar psiBetaEpsilonA = Beta_post * epsA;

const scalar e = psiZFilter - psiBetaEpsilonA;
epsilonA = psiZFilter + ((Rho * epsA) - psiBetaEpsilonA);

eFiltered = (eFiltered * Alpha) + e;

RLTrace = Lambda * RLTrace + eFiltered * ( 1.0 * PG_post - 0.1 * VE_post + 0.00 * PGEps_post );

NoiseTrace *= Alpha;
"""

# --- structured-routing additions (Appendix: routed variants) ---
# These append an extra addToPre(...) term to synapse_dynamics_code, using
# the routed local gradient g^t_j via NormPertEpsTrace/NormNoiseTrace. Left
# as an explicit opt-in since the paper found neither improved on
# broadcast-only and the multiplicative one is harder to reason about.

_ROUTING_ADDITIVE_EXTRA = """
const scalar NormPertEpsTrace = (1.0 - Alpha * Alpha) * PertEpsTrace_post / (Sigma_post * Sigma_post + 1e-6);
const scalar NormNoiseTrace = (1.0 - Alpha * Alpha) * NoiseTrace / (SynSig * SynSig + 1e-6);
addToPre( g * Psi * NormPertEpsTrace );
addToPre( g * Psi * NormNoiseTrace );
"""

_ROUTING_MULTIPLICATIVE_EXTRA = """
const scalar NormPertEpsTrace = (1.0 - Alpha * Alpha) * PertEpsTrace_post / (Sigma_post * Sigma_post + 1e-6);
const scalar NormNoiseTrace = (1.0 - Alpha * Alpha) * NoiseTrace / (SynSig * SynSig + 1e-6);
addToPre( g * Psi * NormPertEpsTrace * (1.0 + PGEps_post) );
addToPre( g * Psi * NormNoiseTrace * (1.0 + PGEps_post) );
"""


def _make_alif_td_lambda_model(pre_spike_syn_code, synapse_dynamics_code,
                               routed=False):
    post_neuron_var_refs = list(_ALIF_TD_POST_NEURON_VAR_REFS)
    if routed:
        post_neuron_var_refs += _ALIF_TD_ROUTING_POST_NEURON_VAR_REFS

    return {
        "params": _ALIF_TD_BASE_PARAMS,
        "vars": _ALIF_TD_BASE_VARS,
        "pre_vars": [("ZFilter", "scalar")],
        "post_vars": _ALIF_TD_POST_VARS,
        "pre_neuron_var_refs": [("Noise1_pre", "scalar")],
        "post_neuron_var_refs": post_neuron_var_refs,
        "pre_spike_code": "ZFilter += 1.0;",
        "pre_dynamics_code": "ZFilter *= Alpha;",
        "post_spike_code": "FAvg += (1.0 - AlphaFAv);\nZ = 1.0;",
        "post_dynamics_code": _ALIF_TD_POST_DYNAMICS_CODE,
        "pre_spike_syn_code": pre_spike_syn_code,
        "synapse_dynamics_code": synapse_dynamics_code,
    }


# The full cross-product the paper actually ablates: noise source x gate x routing.
_ALIF_TD_LAMBDA_VARIANTS = {
    (NoiseSource.NODE, PseudoDerivativeGate.ENABLED, RoutingMode.NONE):
        _make_alif_td_lambda_model(_PRE_SPIKE_SYN_NODE_NOISE, _SYN_DYNAMICS_GATED),
    (NoiseSource.NODE, PseudoDerivativeGate.DISABLED, RoutingMode.NONE):
        _make_alif_td_lambda_model(_PRE_SPIKE_SYN_NODE_NOISE, _SYN_DYNAMICS_UNGATED),
    (NoiseSource.SYNAPTIC_ROTATING, PseudoDerivativeGate.ENABLED, RoutingMode.NONE):
        _make_alif_td_lambda_model(_PRE_SPIKE_SYN_SYNAPTIC_ROTATING_NOISE, _SYN_DYNAMICS_GATED),
    (NoiseSource.SYNAPTIC_ROTATING, PseudoDerivativeGate.DISABLED, RoutingMode.NONE):
        _make_alif_td_lambda_model(_PRE_SPIKE_SYN_SYNAPTIC_ROTATING_NOISE, _SYN_DYNAMICS_UNGATED),
    (NoiseSource.NODE, PseudoDerivativeGate.ENABLED, RoutingMode.ADDITIVE):
        _make_alif_td_lambda_model(_PRE_SPIKE_SYN_NODE_NOISE, _SYN_DYNAMICS_GATED + _ROUTING_ADDITIVE_EXTRA,
                                   routed=True),
    (NoiseSource.NODE, PseudoDerivativeGate.ENABLED, RoutingMode.MULTIPLICATIVE):
        _make_alif_td_lambda_model(_PRE_SPIKE_SYN_NODE_NOISE, _SYN_DYNAMICS_GATED + _ROUTING_MULTIPLICATIVE_EXTRA,
                                   routed=True),
}


def get_alif_td_lambda_model(noise_source: NoiseSource,
                             gate: PseudoDerivativeGate = PseudoDerivativeGate.ENABLED,
                             routing: RoutingMode = RoutingMode.NONE) -> dict:
    return deepcopy(_ALIF_TD_LAMBDA_VARIANTS[(noise_source, gate, routing)])

_HIDDEN_MODELS = {
    HiddenRule.LIF: _EPROP_LIF_PERT,
    HiddenRule.ALIF: _EPROP_ALIF,
}

# Routing modes whose Sigma_post/PertEps_post/PertEpsTrace_post inputs aren't
# actually populated on the hidden neuron yet (see neuron_logic.add_hidden_rl_input_refs
# and CompileState.sigma_eps_connections, which nothing currently appends to).
# Block these until that transport channel is wired up, rather than silently
# training with a channel that's always zero.
_UNWIRED_ROUTING_MODES = {RoutingMode.ADDITIVE, RoutingMode.MULTIPLICATIVE}


def get_hidden_model(rule: HiddenRule,
                      *,
                      noise_source: NoiseSource = NoiseSource.NODE,
                      gate: PseudoDerivativeGate = PseudoDerivativeGate.ENABLED,
                      routing: RoutingMode = RoutingMode.NONE) -> dict:
    """Return a fresh copy of the hidden-layer model dict for `rule`.

    `noise_source`/`gate`/`routing` are only meaningful for
    `HiddenRule.ALIF_TD_LAMBDA`; they're ignored for `LIF`/`ALIF`.
    """
    if rule is HiddenRule.ALIF_TD_LAMBDA:
        if routing in _UNWIRED_ROUTING_MODES:
            raise NotImplementedError(
                f"RoutingMode.{routing.name} is defined in the model cross-product "
                "but its Sigma/PertEps/PertEpsTrace feedback channel isn't wired up "
                "on the hidden neuron yet (see add_hidden_rl_input_refs and "
                "CompileState.sigma_eps_connections). Use RoutingMode.NONE until "
                "that's implemented."
            )
        return get_alif_td_lambda_model(noise_source, gate, routing)
    return deepcopy(_HIDDEN_MODELS[rule])


# ============================================================================
# Output (readout) models
# ============================================================================

_OUTPUT_SUPERVISED_SYMMETRIC = {
    "params": [("Alpha", "scalar")],
    "vars": [("g", "scalar", VarAccess.READ_ONLY), ("DeltaG", "scalar")],
    "pre_vars": [("ZFilter", "scalar")],
    "post_neuron_var_refs": [("E_post", "scalar")],

    "pre_spike_code": "ZFilter += 1.0;",
    "pre_dynamics_code": "ZFilter *= Alpha;",

    "pre_spike_syn_code": "addToPost(g);",
    "synapse_dynamics_code": """
    DeltaG += ZFilter * E_post;
    addToPre(g * E_post);
    """}

_OUTPUT_SUPERVISED_RANDOM = deepcopy(_OUTPUT_SUPERVISED_SYMMETRIC)
_OUTPUT_SUPERVISED_RANDOM["synapse_dynamics_code"] = "DeltaG += ZFilter * E_post;"

_OUTPUT_SUPERVISED_ADAPTIVE_FEEDBACK = deepcopy(_OUTPUT_SUPERVISED_SYMMETRIC)
del _OUTPUT_SUPERVISED_ADAPTIVE_FEEDBACK["pre_spike_syn_code"]

_OUTPUT_VALUE_SYMMETRIC = {
    "params": [("Alpha", "scalar"), ("GammaLambda", "scalar"), ("RetE", "scalar")],
    "vars": [("g", "scalar", VarAccess.READ_ONLY), ("DeltaG", "scalar"), ("RLTrace", "scalar")],
    "pre_vars": [("ZFilter", "scalar")],
    "post_neuron_var_refs": [("E_post", "scalar"), ("ValReg_post", "scalar")],

    "pre_spike_code": "ZFilter += 1.0;",
    "pre_dynamics_code": "ZFilter *= Alpha;",

    "pre_spike_syn_code": "addToPost(g);",
    "synapse_dynamics_code": """
    DeltaG += RLTrace * E_post + ZFilter * ValReg_post;
    RLTrace = GammaLambda * RLTrace - ZFilter;

    if (RetE == 0)
        addToPre(g);
    else
        addToPre(g * E_post);
    """}

_OUTPUT_VALUE_RANDOM = deepcopy(_OUTPUT_VALUE_SYMMETRIC)
_OUTPUT_VALUE_RANDOM["synapse_dynamics_code"] = """
    DeltaG += RLTrace * E_post + ZFilter * ValReg_post;
    RLTrace = GammaLambda * RLTrace - ZFilter;
"""

_OUTPUT_VALUE_ADAPTIVE_FEEDBACK = deepcopy(_OUTPUT_VALUE_SYMMETRIC)
del _OUTPUT_VALUE_ADAPTIVE_FEEDBACK["pre_spike_syn_code"]
_OUTPUT_VALUE_ADAPTIVE_FEEDBACK["post_neuron_var_refs"] += [("F_post", "scalar")]
_OUTPUT_VALUE_ADAPTIVE_FEEDBACK["synapse_dynamics_code"] = """
    DeltaG += RLTrace * E_post + ZFilter * ValReg_post;
    RLTrace = GammaLambda * RLTrace - ZFilter;

    if (RetE == 0)
        addToPre(g);
    else
        addToPre(g * F_post);
"""

_OUTPUT_POLICY_SYMMETRIC = {
    "params": [("Alpha", "scalar"), ("Lambda", "scalar")],
    "vars": [("g", "scalar", VarAccess.READ_ONLY), ("DeltaG", "scalar"), ("RLTrace", "scalar")],
    "pre_vars": [("ZFilter", "scalar")],
    "post_neuron_var_refs": [("E_post", "scalar"), ("PG_post", "scalar"), ("TdE_post", "scalar")],

    "pre_spike_code": "ZFilter += 1.0;",
    "pre_dynamics_code": "ZFilter *= Alpha;",

    "pre_spike_syn_code": "addToPost(g);",
    "synapse_dynamics_code": """
    DeltaG += TdE_post * RLTrace + ZFilter * E_post;
    RLTrace = Lambda * RLTrace + ZFilter * PG_post;

    addToPre(g * PG_post);
    """
}

_OUTPUT_POLICY_RANDOM = deepcopy(_OUTPUT_POLICY_SYMMETRIC)
_OUTPUT_POLICY_RANDOM["synapse_dynamics_code"] = """
    DeltaG += TdE_post * RLTrace + ZFilter * E_post;
    RLTrace = Lambda * RLTrace + ZFilter * PG_post;
"""

_OUTPUT_POLICY_ADAPTIVE_FEEDBACK = deepcopy(_OUTPUT_POLICY_SYMMETRIC)
del _OUTPUT_POLICY_ADAPTIVE_FEEDBACK["pre_spike_syn_code"]
_OUTPUT_POLICY_ADAPTIVE_FEEDBACK["post_neuron_var_refs"] += [("F_post", "scalar")]
_OUTPUT_POLICY_ADAPTIVE_FEEDBACK["synapse_dynamics_code"] = """
    DeltaG += TdE_post * RLTrace + ZFilter * E_post;
    RLTrace = Lambda * RLTrace + ZFilter * PG_post;

    addToPre(g * F_post);
"""

_OUTPUT_MODELS = {
    OutputRule.SUPERVISED_SYMMETRIC: _OUTPUT_SUPERVISED_SYMMETRIC,
    OutputRule.SUPERVISED_RANDOM: _OUTPUT_SUPERVISED_RANDOM,
    OutputRule.SUPERVISED_ADAPTIVE_FEEDBACK: _OUTPUT_SUPERVISED_ADAPTIVE_FEEDBACK,
    OutputRule.VALUE_SYMMETRIC: _OUTPUT_VALUE_SYMMETRIC,
    OutputRule.VALUE_RANDOM: _OUTPUT_VALUE_RANDOM,
    OutputRule.VALUE_ADAPTIVE_FEEDBACK: _OUTPUT_VALUE_ADAPTIVE_FEEDBACK,
    OutputRule.POLICY_SYMMETRIC: _OUTPUT_POLICY_SYMMETRIC,
    OutputRule.POLICY_RANDOM: _OUTPUT_POLICY_RANDOM,
    OutputRule.POLICY_ADAPTIVE_FEEDBACK: _OUTPUT_POLICY_ADAPTIVE_FEEDBACK,
}


def get_output_model(rule: OutputRule) -> dict:
    """Return a fresh copy of the output-layer model dict for `rule`."""
    return deepcopy(_OUTPUT_MODELS[rule])


# ============================================================================
# Pure feedback-connection models (conn.is_feedback == True, non-adaptive)
# ============================================================================

_FEEDBACK_RANDOM_ERROR = {
    "vars": [("g", "scalar", VarAccess.READ_ONLY)],
    "post_neuron_var_refs": [("E_post", "scalar")],
    "synapse_dynamics_code": "addToPre(g * E_post);"}

_FEEDBACK_RANDOM_PLAIN = {
    "vars": [("g", "scalar", VarAccess.READ_ONLY)],
    "post_neuron_var_refs": [("E_post", "scalar")],
    "synapse_dynamics_code": "addToPre(g);"}

_FEEDBACK_MODELS = {
    FeedbackRule.RANDOM_ERROR: _FEEDBACK_RANDOM_ERROR,
    FeedbackRule.RANDOM_PLAIN: _FEEDBACK_RANDOM_PLAIN,
    # POLICY_ADAPTIVE / VALUE_ADAPTIVE reuse the output adaptive-feedback
    # models directly, since they're the same GeNN model applied in reverse.
    FeedbackRule.POLICY_ADAPTIVE: _OUTPUT_POLICY_ADAPTIVE_FEEDBACK,
    FeedbackRule.VALUE_ADAPTIVE: _OUTPUT_VALUE_ADAPTIVE_FEEDBACK,
}


def get_feedback_model(rule: FeedbackRule) -> dict:
    """Return a fresh copy of the feedback-connection model dict for `rule`."""
    return deepcopy(_FEEDBACK_MODELS[rule])


# ============================================================================
# Custom-update models (unrelated to weight-update variant selection,
# used by every configuration)
# ============================================================================

GRADIENT_BATCH_REDUCE_MODEL = {
    "vars": [("ReducedGradient", "scalar", CustomUpdateVarAccess.REDUCE_BATCH_SUM)],
    "var_refs": [("Gradient", "scalar")],
    "update_code": """
    ReducedGradient = Gradient;
    Gradient = 0;
    """}
