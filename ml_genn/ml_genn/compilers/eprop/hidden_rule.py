"""Generate the hidden-layer weight-update model of the RL / TD(lambda) e-prop variant from a
:class:`.variants.HiddenRuleConfig`.

The generated model contains only what the configuration uses. Update order (same as the original
monolithic implementation, ``eprop_compiler.eprop_alif_td_model``):

1. ``DeltaG`` accumulates each term from the traces *before* they are updated this step;
2. e-prop's ALIF eligibility ``e = Psi * (ZFilter - Beta * epsilonA)`` with the old ``epsilonA``;
3. ``eFiltered`` (used by the firing-rate gradient and the e-prop learning signal);
4. the trace updates; 5. the noise trace decays by ``Alpha``.

GeNN runs the weight update's ``post_dynamics_code`` after the neuron's integration and before its
threshold/reset, and the synapse dynamics after the reset (see the paper's Appendix D.6).
"""

from pygenn import VarAccess

from .variants import (HiddenRuleConfig, NoisePlacement, Estimator, LocalRoute, DVMode)


def _f(x: float) -> str:
    """Float literal for generated code (12 significant digits, always with a decimal point)."""
    t = f"{float(x):.12g}"
    return t if any(c in t for c in ".eEn") else t + ".0"


def _scaled(coef: float, expr: str) -> str:
    """'coef * (expr)' with unit coefficients simplified."""
    if coef == 1.0:
        return f"({expr})"
    if coef == -1.0:
        return f"-({expr})"
    return f"{_f(coef)} * ({expr})"


# Independent per-synapse noise: a hash of the pre/post noise samples, the synapse indices and time
# (from the original implementation's commented-out block; '>>' is written as division because
# GeNN's transpiler does not support shifts).
_HASH_NOISE = """
uint32_t bitsPre  = (uint32_t)((Noise1_pre + 20.0) * 1e6);
uint32_t bitsPost = (uint32_t)((Noise1 + 20.0) * 1e6);
uint32_t tBits;
{ scalar tOff = t + 20.0; tBits = (uint32_t)(tOff * 1e6); }
uint32_t synId = (uint32_t)id_pre * 73856093u ^ (uint32_t)id_post * 19349663u;
uint32_t s0 = bitsPre ^ bitsPost ^ synId ^ tBits;
uint32_t x = s0;
x = x ^ (x / 65536u); x = x * 0x7feb352du;
x = x ^ (x / 32768u); x = x * 0x846ca68bu;
x = x ^ (x / 65536u);
uint32_t h1 = x;
x = s0 + 0x9E3779B9u;
x = x ^ (x / 65536u); x = x * 0x7feb352du;
x = x ^ (x / 32768u); x = x * 0x846ca68bu;
x = x ^ (x / 65536u);
uint32_t h2 = x;
const scalar u1 = (h1 + 0.5) / 4294967296.0;
const scalar u2 = (h2 + 0.5) / 4294967296.0;
const scalar synNoise = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265359 * u2);
"""


def build_td_hidden_model(cfg: HiddenRuleConfig, *, c_reg: float, alpha: float, rho: float,
                          f_target: float, alpha_fav: float, v_thresh: float, td_lambda: float):
    """Return the keyword arguments of a ``WeightUpdateModel`` (model dict and initial values)."""
    weight_noise = cfg.noise in (NoisePlacement.WEIGHT_SHARED, NoisePlacement.WEIGHT_INDEPENDENT)
    node_noise = cfg.noise is NoisePlacement.NODE
    exact = cfg.estimator is Estimator.EXACT
    local = cfg.local
    route = local.route if local is not None else None

    # --- which traces ---------------------------------------------------------------------------
    # A single combined trace suffices when drift and gradient share the TD-error modulator and no
    # term needs one of them separately.
    needs_split = (cfg.center_drift or cfg.homeostat != 0 or exact
                   or route is LocalRoute.GRADIENT_TRACE)
    use_P = needs_split and (cfg.drift != 0 or cfg.homeostat != 0)
    use_G = needs_split and (cfg.gradient != 0 or route is LocalRoute.GRADIENT_TRACE)
    use_combined = (not needs_split) and (cfg.drift != 0 or cfg.gradient != 0)
    uses_psibar = (cfg.homeostat != 0 or use_P or (use_G and not exact)
                   or (use_combined and cfg.drift + cfg.gradient != 0)
                   or (route is LocalRoute.GRADIENT_INSTANT and not exact))
    use_eprop = cfg.eprop != 0
    use_local_post = route in (LocalRoute.GRADIENT_INSTANT, LocalRoute.GRADIENT_TRACE)
    use_dvc = local is not None and local.dv != 0 and local.dv_mode is DVMode.CORRECT
    use_prevv = local is not None and local.dv != 0
    vreg_w = (c_reg if local is None or local.vreg is None else local.vreg)
    rate_w = (c_reg if local is None or local.rate is None else local.rate)
    use_rate_local = local is not None and rate_w != 0
    use_favg = cfg.fire_rate_gradient or use_rate_local
    use_efiltered = cfg.fire_rate_gradient or use_eprop

    # --- normalised noise trace and the two scores -------------------------------------------------
    kappa = "(1.0 - Alpha * Alpha) * " if cfg.kappa else ""
    if exact:
        norm_xi = None                     # the exact estimator uses only the fresh noise
    elif weight_noise:
        norm_xi = f"{kappa}NoiseTrace / (SynSig * SynSig + 1e-6)"
    elif node_noise:
        norm_xi = f"{kappa}PertEpsTrace_post / (Sigma_post * Sigma_post + 1e-6)"
    else:
        norm_xi = None
    pre = "(ZFilter - Beta_post * epsA)"
    if exact:
        g_score = "-freshEps / (SynSig * SynSig + 1e-6)"            # REINFORCE sign, no gate, no filter
    elif norm_xi is not None:
        g_score = f"-PsiBar * {pre} * normXi"
    else:
        g_score = None
    p_score = f"(Psi - PsiBar) * {pre} * normXi" if norm_xi is not None else None

    # --- model pieces ----------------------------------------------------------------------------
    params = [("CReg", "scalar"), ("Alpha", "scalar"), ("Rho", "scalar"), ("FTarget", "scalar"),
              ("AlphaFAv", "scalar"), ("Vthresh_post", "scalar"), ("Lambda", "scalar")]
    param_vals = {"CReg": c_reg, "Alpha": alpha, "Rho": rho, "FTarget": f_target,
                  "AlphaFAv": alpha_fav, "Vthresh_post": v_thresh, "Lambda": td_lambda}
    vars_ = [("g", "scalar", VarAccess.READ_ONLY), ("DeltaG", "scalar"), ("epsilonA", "scalar")]
    var_vals = {"DeltaG": 0.0, "epsilonA": 0.0}
    if use_efiltered:
        vars_.append(("eFiltered", "scalar")); var_vals["eFiltered"] = 0.0
    if weight_noise:
        vars_ += [("LogSynSig", "scalar"), ("SynSig", "scalar")]
        var_vals.update({"LogSynSig": cfg.log_syn_sigma, "SynSig": 0.0})
        if not exact:
            vars_.append(("NoiseTrace", "scalar")); var_vals["NoiseTrace"] = 0.0
    if exact:
        vars_.append(("FreshEps", "scalar")); var_vals["FreshEps"] = 0.0
    for name, used in (("RLNoiseTrace", use_combined), ("RLNoisePtrace", use_P),
                       ("RLNoiseGtrace", use_G), ("RLTrace", use_eprop)):
        if used:
            vars_.append((name, "scalar")); var_vals[name] = 0.0

    post_vars = [("Psi", "scalar")]
    post_var_vals = {"Psi": 0.0}
    for name, used in (("FAvg", use_favg), ("FAvgTrace", use_rate_local), ("Z", use_dvc),
                       ("PrevV", use_prevv), ("DVc", use_dvc), ("PsiBar", uses_psibar),
                       ("TdMean", cfg.center_drift), ("AbsTd", cfg.homeostat != 0),
                       ("LocR", use_local_post), ("LocMean", use_local_post)):
        if used:
            post_vars.append((name, "scalar")); post_var_vals[name] = 0.0

    post_refs = {"RefracTime_post": "RefracTime", "V_post": "V", "A_post": "A",
                 "Beta_post": "Beta", "TdE_post": "TdE", "PRew_post": "PRew"}
    if weight_noise:
        post_refs["Noise1"] = "Noise1"
    if node_noise:
        post_refs.update({"PertEpsTrace_post": "PertEpsTrace", "Sigma_post": "Sigma"})
    if use_eprop:
        post_refs.update({"PG_post": "PG", "VE_post": "VE"})
    pre_refs = {"Noise1_pre": "Noise1"} if cfg.noise is NoisePlacement.WEIGHT_INDEPENDENT else {}

    # post spike / dynamics code
    post_spike = []
    if use_favg:
        post_spike.append("FAvg += (1.0 - AlphaFAv);")
    if use_dvc:
        post_spike.append("Z = 1.0;")
    post_dyn = []
    if use_dvc:   # change from the previous post-reset voltage (relative reset by Vthresh)
        post_dyn.append("DVc = fabs(V_post - (PrevV - Vthresh_post * Z));")
        post_dyn.append("Z = 0.0;")
    if use_favg:
        post_dyn.append("FAvg *= AlphaFAv;")
    if use_rate_local:
        post_dyn.append("FAvgTrace = Lambda * FAvgTrace + (1.0 - Lambda) * fabs(FAvg - FTarget);")
    post_dyn.append("""if (RefracTime_post > 0.0) {
    Psi = 0.0;
}
else {
    Psi = (1.0 / Vthresh_post) * 0.3 * fmax(0.0, 1.0 - fabs((V_post - (Vthresh_post + (Beta_post * A_post))) / Vthresh_post));
}""")
    if uses_psibar:
        post_dyn.append(f"PsiBar = {_f(cfg.psibar_decay)} * PsiBar + {_f(1 - cfg.psibar_decay)} * Psi;")
    if cfg.center_drift:
        post_dyn.append(f"TdMean = {_f(cfg.td_stats_decay)} * TdMean + {_f(1 - cfg.td_stats_decay)} * (TdE_post + PRew_post);")
    if cfg.homeostat != 0:
        post_dyn.append(f"AbsTd = {_f(cfg.td_stats_decay)} * AbsTd + {_f(1 - cfg.td_stats_decay)} * fabs(TdE_post + PRew_post);")
    if use_local_post:
        terms = []
        if local.dv != 0:
            terms.append(f"- {_f(local.dv)} * " + ("DVc" if local.dv_mode is DVMode.CORRECT
                                                    else "fabs(PrevV - V_post)"))
        if vreg_w != 0:
            post_dyn.append("const scalar thrL = Vthresh_post + Beta_post * A_post;")
            terms.append(f"- {_f(vreg_w)} * (fmax(0.0, V_post - thrL) + fmax(0.0, -V_post - thrL))")
        if rate_w != 0:
            terms.append(f"+ {_f(rate_w)} * (FAvgTrace - fabs(FAvg - FTarget))")
        post_dyn.append("LocR = 0.0 " + " ".join(terms) + ";")
        post_dyn.append(f"LocMean = {_f(local.baseline_decay)} * LocMean + {_f(1 - local.baseline_decay)} * LocR;")
    if use_prevv:
        post_dyn.append("PrevV = V_post;")

    # presynaptic spike: noise delivery
    accumulate = "FreshEps += eps;" if exact else "NoiseTrace += eps;"
    if cfg.noise is NoisePlacement.WEIGHT_SHARED:
        pre_spike_syn = ("SynSig = exp(LogSynSig - 5.0);\nconst scalar eps = SynSig * Noise1;\n"
                         f"addToPost(g + eps);\n{accumulate}")
    elif cfg.noise is NoisePlacement.WEIGHT_INDEPENDENT:
        pre_spike_syn = (_HASH_NOISE + "SynSig = exp(LogSynSig - 5.0);\nconst scalar eps = SynSig * synNoise;\n"
                         f"addToPost(g + eps);\n{accumulate}")
    else:
        pre_spike_syn = "addToPost(g);"

    # synapse dynamics
    syn = ["const scalar reward = TdE_post + PRew_post;", "const scalar epsA = epsilonA;"]
    if weight_noise:
        syn.append("SynSig = exp(LogSynSig - 5.0);")
    if norm_xi is not None:
        syn.append(f"const scalar normXi = {norm_xi};")
    if exact:
        syn.append("const scalar freshEps = FreshEps;\nFreshEps = 0.0;")
    dg = []
    if use_combined:
        dg.append("reward * RLNoiseTrace")
    if use_P and cfg.drift != 0:
        mod = "(reward - TdMean)" if cfg.center_drift else "reward"
        dg.append(_scaled(cfg.drift, f"{mod} * RLNoisePtrace"))
    if use_G and cfg.gradient != 0:
        dg.append(_scaled(cfg.gradient, "reward * RLNoiseGtrace"))
    if cfg.homeostat != 0:
        dg.append(f"{_f(cfg.homeostat)} * AbsTd * (PsiBar - {_f(cfg.psi_target)}) * RLNoisePtrace")
    if use_eprop:
        dg.append(_scaled(cfg.eprop, "reward * RLTrace"))
    if route is LocalRoute.WHOLE_TRACE:
        # original implementation: local rewards computed here (after the reset) on the whole trace
        whole = ("RLNoiseTrace" if use_combined else
                 " + ".join(x for x in (("RLNoisePtrace" if use_P else ""), ("RLNoiseGtrace" if use_G else "")) if x))
        loc = []
        if local.dv != 0:
            loc.append(f"- {_f(local.dv)} * " + ("DVc" if local.dv_mode is DVMode.CORRECT else "fabs(PrevV - V_post)"))
        if vreg_w != 0:
            syn.append("const scalar thrL = Vthresh_post + Beta_post * A_post;")
            loc.append(f"- {_f(vreg_w)} * (fmax(0.0, V_post - thrL) + fmax(0.0, -V_post - thrL))")
        if rate_w != 0:
            loc.append(f"+ {_f(rate_w)} * (FAvgTrace - fabs(FAvg - FTarget))")
        dg.append(f"(0.0 {' '.join(loc)}) * ({whole})")
    elif route is LocalRoute.GRADIENT_INSTANT:
        dg.append(f"(LocR - LocMean) * ({g_score})")
    elif route is LocalRoute.GRADIENT_TRACE:
        dg.append("(LocR - LocMean) * RLNoiseGtrace")
    if dg:
        syn.append("DeltaG += " + "\n    + ".join(dg) + ";")
    syn.append("const scalar e = Psi * ZFilter - Psi * Beta_post * epsA;")
    syn.append("epsilonA = Psi * ZFilter + (Rho * epsA) - Psi * Beta_post * epsA;")
    if use_efiltered:
        syn.append("eFiltered = (eFiltered * Alpha) + e;")
    if cfg.fire_rate_gradient:
        syn.append("DeltaG += CReg * (FAvg - FTarget) * eFiltered;")
    if use_combined:
        # gate = drift * (Psi - PsiBar) - gradient * PsiBar = drift * Psi - (drift + gradient) * PsiBar
        terms = []
        if cfg.drift != 0:
            terms.append(_scaled(cfg.drift, "Psi"))
        if cfg.drift + cfg.gradient != 0:
            terms.append(f"- {_f(cfg.drift + cfg.gradient)} * PsiBar")
        gate = " ".join(terms)
        syn.append(f"RLNoiseTrace = Lambda * RLNoiseTrace + ({gate}) * {pre} * normXi;")
    if use_P:
        syn.append(f"RLNoisePtrace = Lambda * RLNoisePtrace + {p_score};")
    if use_G:
        syn.append(f"RLNoiseGtrace = Lambda * RLNoiseGtrace + {g_score};")
    if use_eprop:
        syn.append(f"RLTrace = Lambda * RLTrace + eFiltered * (PG_post - {_f(cfg.eprop_value)} * VE_post);")
    if weight_noise and not exact:
        syn.append("NoiseTrace *= Alpha;")

    model = {
        "params": params,
        "vars": vars_,
        "pre_vars": [("ZFilter", "scalar")],
        "post_vars": post_vars,
        "post_neuron_var_refs": [(k, "scalar") for k in post_refs],
        "pre_spike_code": "ZFilter += 1.0;",
        "pre_dynamics_code": "ZFilter *= Alpha;",
        "post_spike_code": "\n".join(post_spike),
        "post_dynamics_code": "\n".join(post_dyn),
        "pre_spike_syn_code": pre_spike_syn,
        "synapse_dynamics_code": "\n".join(syn),
    }
    if pre_refs:
        model["pre_neuron_var_refs"] = [(k, "scalar") for k in pre_refs]
    return dict(model=model, param_vals=param_vals, var_vals=var_vals,
                pre_var_vals={"ZFilter": 0.0}, post_var_vals=post_var_vals,
                pre_neuron_var_refs=pre_refs, post_neuron_var_refs=post_refs)
