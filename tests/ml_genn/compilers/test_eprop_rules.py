"""Tests of the modular e-prop compiler (ml_genn.compilers.eprop).

* every hidden-rule preset compiles and trains a small RL network without errors or NaNs;
* the generated hidden model contains only the terms a configuration uses;
* the "original" preset reproduces the monolithic compiler (ml_genn.compilers.eprop_compiler)
  step by step on the same network, seed and input sequence.

Backend: GeNN's single-threaded CPU backend by default (set MLGENN_TEST_BACKEND to override).
"""
import os
from dataclasses import replace

import numpy as np
import pytest

from ml_genn import Connection, Network, Population
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import AdaptiveLeakyIntegrateFire, LeakyIntegrate, PoissonInput
from ml_genn.optimisers import AdaBelief
from ml_genn.utils.callback_list import CallbackList

from ml_genn.compilers.eprop import (EPropCompiler, PRESETS, PolicyType, HiddenRuleConfig,
                                     NoisePlacement, build_td_hidden_model, default_params)

BACKEND = os.environ.get("MLGENN_TEST_BACKEND", "single_threaded_cpu")
K = 10                       # simulation steps per action
N_IN, N_HID = 8, 12


def _network(seed=0):
    rng = np.random.default_rng(seed)
    w = lambda shape, mean, sd: (mean + sd * rng.standard_normal(shape)).astype(np.float32)
    net = Network(default_params)
    with net:
        inp = Population(PoissonInput(), N_IN)
        hid = Population(AdaptiveLeakyIntegrateFire(v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0,
                                                    tau_adapt=300.0), N_HID)
        pol = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 4)
        val = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 1)
        Connection(inp, hid, Dense(w((N_IN, N_HID), 0.15, 0.05)))
        Connection(hid, hid, Dense(w((N_HID, N_HID), 0.0, 0.05)))
        Connection(hid, pol, Dense(w((N_HID, 4), 0.0, 0.5)))
        Connection(hid, val, Dense(w((N_HID, 1), 0.0, 0.5)))
        Connection(hid, pol, Dense(w((N_HID, 4), 0.0, 0.5)), feedback_name="policy_feedback")
        Connection(hid, val, Dense(w((N_HID, 1), 0.0, 0.5)), feedback_name="value_feedback")
        Connection(pol, val, Dense(weight=1.0), feedback_name="tde_transport")
        Connection(hid, val, Dense(weight=1.0), feedback_name="tde_transport")
    return net, inp, hid, pol, val


def _compile(compiler_cls, name, policy_type, **kw):
    net, inp, hid, pol, val = _network()
    gamma, lam = 0.5 ** (1 / K), 0.8 ** (1 / K)
    compiler = compiler_cls(
        example_timesteps=1, losses={pol: "mean_square_error", val: "mean_square_error"},
        optimiser=AdaBelief(1e-3, beta1=0.99, beta2=0.99999), c_reg=1e-4, batch_size=1,
        feedback_type="random", reward_decay=0.1 ** (1 / K), gamma=gamma, td_lambda=lam,
        train_output_bias=False, reset_time_between_batches=False,
        entropy_coeff=0.0, entropy_coeff_decay=1.0, entropy_coeff_min=0.0,
        policy_heads={pol: policy_type}, value_head=val, rng_seed=1234, backend=BACKEND, **kw)
    return compiler.compile(net, name), inp, hid, pol, val


def _run(compiled, inp, pol, val, steps=200, inspect=None, probe=None):
    """Scripted episode: fixed input rates, a fixed policy-gradient pattern after every action and an
    alternating reward. Returns the hidden-layer weights after every K steps."""
    cb = CallbackList([*set(compiled.base_train_callbacks)], compiled_network=compiled,
                      num_batches=1, num_epochs=1)
    rates = np.linspace(0.2, 0.8, N_IN).astype(np.float32)
    pg = np.array([0.3, -0.7, 0.2, 0.2], np.float32)
    history = []
    with compiled:
        if probe is not None:
            probe(compiled, "start")
        cb.on_epoch_begin(0); cb.on_batch_begin(0)
        compiled.set_input({inp: rates})
        upd = 0
        for t in range(steps):
            if t % K == K - 1:
                p = compiled.neuron_populations[pol]
                p.vars["pre_PG"].view[:] = pg; p.push_var_to_device("pre_PG")
                compiled.losses[val].set_var(compiled.neuron_populations[val], "reward",
                                             1.0 if (t // K) % 2 == 0 else -1.0)
            compiled.step_time(cb)
            compiled.genn_model.custom_update("GradientLearn")
            for o, cus in compiled.optimisers:
                for c in cus:
                    upd += 1; o.set_step(c, upd)
            if t % K == K - 1:
                ws = []
                for conn, pop in compiled.connection_populations.items():
                    if conn.target().name == compiled_hidden_name(compiled) and not conn.is_feedback:
                        pop.vars["g"].pull_from_device(); ws.append(pop.vars["g"].values.copy())
                history.append(np.concatenate([w.ravel() for w in ws]))
            if inspect is not None:
                _accumulate_activity(compiled, inspect)
        if probe is not None:
            probe(compiled, "end")
    return np.array(history)


def _accumulate_activity(compiled, acc):
    """Sum of |value| over time of every hidden-synapse trace and the hidden neurons' PertEpsTrace."""
    hid = compiled_hidden_name(compiled)
    for conn, pop in compiled.connection_populations.items():
        if conn.target().name != hid or conn.is_feedback:
            continue
        for name in ("NoiseTrace", "FreshEps", "RLNoiseTrace", "RLNoisePtrace", "RLNoiseGtrace", "RLTrace"):
            if name in pop.vars:
                pop.vars[name].pull_from_device()
                acc[name] = acc.get(name, 0.0) + float(np.abs(pop.vars[name].values).sum())
    for p, npop in compiled.neuron_populations.items():
        if p.name == hid:
            npop.vars["PertEpsTrace"].pull_from_device()
            acc["PertEpsTrace"] = acc.get("PertEpsTrace", 0.0) + float(np.abs(npop.vars["PertEpsTrace"].view).sum())


def compiled_hidden_name(compiled):
    return [p.name for p in compiled.neuron_populations if isinstance(p.neuron, AdaptiveLeakyIntegrateFire)][0]


# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_presets_compile_and_train(preset, request):
    if preset.startswith("symmetric_hybrid"):
        pytest.skip("needs a core/field network: test_symmetric_hybrid_splits_the_network")
    compiled, inp, hid, pol, val = _compile(EPropCompiler, request.node.name.replace("[", "_").replace("]", ""),
                                            PolicyType.GENERIC, hidden_rule=preset,
                                            f_target=120.0 if preset == "baseline" else 10.0)
    acc = {}
    h = _run(compiled, inp, pol, val, steps=100, inspect=acc)
    assert np.all(np.isfinite(h))
    if preset == "reservoir":                     # no hidden learning by design
        assert np.all(h[-1] == h[0]), "reservoir hidden weights changed"
        return
    assert np.any(h[-1] != h[0]), "hidden weights did not change"
    rule = PRESETS[preset]
    # noise is injected exactly where the configuration says
    if rule.noise is NoisePlacement.NODE:
        assert acc["PertEpsTrace"] > 0 and "NoiseTrace" not in acc
    else:
        assert acc["PertEpsTrace"] == 0, "membrane noise must be off unless noise=NODE"
    if rule.noise in (NoisePlacement.WEIGHT_SHARED, NoisePlacement.WEIGHT_INDEPENDENT):
        assert acc.get("NoiseTrace", 0) > 0 or acc.get("FreshEps", -1) >= 0
    if rule.noise is NoisePlacement.NONE:
        assert "NoiseTrace" not in acc and "FreshEps" not in acc
    # every declared learning trace carries signal
    for name in ("RLNoiseTrace", "RLNoisePtrace", "RLNoiseGtrace", "RLTrace"):
        if name in acc:
            assert acc[name] > 0, f"{name} stayed zero"


def test_generated_models_are_compact():
    kw = dict(c_reg=1e-4, alpha=0.9, rho=0.99, f_target=0.01, alpha_fav=0.998, v_thresh=0.61,
              td_lambda=0.99)
    code = lambda name: build_td_hidden_model(PRESETS[name], **kw)["model"]
    syn = lambda name: code(name)["synapse_dynamics_code"]
    names = lambda name: {v[0] for v in code(name)["vars"]} | {v[0] for v in code(name)["post_vars"]}
    assert "PsiBar" not in names("original")                      # gate psi only
    assert "RLNoiseTrace" in names("proposed") and "RLNoisePtrace" not in names("proposed")
    assert {"RLNoisePtrace", "RLNoiseGtrace", "AbsTd"} <= names("proposed_homeostat")
    assert "NoiseTrace" not in names("unbiased") and "FreshEps" in names("unbiased")
    assert "NoiseTrace" not in names("eprop") and "RLTrace" in names("eprop")
    assert "0.0 *" not in syn("gradient_only") and "0.0 *" not in syn("proposed")
    assert "RLNoise" not in syn("baseline") and "CReg * (FAvg - FTarget) * eFiltered" in syn("baseline")
    with pytest.raises(ValueError):
        HiddenRuleConfig(noise=NoisePlacement.NONE, drift=1.0, gradient=1.0)


def test_original_preset_matches_monolithic_compiler(request):
    """The repository's monolithic compiler implements the original rule without the (1 - alpha^2)
    normalisation; the refactor reproduces it with kappa=False."""
    from ml_genn.compilers.eprop_compiler import EPropCompiler as MonolithicCompiler, PolicyTypes
    base = request.node.name
    mono, inp_m, _, pol_m, val_m = _compile(MonolithicCompiler, base + "_mono", PolicyTypes.GENERIC)
    h_mono = _run(mono, inp_m, pol_m, val_m)
    rule = replace(PRESETS["original"], kappa=False)
    new, inp_n, _, pol_n, val_n = _compile(EPropCompiler, base + "_new", PolicyType.GENERIC, hidden_rule=rule)
    h_new = _run(new, inp_n, pol_n, val_n)
    assert h_mono.shape == h_new.shape
    assert np.any(h_mono[-1] != h_mono[0])
    np.testing.assert_allclose(h_new, h_mono, rtol=1e-5, atol=1e-7)


def test_rule_dict_round_trip():
    from ml_genn.compilers.eprop import hidden_rule_from_dict, hidden_rule_to_dict, LocalRoute
    for name, rule in PRESETS.items():
        assert hidden_rule_from_dict(hidden_rule_to_dict(rule)) == rule, name
    r = hidden_rule_from_dict({"preset": "proposed_full", "homeostat": 2.0, "noise": "NODE",
                               "local": {"route": "GRADIENT_TRACE", "dv": 0.05}})
    assert r.homeostat == 2.0 and r.noise is NoisePlacement.NODE and r.drift == 1.0
    assert r.local.route is LocalRoute.GRADIENT_TRACE and r.local.dv == 0.05
    assert hidden_rule_from_dict({"preset": "proposed_full", "local": None}).local is None
    with pytest.raises(ValueError):
        hidden_rule_from_dict({"preset": "proposed", "homeostatt": 1.0})


def _feedback_weights(compiled, name):
    for conn, pop in compiled.connection_populations.items():
        if conn.is_feedback and conn.name.endswith(name):
            pop.vars["g"].pull_from_device()
            return pop.vars["g"].values.copy()
    raise KeyError(name)


def test_adaptive_feedback_learns_only_when_requested(request):
    for flag in (False, True):
        compiled, inp, hid, pol, val = _compile(EPropCompiler, f"{request.node.name}_{flag}", PolicyType.GENERIC,
                                                hidden_rule="eprop", optimise_feedback=flag)
        seen = {}
        _run(compiled, inp, pol, val, steps=60,
             probe=lambda c, when: seen.__setitem__(when, _feedback_weights(c, "policy_feedback")))
        assert np.any(seen["end"] != seen["start"]) == flag


def test_value_feedback_carries_bv_not_td_error(request):
    """e-prop's critic signal is -c_V B^V (Bellec et al. 2020): VE must equal the feedback weight."""
    compiled, inp, hid, pol, val = _compile(EPropCompiler, request.node.name, PolicyType.GENERIC, hidden_rule="eprop")
    seen = {}

    def probe(c, when):
        if when == "end":
            h = [p for p in c.neuron_populations if isinstance(p.neuron, AdaptiveLeakyIntegrateFire)][0]
            c.neuron_populations[h].vars["VE"].pull_from_device()
            seen["ve"] = c.neuron_populations[h].vars["VE"].view.ravel().copy()
            seen["bv"] = _feedback_weights(c, "value_feedback").ravel()
    _run(compiled, inp, pol, val, steps=30, probe=probe)
    np.testing.assert_allclose(seen["ve"], seen["bv"], rtol=1e-5)


def test_backprop_wiring_and_signal(request):
    """Hidden -> hidden synapses send the backward gradient signal to ISynBack; input -> hidden synapses send
    nothing; the hidden neurons' Back is non-zero with backprop and absent without it."""
    kw = dict(c_reg=1e-4, alpha=0.9, rho=0.99, f_target=0.01, alpha_fav=0.998, v_thresh=0.61, td_lambda=0.99)
    rule = PRESETS["proposed_backprop"]
    sender = build_td_hidden_model(rule, send_back=True, **kw)["model"]["synapse_dynamics_code"]
    receiver = build_td_hidden_model(rule, send_back=False, **kw)["model"]["synapse_dynamics_code"]
    assert "addToPre(g * (-PsiBar * normXi))" in sender and "addToPre" not in receiver
    assert "RLBackTrace" in sender and "RLBackTrace" in receiver           # both learn from the signal
    assert "addToPre" not in build_td_hidden_model(PRESETS["proposed"], send_back=True, **kw)["model"]["synapse_dynamics_code"]

    compiled, inp, hid, pol, val = _compile(EPropCompiler, request.node.name, PolicyType.GENERIC,
                                            hidden_rule="proposed_backprop")
    targets = {(c.source().name, c.target().name): p.pre_target_var
               for c, p in compiled.connection_populations.items() if not c.is_feedback}
    assert targets[(hid.name, hid.name)] == "ISynBack"
    assert targets[(inp.name, hid.name)] != "ISynBack"
    seen = {}

    def probe(c, when):
        if when == "end":
            c.neuron_populations[hid].vars["Back"].pull_from_device()
            seen["back"] = np.asarray(c.neuron_populations[hid].vars["Back"].view).copy()
    h = _run(compiled, inp, pol, val, steps=60, probe=probe)
    assert np.all(np.isfinite(h)) and np.any(seen["back"] != 0)

    plain, *_ = _compile(EPropCompiler, request.node.name + "_off", PolicyType.GENERIC, hidden_rule="proposed")
    hid_pop = [p for p in plain.neuron_populations if isinstance(p.neuron, AdaptiveLeakyIntegrateFire)][0]
    with plain:
        assert "Back" not in plain.neuron_populations[hid_pop].vars


def _hybrid_network(seed=0):
    """Snake-like: input -> recurrent core -> policy / value fields -> heads; no explicit feedback connections."""
    rng = np.random.default_rng(seed)
    w = lambda shape, mean, sd: (mean + sd * rng.standard_normal(shape)).astype(np.float32)
    alif = lambda: AdaptiveLeakyIntegrateFire(v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=300.0)
    net = Network(default_params)
    with net:
        inp = Population(PoissonInput(), N_IN)
        core = Population(alif(), N_HID)
        pf, vf = Population(alif(), 6), Population(alif(), 6)
        pol = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 4)
        val = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 1)
        Connection(inp, core, Dense(w((N_IN, N_HID), 0.15, 0.05)))
        Connection(core, core, Dense(w((N_HID, N_HID), 0.0, 0.05)))
        Connection(core, pf, Dense(w((N_HID, 6), 0.2, 0.05)))
        Connection(core, vf, Dense(w((N_HID, 6), 0.2, 0.05)))
        Connection(pf, pol, Dense(w((6, 4), 0.0, 0.5)))
        Connection(vf, val, Dense(w((6, 1), 0.0, 0.5)))
        Connection(pol, val, Dense(weight=1.0), feedback_name="tde_transport")
        for p in (core, pf, vf):
            Connection(p, val, Dense(weight=1.0), feedback_name="tde_transport")
    return net, inp, core, pf, vf, pol, val


def test_symmetric_hybrid_splits_the_network(request):
    from ml_genn.compilers.eprop import receives_eprop_signal, rule_for_population, EpropScope
    net, inp, core, pf, vf, pol, val = _hybrid_network()
    rule = PRESETS["symmetric_hybrid"]
    assert rule.eprop_scope is EpropScope.SIGNAL_RECIPIENTS
    assert not receives_eprop_signal(core, {pol: None}, val)
    assert receives_eprop_signal(pf, {pol: None}, val) and receives_eprop_signal(vf, {pol: None}, val)
    core_rule = rule_for_population(rule, core, {pol: None}, val)
    field_rule = rule_for_population(rule, pf, {pol: None}, val)
    assert core_rule.eprop == 0 and core_rule.drift == 1 and core_rule.homeostat == 1 and core_rule.center_drift
    assert field_rule.eprop == 1 and field_rule.drift == 0 and field_rule.gradient == 0
    assert field_rule.noise is NoisePlacement.NONE

    gamma, lam = 0.5 ** (1 / K), 0.8 ** (1 / K)
    compiled = EPropCompiler(
        example_timesteps=1, losses={pol: "mean_square_error", val: "mean_square_error"},
        optimiser=AdaBelief(1e-3, beta1=0.99, beta2=0.99999), c_reg=1e-4, batch_size=1,
        feedback_type="symmetric", reward_decay=0.1 ** (1 / K), gamma=gamma, td_lambda=lam,
        train_output_bias=False, reset_time_between_batches=False, entropy_coeff=0.0,
        entropy_coeff_decay=1.0, entropy_coeff_min=0.0, policy_heads={pol: PolicyType.GENERIC},
        value_head=val, rng_seed=1234, backend=BACKEND, hidden_rule="symmetric_hybrid").compile(net, request.node.name)
    pops = compiled.connection_populations
    by_pair = {(c.source(), c.target()): pops[c] for c in pops if not c.is_feedback}
    core_in = by_pair[(inp, core)]
    field_in = by_pair[(core, pf)]
    assert "NoiseTrace" in core_in.vars and "RLTrace" not in core_in.vars          # perturbation rule only
    assert "RLTrace" in field_in.vars and "NoiseTrace" not in field_in.vars        # e-prop only, no noise
    seen = {}

    def probe(c, when):
        snap = {}
        for key, pop in (("core", by_pair[(inp, core)]), ("pf", by_pair[(core, pf)]), ("vf", by_pair[(core, vf)])):
            pop.vars["g"].pull_from_device(); snap[key] = pop.vars["g"].values.copy()
        if when == "end":
            for name, p in (("PG", pf), ("VE", vf)):
                c.neuron_populations[p].vars[name].pull_from_device()
                snap[name] = np.asarray(c.neuron_populations[p].vars[name].view).copy()
        seen[when] = snap
    cb = CallbackList([*set(compiled.base_train_callbacks)], compiled_network=compiled, num_batches=1, num_epochs=1)
    rates = np.linspace(0.2, 0.8, N_IN).astype(np.float32)
    with compiled:
        probe(compiled, "start")
        cb.on_epoch_begin(0); cb.on_batch_begin(0)
        compiled.set_input({inp: rates})
        upd = 0
        for t in range(100):
            if t % K == K - 1:
                p = compiled.neuron_populations[pol]
                p.vars["pre_PG"].view[:] = np.array([0.3, -0.7, 0.2, 0.2], np.float32); p.push_var_to_device("pre_PG")
                compiled.losses[val].set_var(compiled.neuron_populations[val], "reward", 1.0 if (t // K) % 2 else -1.0)
            compiled.step_time(cb)
            compiled.genn_model.custom_update("GradientLearn")
            for o, cus in compiled.optimisers:
                for cu in cus:
                    upd += 1; o.set_step(cu, upd)
            if t == 98:
                probe(compiled, "end")
    for key in ("core", "pf", "vf"):
        assert np.all(np.isfinite(seen["end"][key])) and np.any(seen["end"][key] != seen["start"][key]), key
    assert np.any(seen["end"]["VE"] != 0)                     # the value field receives B^V through the readout


def test_symmetric_hybrid_homeostat_keeps_fields_alive_with_eprop_signal():
    from ml_genn.compilers.eprop import rule_for_population
    net, inp, core, pf, vf, pol, val = _hybrid_network()
    rule = PRESETS["symmetric_hybrid_homeostat"]
    field = rule_for_population(rule, pf, {pol: None}, val)
    assert field.eprop == 1 and field.drift == 0 and field.gradient == 0 and field.homeostat == 1
    assert field.noise is NoisePlacement.WEIGHT_SHARED                      # the homeostat needs the noise trace
    kw = dict(c_reg=1e-4, alpha=0.9, rho=0.99, f_target=0.01, alpha_fav=0.998, v_thresh=0.61, td_lambda=0.99)
    code = build_td_hidden_model(field, **kw)["model"]["synapse_dynamics_code"]
    assert "RLTrace" in code and "AbsTd * (PsiBar" in code and "RLNoiseGtrace" not in code
    core_rule = rule_for_population(rule, core, {pol: None}, val)
    assert core_rule.eprop == 0 and core_rule.drift == 1 and core_rule.homeostat == 1


def test_population_rules_override_the_global_rule(request):
    net, inp, core, pf, vf, pol, val = _hybrid_network()
    gamma, lam = 0.5 ** (1 / K), 0.8 ** (1 / K)
    compiler = EPropCompiler(
        example_timesteps=1, losses={pol: "mean_square_error", val: "mean_square_error"},
        optimiser=AdaBelief(1e-3, beta1=0.99, beta2=0.99999), c_reg=1e-4, batch_size=1,
        feedback_type="symmetric", reward_decay=0.1 ** (1 / K), gamma=gamma, td_lambda=lam,
        train_output_bias=False, reset_time_between_batches=False, entropy_coeff=0.0,
        entropy_coeff_decay=1.0, entropy_coeff_min=0.0, policy_heads={pol: PolicyType.GENERIC},
        value_head=val, rng_seed=1234, backend=BACKEND, hidden_rule="eprop",
        population_rules={core: {"preset": "proposed_homeostat", "homeostat": 1.0, "center_drift": True}})
    assert compiler.rule_for(core).homeostat == 1.0 and compiler.rule_for(pf).eprop == 1.0
    compiled = compiler.compile(net, request.node.name)
    pops = {(c.source(), c.target()): p for c, p in compiled.connection_populations.items() if not c.is_feedback}
    core_in, field_in = pops[(inp, core)], pops[(core, pf)]
    assert "RLNoisePtrace" in core_in.vars and "RLTrace" not in core_in.vars      # perturbation rule + homeostat
    assert "RLTrace" in field_in.vars and "NoiseTrace" not in field_in.vars       # e-prop (global rule)


@pytest.mark.parametrize("MODE", ["rate", "covariance", "psi", "psi_fast"])
def test_plastic_inhibition_updates_i_to_e_locally_and_stays_inhibitory(request, MODE):
    rng = np.random.default_rng(0)
    net = Network(default_params)
    with net:
        inp = Population(PoissonInput(), N_IN)
        alif = dict(v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=300.0)
        e = Population(AdaptiveLeakyIntegrateFire(**alif), N_HID)
        i = Population(AdaptiveLeakyIntegrateFire(**alif), 6)
        pol = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 4)
        val = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 1)
        Connection(inp, e, Dense((0.3 + 0.05 * rng.standard_normal((N_IN, N_HID))).astype(np.float32)))
        Connection(inp, i, Dense((0.3 + 0.05 * rng.standard_normal((N_IN, 6))).astype(np.float32)))
        i_to_e = Connection(i, e, Dense(np.full((6, N_HID), -0.05, np.float32)), exc_inh_sign=-1)
        for src in (e,):
            Connection(src, pol, Dense((0.5 * rng.standard_normal((N_HID, 4))).astype(np.float32)))
            Connection(src, val, Dense((0.5 * rng.standard_normal((N_HID, 1))).astype(np.float32)))
        Connection(pol, val, Dense(weight=1.0), feedback_name="tde_transport")
        for src in (e, i):
            Connection(src, val, Dense(weight=1.0), feedback_name="tde_transport")
    gamma, lam = 0.5 ** (1 / K), 0.8 ** (1 / K)
    compiler = EPropCompiler(
        example_timesteps=1, losses={pol: "mean_square_error", val: "mean_square_error"},
        optimiser=AdaBelief(1e-3, beta1=0.99, beta2=0.99999), c_reg=1e-4, batch_size=1,
        feedback_type="random", reward_decay=0.1 ** (1 / K), gamma=gamma, td_lambda=lam,
        train_output_bias=False, reset_time_between_batches=False, entropy_coeff=0.0,
        entropy_coeff_decay=1.0, entropy_coeff_min=0.0, policy_heads={pol: PolicyType.GENERIC},
        value_head=val, rng_seed=1234, backend=BACKEND, hidden_rule="proposed_homeostat",
        plastic_inhibition={i_to_e: {"eta": 1e-3, "rate": 1.0, "mode": MODE, "w_max": 0.2}})
    compiled = compiler.compile(net, request.node.name)
    pop = compiled.connection_populations[i_to_e]
    assert "DeltaG" not in pop.vars and "XTrace" in pop.pre_vars and "YTrace" in pop.post_vars
    optimised = {id(v) for _, cus in compiled.optimisers for v in cus}
    cb = CallbackList([*set(compiled.base_train_callbacks)], compiled_network=compiled, num_batches=1, num_epochs=1)
    with compiled:
        pop.vars["g"].pull_from_device(); g0 = pop.vars["g"].values.copy()
        cb.on_epoch_begin(0); cb.on_batch_begin(0)
        compiled.set_input({inp: np.full(N_IN, 0.8, np.float32)})
        for _ in range(300):
            compiled.step_time(cb)
        pop.vars["g"].pull_from_device(); g1 = pop.vars["g"].values.copy()
    assert np.all(g1 <= 0.0) and np.all(g1 >= -0.2 - 1e-6)
    if MODE == "rate":
        assert np.mean(g1) < np.mean(g0) - 1e-3      # E fires above the 1 Hz target: inhibition strengthens
    elif MODE in ("psi", "psi_fast"):
        assert "PsiBar" in pop.post_vars and np.abs(g1 - g0).max() > 1e-4
    else:
        assert np.abs(g1 - g0).max() > 1e-4 and "YMean" in pop.post_vars
    assert len(optimised) > 0


@pytest.mark.parametrize("E_INH", [None, -1.2])
def test_inhibitory_reversal_bounds_hyperpolarisation(request, E_INH):
    """Strong inhibition drives V far below rest with current-based synapses, never below E_inh with the reversal."""
    net = Network(default_params)
    with net:
        inp = Population(PoissonInput(), N_IN)
        alif = dict(v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=300.0)
        e = Population(AdaptiveLeakyIntegrateFire(**alif), N_HID)
        i = Population(AdaptiveLeakyIntegrateFire(**alif), 6)
        pol = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 4)
        val = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 1)
        Connection(inp, i, Dense(np.full((N_IN, 6), 0.5, np.float32)))
        Connection(i, e, FixedProbability(1.0, -2.0), exc_inh_sign=-1)       # sparse: sign-constrained (Dale)
        Connection(e, pol, Dense(np.zeros((N_HID, 4), np.float32)))
        Connection(e, val, Dense(np.zeros((N_HID, 1), np.float32)))
        Connection(pol, val, Dense(weight=1.0), feedback_name="tde_transport")
        for src in (e, i):
            Connection(src, val, Dense(weight=1.0), feedback_name="tde_transport")
    gamma, lam = 0.5 ** (1 / K), 0.8 ** (1 / K)
    compiler = EPropCompiler(
        example_timesteps=1, losses={pol: "mean_square_error", val: "mean_square_error"},
        optimiser=AdaBelief(0.0, beta1=0.99, beta2=0.99999), c_reg=0.0, batch_size=1,
        feedback_type="random", reward_decay=0.1 ** (1 / K), gamma=gamma, td_lambda=lam,
        train_output_bias=False, reset_time_between_batches=False, entropy_coeff=0.0,
        entropy_coeff_decay=1.0, entropy_coeff_min=0.0, policy_heads={pol: PolicyType.GENERIC},
        value_head=val, rng_seed=1234, backend=BACKEND, hidden_rule="reservoir", inhibitory_reversal=E_INH)
    compiled = compiler.compile(net, request.node.name)
    cb = CallbackList([*set(compiled.base_train_callbacks)], compiled_network=compiled, num_batches=1, num_epochs=1)
    vmin = 0.0
    with compiled:
        cb.on_epoch_begin(0); cb.on_batch_begin(0)
        compiled.set_input({inp: np.full(N_IN, 0.9, np.float32)})
        for _ in range(200):
            compiled.step_time(cb)
            npop = compiled.neuron_populations[e]
            npop.vars["V"].pull_from_device()
            vmin = min(vmin, float(np.min(npop.vars["V"].view)))
    if E_INH is None:
        assert vmin < -3.0                       # current-based: far below any reversal potential
    else:
        assert vmin >= E_INH - 1e-4 and vmin < -0.3


@pytest.mark.parametrize("MODE", ["psi", "psi_fast"])
def test_psi_mode_releases_inhibition_onto_hyperpolarised_neurons(request, MODE):
    """E held far below threshold by strong inhibition has psibar ~ 0 < psi_target, so the psi mode weakens the
    inhibition onto it."""
    net = Network(default_params)
    with net:
        inp = Population(PoissonInput(), N_IN)
        alif = dict(v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=300.0)
        e = Population(AdaptiveLeakyIntegrateFire(**alif), N_HID)
        i = Population(AdaptiveLeakyIntegrateFire(**alif), 6)
        pol = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 4)
        val = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 1)
        Connection(inp, i, Dense(np.full((N_IN, 6), 0.5, np.float32)))
        i_to_e = Connection(i, e, FixedProbability(1.0, -1.0), exc_inh_sign=-1)
        Connection(e, pol, Dense(np.zeros((N_HID, 4), np.float32)))
        Connection(e, val, Dense(np.zeros((N_HID, 1), np.float32)))
        Connection(pol, val, Dense(weight=1.0), feedback_name="tde_transport")
        for src in (e, i):
            Connection(src, val, Dense(weight=1.0), feedback_name="tde_transport")
    gamma, lam = 0.5 ** (1 / K), 0.8 ** (1 / K)
    compiler = EPropCompiler(
        example_timesteps=1, losses={pol: "mean_square_error", val: "mean_square_error"},
        optimiser=AdaBelief(0.0, beta1=0.99, beta2=0.99999), c_reg=0.0, batch_size=1,
        feedback_type="random", reward_decay=0.1 ** (1 / K), gamma=gamma, td_lambda=lam,
        train_output_bias=False, reset_time_between_batches=False, entropy_coeff=0.0,
        entropy_coeff_decay=1.0, entropy_coeff_min=0.0, policy_heads={pol: PolicyType.GENERIC},
        value_head=val, rng_seed=1234, backend=BACKEND, hidden_rule="reservoir",
        plastic_inhibition={i_to_e: {"eta": 1e-3, "mode": MODE}})
    compiled = compiler.compile(net, request.node.name)
    pop = compiled.connection_populations[i_to_e]
    cb = CallbackList([*set(compiled.base_train_callbacks)], compiled_network=compiled, num_batches=1, num_epochs=1)
    with compiled:
        pop.vars["g"].pull_from_device(); g0 = pop.vars["g"].values.copy()
        cb.on_epoch_begin(0); cb.on_batch_begin(0)
        compiled.set_input({inp: np.full(N_IN, 0.9, np.float32)})
        for _ in range(300):
            compiled.step_time(cb)
        pop.vars["g"].pull_from_device(); g1 = pop.vars["g"].values.copy()
    assert np.all(g1 <= 0.0) and np.mean(g1) > np.mean(g0) + 1e-3     # |w| shrinks: inhibition released


def _dale_net(seed=0):
    from ml_genn.connectivity import ToroidalGaussian2D
    np.random.seed(seed)
    net = Network(default_params)
    with net:
        inp = Population(PoissonInput(), N_IN)
        alif = dict(v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=300.0)
        a = Population(AdaptiveLeakyIntegrateFire(**alif), (16, 16, 1))
        b = Population(AdaptiveLeakyIntegrateFire(**alif), (16, 16, 1))
        pol = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 4)
        val = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 1)
        Connection(inp, a, Dense(weight=0.2))
        # excitatory by sign, but every weight starts negative: all of them violate Dale's law
        conn = Connection(a, b, ToroidalGaussian2D(sigma=0.06, fan_in=12, weight=Normal(mean=-0.05, sd=0.001),
                                                   exact=True), exc_inh_sign=+1)
        Connection(b, pol, Dense(weight=0.0))
        Connection(b, val, Dense(weight=0.0))
        Connection(pol, val, Dense(weight=1.0), feedback_name="tde_transport")
        for src in (a, b):
            Connection(src, val, Dense(weight=1.0), feedback_name="tde_transport")
    return net, conn, pol, val


def _toroidal_dist(pre, post, n=16):
    dy = np.abs(pre // n - post // n); dy = np.minimum(dy, n - dy)
    dx = np.abs(pre % n - post % n); dx = np.minimum(dx, n - dx)
    return np.sqrt(dx ** 2 + dy ** 2)


@pytest.mark.parametrize("LOCAL", [True, False])
def test_dale_rewiring_restores_signs_and_local_mode_keeps_new_synapses_local(request, LOCAL):
    net, conn, pol, val = _dale_net()
    original = _toroidal_dist(conn.connectivity.pre_ind, conn.connectivity.post_ind)
    gamma, lam = 0.5 ** (1 / K), 0.8 ** (1 / K)
    compiler = EPropCompiler(
        example_timesteps=1, losses={pol: "mean_square_error", val: "mean_square_error"},
        optimiser=AdaBelief(0.0, beta1=0.99, beta2=0.99999), c_reg=0.0, batch_size=1,
        feedback_type="random", reward_decay=0.1 ** (1 / K), gamma=gamma, td_lambda=lam,
        train_output_bias=False, reset_time_between_batches=False, entropy_coeff=0.0,
        entropy_coeff_decay=1.0, entropy_coeff_min=0.0, policy_heads={pol: PolicyType.GENERIC},
        value_head=val, rng_seed=1234, backend=BACKEND, hidden_rule="proposed", dale_local_rewiring=LOCAL)
    compiled = compiler.compile(net, request.node.name)
    sg = compiled.connection_populations[conn]
    with compiled:
        compiled.genn_model.custom_update("DaleInit")
        for _ in range(30):                       # resistance 0.9: ~10% of the violating synapses rewire per call
            compiled.genn_model.custom_update("DalePrune")
            compiled.genn_model.custom_update("DaleRewire")
        sg.pull_connectivity_from_device()
        sg.vars["g"].pull_from_device()
        pre, post = np.asarray(sg.get_sparse_pre_inds()), np.asarray(sg.get_sparse_post_inds())
        g = np.asarray(sg.vars["g"].values)
    assert len(pre) == len(original) and np.all(g >= 0.0)        # same synapse count, all excitatory again
    # most violating synapses are reset to a tiny weight of the right sign (resistance 0.9); look at the rewired ones
    old = set(zip(conn.connectivity.pre_ind.tolist(), conn.connectivity.post_ind.tolist()))
    new = np.array([(i, j) for i, j in zip(pre.tolist(), post.tolist()) if (i, j) not in old])
    assert len(new) > 20
    far = np.mean(_toroidal_dist(new[:, 0], new[:, 1]) > original.max() + 2)
    if LOCAL:
        assert far < 0.05                         # rewired synapses stay in the original neighbourhood
    else:
        assert far > 0.3                          # uniform rewiring scatters them across the layer


def test_dale_connections_restricts_rewiring_to_the_chosen_connections(request):
    """dale_connections: only the listed connections keep their sign (here the inhibitory one); the excitatory
    connection, every weight of which violates Dale's law, is left as it is."""
    np.random.seed(0)
    net = Network(default_params)
    with net:
        inp = Population(PoissonInput(), N_IN)
        alif = dict(v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=300.0)
        e = Population(AdaptiveLeakyIntegrateFire(**alif), N_HID)
        i = Population(AdaptiveLeakyIntegrateFire(**alif), 6)
        pol = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 4)
        val = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 1)
        Connection(inp, e, Dense(weight=0.3))
        Connection(inp, i, Dense(weight=0.3))
        # every weight violates its sign: excitatory weights negative, inhibitory weights positive
        e_to_e = Connection(e, e, FixedProbability(0.5, Normal(mean=-0.05, sd=0.001)), exc_inh_sign=+1)
        i_to_e = Connection(i, e, FixedProbability(0.8, Normal(mean=0.05, sd=0.001)), exc_inh_sign=-1)
        Connection(e, pol, Dense(weight=0.0))
        Connection(e, val, Dense(weight=0.0))
        Connection(pol, val, Dense(weight=1.0), feedback_name="tde_transport")
        for src in (e, i):
            Connection(src, val, Dense(weight=1.0), feedback_name="tde_transport")
    gamma, lam = 0.5 ** (1 / K), 0.8 ** (1 / K)
    compiler = EPropCompiler(
        example_timesteps=1, losses={pol: "mean_square_error", val: "mean_square_error"},
        optimiser=AdaBelief(0.0, beta1=0.99, beta2=0.99999), c_reg=0.0, batch_size=1,
        feedback_type="random", reward_decay=0.1 ** (1 / K), gamma=gamma, td_lambda=lam,
        train_output_bias=False, reset_time_between_batches=False, entropy_coeff=0.0,
        entropy_coeff_decay=1.0, entropy_coeff_min=0.0, policy_heads={pol: PolicyType.GENERIC},
        value_head=val, rng_seed=1234, backend=BACKEND, hidden_rule="proposed", dale_connections=[i_to_e])
    compiled = compiler.compile(net, request.node.name)
    with compiled:
        compiled.genn_model.custom_update("DaleInit")
        for _ in range(30):
            compiled.genn_model.custom_update("DalePrune")
            compiled.genn_model.custom_update("DaleRewire")
        g = {}
        for name, c in (("ee", e_to_e), ("ie", i_to_e)):
            sg = compiled.connection_populations[c]
            sg.vars["g"].pull_from_device()
            g[name] = np.asarray(sg.vars["g"].values)
    assert np.all(g["ie"] <= 0.0)                     # the chosen inhibitory connection is inhibitory again
    assert np.all(g["ee"] < 0.0)                      # the excitatory one was not touched: still all negative


@pytest.mark.parametrize("SCALING", [0.0, 0.02])
def test_local_scaling_holds_received_inhibition_near_its_warmup_level(request, SCALING):
    """Covariance plasticity moves the total inhibition (here it shrinks it); local synaptic scaling (synapse weight x
    the neuron's own InhAvg / InhTarget) pulls the received inhibition back towards the level captured after the
    warm-up, in either direction."""
    net = Network(default_params)
    with net:
        inp = Population(PoissonInput(), N_IN)
        alif = dict(v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=300.0)
        e = Population(AdaptiveLeakyIntegrateFire(**alif), N_HID)
        i = Population(AdaptiveLeakyIntegrateFire(**alif), 6)
        pol = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 4)
        val = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 1)
        Connection(inp, e, Dense(weight=0.3))
        Connection(inp, i, Dense(weight=0.3))
        i_to_e = Connection(i, e, FixedProbability(1.0, -0.05), exc_inh_sign=-1)
        Connection(e, pol, Dense(weight=0.0))
        Connection(e, val, Dense(weight=0.0))
        Connection(pol, val, Dense(weight=1.0), feedback_name="tde_transport")
        for src in (e, i):
            Connection(src, val, Dense(weight=1.0), feedback_name="tde_transport")
    gamma, lam = 0.5 ** (1 / K), 0.8 ** (1 / K)
    cfg = {"eta": 5e-3, "mode": "covariance", "tau_mean": 50.0, "scaling": SCALING, "scaling_tau": 20.0,
           "scaling_warmup": 60.0}
    compiler = EPropCompiler(
        example_timesteps=1, losses={pol: "mean_square_error", val: "mean_square_error"},
        optimiser=AdaBelief(0.0, beta1=0.99, beta2=0.99999), c_reg=0.0, batch_size=1,
        feedback_type="random", reward_decay=0.1 ** (1 / K), gamma=gamma, td_lambda=lam,
        train_output_bias=False, reset_time_between_batches=False, entropy_coeff=0.0,
        entropy_coeff_decay=1.0, entropy_coeff_min=0.0, policy_heads={pol: PolicyType.GENERIC},
        value_head=val, rng_seed=1234, backend=BACKEND, hidden_rule="reservoir",
        plastic_inhibition={i_to_e: cfg})
    compiled = compiler.compile(net, request.node.name)
    sg, npop = compiled.connection_populations[i_to_e], compiled.neuron_populations[e]
    if SCALING > 0:
        assert "InhAvg" in npop.vars and sg.post_target_var == "ISynPI"
    cb = CallbackList([*set(compiled.base_train_callbacks)], compiled_network=compiled, num_batches=1, num_epochs=1)
    with compiled:
        cb.on_epoch_begin(0); cb.on_batch_begin(0)
        compiled.set_input({inp: np.full(N_IN, 0.9, np.float32)})
        for _ in range(60):
            compiled.step_time(cb)
        sg.vars["g"].pull_from_device(); g_warm = float(np.abs(sg.vars["g"].values).mean())
        for _ in range(600):
            compiled.step_time(cb)
        sg.vars["g"].pull_from_device(); g_end = float(np.abs(sg.vars["g"].values).mean())
        if SCALING > 0:
            npop.vars["InhTarget"].pull_from_device()
            assert np.all(npop.vars["InhTarget"].view > 0)            # captured after the warm-up
    growth = g_end / g_warm
    _SCALING_GROWTH[SCALING] = growth
    print(f"scaling {SCALING}: mean |g| warm-up -> end x{growth:.2f}")
    if len(_SCALING_GROWTH) == 2:                # with scaling the total ends much closer to its warm-up level
        assert abs(np.log(_SCALING_GROWTH[0.02])) < 0.5 * abs(np.log(_SCALING_GROWTH[0.0]))


_SCALING_GROWTH = {}


def test_per_neuron_homeostat_target_is_captured_after_the_warmup(request):
    rule = {"preset": "proposed_homeostat", "homeostat": 1.0, "psi_target_warmup": 30.0}
    compiled, inp, hid, pol, val = _compile(EPropCompiler, request.node.name, PolicyType.GENERIC, hidden_rule=rule)
    pops = [p for c, p in compiled.connection_populations.items()
            if not c.is_feedback and c.target() is hid and "PsiTargetN" in p.post_vars]
    assert pops
    cb = CallbackList([*set(compiled.base_train_callbacks)], compiled_network=compiled, num_batches=1, num_epochs=1)
    with compiled:
        cb.on_epoch_begin(0); cb.on_batch_begin(0)
        compiled.set_input({inp: np.linspace(0.2, 0.8, N_IN).astype(np.float32)})
        for t in range(60):
            compiled.step_time(cb)
            if t == 20:
                pops[0].post_vars["PsiTargetN"].pull_from_device()
                assert np.all(pops[0].post_vars["PsiTargetN"].view == 0)      # not captured before the warm-up
        targets = []
        for p in pops:
            p.post_vars["PsiTargetN"].pull_from_device(); targets.append(p.post_vars["PsiTargetN"].view.copy())
    assert np.all(targets[0] > 0)                                              # captured, one value per neuron
    for t in targets[1:]:
        assert np.allclose(t, targets[0])                                      # the same in every synapse group


@pytest.mark.parametrize("DECORR,NORM", [(0.0, "covariance"), (0.5, "covariance"), (0.5, "correlation")])
def test_decorrelation_reward_routes_inhibition_and_changes_learning(request, DECORR, NORM):
    """The decorrelation local reward: inhibitory synapses go to the ISynInh channel, the neuron tracks InhNow /
    InhMean, and the reward (same seed, same everything else) changes how the excitatory weights learn."""
    rule = {"preset": "proposed", "local": {"route": "GRADIENT_INSTANT", "dv": 0.0, "vreg": 0.0, "rate": 0.0,
                                            "decorr": DECORR, "decorr_norm": NORM}}
    net = Network(default_params)
    with net:
        inp = Population(PoissonInput(), N_IN)
        alif = dict(v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=300.0)
        e = Population(AdaptiveLeakyIntegrateFire(**alif), N_HID)
        i = Population(AdaptiveLeakyIntegrateFire(**alif), 6)
        pol = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 4)
        val = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 1)
        # moderate drive (a saturated, always-refractory neuron has psi = 0 and cannot learn); a matrix: learnable
        inp_e = Connection(inp, e, Dense(np.full((N_IN, N_HID), 0.06, np.float32)))
        Connection(inp, i, Dense(weight=0.08))
        i_to_e = Connection(i, e, FixedProbability(1.0, -0.05), exc_inh_sign=-1)
        Connection(e, pol, Dense(weight=0.0))
        Connection(e, val, Dense(weight=0.0))
        Connection(pol, val, Dense(weight=1.0), feedback_name="tde_transport")
        for src in (e, i):
            Connection(src, val, Dense(weight=1.0), feedback_name="tde_transport")
    gamma, lam = 0.5 ** (1 / K), 0.8 ** (1 / K)
    compiler = EPropCompiler(
        example_timesteps=1, losses={pol: "mean_square_error", val: "mean_square_error"},
        optimiser=AdaBelief(1e-3, beta1=0.99, beta2=0.99999), c_reg=0.0, batch_size=1,
        feedback_type="random", reward_decay=0.1 ** (1 / K), gamma=gamma, td_lambda=lam,
        train_output_bias=False, reset_time_between_batches=False, entropy_coeff=0.0,
        entropy_coeff_decay=1.0, entropy_coeff_min=0.0, policy_heads={pol: PolicyType.GENERIC},
        value_head=val, rng_seed=1234, backend=BACKEND, hidden_rule=rule)
    compiled = compiler.compile(net, request.node.name)
    sg, npop = compiled.connection_populations[i_to_e], compiled.neuron_populations[e]
    if DECORR != 0:
        assert sg.post_target_var == "ISynInh" and "InhMean" in npop.vars
    cb = CallbackList([*set(compiled.base_train_callbacks)], compiled_network=compiled, num_batches=1, num_epochs=1)
    upd = 0
    with compiled:
        cb.on_epoch_begin(0); cb.on_batch_begin(0)
        compiled.set_input({inp: np.full(N_IN, 0.9, np.float32)})
        for _ in range(300):
            compiled.step_time(cb)
            compiled.genn_model.custom_update("GradientLearn")
            for o, cus in compiled.optimisers:
                for c in cus:
                    upd += 1; o.set_step(c, upd)
        if DECORR != 0:
            npop.vars["InhMean"].pull_from_device()
            assert np.all(npop.vars["InhMean"].view > 0)               # the neuron senses its inhibition
        w = compiled.connection_populations[inp_e]
        w.vars["g"].pull_from_device()
        _DECORR_W[(DECORR, NORM)] = np.asarray(w.vars["g"].values).copy()
    base = _DECORR_W.get((0.0, "covariance"))
    if DECORR != 0 and base is not None:                               # the reward changes what is learned
        assert np.abs(_DECORR_W[(DECORR, NORM)] - base).max() > 1e-6
        assert np.all(np.isfinite(_DECORR_W[(DECORR, NORM)]))


_DECORR_W = {}


@pytest.mark.parametrize("CANCEL", [0.0, 0.5])
def test_cancellation_reward_tracks_excitatory_drive_and_changes_learning(request, CANCEL):
    """The cancellation local reward -|dV| / (|dI_exc| + eps mean): inhibition goes to its own channel, the neuron
    tracks the change of its excitatory drive (DExc / DExcMean), and the reward changes how the weights learn."""
    rule = {"preset": "proposed", "local": {"route": "GRADIENT_INSTANT", "dv": 0.0, "vreg": 0.0, "rate": 0.0,
                                            "cancel": CANCEL}}
    net = Network(default_params)
    with net:
        inp = Population(PoissonInput(), N_IN)
        alif = dict(v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=300.0)
        e = Population(AdaptiveLeakyIntegrateFire(**alif), N_HID)
        i = Population(AdaptiveLeakyIntegrateFire(**alif), 6)
        pol = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 4)
        val = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 1)
        inp_e = Connection(inp, e, Dense(np.full((N_IN, N_HID), 0.06, np.float32)))
        Connection(inp, i, Dense(weight=0.08))
        i_to_e = Connection(i, e, FixedProbability(1.0, -0.05), exc_inh_sign=-1)
        Connection(e, pol, Dense(weight=0.0))
        Connection(e, val, Dense(weight=0.0))
        Connection(pol, val, Dense(weight=1.0), feedback_name="tde_transport")
        for src in (e, i):
            Connection(src, val, Dense(weight=1.0), feedback_name="tde_transport")
    gamma, lam = 0.5 ** (1 / K), 0.8 ** (1 / K)
    compiler = EPropCompiler(
        example_timesteps=1, losses={pol: "mean_square_error", val: "mean_square_error"},
        optimiser=AdaBelief(1e-3, beta1=0.99, beta2=0.99999), c_reg=0.0, batch_size=1,
        feedback_type="random", reward_decay=0.1 ** (1 / K), gamma=gamma, td_lambda=lam,
        train_output_bias=False, reset_time_between_batches=False, entropy_coeff=0.0,
        entropy_coeff_decay=1.0, entropy_coeff_min=0.0, policy_heads={pol: PolicyType.GENERIC},
        value_head=val, rng_seed=1234, backend=BACKEND, hidden_rule=rule)
    compiled = compiler.compile(net, request.node.name)
    sg, npop = compiled.connection_populations[i_to_e], compiled.neuron_populations[e]
    if CANCEL != 0:
        assert sg.post_target_var == "ISynInh" and "DExcMean" in npop.vars
    cb = CallbackList([*set(compiled.base_train_callbacks)], compiled_network=compiled, num_batches=1, num_epochs=1)
    upd = 0
    rng = np.random.default_rng(0)
    with compiled:
        cb.on_epoch_begin(0); cb.on_batch_begin(0)
        for step in range(300):
            if step % 30 == 0:                            # the input changes: something to cancel
                compiled.set_input({inp: rng.uniform(0.2, 0.9, N_IN).astype(np.float32)})
            compiled.step_time(cb)
            compiled.genn_model.custom_update("GradientLearn")
            for o, cus in compiled.optimisers:
                for c in cus:
                    upd += 1; o.set_step(c, upd)
        if CANCEL != 0:
            npop.vars["DExcMean"].pull_from_device()
            assert np.all(npop.vars["DExcMean"].view > 0)              # the neuron senses its drive changing
        w = compiled.connection_populations[inp_e]
        w.vars["g"].pull_from_device()
        _CANCEL_W[CANCEL] = np.asarray(w.vars["g"].values).copy()
    if CANCEL != 0 and 0.0 in _CANCEL_W:                               # the reward changes what is learned
        assert np.abs(_CANCEL_W[CANCEL] - _CANCEL_W[0.0]).max() > 1e-6
        assert np.all(np.isfinite(_CANCEL_W[CANCEL]))


_CANCEL_W = {}


def test_homeostat_floor_is_one_sided():
    from ml_genn.compilers.eprop import hidden_rule_from_dict
    kw = dict(c_reg=1e-4, alpha=0.9, rho=0.99, f_target=0.01, alpha_fav=0.998, v_thresh=0.61, td_lambda=0.99)
    syn = lambda cfg: build_td_hidden_model(cfg, **kw)["model"]["synapse_dynamics_code"]
    two = syn(HiddenRuleConfig(homeostat=1.0, psi_target=0.02))
    one = syn(HiddenRuleConfig(homeostat=1.0, psi_target=0.02, homeostat_floor=True))
    assert "fmin((PsiBar - 0.02), 0.0)" in one and "fmin(" not in two
    assert hidden_rule_from_dict({"preset": "proposed", "homeostat": 1.0, "homeostat_floor": True}).homeostat_floor


def test_surprise_drift_modulator():
    kw = dict(c_reg=1e-4, alpha=0.9, rho=0.99, f_target=0.01, alpha_fav=0.998, v_thresh=0.61, td_lambda=0.99)
    model = lambda cfg: build_td_hidden_model(cfg, **kw)["model"]
    syn = lambda cfg: model(cfg)["synapse_dynamics_code"]
    base = syn(HiddenRuleConfig())
    mixed = model(HiddenRuleConfig(surprise_drift=0.5))
    only = syn(HiddenRuleConfig(drift_td=0.0, surprise_drift=1.0))
    assert "AbsTd" not in base
    assert "(reward + 0.5 * (fabs(reward) - AbsTd)) * RLNoisePtrace" in mixed["synapse_dynamics_code"]
    assert "AbsTd = " in mixed["post_dynamics_code"]                        # running mean |delta| is tracked
    assert "(1.0 * (fabs(reward) - AbsTd)) * RLNoisePtrace" in only        # outcome sign removed from the drift
    assert "reward * RLNoiseGtrace" in only                                 # the gradient part keeps delta


def test_value_afterglow_weight_learns_the_remaining_reward(request):
    """The learned afterglow weight starts at 0, grows while rewards leave a decaying trace the head does not
    predict, and stays within [0, max]; without the option the head has no such variable."""
    compiled, inp, hid, pol, val = _compile(EPropCompiler, request.node.name + "_ag", PolicyType.GENERIC,
                                            hidden_rule="reservoir", value_afterglow={"lr": 0.05, "max": 50.0})
    seen = {}
    def probe(c, when):
        v = c.neuron_populations[val].vars["AgW"]; v.pull_from_device()
        seen[when] = float(np.asarray(v.view).ravel()[0])
    _run(compiled, inp, pol, val, steps=40 * K, probe=probe)
    assert seen["start"] == 0.0 and 0.0 < seen["end"] <= 50.0
    plain, _, _, _, pval = _compile(EPropCompiler, request.node.name + "_plain", PolicyType.GENERIC,
                                    hidden_rule="reservoir")
    assert "AgW" not in plain.neuron_populations[pval].vars


def test_value_transform_is_invertible_and_matches_the_generated_code():
    import math
    from ml_genn.compilers.eprop.neuron_logic import (value_transform_h, value_transform_h_inv, _h_code,
                                                      _h_inv_code, VALUE_TRANSFORM_EPS as eps)
    x = np.array([-500.0, -20.0, -1.0, -0.01, 0.0, 0.3, 2.0, 50.0, 1e4])
    assert np.allclose(value_transform_h_inv(value_transform_h(x)), x, rtol=1e-6, atol=1e-8)
    assert abs(value_transform_h(100.0)) < 10.0                       # large values are compressed
    env = {"copysign": math.copysign, "fabs": abs, "sqrt": math.sqrt}
    for v in x:
        assert eval(_h_code(repr(float(v)), eps), env) == pytest.approx(value_transform_h(v), rel=1e-9, abs=1e-12)
        assert eval(_h_inv_code(repr(float(v)), eps), env) == pytest.approx(value_transform_h_inv(v), rel=1e-9,
                                                                             abs=1e-12)


def test_value_transform_runs_in_the_value_head(request):
    compiled, inp, hid, pol, val = _compile(EPropCompiler, request.node.name, PolicyType.GENERIC,
                                            hidden_rule="proposed", value_transform="sqrt")
    seen = {}
    def probe(c, when):
        e = c.neuron_populations[val].vars["E"]; e.pull_from_device()
        seen[when] = float(np.asarray(e.view).ravel()[0])
    hist = _run(compiled, inp, pol, val, steps=20 * K, probe=probe)
    assert np.isfinite(seen["end"]) and np.all(np.isfinite(hist))
    with pytest.raises(ValueError):
        _compile(EPropCompiler, request.node.name + "_bad", PolicyType.GENERIC, hidden_rule="proposed",
                 value_transform="sqrt", value_afterglow={"lr": 0.1})


def test_value_transform_head_only_broadcasts_the_original_units(request):
    """sqrt_head: the hidden neurons receive r + gamma h^-1(V_h) - h^-1(V_h previous), not the transformed error."""
    compiled, inp, hid, pol, val = _compile(EPropCompiler, request.node.name, PolicyType.GENERIC,
                                            hidden_rule="proposed", value_transform="sqrt_head")
    cb = CallbackList([*set(compiled.base_train_callbacks)], compiled_network=compiled, num_batches=1, num_epochs=1)
    bc, err, tde = [], [], []
    with compiled:
        cb.on_epoch_begin(0); cb.on_batch_begin(0)
        compiled.set_input({inp: np.linspace(0.2, 0.8, N_IN).astype(np.float32)})
        vpop, hpop = compiled.neuron_populations[val], compiled.neuron_populations[hid]
        for t in range(20 * K):
            if t % K == K - 1:
                compiled.losses[val].set_var(vpop, "reward", 3.0 if (t // K) % 2 == 0 else -3.0)
            compiled.step_time(cb)
            compiled.genn_model.custom_update("GradientLearn")
            for name, out in (("TdBroadcast", bc), ("E", err)):
                v = vpop.vars[name]; v.pull_from_device(); out.append(float(np.asarray(v.view).ravel()[0]))
            hpop.vars["TdE"].pull_from_device(); tde.append(float(np.asarray(hpop.vars["TdE"].view).ravel()[0]))
    bc, err, tde = map(np.array, (bc, err, tde))
    lag = lambda a, k: (a[k:], tde[:len(tde) - k]) if k == 0 else (a[:-k], tde[k:])
    best = min(np.max(np.abs(x - y)) for x, y in (lag(bc, 0), lag(bc, 1)))
    worst_e = min(np.max(np.abs(x - y)) for x, y in (lag(err, 0), lag(err, 1)))
    assert best < 1e-4 * max(1.0, np.abs(bc).max())                 # the hidden TD error is the broadcast
    assert worst_e > 1e-3                                           # ... and not the transformed error


def test_populations_with_different_time_constants_get_their_own_eligibility_decays(request):
    """Every connection's eligibility decays with its TARGET population's constants (alpha from tau_mem, rho from
    tau_adapt); before, the compiler refused networks with more than one tau."""
    rng = np.random.default_rng(0)
    w = lambda shape, mean, sd: (mean + sd * rng.standard_normal(shape)).astype(np.float32)
    alif = lambda tm, ta: AdaptiveLeakyIntegrateFire(v_thresh=0.61, tau_mem=tm, tau_refrac=3.0, tau_adapt=ta)
    net = Network(default_params)
    with net:
        inp = Population(PoissonInput(), N_IN)
        fast, slow = Population(alif(10.0, 300.0), N_HID), Population(alif(40.0, 1000.0), N_HID)
        pol = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 4)
        val = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 1)
        c_in = Connection(inp, fast, Dense(w((N_IN, N_HID), 0.15, 0.05)))
        c_fs = Connection(fast, slow, Dense(w((N_HID, N_HID), 0.1, 0.05)))
        c_ss = Connection(slow, slow, Dense(w((N_HID, N_HID), 0.0, 0.05)))
        c_out = Connection(slow, pol, Dense(w((N_HID, 4), 0.0, 0.5)))
        Connection(slow, val, Dense(w((N_HID, 1), 0.0, 0.5)))
        Connection(slow, pol, Dense(w((N_HID, 4), 0.0, 0.5)), feedback_name="policy_feedback")
        Connection(slow, val, Dense(w((N_HID, 1), 0.0, 0.5)), feedback_name="value_feedback")
        Connection(pol, val, Dense(weight=1.0), feedback_name="tde_transport")
        for h in (fast, slow):
            Connection(h, val, Dense(weight=1.0), feedback_name="tde_transport")
    compiler = EPropCompiler(
        example_timesteps=1, losses={pol: "mean_square_error", val: "mean_square_error"},
        optimiser=AdaBelief(1e-3, beta1=0.99, beta2=0.99999), c_reg=1e-4, batch_size=1,
        feedback_type="random", reward_decay=0.1 ** (1 / K), gamma=0.5 ** (1 / K), td_lambda=0.8 ** (1 / K),
        train_output_bias=False, reset_time_between_batches=False,
        entropy_coeff=0.0, entropy_coeff_decay=1.0, entropy_coeff_min=0.0,
        policy_heads={pol: PolicyType.GENERIC}, value_head=val, rng_seed=1234, backend=BACKEND, hidden_rule="eprop")
    seen = {}                                   # the parameters the compiler gives every connection's update model
    build = compiler.build_weight_update_model
    def spy(conn, *args, **kw):
        wum = build(conn, *args, **kw)
        seen[conn] = dict(wum.param_vals)
        return wum
    compiler.build_weight_update_model = spy
    compiled = compiler.compile(net, request.node.name)
    params = lambda c: seen[c]
    a = lambda tau: np.exp(-1.0 / tau)
    assert params(c_in)["Alpha"] == pytest.approx(a(10.0)) and params(c_in)["Rho"] == pytest.approx(a(300.0))
    assert params(c_fs)["Alpha"] == pytest.approx(a(40.0)) and params(c_fs)["Rho"] == pytest.approx(a(1000.0))
    assert params(c_ss)["Alpha"] == pytest.approx(a(40.0))
    assert params(c_out)["Alpha"] == pytest.approx(a(10.0))            # the readout's own membrane decay
    # the filtered eligibility (e-prop's F_kappa) uses the READOUT's decay, whatever the target's tau_mem
    assert params(c_fs)["Kappa"] == pytest.approx(a(10.0)) and params(c_in)["Kappa"] == pytest.approx(a(10.0))
    _run(compiled, inp, pol, val, steps=4 * K)                           # and it runs
