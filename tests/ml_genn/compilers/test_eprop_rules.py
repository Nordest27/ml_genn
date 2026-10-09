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
