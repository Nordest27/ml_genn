"""Search studies for Snake (5x5, the paper's architecture). Edit freely; `python sweep.py plan` shows the cost.

A study:
  base     config keys (snake_hparams.DEFAULTS) fixed for the study; give "hidden_rule" as a dict
  inherit  start from another study's best.json (its rule fields are kept, the study's preset wins)
  space    {key: ("log", lo, hi) | ("lin", lo, hi) | ("choice", [values])}; "hidden_rule.<field>" sets
           a HiddenRuleConfig field. Config 0 is always the base point (the hand-tuned / inherited one).
  n, rungs, seeds, keep, window   successive-halving schedule (DEFAULT_SCHEDULE unless overridden)
  gates    optional early kills: {"max_hz": 300, "min_hz": 1, "hz_after": 0.2, "timeout_s": 6 * 3600}

Principles: architecture and task constants stay at their hand-tuned values; each main-comparison rule gets
the same budget over the learning rate and its own noise scale; ablations inherit the proposed rule's tuned
values and only re-tune the learning rate, so they change one thing at a time.
"""

DEFAULT_SCHEDULE = {
    "n": 16,                       # configurations at the first rung (base point + 15 Sobol points)
    "rungs": [1e6, 3e6, 9e6],      # simulation timesteps per run at each rung (30 per environment step)
    "seeds": [1, 2, 2],            # seeds per configuration at each rung (common across configurations)
    "keep": 1 / 3,                 # fraction promoted to the next rung
    "window": 1 / 3,               # score = reward per env step over the last third of the run
    "gates": {},
    "sobol_seed": 0,
}

LR = ("log", 2e-6, 5e-5)                     # hand-tuned: 1e-5
WEIGHT_NOISE = ("lin", -1.5, 1.5)            # LogSynSig: sigma = exp(x - 5), i.e. x0.22 .. x4.5 of the default
NODE_NOISE = ("log", 2e-3, 5e-2)             # membrane noise s.d. (default 1e-2)
ONE_D = {"n": 6, "rungs": [1e6, 3e6], "seeds": [1, 2], "keep": 1 / 3}   # learning-rate-only studies

STUDIES = {
    # ---- main comparison -------------------------------------------------------------------------
    "proposed": {
        "base": {"hidden_rule": {"preset": "proposed"}},
        "space": {"lr": LR, "hidden_rule.log_syn_sigma": WEIGHT_NOISE},
    },
    "node_proposed": {
        "base": {"hidden_rule": {"preset": "node_proposed"}},
        "space": {"lr": LR, "node_sigma": NODE_NOISE},
    },
    "ind_noise_proposed": {
        "base": {"hidden_rule": {"preset": "ind_noise_proposed"}},
        "space": {"lr": LR, "hidden_rule.log_syn_sigma": WEIGHT_NOISE},
    },
    "adaptive_eprop": {
        # learned feedback weights; AdaBelief like the other rules (set "optimiser": "adam", "lr": 7e-6 for the
        # earlier hand-tuned Adam setting)
        "base": {"hidden_rule": {"preset": "eprop"}, "feedback_type": "adaptive", "optimise_feedback": True},
        "space": {"lr": LR, "c_reg": ("log", 1e-5, 1e-3)},
    },
    "baseline_120hz": {
        "base": {"hidden_rule": {"preset": "baseline"}, "f_target": 120.0},
        "space": {"lr": LR}, **ONE_D,
    },
    # ---- additions to the proposed rule (start from its tuned values) ----------------------------
    "proposed_homeostat": {
        "inherit": "proposed",
        "base": {"hidden_rule": {"preset": "proposed_homeostat"}},
        "space": {"hidden_rule.homeostat": ("log", 0.5, 20.0), "hidden_rule.psi_target": ("lin", 0.03, 0.25)},
        "n": 12,
    },
    "proposed_full": {
        "inherit": "proposed_homeostat",
        "base": {"hidden_rule": {"preset": "proposed_full"}},
        "space": {"hidden_rule.local.dv": ("log", 1e-3, 1e-1)}, **ONE_D,
    },
    # ---- dissection: no drift -> drift (inherit proposed / proposed_homeostat, re-tune lr only) ---
    "gradient_only": {"inherit": "proposed", "base": {"hidden_rule": {"preset": "gradient_only"}},
                      "space": {"lr": LR}, **ONE_D},
    "gradient_only_homeostat": {"inherit": "proposed_homeostat",
                                "base": {"hidden_rule": {"preset": "gradient_only_homeostat"}},
                                "space": {"lr": LR}, **ONE_D},
    "drift_only": {"inherit": "proposed", "base": {"hidden_rule": {"preset": "drift_only"}},
                   "space": {"lr": LR}, **ONE_D},
    "unbiased": {"inherit": "proposed", "base": {"hidden_rule": {"preset": "unbiased"}},
                 "space": {"lr": LR}, **ONE_D},
}


# ---- combined rules: e-prop's feedback learning signal plus perturbation terms ----------------------------
# Inherit the noise level (and homeostat) from the perturbation study and the learning rate, c_reg, optimiser and
# feedback learning from adaptive e-prop; search the learning rate and the weight of the e-prop term relative to
# the perturbation terms (hidden_rule.eprop; the perturbation coefficients stay 1).
MIX = ("log", 0.1, 10.0)
for _combo, _parents in (("eprop_plus_proposed", ["proposed", "adaptive_eprop"]),
                         ("eprop_plus_gradient", ["proposed", "adaptive_eprop"]),
                         ("eprop_plus_drift", ["proposed", "adaptive_eprop"]),
                         ("eprop_plus_proposed_homeostat", ["proposed_homeostat", "adaptive_eprop"]),
                         ("eprop_plus_homeostat", ["proposed_homeostat", "adaptive_eprop"])):
    STUDIES[_combo] = {"inherit": _parents, "base": {"hidden_rule": {"preset": _combo}},
                       "space": {"lr": LR, "hidden_rule.eprop": MIX}, "n": 12}

# ---- harder Snake: longer horizon (gamma) and a larger board ------------------------------------------------
# Each tuned rule run on a grid, every value with 2 seeds for 9M timesteps (screen; rerun winners at full length).
# gamma is per environment step (0.5 = a horizon of about 2 moves). The PPO baseline (ppo_snake_baseline) uses 0.99:
# run it at the same gamma values for the ceiling of each objective.
HARDER = {"grid": True, "rungs": [9e6], "seeds": [2], "keep": 1.0}
for _name in ("proposed", "adaptive_eprop", "eprop_plus_proposed"):
    STUDIES[f"gamma_{_name}"] = {"inherit": _name, "space": {"gamma_env": ("choice", [0.5, 0.8, 0.9, 0.95])},
                                 **HARDER}
    STUDIES[f"board7_{_name}"] = {"inherit": _name, "base": {"board_size": 7},
                                  "space": {"gamma_env": ("choice", [0.5, 0.9])}, **HARDER}

# ---- snake_switch: rules compared on performance-triggered task switches (score = switches reached) --------
# Each study inherits a tuned rule and runs it 5 times at full length (no search). Set the criterion from your
# reward-rate curves: reachable by the good rules in a few million timesteps, not trivially.
SWITCH = {"criterion": 0.08, "window": 20000, "mangles": ["channels", "actions"], "seed": 0}
SWITCH_RUN = {"n": 1, "rungs": [30e6], "seeds": [5], "keep": 1.0}
for _name in ("proposed", "proposed_homeostat", "gradient_only", "gradient_only_homeostat", "adaptive_eprop",
              "eprop_plus_proposed", "eprop_plus_gradient", "eprop_plus_drift", "eprop_plus_proposed_homeostat",
              "eprop_plus_homeostat"):
    STUDIES[f"switch_{_name}"] = {"inherit": _name, "base": {"switch": SWITCH, "monitor": True},
                                  "space": {}, **SWITCH_RUN}
