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
        "base": {"hidden_rule": {"preset": "eprop"}, "feedback_type": "adaptive"},
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
