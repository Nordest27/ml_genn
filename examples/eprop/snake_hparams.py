"""Hyperparameters of the Snake script, from one JSON file.

    SNAKE_CONFIG=path/to/config.json python run_headless.py

Keys missing from the file keep the hand-tuned defaults below, so a config only needs what differs.
Without SNAKE_CONFIG everything is the default and the script behaves as before (monolithic compiler,
original rule). The sweep tool (hpo/sweep.py) writes complete configs that can be rerun this way.

"hidden_rule" selects the modular compiler (ml_genn.compilers.eprop) and is either a preset name or a
dict, e.g. {"preset": "proposed_homeostat", "homeostat": 2.0, "log_syn_sigma": 0.5}
(see ml_genn.compilers.eprop.hidden_rule_from_dict).
"""
import json
import os
import random

import numpy as np

DEFAULTS = {
    # learning rule
    "hidden_rule": None,         # None: monolithic compiler (the original rule); else preset or dict
    "lr": 1e-5,                  # AdaBelief learning rate (all trained weights)
    "c_reg": 1e-4,               # firing-rate regulariser
    "f_target": 10.0,            # target rate (Hz) of the regulariser
    "node_sigma": 1e-2,          # membrane-noise s.d. (used by noise=NODE rules only)
    "feedback_type": "random",   # "random" | "symmetric" | "adaptive"
    "optimise_feedback": False,  # learn the feedback weights (adaptive e-prop; modular compiler only)
    "optimiser": "adabelief",    # "adabelief" (beta 0.99/0.99999) | "adam" (beta 0.9/0.999, adaptive e-prop)
    "entropy_coeff": 1.0,
    "entropy_decay_env": 0.99999,    # per environment step
    # time constants, per environment step (raised to 1/WAIT_INC in the script)
    "gamma_env": 0.5,
    "td_lambda_env": 0.8,
    "reward_decay_env": 0.1,
    # task and architecture
    "board_size": 5,             # larger boards are harder (the view stays visible_range = 5)
    "hid_e": 20,
    "hid_i": 15,
    "fan_in": 300,
    # snake_switch: performance-triggered task switches and the dead-neuron monitor (snake_switch.py)
    "switch": None,              # e.g. {"criterion": 0.08, "window": 20000, "mangles": ["channels", "actions"], "seed": 0}
    "monitor": False,            # log dead / silent / firing-dead hidden neurons per population
    "monitor_every": 10000,      # environment moves per monitor report
    # run
    "seed": None,                # None: unseeded (as before)
    "max_timesteps": 30e6,       # stop after this many simulation timesteps ...
    "min_episodes": 20000,       # ... and at least this many episodes (both as before)
    "csv_prefix": "x2",
    "repetition": 5,
    "trace_log": True,           # per-timestep value trace (large; sweeps turn it off)
}


def load(defaults=None, env_var="SNAKE_CONFIG"):
    """Defaults updated with the json file named by `env_var` (other scripts pass their own defaults)."""
    defaults = DEFAULTS if defaults is None else defaults
    hp = dict(defaults)
    path = os.environ.get(env_var)
    if path:
        with open(path) as f:
            user = json.load(f)
        unknown = set(user) - set(defaults) - {"_meta"}
        if unknown:
            raise KeyError(f"{path}: unknown hyperparameters {sorted(unknown)}")
        hp.update(user)
    if hp["seed"] is not None:
        random.seed(hp["seed"])
        np.random.seed(hp["seed"])
    return hp


class NullLogger:
    """Stand-in for AsyncTraceLogger when trace_log is off."""
    def log(self, *args):
        pass

    def close(self):
        pass
