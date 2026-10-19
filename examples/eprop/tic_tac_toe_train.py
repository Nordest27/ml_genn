##################### TIC TAC TOE — SNN AGENT #####################
#####################################################################
# Drop-in replacement for the connect_four / snake training scripts.
# Uses the same EProp / mlGeNN stack; imports PerformanceVisualizer
# from the generic visualizer module.
#
# UPDATED for the new TicTacToeEnv turn model:
#   - Both players get an equal wait_inc "thinking" phase before each of
#     their moves (previously only the agent waited; the opponent moved
#     for free inside the same step() call as soon as wait_count hit 0).
#   - Either player can move first each episode (env's first_player=
#     "random" by default).
#   - env.wait_count == 0 no longer implies "it's the agent's turn" — use
#     env.awaiting_agent_action instead, which is only True when the next
#     step() call will consume a real `action` to place the agent's piece.
#   - Every timestep (whichever player is thinking/moving) re-encodes and
#     feeds spikes for the current observation, since the opponent's move
#     now also changes what's on the board mid-episode outside of the
#     single combined step() call the old code relied on.
#####################################################################

import numpy as np
import cv2
import random
import time
import multiprocessing as mp
import csv
import snake_hparams

import snake_network

# Hyperparameters: TTT_CONFIG=config.json overrides these. Every key of snake_hparams.DEFAULTS is accepted, so the
# network and rule settings of a Snake config ("toroidal", "local", "network", "ei_layers", "hid_e", "hidden_rule",
# "dale", "plastic_inhibition", "backend", ...) build the same network here (snake_network.py). The defaults below
# keep tic-tac-toe's previous network and training settings where they differ from Snake's.
TTT_DEFAULTS = {
    **snake_hparams.DEFAULTS,
    "hidden_rule": None,         # None: monolithic compiler; else a preset or dict (ml_genn.compilers.eprop)
    "feedback_type": "random",
    "optimise_feedback": False,  # learn the feedback weights (adaptive e-prop; modular compiler only)
    "optimiser": "adam",         # "adam" (beta 0.9/0.999) | "adabelief" (beta 0.99/0.99999)
    "lr": 1e-5,
    "c_reg": 1e-2,
    "f_target": 10.0,
    "seed": None,
    "toroidal": "legacy",        # the original sampler (tic-tac-toe's previous network)
    "explicit_feedback": False,  # tic-tac-toe had no explicit feedback connections to the heads
    "csv_output": "outputs_ttt/weight_dist(1).csv",
}
HP = snake_hparams.load(TTT_DEFAULTS, "TTT_CONFIG")
# Snake keys that tic-tac-toe does not use (its timing, discounting and entropy settings are the constants below)
SNAKE_ONLY_KEYS = ("entropy_coeff", "entropy_decay_env", "entropy_bonus", "entropy_bonus_decay_env", "entropy_scale",
                   "entropy_td_decay_env", "entropy_reward", "gamma_env", "td_lambda_env", "reward_decay_env",
                   "board_size", "memory", "switch", "monitor", "monitor_every", "probe", "max_timesteps",
                   "min_episodes", "csv_prefix", "repetition", "trace_log")
_ignored = [k for k in SNAKE_ONLY_KEYS if HP[k] != snake_hparams.DEFAULTS[k]]
if _ignored:
    print(f"tic-tac-toe ignores these Snake settings of the config: {_ignored}")
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from pygenn import SynapseMatrixConnectivity
from ml_genn import Population, Connection, Network
from ml_genn.callbacks import Checkpoint
from ml_genn.compilers import EPropCompiler, PolicyTypes
from ml_genn.connectivity import Dense, FixedProbability, ToroidalGaussian2D
from ml_genn.initializers import Normal
from ml_genn.neurons import (LeakyIntegrate, AdaptiveLeakyIntegrateFire,
                              SpikeInput, PoissonInput)
from ml_genn.serialisers import Numpy
from ml_genn.optimisers import Adam, AsyncLocalAdam, CAdam, AdaBelief
from ml_genn.utils.data import preprocess_spikes
from ml_genn.utils.callback_list import CallbackList
from ml_genn.compilers.eprop_compiler import default_params

from tic_tac_toe_env import TicTacToeEnv
from performance_visualizer import PerformanceVisualizer


def compute_returns(rewards, gamma):
    """
    Backward discounted return G_t = r_t + gamma * G_{t+1}, computed
    per-timestep over a full episode of rewards.

    rewards: sequence of raw per-timestep rewards actually seen by the
             network, in time order (NOT the forward-decayed
             `reward_trace` — that's a different, causal-only quantity).
    gamma:   per-timestep discount factor.

    Returns a list the same length as `rewards`, where result[t] is the
    true return-to-go from timestep t onward.
    """
    G = list(rewards)
    for i in range(len(G) - 2, -1, -1):
        G[i] += gamma * G[i + 1]
    return G

# ─── Board / timing constants ────────────────────────────────────
BOARD_ROWS  = 3
BOARD_COLS  = 3
WAIT_INC    = 30          # timesteps per move ("thinking time") — SAME for both players
PIXEL_SCALE = 120

NUM_ACTIONS = BOARD_ROWS * BOARD_COLS  # one action per cell (flattened)
OBS_SCALE   = 5                        # 4x4 px per cell -> 12x12 observation
FIRST_PLAYER = "random"                # "agent" | "opponent" | "random" | 1 | -1

# ─── Network topology ────────────────────────────────────────────
INPUT_SHAPE = (BOARD_ROWS * OBS_SCALE, BOARD_COLS * OBS_SCALE, 3)  # (12, 12, 3)
INPUT_SIZE     = int(np.prod(INPUT_SHAPE))

# The hidden network (layers, sizes, connectivity, fields) is built by snake_network.py from HP, as in snake.py:
# "hid_e" / "hid_i" (default 20 / 15), "ei_layers", "toroidal" (default here: "legacy", the original network),
# "local" and "network". See snake_hparams.py.
CONNECTIVITY_TYPE = "toroidal"   # kept for tic_tac_toe_2agents.py (no longer used by the builder)

# ─── Temporal / learning constants ───────────────────────────────
reward_decay          = 0.1  ** (1 / WAIT_INC)
gamma                 = 0.7   ** (1 / WAIT_INC)
td_lambda             = 0.5   ** (1 / WAIT_INC)
entropy_coeff         = 1e-2
entropy_decay         = 0.99999 ** (1 / WAIT_INC)
entropy_coeff_min     = 0.0

TRAIN             = True
KERNEL_PROFILING  = False
CHECKPOINT_NAME   = None # "ttt_ep76000"
CHECKPOINT_SAVE_NAME = "ttt"
if CHECKPOINT_NAME is not None:
    CONNECTIVITY_TYPE = "fixed"

print("CHEKPOINT_NAME", CHECKPOINT_NAME)
serialiser = Numpy("ttt_checkpoints")

# ─── CSV output ──────────────────────────────────────────────────
CSV_OUTPUT = HP["csv_output"]
def init_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)          # or rename with a timestamp instead
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "score", "ep_steps", "avg_abs_td_error",
            "reward_rate", "voltage", "voltage_loss", "frequency", "first_player",
        ])

# ─── Build & compile network ─────────────────────────────────────

BUILT = None   # the snake_network.Built of the last network built (populations, labels, layer lists)


def plastic_inhibition_connections(built):
    """{Connection: settings} of the plastic inhibitory synapses (as snake.py): every layer's I -> E
    ("plastic_inhibition") and I -> I ("plastic_inhibition_i"), restricted to "plastic_inhibition_layers"."""
    layers = HP["plastic_inhibition_layers"]
    if layers is not None and any(not 1 <= k <= len(built.inhibitory_to_e) for k in layers):
        raise ValueError(f"plastic_inhibition_layers {layers}: the network has {len(built.inhibitory_to_e)} layers")
    out = {}
    for key, conns in (("plastic_inhibition", built.inhibitory_to_e), ("plastic_inhibition_i", built.inhibitory_to_i)):
        if HP[key] is not None:
            out.update({c: HP[key] for k, c in enumerate(conns, 1) if layers is None or k in layers})
    return out


def build_compiled_network(connectivity_type=None):
    """The network of snake_network.py from HP (the same builder and keys as snake.py), compiled.
    connectivity_type: unused (kept for tic_tac_toe_2agents.py); checkpoints restore the saved connectivity."""
    global BUILT
    spec = snake_network.spec_from_hparams(HP, INPUT_SHAPE, NUM_ACTIONS, channels=INPUT_SHAPE[2])
    built = BUILT = snake_network.build_network(spec, default_params)
    network, input_pop, policy, value = built.network, built.input, built.policy, built.value

    # ── Compiler ─────────────────────────────────────────────────
    if HP["hidden_rule"] is None:
        compiler_cls, rule_kwargs = EPropCompiler, {}
    else:
        from ml_genn.compilers.eprop import EPropCompiler as ModularEPropCompiler
        compiler_cls, rule_kwargs = ModularEPropCompiler, {"hidden_rule": HP["hidden_rule"],
                                                           "optimise_feedback": HP["optimise_feedback"],
                                                           "train_readout": HP["train_readout"],
                                                           "population_rules": built.population_rules}
        if HP["dale"] is not None:
            rule_kwargs["dale_local_rewiring"] = bool(HP["dale"].get("local", True))
        if HP["inhibitory_reversal"] is not None:
            rule_kwargs["inhibitory_reversal"] = HP["inhibitory_reversal"]
        plastic = plastic_inhibition_connections(built)
        if plastic:
            rule_kwargs["plastic_inhibition"] = plastic
    if HP["optimiser"] == "adam":
        optimiser = Adam(HP["lr"], beta1=0.9, beta2=0.999, l2_init_strength=1e-5)
    else:
        optimiser = AdaBelief(HP["lr"], beta1=0.99, beta2=0.99999, l2_init_strength=1e-5)
    compiler = compiler_cls(
        **rule_kwargs,
        rng_seed=0 if HP["seed"] is None else HP["seed"] + 1,   # GeNN: 0 = random seed
        # the CPU backend compiles without optimisation unless asked
        **({"backend": HP["backend"], "optimize_code": True} if HP["backend"] == "single_threaded_cpu"
           else {"backend": HP["backend"]} if HP["backend"] else {}),
        example_timesteps=1,
        losses={
            policy: "mean_square_error",
            value:  "mean_square_error",
        },
        # optimiser=AdaBelief(1e-5, task_steps=1, beta1=0.99, beta2=0.99999, l2_init_strength=1e-5),
        optimiser=optimiser,
        # optimiser=Adam(5e-5, beta1=0.9, beta2=0.999),
        c_reg=HP["c_reg"],
        # c_reg=1e-4,
        f_target=HP["f_target"],
        # f_target=100,
        batch_size=1,
        kernel_profiling=KERNEL_PROFILING,
        feedback_type=HP["feedback_type"],
        reward_decay=reward_decay,
        gamma=gamma,
        td_lambda=td_lambda,
        train_output_bias=False,
        reset_time_between_batches=False,
        entropy_coeff=entropy_coeff,
        entropy_coeff_decay=entropy_decay,
        entropy_coeff_min=entropy_coeff_min,
        dale_rewiring_l1_strength=0.0,
        policy_heads={policy: PolicyTypes.GENERIC},
        value_head=value,
    )

    if CHECKPOINT_NAME is not None:
        print("Loading", CHECKPOINT_NAME)
        network.load((CHECKPOINT_NAME,), serialiser)

    compiled_net = compiler.compile(network)

    hidden_layers = {i: pop for i, pop in enumerate(built.layers[0])}      # the first layer's E and I
    return compiled_net, network, input_pop, hidden_layers, policy, value


# ─── Spike encoding ──────────────────────────────────────────────

def make_rate_coded_spikes(values, base_timestep, input_size, K):
    values = np.clip(values, 0.0, 1.0) * 0.3
    times, idxs = [], []
    for i, v in enumerate(values):
        if v <= 0:
            continue
        fired = np.nonzero(np.random.rand(K) < v)[0]
        if fired.size == 0:
            continue
        times.append((base_timestep + fired).astype(np.int64))
        idxs.append(np.full(fired.size, i, dtype=np.int64))
    if not times:
        return preprocess_spikes(np.empty(0, np.int64),
                                 np.empty(0, np.int64), input_size)
    return preprocess_spikes(np.concatenate(times),
                             np.concatenate(idxs), input_size)


def softmax_masked(logits, mask):
    logits = logits.copy()
    logits[~mask] = -1e9
    e = np.exp(logits - logits.max())
    return e / (e.sum() + 1e-8)

INPUT_RATE_SCALE = 0.5

def obs_to_poisson_rate(obs):
    return (
        np.clip(obs, 0.0, 1.0)
        * INPUT_RATE_SCALE
    ).astype(np.float32)

# ─── Main training loop ──────────────────────────────────────────
def train(compiled_net, input_pop, hidden_layers, policy, value,
          train_callback_list, visualizer, episodes=int(1e10)):

    env         = TicTacToeEnv(rows=BOARD_ROWS, cols=BOARD_COLS,
                                wait_inc=WAIT_INC, scale=PIXEL_SCALE,
                                obs_scale=OBS_SCALE, opponent="minimax",
                                first_player=FIRST_PLAYER)
    opt_updt    = 0
    best_reward_ep = -10000
    best_reward = -np.inf
    best_run    = []
    avg         = 0.0
    smoothing   = 0.95

    # Running diagnostic averages (mirrors connect_four_snn.py)
    v_avg         = 0.0
    v_reg_loss_avg = 0.0
    freq_avg      = 0.0

    train_callback_list.on_epoch_begin(0)
    train_callback_list.on_batch_begin(0)

    for ep in range(episodes):

        obs  = env.reset()
        done = False
        total_reward   = 0.0
        reward_trace   = 0.0
        current_run    = []
        current_values = []
        current_rt     = []
        current_probs  = []
        ep_frames      = 0
        td_error_sum_abs = 0.0
        first_player_this_ep = env.current_player  # 1=agent, -1=opponent

        # ── initial spike encoding ───────────────────────────────
        compiled_net.set_input({
            input_pop: obs_to_poisson_rate(obs)
        })

        # Tracks whether we just crossed into a new mover's think-phase,
        # so we know when to re-encode/re-render (mirrors the old
        # `env.wait_count == env.wait_inc` check, but now fires at the
        # start of EITHER player's think-phase, not just the agent's).
        last_wait_count = env.wait_count

        # ── episode loop ─────────────────────────────────────────
        while not done:

            current_values.append(compiled_net.get_readout(value)[0].mean())
            current_rt.append(reward_trace)

            if env.awaiting_agent_action:
                logits = compiled_net.get_readout(policy).flatten()
                mask   = env.legal_mask()
                probs  = softmax_masked(logits, mask)

                action_label = np.random.choice(NUM_ACTIONS, p=probs)

                y_true = np.zeros(NUM_ACTIONS)
                y_true[action_label] = 1.0
                PG = probs - y_true

                log_p = np.log(probs + 1e-8)
                entropy = -np.sum(probs * log_p)

                compiled_net.neuron_populations[policy].vars["pre_PG"].view[:] = PG.astype(np.float32)
                compiled_net.neuron_populations[policy].push_var_to_device("pre_PG")

                current_probs.append(probs)
                action_to_step = action_label
            else:
                # Either still in a think-phase (either player), or it's
                # the opponent's turn to move — `action` is ignored by
                # env.step() in both cases, so any legal-looking value is
                # fine here (kept for readability / debuggability only).
                legal = np.where(env.legal_mask())[0]
                action_to_step = int(random.choice(legal)) if len(legal) else 0

            obs, reward, done = env.step(action_to_step)
            total_reward += reward
            reward_trace  = reward_trace * reward_decay + reward

            if reward != 0:
                compiled_net.losses[value].set_var(
                    compiled_net.neuron_populations[value], "reward", reward
                )

            # ── per-step diagnostics (mirrors connect_four_snn.py) ──
            # Fires at the START of every think-phase (agent's OR
            # opponent's) rather than only the agent's, since wait_count
            # resets to wait_inc for whichever player's turn it now is.
            started_new_think_phase = (
                (not done) and env.wait_count == env.wait_inc
                and last_wait_count != env.wait_inc
            )
            if started_new_think_phase:
                ep_frames += 1
                syn_sig_vals = []
                # Firing frequency
                for conn_pop in list(compiled_net.connection_populations.values())[::-1]:
                    try:
                        conn_pop.post_vars["FAvg"].pull_from_device()
                        f = conn_pop.post_vars["FAvg"].view
                    except Exception:
                        f = 0
                    try:
                        conn_pop.vars["SynSig"].pull_from_device()
                        syn_sig_vals.append(conn_pop.vars["SynSig"].values.mean())
                    except Exception:
                        pass
                freq_avg = np.mean(abs(f))
                syn_sig_avg = np.mean(syn_sig_vals)

                # Voltage diagnostics on first hidden population
                first_hidden = list(hidden_layers.values())[0]
                compiled_net.neuron_populations[first_hidden].vars["V"].pull_from_device()
                compiled_net.neuron_populations[first_hidden].vars["A"].pull_from_device()
                compiled_net.neuron_populations[first_hidden].vars["Beta"].pull_from_device()

                v_view    = compiled_net.neuron_populations[first_hidden].vars["V"].view
                A_view    = compiled_net.neuron_populations[first_hidden].vars["A"].view
                beta_view = compiled_net.neuron_populations[first_hidden].vars["Beta"].view

                v_avg = v_avg * 0.999 + 0.001 * np.mean(v_view)
                v_reg_loss_avg = v_reg_loss_avg * 0.999 + 0.001 * np.mean(abs(
                    np.maximum( v_view - (0.61 + beta_view * A_view), 0.0) +
                    np.maximum(-v_view - (0.61 + beta_view * A_view), 0.0)
                ))

                # Capture frame + re-encode spikes for the (possibly just
                # updated, e.g. after the opponent moved) board state.
                current_run.append(env.render())
                compiled_net.set_input({
                    input_pop: obs_to_poisson_rate(obs)
                })


            last_wait_count = env.wait_count if not done else last_wait_count

            compiled_net.step_time(train_callback_list)

            compiled_net.genn_model.custom_update("GradientLearn")
            for o, custom_updates in compiled_net.optimisers:
                for c in custom_updates:
                    o.set_step(c, opt_updt := opt_updt + 1)

        # ── terminal drain ───────────────────────────────────────
        compiled_net.set_input({
            input_pop: obs_to_poisson_rate(obs)
        })
        for _ in range(WAIT_INC):
            current_values.append(compiled_net.get_readout(value)[0].mean())
            reward_trace = reward_trace * reward_decay
            current_rt.append(reward_trace)
            compiled_net.step_time(train_callback_list)

            compiled_net.genn_model.custom_update("GradientLearn")
            for o, custom_updates in compiled_net.optimisers:
                for c in custom_updates:
                    o.set_step(c, opt_updt := opt_updt + 1)

        for _ in range(3):
            current_run.append(env.render())
        # Dale's law during training (as snake.py): wrong-sign synapses reset or pruned and rewired at episode ends
        if HP["dale"] is not None and ep % int(HP["dale"].get("every_episodes", 1)) == 0:
            compiled_net.genn_model.custom_update("DalePrune")
            compiled_net.genn_model.custom_update("DaleRewire")
        # ── periodic checkpoint ───────────────────────────────────
        if (ep + 1) % 1000 == 0:
            compiled_net.save_connectivity((f"{CHECKPOINT_SAVE_NAME}_ep{ep+1}",), serialiser)
            compiled_net.save((f"{CHECKPOINT_SAVE_NAME}_ep{ep+1}",), serialiser)
            print(f"  [checkpoint saved at ep {ep+1}]")

        # ── best-run tracking + visualizer ────────────────────────
        if current_probs and ep > best_reward_ep + 100:
            best_reward_ep = ep
            best_reward    = total_reward
            best_run       = list(current_run)[-6:]
            if visualizer:
                visualizer.push_best_sequence(best_run)
                current_G = compute_returns(current_rt, gamma)
                visualizer.push_metrics(
                    values=current_values,
                    reward_trace=np.array(current_G),
                    probs=current_probs,
                )

        avg = smoothing * avg + (1 - smoothing) * total_reward if avg else total_reward
        if visualizer:
            visualizer.push_metrics(reward=total_reward)

        # ── CSV logging (mirrors connect_four_snn.py) ─────────────
        safe_ep_frames = max(ep_frames, 1)
        with open(CSV_OUTPUT, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep,
                total_reward,
                ep_frames,
                td_error_sum_abs / safe_ep_frames,
                total_reward / safe_ep_frames,
                v_avg,
                v_reg_loss_avg,
                freq_avg,
                "agent" if first_player_this_ep == 1 else "opponent",
            ])

        if ep % 10 == 0:
            print(
                f"Ep {ep+1:6d} | "
                f"reward {total_reward:+7.2f} | "
                f"best {best_reward:+7.2f} | "
                f"avg {avg:+7.2f} | "
                f"outcome {env.winner} | "
                f"moves {env.moves} | "
                f"first {'agent' if first_player_this_ep == 1 else 'opponent'} | "
                f"voltage {v_avg:.4f} | "
                f"freq {1000 * freq_avg:.4f}"
            )


# ─── Entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    init_csv(CSV_OUTPUT)

    compiled_net, network, input_pop, hidden_layers, policy, value = \
        build_compiled_network(connectivity_type=CONNECTIVITY_TYPE)

    train_callback_list = CallbackList(
        [*set(compiled_net.base_train_callbacks),
         Checkpoint(serialiser)],
        compiled_network=compiled_net,
        num_batches=1,
        num_epochs=1,
    )

    vis = PerformanceVisualizer(window=100)

    try:
        with compiled_net:
            train(
                compiled_net, input_pop, hidden_layers, policy, value,
                train_callback_list, vis,
                episodes=int(1e10),
            )
    finally:
        vis.close()