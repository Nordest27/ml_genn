##################### CONNECT FOUR — SNN AGENT #####################
#####################################################################
# Drop-in replacement for the snake training script.
# Uses the same EProp / mlGeNN stack; imports PerformanceVisualizer
# from the generic visualizer module.
#####################################################################

import numpy as np
import cv2
import random
import time
import multiprocessing as mp
import csv
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
                              SpikeInput)
from ml_genn.serialisers import Numpy
from ml_genn.optimisers import Adam
from ml_genn.utils.data import preprocess_spikes
from ml_genn.utils.callback_list import CallbackList
from ml_genn.compilers.eprop_compiler import default_params

from connect_four_env import ConnectFourEnv
from performance_visualizer import PerformanceVisualizer

# ─── Board / timing constants ────────────────────────────────────
BOARD_ROWS  = 6
BOARD_COLS  = 7
WAIT_INC    = 30          # timesteps per move ("thinking time")
PIXEL_SCALE = 40

NUM_ACTIONS = BOARD_COLS  # one action per column
OBS_SCALE   = 2
# ─── Network topology ────────────────────────────────────────────
INPUT_SHAPE = (BOARD_ROWS * OBS_SCALE, BOARD_COLS * OBS_SCALE, 3) 
INPUT_SIZE     = int(np.prod(INPUT_SHAPE)) 

HIDDEN_E_SHAPE = (20, 20, 3)
HIDDEN_I_SHAPE = (15, 15, 3)
NUM_HIDDEN_E   = int(np.prod(HIDDEN_E_SHAPE))
NUM_HIDDEN_I   = int(np.prod(HIDDEN_I_SHAPE))

CONNECTIVITY_TYPE = "toroidal"   # "toroidal" | "fixed"

SIGMA_IN = 0.1
SIGMA_H  = 0.05

DESIRED_FAN_IN_IN = 300
DESIRED_FAN_IN_H1 = 300
DESIRED_FAN_IN_H2 = 300

CONN_P = {"I-H": 0.1, "H-H": 0.1, "H-P": 0.5, "H-V": 0.5, "F": 1.0}

# ─── Temporal / learning constants ───────────────────────────────
reward_decay          = 0.1  ** (1 / WAIT_INC)
gamma                 = 0.5   ** (1 / WAIT_INC)
td_lambda             = 0.1   ** (1 / WAIT_INC)
entropy_coeff         = 1e-2
entropy_decay         = 0.99999 ** (1 / WAIT_INC)
entropy_coeff_min     = 0.0

TRAIN             = True
KERNEL_PROFILING  = False
CHECKPOINT_NAME   = None      # set to e.g. "c4_mid" to resume

serialiser = Numpy("c4_checkpoints")

# ─── CSV output ──────────────────────────────────────────────────
CSV_OUTPUT = "outputs/c4_experiment.csv"
os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)

if os.path.exists(CSV_OUTPUT):
    os.remove(CSV_OUTPUT)

with open(CSV_OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "episode",
        "score",
        "ep_steps",
        "avg_abs_td_error",
        "reward_rate",
        "voltage",
        "voltage_loss",
        "frequency",
    ])


# ─── Connectivity helpers (mirrors snake.py) ─────────────────────

def make_connectivity(
    connectivity_type,
    src_shape,
    desired_fan_in=None,
    fan_in_scale=None,
    p=None,
    sigma=None,
    sign=None,
    mean_scale=0.1,
    sd_scale=0.05,
):
    if connectivity_type == "fixed":
        if p is None:
            raise ValueError("Fixed connectivity requires p")
        if sign is None:
            sd_scale = 1.0
        fan_in = p * np.prod(src_shape)
        mean = (sign or 0) * mean_scale / np.sqrt(fan_in)
        sd   = sd_scale / np.sqrt(fan_in)
        return FixedProbability(p, Normal(mean=mean, sd=sd))

    elif connectivity_type == "toroidal":
        if sigma is None:
            raise ValueError("Toroidal connectivity requires sigma")
        if desired_fan_in is None:
            raise ValueError("Toroidal connectivity requires desired_fan_in")
        fan_in = desired_fan_in
        if sign == -1:
            mean_scale *= 3
        elif sign is None:
            sd_scale = 1.0
        mean = (sign or 0) * mean_scale / np.sqrt(fan_in)
        sd   = sd_scale / np.sqrt(fan_in)
        return ToroidalGaussian2D(
            sigma=sigma,
            fan_in=desired_fan_in,
            fan_in_scale=fan_in_scale,
            weight=Normal(mean=mean, sd=sd),
        )
    else:
        raise ValueError(f"Unknown connectivity_type: {connectivity_type}")


@dataclass
class EILayerConfig:
    """Configuration for a single EI layer (identical to snake.py)."""
    e_shape: Tuple[int, ...]
    i_shape: Tuple[int, ...]

    v_thresh:   float = 0.61
    tau_mem:    float = 10.0
    tau_refrac: float = 3.0
    tau_adapt:  float = 300.0
    beta:       float = 0.0174

    connectivity_type: str = "toroidal"
    sigma: float = 0.05
    desired_fan_in_ee: int = 300
    desired_fan_in_ei: int = 300
    desired_fan_in_ie: int = 300
    desired_fan_in_ii: int = 300
    p_ee: float = 0.005
    p_ei: float = 0.005
    p_ie: float = 0.005
    p_ii: float = 0.005

    mean_scale: float = 0.1
    sd_scale:   float = 0.05


class EILayer:
    """
    A single Excitatory-Inhibitory layer (mirrors snake.py EILayer).
    """

    def __init__(self, cfg: EILayerConfig, name: str = ""):
        self.cfg  = cfg
        self.name = name
        self.e: Optional[Population] = None
        self.i: Optional[Population] = None
        self._internal_connections: list = []

    def build(self):
        cfg    = self.cfg
        neuron_kwargs = dict(
            v_thresh=cfg.v_thresh,
            tau_mem=cfg.tau_mem,
            tau_refrac=cfg.tau_refrac,
            tau_adapt=cfg.tau_adapt,
            beta=cfg.beta,
            integrate_during_refrac=False
        )
        self.e = Population(AdaptiveLeakyIntegrateFire(**neuron_kwargs), cfg.e_shape)
        self.i = Population(AdaptiveLeakyIntegrateFire(**neuron_kwargs), cfg.i_shape)

        internal = [
            (self.e, self.e, cfg.e_shape, cfg.desired_fan_in_ee, cfg.p_ee, +1),
            (self.e, self.i, cfg.e_shape, cfg.desired_fan_in_ei, cfg.p_ei, +1),
            (self.i, self.e, cfg.i_shape, cfg.desired_fan_in_ie, cfg.p_ie, -1),
            (self.i, self.i, cfg.i_shape, cfg.desired_fan_in_ii, cfg.p_ii, -1),
        ]
        for pre, post, src_shape, fan_in, p, sign in internal:
            conn = Connection(
                pre, post,
                make_connectivity(
                    connectivity_type=cfg.connectivity_type,
                    src_shape=src_shape,
                    p=p,
                    sigma=cfg.sigma,
                    desired_fan_in=fan_in,
                    sign=sign,
                    mean_scale=cfg.mean_scale,
                    sd_scale=cfg.sd_scale,
                ),
                exc_inh_sign=sign,
            )
            self._internal_connections.append(conn)
        return self

    def connect_from(
        self,
        source: Population,
        src_shape: Tuple,
        connectivity_type: str = None,
        desired_fan_in: int = 100,
        p: float = 0.01,
        sigma: float = None,
        fan_in_scale: float = None,
    ):
        cfg    = self.cfg
        c_type = connectivity_type or cfg.connectivity_type
        sig    = sigma or cfg.sigma
        for target in (self.e, self.i):
            Connection(
                source, target,
                make_connectivity(
                    connectivity_type=c_type,
                    src_shape=src_shape,
                    p=p,
                    sigma=sig,
                    desired_fan_in=desired_fan_in,
                    fan_in_scale=fan_in_scale,
                    sign=+1,
                    mean_scale=cfg.mean_scale,
                    sd_scale=cfg.sd_scale,
                ),
                exc_inh_sign=+1,
            )

    def connect_to_field(
        self,
        field: Population,
        p: float = 0.5,
        sigma: float = None,
        fan_in_scale: float = None,
    ):
        cfg = self.cfg
        sig = sigma or cfg.sigma
        for src, sign, src_shape, fan_in in (
            (self.e, +1, cfg.e_shape, cfg.desired_fan_in_ee),
            (self.i, -1, cfg.i_shape, cfg.desired_fan_in_ie),
        ):
            Connection(
                src, field,
                make_connectivity(
                    connectivity_type=cfg.connectivity_type,
                    src_shape=src_shape,
                    p=p,
                    sigma=sig,
                    desired_fan_in=fan_in,
                    fan_in_scale=fan_in_scale,
                    sign=sign,
                    mean_scale=cfg.mean_scale,
                    sd_scale=cfg.sd_scale,
                ),
                exc_inh_sign=sign,
            )

    def populations(self):
        return self.e, self.i


# ─── Build & compile network ─────────────────────────────────────

def build_compiled_network(connectivity_type=CONNECTIVITY_TYPE):
    network = Network(default_params)

    ei_cfg = EILayerConfig(
        e_shape=HIDDEN_E_SHAPE,
        i_shape=HIDDEN_I_SHAPE,
        connectivity_type=connectivity_type,
        sigma=SIGMA_H,
        desired_fan_in_ee=DESIRED_FAN_IN_H1,
        desired_fan_in_ei=DESIRED_FAN_IN_H1,
        desired_fan_in_ie=DESIRED_FAN_IN_H2,
        desired_fan_in_ii=DESIRED_FAN_IN_H2,
    )

    with network:

        # ── Populations ──────────────────────────────────────────
        input_pop = Population(
            SpikeInput(max_spikes=INPUT_SIZE * WAIT_INC), INPUT_SHAPE
        )

        alif_params = dict(
            v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=300,
            integrate_during_refrac=False
        )

        ei_layers = []
        for i in range(1):
            ei_layers.append(EILayer(ei_cfg, name=f"L{i+1}").build())

        policy_field = Population(
            AdaptiveLeakyIntegrateFire(**alif_params), HIDDEN_I_SHAPE
        )
        value_field = Population(
            AdaptiveLeakyIntegrateFire(**alif_params), HIDDEN_I_SHAPE
        )

        policy = Population(
            LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), NUM_ACTIONS
        )
        value = Population(
            LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 1
        )

        # ── Input → first EI layer ────────────────────────────────
        ei_layers[0].connect_from(
            input_pop, INPUT_SHAPE,
            connectivity_type=connectivity_type,
            desired_fan_in=DESIRED_FAN_IN_IN,
            sigma=SIGMA_IN,
        )

        # ── Last EI layer → field populations ────────────────────
        ei_layers[-1].connect_to_field(policy_field, p=CONN_P["H-H"])
        ei_layers[-1].connect_to_field(value_field,  p=CONN_P["H-H"])

        # ── Field → output heads (forward + feedback) ────────────
        for field, head, feedback_name, n_out in (
            (policy_field, policy, "policy_feedback", NUM_ACTIONS),
            (value_field,  value,  "value_feedback",  1),
        ):
            Connection(
                field, head,
                make_connectivity("fixed", src_shape=HIDDEN_I_SHAPE,
                                  p=0.99999, sign=None),
                exc_inh_sign=None,
            )
            # Connection(
            #     field, head,
            #     FixedProbability(0.99999, Normal(sd=1.0 / np.sqrt(n_out))),
            #     feedback_name=feedback_name,
            #     exc_inh_sign=None,
            # )

        # ── tde_transport: policy + all hidden → value ────────────
        Connection(policy, value, Dense(weight=1.0),
                   feedback_name="tde_transport")
        for layer in ei_layers:
            for pop in layer.populations():
                Connection(pop, value, Dense(weight=1.0),
                           feedback_name="tde_transport")
        for field in (policy_field, value_field):
            Connection(field, value, Dense(weight=1.0),
                       feedback_name="tde_transport")

        # # ── policy/value feedback from EI layers ──────────────────
        # for layer in ei_layers:
        #     for pop in layer.populations():
        #         Connection(
        #             pop, policy,
        #             FixedProbability(CONN_P["F"],
        #                              Normal(sd=1.0 / np.sqrt(NUM_ACTIONS))),
        #             feedback_name="policy_feedback",
        #             exc_inh_sign=None,
        #         )
        #         Connection(
        #             pop, value,
        #             FixedProbability(CONN_P["F"],
        #                              Normal(sd=1.0 / np.sqrt(1))),
        #             feedback_name="value_feedback",
        #             exc_inh_sign=None,
        #         )

    # ── Compiler ─────────────────────────────────────────────────
    compiler = EPropCompiler(
        example_timesteps=1,
        losses={
            policy: "mean_square_error",
            value:  "mean_square_error",
        },
        optimiser=Adam(1e-4),
        # c_reg=1e-1,
        batch_size=1,
        kernel_profiling=KERNEL_PROFILING,
        feedback_type="symmetric",
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
        network.load((CHECKPOINT_NAME,), serialiser)

    compiled_net = compiler.compile(network)

    hidden_layers = {i: pop
                     for i, pop in enumerate(ei_layers[0].populations())}
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


# ─── Main training loop ──────────────────────────────────────────

def train(compiled_net, input_pop, hidden_layers, policy, value,
          train_callback_list, visualizer, episodes=int(1e10)):

    env         = ConnectFourEnv(rows=BOARD_ROWS, cols=BOARD_COLS,
                                  wait_inc=WAIT_INC, scale=PIXEL_SCALE,
                                  obs_scale = OBS_SCALE, opponent="opportunistic")
    opt_updt    = 0
    best_reward_ep = -10000
    best_reward = -np.inf
    best_run    = []
    avg         = 0.0
    smoothing   = 0.95

    # Running diagnostic averages (mirrors snake.py)
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

        # ── initial spike encoding ───────────────────────────────
        spikes = make_rate_coded_spikes(
            obs.reshape(-1), compiled_net.genn_model.timestep,
            INPUT_SIZE, WAIT_INC
        )
        compiled_net.set_input({input_pop: [spikes]})
        env.wait_count = WAIT_INC

        # ── episode loop ─────────────────────────────────────────
        while not done:

            current_values.append(compiled_net.get_readout(value)[0].mean())
            current_rt.append(reward_trace)

            if env.wait_count == 0:
                logits = compiled_net.get_readout(policy).flatten()
                mask   = env.legal_mask()
                probs  = softmax_masked(logits, mask)

                action_label = np.random.choice(NUM_ACTIONS, p=probs)

                y_true = np.zeros(NUM_ACTIONS)
                y_true[action_label] = 1.0
                PG = probs - y_true

                log_p = np.log(probs + 1e-8)
                entropy = -np.sum(probs * log_p)

                # entropy_grad_logits = -probs * (log_p + entropy)

                # E = entropy_coeff * entropy_grad_logits

                compiled_net.neuron_populations[policy].vars["pre_PG"].view[:] = PG.astype(np.float32)
                compiled_net.neuron_populations[policy].push_var_to_device("pre_PG")

                # compiled_net.neuron_populations[policy].vars["pre_E"].view[:] = E.astype(np.float32)
                # compiled_net.neuron_populations[policy].push_var_to_device("pre_E")
                
                current_probs.append(probs)

            else:
                legal = np.where(env.legal_mask())[0]
                action_label = int(random.choice(legal)) if len(legal) else 0

            obs, reward, done = env.step(action_label)
            total_reward += reward
            reward_trace  = reward_trace * reward_decay + reward

            if reward != 0:
                compiled_net.losses[value].set_var(
                    compiled_net.neuron_populations[value], "reward", reward
                )

            # ── per-step diagnostics (mirrors snake.py) ──────────
            if env.wait_count == env.wait_inc:
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

                v_avg = v_avg * 0.999 + 0.001 * np.mean(abs(v_view))
                v_reg_loss_avg = v_reg_loss_avg * 0.999 + 0.001 * np.mean(abs(
                    np.maximum( v_view - (0.61 + beta_view * A_view), 0.0) +
                    np.maximum(-v_view - (0.61 + beta_view * A_view), 0.0)
                ))

                # Capture frame + new spikes
                current_run.append(env.render())
                spikes = make_rate_coded_spikes(
                    obs.reshape(-1), compiled_net.genn_model.timestep,
                    INPUT_SIZE, WAIT_INC
                )
                compiled_net.set_input({input_pop: [spikes]})

            compiled_net.step_time(train_callback_list)

            # if env.wait_count == env.wait_inc:
            #     compiled_net.genn_model.custom_update("GradientLearn")
            #     for o, custom_updates in compiled_net.optimisers:
            #         for c in custom_updates:
            #             o.set_step(c, opt_updt := opt_updt + 1)

        # ── terminal drain ───────────────────────────────────────
        spikes = make_rate_coded_spikes(
            obs.reshape(-1), compiled_net.genn_model.timestep,
            INPUT_SIZE, WAIT_INC
        )
        compiled_net.set_input({input_pop: [spikes]})
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
        # ── periodic checkpoint ───────────────────────────────────
        if (ep + 1) % 1000 == 0:
            compiled_net.save((f"c4_ep{ep+1}",), serialiser)
            print(f"  [checkpoint saved at ep {ep+1}]")

        # ── best-run tracking + visualizer ────────────────────────
        if current_probs and ep > best_reward_ep + 100:
            best_reward_ep = ep
            best_reward    = total_reward
            best_run       = list(current_run)[-6:]
            if visualizer:
                visualizer.push_best_sequence(best_run)
                visualizer.push_metrics(
                    values=current_values,
                    reward_trace=np.array(current_rt),
                    probs=current_probs,
                )

        avg = smoothing * avg + (1 - smoothing) * total_reward if avg else total_reward
        if visualizer:
            visualizer.push_metrics(reward=total_reward)

        # ── CSV logging (mirrors snake.py) ────────────────────────
        safe_ep_frames = max(ep_frames, 1)
        with open(CSV_OUTPUT, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep,
                total_reward,
                ep_frames,
                td_error_sum_abs / safe_ep_frames,
                WAIT_INC * total_reward / safe_ep_frames,
                v_avg,
                v_reg_loss_avg,
                freq_avg,
            ])

        if ep % 10 == 0:
            print(
                f"Ep {ep+1:6d} | "
                f"reward {total_reward:+7.2f} | "
                f"best {best_reward:+7.2f} | "
                f"avg {avg:+7.2f} | "
                f"outcome {env.winner} | "
                f"moves {env.moves} | "
                f"voltage {v_avg:.4f} | "
                f"freq {1000 * freq_avg:.4f}"
            )


# ─── Entry point ─────────────────────────────────────────────────

if __name__ == "__main__":

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