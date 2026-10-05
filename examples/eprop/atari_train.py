##################### ATARI — SNN AGENT #####################
###############################################################
# Drop-in replacement for the Connect Four training script.
# Uses the same EProp / mlGeNN stack; imports PerformanceVisualizer
# from the generic visualizer module (unchanged from connect_four.py)
# and AtariEnv from atari_env.py for the environment side.
#
# POLICY HEAD: softmax over a 3-unit readout (down / stay / up),
# exactly the PolicyTypes.GENERIC branch used in pacmanAgents.py's
# getAction(): logits -> softmax -> categorical sample -> PG = probs
# - one_hot(action), pushed straight to the policy population's
# `pre_PG` var. No injected noise, no perturbation trace, no tanh
# derivative -- this is the same analytic score-function gradient
# pacman uses, just with 3 outcomes instead of pacman's 4.
#
# As before, the action is resampled fresh on *every* simulated
# timestep, not just once per WAIT_INC -- the environment reacts to
# the action every timestep, it's only the *input frame* (what the
# SNN sees, and hence the logits/probs) that's held/refreshed on the
# WAIT_INC cadence, purely for encoding/render performance. So the
# paddle can be corrected continuously (via fresh categorical draws
# from the same softmax distribution) even while the network is
# "looking at" the same frame for several ticks. The gradient credit
# (PG) is only pushed to the device once per input-frame refresh,
# matching pacman's "one PG push per getAction() call" cadence.
###############################################################

import numpy as np
import random
import csv
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from ml_genn import Population, Connection, Network
from ml_genn.callbacks import Checkpoint
from ml_genn.compilers import EPropCompiler, PolicyTypes
from ml_genn.connectivity import Dense, FixedProbability, ToroidalGaussian2D
from ml_genn.initializers import Normal
from ml_genn.neurons import (LeakyIntegrate, AdaptiveLeakyIntegrateFire,
                              PoissonInput)
from ml_genn.serialisers import Numpy
from ml_genn.optimisers import CAdam, AdaBelief
from ml_genn.utils.callback_list import CallbackList
from ml_genn.compilers.eprop_compiler import default_params

from atari_env import AtariEnv
from performance_visualizer import PerformanceVisualizer

def compute_returns(rewards, gamma):
    """
    Backward discounted return G_t = r_t + gamma * G_{t+1}, computed
    per-timestep over a full episode (or trajectory) of rewards.

    rewards: sequence of raw per-timestep rewards actually seen by the
             network, in time order.
    gamma:   per-timestep discount factor.

    Returns a list the same length as `rewards`, where result[t] is the
    true return-to-go from timestep t onward.
    """
    G = list(rewards)
    for i in range(len(G) - 2, -1, -1):
        G[i] += gamma * G[i + 1]
    return G

# ─── Game / timing constants ─────────────────────────────────────
GAME_ID       = "ALE/Pong-v5"   # swap for any ALE/<Game>-v5 id
RENDER_SCALE  = 1
FRAME_SKIP    = 10
CLIP_REWARDS  = True

# ─── Network topology ────────────────────────────────────────────
# Simplest possible version: single grayscale frame, no frame
# stacking. Any motion/velocity signal has to come from the SNN's
# own recurrent state (membrane potential + adaptation + recurrent
# E/I connectivity), not from pre-stacked input frames. If it
# plateaus on games where velocity matters (Pong is the classic
# case), that's a signal to revisit tau_mem/tau_adapt vs WAIT_INC
# before reaching for frame stacking again.
#
# Keep the input small at first — every input pixel gets rate
# encoded WAIT_INC times per env step, so resolution costs sim time
# and connection counts fast. Scale up once training is confirmed stable.
INPUT_SHAPE = (42, 42, 1)
INPUT_SIZE  = int(np.prod(INPUT_SHAPE))

HIDDEN_E_SHAPE = (21, 21, 1)
HIDDEN_I_SHAPE = (15, 15, 1)
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
reward_decay      = 0.1  
gamma             = 0.99
td_lambda         = 0.99
entropy_coeff     = 1e-2
entropy_decay     = 0.99999 
entropy_coeff_min = 0.0

# ─── Policy-head constants ────────────────────────────────────────
# Number of policy-head output units. Ordinal outcomes are
# 0=down, 1=stay, 2=up (see ACTION_MAP below) -- softmax turns the
# 3-unit readout into a genuine categorical over these exactly the
# way pacmanAgents.py's PolicyTypes.GENERIC branch turns its 4-unit
# readout into a categorical over WEST/NORTH/EAST/SOUTH.
NUM_POLICY_OUTPUTS = 3

TRAIN            = True
KERNEL_PROFILING = False
CHECKPOINT_NAME  = None      # set e.g. "atari_mid" to resume

serialiser = Numpy("atari_checkpoints")

# ─── Action-index <-> real ALE action-id resolution ──────────────
# Ordinal outcome indices used throughout this file: 0=down, 1=stay, 2=up.
#
# AtariEnv.n_actions comes straight from gym.make(...).action_space.n
# (AtariPreprocessing does not remap the action space), and
# AtariEnv.step(action) passes int(action) straight through to the
# underlying gym env. So the ids below are real ALE action ids.
#
# For ALE/Pong-v5 specifically, gym's action meanings are:
#   ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
# and (this is a known ALE-Pong quirk) RIGHT moves the paddle UP,
# LEFT moves the paddle DOWN. Verified directly against
# gym.make("ALE/Pong-v5").unwrapped.get_action_meanings().
_PONG_ACTION_MAP = {"down": 3, "stay": 0, "up": 2}  # LEFT, NOOP, RIGHT

def resolve_ordinal_action_map(env):
    """
    Build {0: down_id, 1: stay_id, 2: up_id} for the joystick head.

    Tries to introspect the wrapped env's action meanings so this isn't
    silently wrong if you point GAME_ID at something other than Pong.
    Falls back to the known Pong mapping otherwise. Prints what it
    resolved to -- check this line against your actual game before a
    long training run.
    """
    meanings = None
    for attr_path in ("env.unwrapped.get_action_meanings",
                       "unwrapped.get_action_meanings",
                       "get_action_meanings"):
        obj = env
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            meanings = obj()
            break
        except AttributeError:
            continue

    if meanings is not None:
        meanings = [m.upper() for m in meanings]
        try:
            down_id = meanings.index("LEFT")
            up_id   = meanings.index("RIGHT")
            stay_id = meanings.index("NOOP")
            action_map = {0: down_id, 1: stay_id, 2: up_id}
            print(f"[action map] resolved from env.get_action_meanings() = {meanings}")
            print(f"[action map] down={down_id} stay={stay_id} up={up_id}")
            return action_map
        except ValueError:
            print(f"[action map] WARNING: couldn't find LEFT/RIGHT/NOOP in "
                  f"{meanings}, falling back to hardcoded Pong map")

    action_map = {0: _PONG_ACTION_MAP["down"],
                  1: _PONG_ACTION_MAP["stay"],
                  2: _PONG_ACTION_MAP["up"]}
    print(f"[action map] WARNING: couldn't introspect env action meanings "
          f"(game_id={GAME_ID}); falling back to hardcoded Pong map: {action_map}. "
          f"Verify this is correct for your game before training.")
    return action_map


# ─── Probe the env once to get the action-space size + action map ─
_probe_env  = AtariEnv(game_id=GAME_ID, inp_shape=INPUT_SHAPE,
                       frame_skip=1,
                       render_scale=RENDER_SCALE, clip_rewards=CLIP_REWARDS)
NUM_ACTIONS = _probe_env.n_actions          # kept for value-head "F" connection etc.
ACTION_MAP  = resolve_ordinal_action_map(_probe_env)
_probe_env.close()
del _probe_env

# ─── CSV output ──────────────────────────────────────────────────
CSV_OUTPUT = f"outputs/atari_{GAME_ID.split('/')[-1]}_experiment.csv"
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


# ─── Connectivity helpers (identical to connect_four.py / snake.py) ─

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
    """Configuration for a single EI layer (identical to connect_four.py)."""
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
    """A single Excitatory-Inhibitory layer (mirrors connect_four.py)."""

    def __init__(self, cfg: EILayerConfig, name: str = ""):
        self.cfg  = cfg
        self.name = name
        self.e: Optional[Population] = None
        self.i: Optional[Population] = None
        self._internal_connections: list = []

    def build(self):
        cfg = self.cfg
        neuron_kwargs = dict(
            v_thresh=cfg.v_thresh,
            tau_mem=cfg.tau_mem,
            tau_refrac=cfg.tau_refrac,
            tau_adapt=cfg.tau_adapt,
            beta=cfg.beta,
            integrate_during_refrac=True,
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

    def connect_to_next(
        self,
        next_layer: "EILayer",
        p: float = 0.01,
        sigma: float = None,
        fan_in_scale: float = None,
    ):
        cfg = self.cfg
        sig = sigma or cfg.sigma

        for target in (next_layer.e, next_layer.i):
            Connection(
                self.e, target,
                make_connectivity(
                    connectivity_type=cfg.connectivity_type,
                    src_shape=cfg.e_shape,
                    p=p,
                    sigma=sig,
                    desired_fan_in=cfg.desired_fan_in_ee,
                    fan_in_scale=fan_in_scale,
                    sign=+1,
                    mean_scale=cfg.mean_scale,
                    sd_scale=cfg.sd_scale,
                ),
                exc_inh_sign=+1,
            )

        for target in (next_layer.e, next_layer.i):
            Connection(
                self.i, target,
                make_connectivity(
                    connectivity_type=cfg.connectivity_type,
                    src_shape=cfg.i_shape,
                    p=p,
                    sigma=sig,
                    desired_fan_in=cfg.desired_fan_in_ie,
                    fan_in_scale=fan_in_scale,
                    sign=-1,
                    mean_scale=cfg.mean_scale,
                    sd_scale=cfg.sd_scale,
                ),
                exc_inh_sign=-1,
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
        # PoissonInput: rate-based spike generation happens on-device
        # (gennrand_uniform() >= exp(-rate*dt) each timestep), so the
        # CPU only needs to push one rate value per pixel per env step
        # -- not a precomputed spike train. See obs_to_poisson_rate().
        input_pop = Population(PoissonInput(), INPUT_SHAPE)

        alif_params = dict(
            v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=300,
            integrate_during_refrac=True
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

        # Policy head is a NUM_POLICY_OUTPUTS-unit softmax readout,
        # exactly mirroring pacmanAgents.py's PolicyTypes.GENERIC head
        # (there NUM_OUTPUT=4 for WEST/NORTH/EAST/SOUTH; here it's 3
        # for down/stay/up). Action selection happens in
        # softmax_policy() below.
        policy = Population(
            LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"),
            NUM_POLICY_OUTPUTS,
        )
        value = Population(
            LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 1
        )

        # Input → first EI layer (excitatory only)
        ei_layers[0].connect_from(
            input_pop, INPUT_SHAPE,
            connectivity_type=connectivity_type,
            desired_fan_in=DESIRED_FAN_IN_IN,
            sigma=SIGMA_IN,
        )

        # Stack EI layers
        for i in range(len(ei_layers) - 1):
            ei_layers[i].connect_to_next(ei_layers[i + 1])

        # ── Last EI layer → field populations ────────────────────
        ei_layers[-1].connect_to_field(policy_field, p=CONN_P["H-H"])
        ei_layers[-1].connect_to_field(value_field,  p=CONN_P["H-H"])

        # ── Field → output heads ──────────────────────────────────
        for field, head, feedback_name, n_out in (
            (policy_field, policy, "policy_feedback", NUM_POLICY_OUTPUTS),
            (value_field,  value,  "value_feedback",  1),
        ):
            Connection(
                field, head,
                make_connectivity("fixed", src_shape=HIDDEN_I_SHAPE,
                                  p=0.99999, sign=None),
                exc_inh_sign=None,
            )

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

    # ── Compiler ─────────────────────────────────────────────────
    compiler = EPropCompiler(
        example_timesteps=1,
        losses={
            policy: "mean_square_error",
            value:  "mean_square_error",
        },
        optimiser=AdaBelief(1e-6, task_steps=1, beta1=0.99, beta2=0.99999),
        c_reg=1e-2,
        batch_size=1,
        kernel_profiling=KERNEL_PROFILING,
        feedback_type="random",
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

# Per-timestep firing probability of PoissonInput is 1 - exp(-rate*dt).
# dt = 1.0 ms here (EPropCompiler default, unchanged in this script), so
# INPUT_RATE_SCALE = 0.36 gives 1 - exp(-0.36) ≈ 0.30 at full pixel
# intensity -- matching the old make_rate_coded_spikes' peak probability
# (values*0.3) as closely as a rate-based encoding can. Retune this if
# the network's spike statistics look off after the swap.
INPUT_RATE_SCALE = 0.5

def obs_to_poisson_rate(obs):
    """
    Pixel intensities in [0,1], shaped like INPUT_SHAPE -> per-timestep
    Poisson rate for PoissonInput. The GPU resamples spikes from this
    rate every timestep on its own, so this only needs to be pushed once
    per env step (while the frame is held for WAIT_INC timesteps), not
    once per individual timestep.
    """
    return (np.clip(obs, 0.0, 1.0) * INPUT_RATE_SCALE).astype(np.float32)


def softmax_policy(logits):
    """
    Softmax policy head -- the PolicyTypes.GENERIC branch from
    pacmanAgents.py's getAction(), applied to the 3-unit (down/stay/up)
    readout instead of pacman's 4-unit (WEST/NORTH/EAST/SOUTH) readout:

      1. logits -> softmax -> probs, a genuine 3-way categorical with
         no separate temperature or threshold hyperparameters to tune
         (same property the old perturbation head advertised, just via
         softmax instead of tanh(h + eps)).
      2. Sample one action from that categorical.
      3. PG = probs - one_hot(action), the analytic score-function
         gradient for softmax + categorical sampling. This is pushed
         straight to the policy population's `pre_PG` var -- same as
         pacman does, and no perturbation trace or tanh-derivative
         surrogate is needed since this is the closed-form gradient,
         not a node-perturbation estimator.

    Stateless (unlike the old GaussianPerturbationHead) -- there's no
    trace to carry across calls, so no reset() is needed between
    episodes.
    """
    logits = np.asarray(logits, dtype=np.float64).flatten()
    shifted = logits - logits.max()
    exp_l = np.exp(shifted)
    probs = exp_l / (exp_l.sum() + 1e-8)

    action_idx = int(np.random.choice(len(probs), p=probs))

    y_true = np.zeros_like(probs)
    y_true[action_idx] = 1.0
    pg = probs - y_true

    return action_idx, probs, pg

# ─── Main training loop ──────────────────────────────────────────

def train(compiled_net, input_pop, hidden_layers, policy, value,
          train_callback_list, visualizer, episodes=int(1e10)):

    env = AtariEnv(
        game_id=GAME_ID, inp_shape=INPUT_SHAPE,
        frame_skip=1, render_scale=RENDER_SCALE,
        clip_rewards=CLIP_REWARDS,
    )

    opt_updt       = 0
    best_reward_ep = -10000
    best_reward    = -np.inf
    avg            = 0.0
    smoothing      = 0.95

    # Running diagnostic averages (mirrors connect_four.py / snake.py)
    v_avg          = 0.0
    v_reg_loss_avg = 0.0
    freq_avg       = 0.0

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

        # ── initial input encoding ───────────────────────────────
        compiled_net.set_input({input_pop: obs_to_poisson_rate(obs)})
        frame_skip = FRAME_SKIP

        # ── episode loop ─────────────────────────────────────────
        while not done:

            # ── per-step diagnostics (mirrors connect_four.py) ───
            if frame_skip == FRAME_SKIP:
                h = compiled_net.get_readout(policy).flatten()
                action_idx, probs, pg = softmax_policy(h)

                compiled_net.neuron_populations[policy].vars["pre_PG"].view[:] = \
                    pg.astype(np.float32)
                compiled_net.neuron_populations[policy].push_var_to_device("pre_PG")
                
                ep_frames += 1
                current_probs.append(probs)

                syn_sig_vals = []
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

                current_run.append(env.render())
                compiled_net.set_input({input_pop: obs_to_poisson_rate(obs)})

            current_values.append(compiled_net.get_readout(value)[0].mean())
            current_rt.append(reward_trace)

            # Fresh categorical draw every timestep from the same
            # softmax(h) distribution -- h (and hence probs) only
            # changes when the input frame is refreshed above, but the
            # actual action taken is resampled continuously so the
            # paddle can still be corrected mid-frame.
            action_idx, probs, pg = softmax_policy(h)

            action_label = ACTION_MAP[action_idx]

            obs, reward, done = env.step(action_label)
            total_reward += reward
            reward_trace  = reward_trace * reward_decay + reward

            if reward != 0:
                compiled_net.losses[value].set_var(
                    compiled_net.neuron_populations[value], "reward", reward
                )

            compiled_net.step_time(train_callback_list)

            compiled_net.genn_model.custom_update("GradientLearn")
            for o, custom_updates in compiled_net.optimisers:
                for c in custom_updates:
                    o.set_step(c, opt_updt := opt_updt + 1)

            frame_skip -= 1
            if frame_skip == 0:
                frame_skip = FRAME_SKIP

        # ── periodic checkpoint ───────────────────────────────────
        if (ep + 1) % 1000 == 0:
            compiled_net.save((f"atari_ep{ep+1}",), serialiser)
            print(f"  [checkpoint saved at ep {ep+1}]")

        # ── best-run tracking + visualizer ────────────────────────
        if current_probs and (total_reward >= best_reward or (ep - best_reward_ep) > 50):
            best_reward_ep = ep
            best_reward    = total_reward
            best_run       = list(current_run)
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

        # ── CSV logging ────────────────────────────────────────────
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
            ])

        if ep % 10 == 0:
            print(
                f"Ep {ep+1:6d} | "
                f"reward {total_reward:+7.2f} | "
                f"best {best_reward:+7.2f} | "
                f"avg {avg:+7.2f} | "
                f"env steps {env.steps_taken:4d} | "
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