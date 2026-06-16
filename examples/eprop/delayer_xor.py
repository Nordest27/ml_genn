"""
delayed_recall_env.py
=====================
Delayed evidence-accumulation task — closely following Bellec et al. 2020
(e-prop paper, Fig. 3 / store-recall variant).

Task
----
  Each trial has three phases:
    1. EVIDENCE  (CUE_DUR sim-steps)
       Population A and population B each spike with Poisson statistics.
       One population is designated the "target" and fires at HIGH_RATE;
       the other fires at LOW_RATE (background noise).
       The network must accumulate which population was more active.

    2. DELAY  (DELAY_DUR sim-steps)
       All input populations are silent.
       The network must hold the evidence in recurrent dynamics.

    3. RECALL  (1 decision step)
       A dedicated RECALL input population fires briefly.
       The agent outputs a direction: LEFT (→ pop A was dominant)
                                  or RIGHT (→ pop B was dominant).
       Reward: +1 if correct, -1 if wrong.

Input layout  (flat, length = NUM_INPUTS)
-----------------------------------------
  [0  .. N-1]      pop A  (N neurons)
  [N  .. 2N-1]     pop B  (N neurons)
  [2N .. 3N-1]     RECALL signal  (N neurons, fires only at recall step)

Observation encoding
--------------------
  Rate-coded Poisson spikes over WAIT_INC sim-steps per decision step,
  identical to the ml_genn spike-encoding used in snake/pacman.

Visualization
-------------
  cv2 frame shows:
    LEFT   — spike raster for pop A and pop B during the evidence window
    CENTRE — trial phase timeline (evidence → delay → recall)
    RIGHT  — single policy probability bar (P(correct direction))
              + rolling accuracy sparkline

Usage
-----
    python delayed_recall_env.py                 # default: delay=200 steps
    python delayed_recall_env.py --delay 1000
    python delayed_recall_env.py --load tag      # resume checkpoint
"""

from __future__ import annotations
import argparse
import math
import numpy as np
import cv2

from ml_genn import Population, Connection, Network
from ml_genn.neurons import SpikeInput, AdaptiveLeakyIntegrateFire, LeakyIntegrate
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal
from ml_genn.compilers import EPropCompiler, PolicyTypes
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.compilers.eprop_compiler import default_params
from ml_genn.utils.data import preprocess_spikes
from ml_genn.utils.callback_list import CallbackList

from performance_visualizer import PerformanceVisualizer


# ===========================================================================
# Hyper-parameters
# ===========================================================================

# Neurons per population slot
N_POP = 10

# Firing rates (spikes per sim-step, i.e. probability per step)
HIGH_RATE = 0.5   # target population during evidence
LOW_RATE  = 0.1   # distractor population during evidence

# Evidence window length (decision steps, each = WAIT_INC sim-steps)
CUE_DUR   = 10

# Recall window length (decision steps)
RECALL_DUR = 3


# ===========================================================================
# Temporal config — derived from delay length
# ===========================================================================

def compute_temporal_cfg(delay: int, wait_inc: int) -> dict:
    """
    Scale discount / trace decay to the delay so credit can bridge the gap.
      gamma^delay ≈ 0.5  →  gamma = 0.5^(1/delay)   (per decision step)
    We then convert to per-sim-step for the compiler.
    """
    gamma        = 0.5  ** (1.0 / max(delay, 1))
    td_lambda    = 0.95 ** (1.0 / max(delay, 1))
    reward_decay = 0.1  ** (1.0 / max(delay, 1))

    # per sim-step
    gamma_sim        = gamma        ** (1.0 / wait_inc)
    td_lambda_sim    = td_lambda    ** (1.0 / wait_inc)
    reward_decay_sim = reward_decay ** (1.0 / wait_inc)

    entropy_coeff        = 5e-2
    entropy_coeff_decay  = 0.9999 ** (1.0 / wait_inc)
    entropy_coeff_min    = 1e-3

    return dict(
        gamma               = gamma_sim,
        td_lambda           = td_lambda_sim,
        reward_decay        = reward_decay_sim,
        entropy_coeff       = entropy_coeff,
        entropy_coeff_decay = entropy_coeff_decay,
        entropy_coeff_min   = entropy_coeff_min,
    )


# ===========================================================================
# Spike encoding
# ===========================================================================

def make_rate_coded_spikes(values: np.ndarray, base_timestep: int,
                            input_size: int, K: int) -> object:
    values = np.clip(values, 0.0, 1.0)
    times, idxs = [], []
    for i, v in enumerate(values):
        if v <= 0:
            continue
        spikes = np.random.rand(K) < v
        if not np.any(spikes):
            continue
        spike_times = base_timestep + np.nonzero(spikes)[0]
        times.append(spike_times.astype(np.int64))
        idxs.append(np.full(len(spike_times), i, dtype=np.int64))
    if not times:
        return preprocess_spikes(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            input_size
        )
    return preprocess_spikes(
        np.concatenate(times),
        np.concatenate(idxs),
        input_size
    )


# ===========================================================================
# Visualization
# ===========================================================================

_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_BG     = (18,  18,  24)
_DIM    = (50,  50,  60)
_COL_A  = (60, 200,  80)   # green  — pop A
_COL_B  = (60, 100, 220)   # blue   — pop B
_COL_RC = (255, 180,  20)  # amber  — recall
_OK     = (40, 210,  80)
_FAIL   = (40,  60, 210)
_TEXT   = (200, 200, 210)
_DIMT   = (100, 100, 110)


def _put(img, text, xy, scale=0.42, col=_TEXT, thick=1):
    cv2.putText(img, text, xy, _FONT, scale, col, thick, cv2.LINE_AA)


def render_trial(
    target: int,           # 0=A dominant, 1=B dominant
    spike_raster_A: list,  # [(neuron_idx, t_step), ...] — evidence spikes for pop A
    spike_raster_B: list,  # [(neuron_idx, t_step), ...] — evidence spikes for pop B
    last_action: int,      # 0=chose A, 1=chose B
    last_reward: float,    # +1 correct, -1 wrong
    prob_A: float,         # policy P(A) at decision time
    acc_history: list,     # rolling accuracy per episode
    frame_w: int = 640,
    frame_h: int = 300,
) -> np.ndarray:
    """
    One image per trial, rendered at the end.

    Layout (640 x 300):

      ┌─────────────────────────────┬──────────────────────┐
      │  SPIKE RASTER               │  DECISION            │
      │  Pop A  ·  · ·  ·           │                      │
      │  Pop B   ··   ·  · ·        │  A ████████░░░░ B    │
      │                             │  chose A  ✓ CORRECT  │
      │                             │  ── accuracy ────    │
      └─────────────────────────────┴──────────────────────┘

    Left: spike raster — each row is one neuron, each dot is one spike.
          Pop A (green, top half) and pop B (blue, bottom half) share the panel.
          The dominant population has visibly more dots.
    Right: split A/B bar showing the policy's probability at decision time,
           plus the verdict and rolling accuracy sparkline.
    """
    PAD      = 14
    RAST_W   = 360
    RAST_H   = frame_h - 2 * PAD
    rast_x   = PAD
    rast_y   = PAD
    dec_x    = rast_x + RAST_W + PAD
    dec_w    = frame_w - dec_x - PAD

    img = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    img[:] = _BG

    # ── LEFT: spike raster ──────────────────────────────────────────
    half_h = RAST_H // 2

    # Pop A panel (top)
    ay0 = rast_y
    cv2.rectangle(img, (rast_x, ay0), (rast_x + RAST_W, ay0 + half_h), _DIM, 1)
    _put(img, "Pop A", (rast_x + 4, ay0 + 14), col=_COL_A, scale=0.5, thick=1)

    for (ni, ti) in spike_raster_A:
        sx = rast_x + int(ti / max(CUE_DUR, 1) * RAST_W)
        sy = ay0 + int(ni / N_POP * half_h)
        cv2.circle(img, (int(sx), int(sy)), 2, _COL_A, -1)

    # Pop B panel (bottom)
    by0 = ay0 + half_h
    cv2.rectangle(img, (rast_x, by0), (rast_x + RAST_W, by0 + half_h), _DIM, 1)
    _put(img, "Pop B", (rast_x + 4, by0 + 14), col=_COL_B, scale=0.5, thick=1)

    for (ni, ti) in spike_raster_B:
        sx = rast_x + int(ti / max(CUE_DUR, 1) * RAST_W)
        sy = by0 + int(ni / N_POP * half_h)
        cv2.circle(img, (int(sx), int(sy)), 2, _COL_B, -1)

    # ── RIGHT: decision ─────────────────────────────────────────────
    # A|B split probability bar
    bar_y = rast_y + 30
    bar_h = 32
    fill_a = int(prob_A * dec_w)

    cv2.rectangle(img, (dec_x, bar_y), (dec_x + dec_w, bar_y + bar_h), _DIM, -1)
    if fill_a > 0:
        cv2.rectangle(img, (dec_x, bar_y),
                      (dec_x + fill_a, bar_y + bar_h), _COL_A, -1)
    if fill_a < dec_w:
        cv2.rectangle(img, (dec_x + fill_a, bar_y),
                      (dec_x + dec_w, bar_y + bar_h), _COL_B, -1)
    cv2.rectangle(img, (dec_x, bar_y), (dec_x + dec_w, bar_y + bar_h), (80, 80, 90), 1)

    # A / B end labels inside bar
    _put(img, "A", (dec_x + 4, bar_y + bar_h - 8),
         col=(10, 10, 10), scale=0.55, thick=2)
    _put(img, "B", (dec_x + dec_w - 16, bar_y + bar_h - 8),
         col=(10, 10, 10), scale=0.55, thick=2)

    # percentages above bar ends
    _put(img, f"{prob_A:.0%}",     (dec_x + 2,          bar_y - 12), col=_COL_A, scale=0.4)
    _put(img, f"{1-prob_A:.0%}",   (dec_x + dec_w - 34, bar_y - 12), col=_COL_B, scale=0.4)

    # Verdict
    r_col   = _OK if last_reward > 0 else _FAIL
    verdict = "CORRECT" if last_reward > 0 else "WRONG"
    chose   = "A" if last_action == 0 else "B"
    _put(img, f"chose {chose}  —  {verdict}",
         (dec_x, bar_y + bar_h + 18), col=r_col, scale=0.52, thick=1)

    # Rolling accuracy sparkline
    sp_y0 = bar_y + bar_h + 50
    sp_h  = frame_h - sp_y0 - PAD
    cv2.rectangle(img, (dec_x, sp_y0), (dec_x + dec_w, sp_y0 + sp_h), _DIM, 1)
    _put(img, "accuracy (last 100)", (dec_x, sp_y0 - 10), col=_DIMT, scale=0.36)

    if len(acc_history) > 1:
        accs = np.array(acc_history[-dec_w:], dtype=float)
        pts  = []
        for xi, a in enumerate(accs):
            px = dec_x + xi
            py = sp_y0 + sp_h - int(a * (sp_h - 4)) - 2
            py = int(np.clip(py, sp_y0 + 1, sp_y0 + sp_h - 1))
            pts.append((px, py))
        for i in range(1, len(pts)):
            frac = float(accs[min(i, len(accs) - 1)])
            g    = int(40 + 170 * frac)
            r    = int(40 + 170 * (1.0 - frac))
            cv2.line(img, pts[i-1], pts[i], (20, g, r), 1, cv2.LINE_AA)
        cur_acc = float(np.mean(acc_history[-100:]))
        _put(img, f"{cur_acc:.0%}", (dec_x + dec_w + 4, sp_y0 + sp_h // 2 + 4),
             col=_TEXT, scale=0.42)

    mid_sy = sp_y0 + sp_h // 2
    cv2.line(img, (dec_x, mid_sy), (dec_x + dec_w, mid_sy), _DIMT, 1)
    _put(img, "50%", (dec_x - 30, mid_sy + 4), col=_DIMT, scale=0.3)

    return img


# ===========================================================================
# Environment
# ===========================================================================

class DelayedRecallEnv:
    """
    Delayed evidence-accumulation environment (e-prop paper style).

    Trial structure (in decision steps, each = WAIT_INC sim-steps):
      t = 0 … CUE_DUR-1          : evidence — pop A and pop B spike
      t = CUE_DUR … CUE_DUR+delay-1 : delay — silence
      t = CUE_DUR+delay … +RECALL_DUR-1 : recall signal fires
      At the last recall step the agent's action is evaluated.
    """

    name = "Delayed Recall"

    def __init__(self, delay: int = 200, wait_inc: int = 30):
        self.delay    = delay
        self.wait_inc = wait_inc

        self.num_inputs  = 3 * N_POP   # pop A | pop B | recall
        self.num_outputs = 2            # 0=A, 1=B

        self.total_steps = CUE_DUR + delay + RECALL_DUR

        # trial state
        self.target      = 0
        self.t           = 0
        self.done        = False
        self.last_action = 0
        self.last_reward = 0.0

        # raster storage for renderer (evidence phase only)
        self._raster_A: list = []
        self._raster_B: list = []

        # render data
        self._prob_A      : float       = 0.5
        self._acc_history : list[float] = []

    # ------------------------------------------------------------------
    def reset(self):
        self.target      = np.random.randint(0, 2)
        self.t           = 0
        self.done        = False
        self.last_action = 0
        self.last_reward = 0.0
        self._raster_A   = []
        self._raster_B   = []
        return self._obs()

    # ------------------------------------------------------------------
    @property
    def phase(self) -> str:
        if self.t < CUE_DUR:
            return "evidence"
        if self.t < CUE_DUR + self.delay:
            return "delay"
        return "recall"

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.num_inputs, dtype=np.float32)
        ph  = self.phase

        if ph == "evidence":
            # target pop fires at HIGH_RATE, distractor at LOW_RATE
            if self.target == 0:
                obs[0:N_POP]     = HIGH_RATE   # pop A is target
                obs[N_POP:2*N_POP] = LOW_RATE
            else:
                obs[0:N_POP]     = LOW_RATE
                obs[N_POP:2*N_POP] = HIGH_RATE  # pop B is target

        # delay phase: all zeros
        return obs

    # ------------------------------------------------------------------
    def step(self, action: int):
        self.last_action = action
        self.last_reward = 0.0

        self.t += 1
        obs = self._obs()

        # Record spikes for raster (evidence phase, 0-indexed step before increment)
        t_prev = self.t - 1
        if t_prev < CUE_DUR:
            # we'll let the renderer sample these from the obs rates
            # store the *step index* so render knows when during evidence
            for ni in range(N_POP):
                if np.random.rand() < HIGH_RATE if (
                    (self.target == 0 and ni < N_POP) or
                    (self.target == 1 and ni >= N_POP)
                ) else LOW_RATE:
                    pass  # handled by make_rate_coded_spikes
            # store a lightweight raster from the previous obs
            # (we re-sample here just for display — doesn't affect training)
            prev_rates_A = HIGH_RATE if self.target == 0 else LOW_RATE
            prev_rates_B = HIGH_RATE if self.target == 1 else LOW_RATE
            for ni in range(N_POP):
                if np.random.rand() < prev_rates_A:
                    self._raster_A.append((ni, t_prev))
                if np.random.rand() < prev_rates_B:
                    self._raster_B.append((ni, t_prev))

        # Evaluate action on the last recall step
        if self.t >= self.total_steps:
            correct          = int(action == self.target)
            self.last_reward = 1.0 if correct else -1.0
            self.done        = True

        return obs, self.last_reward, self.done, {
            "correct": (self.last_reward > 0),
            "target" : self.target,
        }

    # ------------------------------------------------------------------
    def push_prob_correct(self, probs: np.ndarray):
        """Feed policy softmax; store P(A) for render."""
        self._prob_A = float(probs[0])

    def push_rolling_acc(self, acc: float):
        self._acc_history.append(acc)

    # ------------------------------------------------------------------
    def render_trial_frame(self) -> np.ndarray:
        """Call once at end of trial to get the single summary image."""
        return render_trial(
            target         = self.target,
            spike_raster_A = self._raster_A,
            spike_raster_B = self._raster_B,
            last_action    = self.last_action,
            last_reward    = self.last_reward,
            prob_A         = self._prob_A,
            acc_history    = self._acc_history,
        )


# ===========================================================================
# Network
# ===========================================================================

def build_network(env: DelayedRecallEnv, cfg: dict):
    INPUT_SIZE  = env.num_inputs
    INPUT_SHAPE = (INPUT_SIZE, 1, 1)
    NUM_OUTPUT  = env.num_outputs
    WAIT_INC    = cfg['wait_inc']

    H_E = cfg['hidden_e_shape']
    H_I = cfg['hidden_i_shape']

    network = Network(default_params)
    with network:

        input_pop = Population(
            SpikeInput(max_spikes=INPUT_SIZE * WAIT_INC), INPUT_SHAPE
        )

        tau_adapt = cfg.get('tau_adapt', 2000)

        hidden_E     = Population(AdaptiveLeakyIntegrateFire(
            v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=tau_adapt), H_E)
        hidden_I     = Population(AdaptiveLeakyIntegrateFire(
            v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=tau_adapt), H_I)
        policy_field = Population(AdaptiveLeakyIntegrateFire(
            v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=tau_adapt), H_I)
        value_field  = Population(AdaptiveLeakyIntegrateFire(
            v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=tau_adapt), H_I)

        policy = Population(
            LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), NUM_OUTPUT)
        value  = Population(
            LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 1)

        # ---- connectivity helpers ----
        def fp(p, mean=0.0, sd=0.1):
            return FixedProbability(p, Normal(mean=mean, sd=sd))

        inp_sd = 1.0 / math.sqrt(max(int(0.5 * INPUT_SIZE), 1))
        for layer in [hidden_E, hidden_I]:
            Connection(input_pop, layer, fp(0.5, sd=inp_sd), exc_inh_sign=1)

        e_sd = 1.0 / math.sqrt(max(int(0.15 * np.prod(H_E)), 1))
        for layer in [hidden_I, hidden_E, policy_field, value_field]:
            Connection(hidden_E, layer, fp(0.15, sd=e_sd), exc_inh_sign=1)

        i_sd = 1.0 / math.sqrt(max(int(0.15 * np.prod(H_I)), 1))
        for layer in [hidden_I, hidden_E, policy_field, value_field]:
            Connection(hidden_I, layer, fp(0.15, sd=i_sd), exc_inh_sign=-1)

        out_sd = 1.0 / math.sqrt(NUM_OUTPUT)
        Connection(policy_field, policy,
                   FixedProbability(0.99999, Normal(sd=out_sd)), exc_inh_sign=None)
        Connection(value_field,  value,
                   FixedProbability(0.99999, Normal(sd=1.0)), exc_inh_sign=None)

        Connection(policy_field, policy,
                   FixedProbability(0.99999, Normal(sd=out_sd)),
                   feedback_name="policy_feedback", exc_inh_sign=None)
        Connection(value_field, value,
                   FixedProbability(0.99999, Normal(sd=1.0)),
                   feedback_name="value_feedback", exc_inh_sign=None)

        Connection(policy, value, Dense(weight=1.0), feedback_name="tde_transport")
        for hl in [hidden_E, hidden_I, policy_field, value_field]:
            Connection(hl, value, Dense(weight=1.0), feedback_name="tde_transport")
            Connection(hl, value, Dense(weight=1.0), feedback_name="policy_reward")

        for hl, sign in [(hidden_E, 1), (hidden_I, -1)]:
            for fb in ["policy_feedback", "policy_regularisation"]:
                Connection(hl, policy, fp(0.5, sd=out_sd),
                           feedback_name=fb, exc_inh_sign=sign)
            for fb in ["value_feedback", "value_regularisation"]:
                Connection(hl, value, fp(0.5, sd=1.0),
                           feedback_name=fb, exc_inh_sign=sign)

    compiler = EPropCompiler(
        example_timesteps     = 1,
        losses                = {policy: "mean_square_error",
                                 value:  "mean_square_error"},
        optimiser             = Adam(cfg['lr']),
        batch_size            = 1,
        kernel_profiling      = False,
        feedback_type         = "random",
        reward_decay          = cfg['reward_decay'],
        gamma                 = cfg['gamma'],
        td_lambda             = cfg['td_lambda'],
        train_output_bias     = False,
        reset_time_between_batches = False,
        entropy_coeff         = cfg['entropy_coeff'],
        entropy_coeff_decay   = cfg['entropy_coeff_decay'],
        entropy_coeff_min     = cfg['entropy_coeff_min'],
        policy_heads          = {policy: PolicyTypes.GENERIC},
        value_head            = value,
    )

    serialiser = Numpy("checkpoints_recall")

    if cfg.get('checkpoint_load'):
        network.load((cfg['checkpoint_load'],), serialiser)

    compiled_net = compiler.compile(network)
    return compiled_net, network, input_pop, policy, value, serialiser


# ===========================================================================
# Training loop
# ===========================================================================

def train(env: DelayedRecallEnv, cfg: dict, episodes: int = int(1e6)):
    WAIT_INC   = cfg['wait_inc']
    INPUT_SIZE = env.num_inputs

    compiled_net, network, input_pop, policy, value, serialiser = \
        build_network(env, cfg)

    train_cb = CallbackList(
        [*set(compiled_net.base_train_callbacks)],
        compiled_network=compiled_net,
        num_batches=1, num_epochs=1,
    )
    all_metrics = {}

    viz         = PerformanceVisualizer(window=100)
    opt_updt    = 0
    win_accs    : list[float] = []
    smoothing   = 0.99
    avg_reward  = 0.0

    print(f"\n{'='*65}")
    print(f"  Delayed Recall  |  delay={env.delay} steps  wait={WAIT_INC}")
    print(f"  Evidence={CUE_DUR} steps  Recall={RECALL_DUR} steps")
    print(f"  High rate={HIGH_RATE}  Low rate={LOW_RATE}")
    print(f"  gamma={cfg['gamma']:.6f}  td_lambda={cfg['td_lambda']:.6f}")
    print(f"{'='*65}\n")

    with compiled_net:
        train_cb.on_epoch_begin(0)
        train_cb.on_batch_begin(0)

        for ep in range(episodes):
            obs          = env.reset()
            done         = False
            total_reward = 0.0
            correct      = False
            reward_trace = 0.0
            frames       = []
            probs_hist        = []
            values_hist       = []
            reward_trace_hist = []

            spikes = make_rate_coded_spikes(
                obs, compiled_net.genn_model.timestep, INPUT_SIZE, K=WAIT_INC)
            compiled_net.set_input({input_pop: [spikes]})

            while not done:
                # Run WAIT_INC sim-steps
                compiled_net.step_time(train_cb)

                # Policy + value readout
                logits  = compiled_net.get_readout(policy).flatten()
                shifted = logits - logits.max()
                exp_l   = np.exp(shifted)
                probs   = exp_l / (exp_l.sum() + 1e-8)
                probs_hist.append(probs.copy())

                v_est = float(compiled_net.get_readout(value)[0].mean())
                values_hist.append(v_est)
                reward_trace_hist.append(reward_trace)

                # Feed P(correct) to env for rendering
                env.push_prob_correct(probs)

                action_idx = np.random.choice(env.num_outputs, p=probs)

                # Policy gradient
                y_true             = np.zeros(env.num_outputs, dtype=np.float32)
                y_true[action_idx] = 1.0
                PG                 = (probs - y_true).astype(np.float32)
                log_p              = np.log(probs + 1e-8)
                entropy            = -float(np.sum(probs * log_p))
                E                  = (cfg['entropy_coeff']
                                      * (-probs * (log_p + entropy))).astype(np.float32)

                pop = compiled_net.neuron_populations[policy]
                pop.vars["pre_PG"].view[:] = PG
                pop.push_var_to_device("pre_PG")
                pop.vars["pre_E"].view[:] = E
                pop.push_var_to_device("pre_E")

                obs, reward, done, info = env.step(action_idx)
                total_reward += reward
                reward_trace  = reward_trace * cfg['reward_decay'] + reward
                correct       = info.get("correct", False)

                if reward != 0.0:
                    compiled_net.losses[value].set_var(
                        compiled_net.neuron_populations[value],
                        "reward", float(reward)
                    )

                # Single frame captured at end of trial only
                if done:
                    frames.append(env.render_trial_frame())


            for _ in range(WAIT_INC * 5):
                reward_trace *= cfg['reward_decay']
                compiled_net.step_time(train_cb)

            compiled_net.genn_model.custom_update("GradientLearn")
            for o, custom_updates in compiled_net.optimisers:
                for c in custom_updates:
                    o.set_step(c, opt_updt)
                    opt_updt += 1

            # Stats
            win_accs.append(float(correct))
            if len(win_accs) > 100:
                win_accs.pop(0)
            rolling_acc = float(np.mean(win_accs))
            env.push_rolling_acc(rolling_acc)

            avg_reward = (smoothing * avg_reward + (1 - smoothing) * total_reward
                          if avg_reward != 0 else total_reward)

            viz.push_metrics(ep=ep, reward=total_reward,
                             accuracy=float(correct), grad_norm=0.0)
            viz.push_metrics(
                values=values_hist,
                reward_trace=reward_trace_hist,
                probs=probs_hist,
            )
            if frames:
                viz.push_best_sequence(frames)

            if ep % 50 == 0:
                print(
                    f"Ep {ep+1:7d}  |  "
                    f"target={'A' if env.target==0 else 'B'}  "
                    f"out={'A' if env.last_action==0 else 'B'}  "
                    f"{'✓' if correct else '✗'}  |  "
                    f"roll_acc(100)={rolling_acc:.3f}  "
                    f"avg_rew={avg_reward:+.3f}"
                )

            if (ep + 1) % 2000 == 0:
                tag = f"recall_delay{env.delay}_ep{ep+1}"
                compiled_net.save((tag,), serialiser)
                compiled_net.save_connectivity((tag,), serialiser)
                print(f"  → Checkpoint: {tag}")

    viz.stop()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delayed Recall — e-prop evidence accumulation task"
    )
    parser.add_argument("--delay",     type=int,   default=200,
                        help="Delay length in decision steps (default 200)")
    parser.add_argument("--wait",      type=int,   default=30,
                        help="Sim timesteps per decision step (default 30)")
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--tau_adapt", type=int,   default=2000,
                        help="Adaptation time constant ms (default 2000)")
    parser.add_argument("--episodes",  type=int,   default=int(1e6))
    parser.add_argument("--load",      default=None,
                        help="Checkpoint tag to resume from")
    parser.add_argument("--hidden_e",  type=int,   default=128)
    parser.add_argument("--hidden_i",  type=int,   default=64)
    args = parser.parse_args()

    temporal = compute_temporal_cfg(args.delay, args.wait)

    def _shape(n):
        s = int(math.isqrt(n))
        while s * s < n:
            s += 1
        return (s, s, 1)

    cfg = {
        **temporal,
        "wait_inc"        : args.wait,
        "lr"              : args.lr,
        "tau_adapt"       : args.tau_adapt,
        "hidden_e_shape"  : _shape(args.hidden_e),
        "hidden_i_shape"  : _shape(args.hidden_i),
        "checkpoint_load" : args.load,
    }

    print(f"Temporal cfg for delay={args.delay}:")
    for k in ("gamma", "td_lambda", "reward_decay", "entropy_coeff"):
        print(f"  {k:30s} = {cfg[k]:.8f}")

    env = DelayedRecallEnv(delay=args.delay, wait_inc=args.wait)
    train(env, cfg, episodes=args.episodes)