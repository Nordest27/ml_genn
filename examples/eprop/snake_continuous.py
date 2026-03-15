##################### SNAKE ENV #####################
#####################################################
from pygenn import SynapseMatrixConnectivity
import multiprocessing as mp
from multiprocessing import Manager, Queue
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from ml_genn.metrics.metric import Metric
from ml_genn.metrics import default_metrics
from ml_genn.callbacks.custom_update import CustomUpdateOnBatchEnd
from ml_genn.utils.module import get_object_mapping
from ml_genn.utils.data import preprocess_spikes
from ml_genn.utils.callback_list import CallbackList
import random
from line_profiler import profile

from ml_genn import InputLayer, Layer, Network, Population, Connection
from ml_genn.callbacks import Checkpoint
from ml_genn.compilers import EPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense, FixedProbability, Conv2D, ToroidalGaussian2D
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, AdaptiveLeakyIntegrateFire, SpikeInput
from ml_genn.serialisers import Numpy
from ml_genn.optimisers import Adam
from scipy.ndimage import gaussian_filter

from ml_genn.compilers.eprop_compiler import default_params
from collections import defaultdict, deque


# ============================================================
#  CONTINUOUS ACTION HEAD  (threshold-queue version)
#
#  Policy outputs 2 neurons whose membrane potentials are read directly:
#    neuron 0 = horizontal   value > +threshold → enqueue RIGHT (2)
#                            value < -threshold → enqueue LEFT  (0)
#    neuron 1 = vertical     value > +threshold → enqueue UP    (1)
#                            value < -threshold → enqueue DOWN  (3)
#
#  The SRNN neuron itself is the integrator — no external accumulator.
#  We just check the readout each timestep and push to a discrete queue.
#  The env pops one action per move from the front of the deque.
# ============================================================


class ThresholdQueue:
    """
    Edge-triggered threshold checker — enqueues a discrete action only when
    a neuron readout CROSSES the threshold upward (i.e. was below last step,
    is at-or-above this step).  Holding above the threshold does not re-fire.

    Action encoding (matches SnakeEnv.step):
        0 = left   1 = up   2 = right   3 = down
    """
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.queue: deque = deque()
        self._values = np.zeros(NUM_OUTPUT)

    def reset(self):
        self.queue.clear()

    def check(self, values):
        """Detect rising-edge threshold crossings and enqueue actions."""
        # Horizontal: rising cross of +threshold → RIGHT

        self._values = self._values*(0.5**1/WAIT_INC) + values
        # for i in range(NUM_OUTPUT):
        #     if self._values[i] > self.threshold:
        #         self.queue.append(i)
        self.queue.append(np.argmax(self._values))

        if len(self.queue) > 1:
            self.queue.popleft()

    def pop(self, default_action: int) -> int:
        """Return next queued action, or default if queue is empty."""
        return self.queue.popleft() if self.queue else default_action

def extract_actual_sparse_connections(compiled_net):
    connection_counts = {}
    for c, genn_pop in compiled_net.connection_populations.items():
        if genn_pop.matrix_type & SynapseMatrixConnectivity.SPARSE:
            genn_pop.pull_connectivity_from_device()
            pre_inds = genn_pop.get_sparse_pre_inds()
            connection_counts[c] = len(pre_inds)
    return connection_counts


def extract_fanin_statistics(compiled_net):
    stats = {}
    for c, genn_pop in compiled_net.connection_populations.items():
        if genn_pop.matrix_type & SynapseMatrixConnectivity.SPARSE:
            genn_pop.pull_connectivity_from_device()
            post_inds = genn_pop.get_sparse_post_inds()
            fanin = np.bincount(post_inds)
            stats[c] = {
                "total": len(post_inds),
                "mean_fanin": np.mean(fanin),
                "std_fanin": np.std(fanin),
                "min_fanin": np.min(fanin),
                "max_fanin": np.max(fanin)
            }
    return stats


class SnakeEnv:
    def __init__(self, size=28, visible_range=5, scale=1, wait_inc=5, inp_shape=(5,5,3)):
        assert visible_range % 2 == 1, "visible_range must be odd"
        self.size = size
        self.visible_range = visible_range
        self.scale = scale
        self.wait_inc = wait_inc
        self.won = False
        self.inp_shape = inp_shape
        self.reset()

    def reset(self):
        self.snake = [(self.size // 2, self.size // 2)]
        self.dir_idx = 1
        self.direction = 'up'
        self.apples = []
        self.spawn_apples()
        self.done = False
        self.steps_since_last_apple = 0
        self.wait_count = self.wait_inc
        return self.get_local_img_observation()

    def spawn_apples(self):
        empty_cells = [(i, j) for i in range(self.size)
                       for j in range(self.size) if (i, j) not in self.snake and (i, j) not in self.apples]
        if len(empty_cells) == 0:
            return
        self.apples = [random.choice(empty_cells)]

    def step(self, action):
        if self.done:
            raise Exception("Environment needs reset. Call env.reset().")

        if self.wait_count > 0:
            self.wait_count -= 1
            return self.get_local_img_observation(), 0.0, self.done

        dirs = ['left', 'up', 'right', 'down']
        new_dir = dirs[action]
        reward = -5

        # Prevent 180° turn
        if (self.direction == 'up' and new_dir == 'down') or \
           (self.direction == 'down' and new_dir == 'up') or \
           (self.direction == 'left' and new_dir == 'right') or \
           (self.direction == 'right' and new_dir == 'left'):
            new_dir = self.direction

        self.direction = new_dir

        head_y, head_x = self.snake[0]
        if self.direction == 'up':    head_y -= 1
        elif self.direction == 'down':  head_y += 1
        elif self.direction == 'left':  head_x -= 1
        elif self.direction == 'right': head_x += 1

        new_head = (head_y, head_x)
        apple_found = None
        for apple in self.apples:
            if new_head == apple:
                apple_found = apple
                break

        if (head_y < 0 or head_y >= self.size or
                head_x < 0 or head_x >= self.size or
                new_head in self.snake[:-1]):
            reward -= 100
            self.done = True
            return self.get_local_img_observation(), reward / 100, True

        self.snake.insert(0, new_head)

        if apple_found is not None:
            self.apples.remove(apple_found)
            self.spawn_apples()
            reward += 100.0
            if len(self.apples) == 0:
                self.done = True
                self.won = True
            if len(self.snake) > 0.75 * self.size ** 2 and self.size > 5:
                self.won = True
            self.steps_since_last_apple = 0
        else:
            self.snake.pop()
            self.steps_since_last_apple += 1

        if self.steps_since_last_apple > self.size ** 2:
            self.done = True
            return self.get_local_img_observation(), reward / 100, self.done

        self.wait_count = self.wait_inc
        return self.get_local_img_observation(), reward / 100, self.done

    def get_local_img_observation(self):
        img = self.local_img(scale=self.scale)
        return cv2.resize(img, (self.inp_shape[0], self.inp_shape[1]),
                          interpolation=cv2.INTER_NEAREST)

    def img(self, scale=10):
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8) + 100
        for y, x in self.snake[1:]:
            img[y, x, 1] += 63
        head_y, head_x = self.snake[0]
        img[head_y, head_x, 1] += 31
        for apple in self.apples:
            ay, ax = apple
            img[ay, ax, 2] += 63
        v = self.visible_range
        r = v // 2
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y = head_y + dy
                x = head_x + dx
                if y < 0 or y >= self.size or x < 0 or x >= self.size:
                    continue
                img[y, x] = [0, 0, 0]
                if (y, x) in self.snake[1:]:
                    img[y, x] = [0, 255, 0]
                if (y, x) == (head_y, head_x):
                    img[y, x] = [0, 155, 0]
                for apple in self.apples:
                    if (y, x) == apple:
                        img[y, x] = [0, 0, 255]
        return cv2.resize(img, (self.size * scale, self.size * scale),
                          interpolation=cv2.INTER_NEAREST)

    def local_img(self, scale=10):
        v = self.visible_range
        r = v // 2
        head_y, head_x = self.snake[0]
        img = np.zeros((v, v, 3), dtype=np.uint8)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y = head_y + dy
                x = head_x + dx
                local_y = dy + r
                local_x = dx + r
                if ((x == -1 and -1 <= y < self.size + 1) or
                        (x == self.size and -1 <= y < self.size + 1) or
                        (y == -1 and -1 <= x < self.size + 1) or
                        (y == self.size and -1 <= x < self.size + 1)):
                    img[local_y, local_x] = [100, 100, 100]
                    continue
                if (y, x) in self.snake[1:]:
                    img[local_y, local_x] = [0, 255, 0]
                if (y, x) == (head_y, head_x):
                    img[local_y, local_x] = [0, 155, 0]
                for apple in self.apples:
                    if (y, x) == apple:
                        img[local_y, local_x] = [0, 0, 255]
        return cv2.resize(img, (v * scale, v * scale), interpolation=cv2.INTER_NEAREST)


################### DEFINE MODEL ####################
#####################################################

def expected_toroidal_connections(src_size, dst_size, sigma, desired_fan_in=None, p_max=None):
    if desired_fan_in is None and p_max is None:
        raise ValueError("Provide either desired_fan_in or p_max")
    if desired_fan_in is None:
        desired_fan_in = src_size * 2.0 * np.pi * sigma ** 2 * p_max
    return dst_size * desired_fan_in


CONNECTIVITY_TYPE = "toroidal"
WINDOW_EPISODES = 100
BOARD_SIZE = 2

VISIBLE_RANGE = 5
SCALE = 1
WAIT_INC = 30
INPUT_C = 3
INPUT_SHAPE = (15, 15, INPUT_C)
HIDDEN_E_SHAPE = (25, 25, INPUT_C)
HIDDEN_I_SHAPE = (20, 20, INPUT_C)
INPUT_SIZE = np.prod(INPUT_SHAPE)
NUM_HIDDEN_E = np.prod(HIDDEN_E_SHAPE)
NUM_HIDDEN_I = np.prod(HIDDEN_I_SHAPE)

SIGMA_IN = 0.1
SIGMA_H = 0.05
DESIRED_FAN_IN_IN = 100
DESIRED_FAN_IN_H1 = 100
DESIRED_FAN_IN_H2 = 100
FAN_IN_SCALE_IN = 0.25

# ── Continuous policy: 4 outputs ─────────────────────────────────────────────
ACTION_THRESHOLD = 1.0   # readout value that triggers an enqueue
NUM_OUTPUT = 4   # continuous head

CONN_P = {
    "I-H": 0.001,
    "D-O": 0.001,
    "H-H": 0.005,
    "H-P": 0.1,
    "H-V": 0.1,
    "F":   0.1
}

print("Expected random connections:")
expected_conns = 0
print("- I-H:", aux_conns := INPUT_SIZE * (NUM_HIDDEN_E + NUM_HIDDEN_I) * CONN_P["I-H"])
expected_conns += aux_conns * int(CONNECTIVITY_TYPE == "fixed")
print("- H-H:", aux_conns := ((NUM_HIDDEN_E + NUM_HIDDEN_I) ** 2) * CONN_P["H-H"])
expected_conns += aux_conns * int(CONNECTIVITY_TYPE == "fixed")
print("- H-P:", aux_conns := (NUM_HIDDEN_E + NUM_HIDDEN_I) * NUM_OUTPUT * CONN_P["H-P"])
expected_conns += aux_conns
print("- H-V:", aux_conns := (NUM_HIDDEN_E + NUM_HIDDEN_I) * CONN_P["H-V"])
expected_conns += aux_conns

print("Expected toroidal connections:")
print("- I-H:", aux_conns := expected_toroidal_connections(
    INPUT_SIZE, NUM_HIDDEN_E + NUM_HIDDEN_I, SIGMA_IN, desired_fan_in=DESIRED_FAN_IN_IN))
expected_conns += aux_conns * int(CONNECTIVITY_TYPE == "toroidal")
print("- H-H:", aux_conns := expected_toroidal_connections(
    NUM_HIDDEN_E + NUM_HIDDEN_I, NUM_HIDDEN_E + NUM_HIDDEN_I, SIGMA_H, desired_fan_in=DESIRED_FAN_IN_H1))
expected_conns += aux_conns * int(CONNECTIVITY_TYPE == "toroidal")

TRAIN = True
CHECKPOINT_BOARD_SIZE = None
if CHECKPOINT_BOARD_SIZE is not None:
    CONNECTIVITY_TYPE = "fixed"
KERNEL_PROFILING = False

gamma = 0.1 ** (1 / WAIT_INC)
td_lambda = 0.8 ** (1 / WAIT_INC)
entropy_coeff = 1e-3
entropy_decay = 0.99999 ** (1 / WAIT_INC)
entropy_coeff_min = 1e-5
dale_l1_reg = 0.0

serialiser = Numpy("continuos_snake_checkpoints")
network = Network(default_params)
hidden_layers = {}


def make_connectivity(connectivity_type, src_shape, dst_shape=None, desired_fan_in=None,
                      fan_in_scale=None, p=None, sigma=None, sign=None,
                      mean_scale=0.1, sd_scale=0.05):
    if connectivity_type == "fixed":
        if p is None:
            raise ValueError("Fixed connectivity requires p")
        if sign is None:
            sd_scale = 1.0
        fan_in = p * np.prod(src_shape)
        mean = (sign or 0) * mean_scale / np.sqrt(fan_in)
        sd = sd_scale / np.sqrt(fan_in)
        return FixedProbability(p, Normal(mean=mean, sd=sd))

    elif connectivity_type == "toroidal":
        if sigma is None or desired_fan_in is None:
            raise ValueError("Toroidal connectivity requires sigma and desired_fan_in")
        fan_in = desired_fan_in
        if sign == -1:
            mean_scale *= 3
        elif sign is None:
            sd_scale = 1.0
        mean = (sign or 0) * mean_scale / np.sqrt(fan_in)
        sd = sd_scale / np.sqrt(fan_in)
        return ToroidalGaussian2D(
            sigma=sigma, fan_in=desired_fan_in, fan_in_scale=fan_in_scale,
            weight=Normal(mean=mean, sd=sd))
    else:
        raise ValueError(f"Unknown connectivity_type: {connectivity_type}")


def build_compiled_network(connectivity_type="fixed"):
    global dale_l1_reg
    network = Network(default_params)
    hidden_layers = {}

    with network:
        # ── Populations ──────────────────────────────────────────────────────
        input_pop = Population(SpikeInput(max_spikes=INPUT_SIZE * WAIT_INC), INPUT_SHAPE)

        hidden_layers["E"] = Population(
            AdaptiveLeakyIntegrateFire(v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=300),
            HIDDEN_E_SHAPE)

        hidden_layers["I"] = Population(
            AdaptiveLeakyIntegrateFire(v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=300),
            HIDDEN_I_SHAPE)

        # ── Continuous policy head: 4 outputs (mu_x, log_sig_x, mu_y, log_sig_y) ──
        # Using LeakyIntegrate (linear readout) — no softmax, raw real values
        policy = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), NUM_OUTPUT)

        value = Population(LeakyIntegrate(tau_mem=30.0, bias=0.0, readout="var"), 1)

        # ── Input → Hidden ────────────────────────────────────────────────────
        for layer, dst_shape in [(hidden_layers["I"], HIDDEN_I_SHAPE),
                                  (hidden_layers["E"], HIDDEN_E_SHAPE)]:
            Connection(input_pop, layer,
                       make_connectivity(connectivity_type=connectivity_type,
                                         src_shape=INPUT_SHAPE, dst_shape=dst_shape,
                                         p=CONN_P["I-H"], sigma=SIGMA_IN,
                                         desired_fan_in=DESIRED_FAN_IN_IN,
                                         fan_in_scale=FAN_IN_SCALE_IN, sign=1),
                       exc_inh_sign=1)

        # ── Excitatory ────────────────────────────────────────────────────────
        for layer, dst_shape, prob, c_type in [
            (hidden_layers["I"], HIDDEN_I_SHAPE, CONN_P["H-H"], connectivity_type),
            (hidden_layers["E"], HIDDEN_E_SHAPE, CONN_P["H-H"], connectivity_type),
            (policy, None, CONN_P["H-P"], "fixed"),
            (value,  None, CONN_P["H-V"], "fixed"),
        ]:
            Connection(hidden_layers["E"], layer,
                       make_connectivity(connectivity_type=c_type,
                                         src_shape=HIDDEN_E_SHAPE, dst_shape=dst_shape,
                                         p=prob, sigma=SIGMA_H,
                                         desired_fan_in=DESIRED_FAN_IN_H1, sign=1),
                       exc_inh_sign=1)

        # ── Inhibitory ────────────────────────────────────────────────────────
        for layer, dst_shape, prob, c_type in [
            (hidden_layers["I"], HIDDEN_I_SHAPE, CONN_P["H-H"], connectivity_type),
            (hidden_layers["E"], HIDDEN_E_SHAPE, CONN_P["H-H"], connectivity_type),
            (policy, None, CONN_P["H-P"], "fixed"),
            (value,  None, CONN_P["H-V"], "fixed"),
        ]:
            Connection(hidden_layers["I"], layer,
                       make_connectivity(connectivity_type=c_type,
                                         src_shape=HIDDEN_I_SHAPE, dst_shape=dst_shape,
                                         p=prob, sigma=SIGMA_H,
                                         desired_fan_in=DESIRED_FAN_IN_H2, sign=-1),
                       exc_inh_sign=-1)

        # ── Feedback connections ──────────────────────────────────────────────
        Connection(policy, value, Dense(weight=1.0), feedback_name="tde_transport")
        for hidden_layer in hidden_layers.values():
            Connection(hidden_layer, value, Dense(weight=1.0), feedback_name="tde_transport")
            sign = 1 if hidden_layer is hidden_layers["E"] else -1
            for fb_name in ["policy_feedback", "policy_regularisation"]:
                Connection(hidden_layer, policy,
                           FixedProbability(CONN_P["F"], Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
                           feedback_name=fb_name, exc_inh_sign=sign)
            for fb_name in ["value_feedback", "value_regularisation"]:
                Connection(hidden_layer, value,
                           FixedProbability(CONN_P["F"], Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
                           feedback_name=fb_name, exc_inh_sign=sign)

    # ── Compiler ──────────────────────────────────────────────────────────────
    dale_l1_reg = 0.0001 / np.sqrt(DESIRED_FAN_IN_IN) if CONNECTIVITY_TYPE == "toroidal" \
        else 0.01 / np.sqrt(max(INPUT_SIZE, NUM_HIDDEN_E))
    dale_l1_reg = 0
    print("L1 reg strength:", dale_l1_reg)

    # NOTE: The continuous policy uses mean_square_error as a surrogate loss;
    # the actual policy gradient signal is injected via "tdError" as before.
    compiler = EPropCompiler(
        example_timesteps=1,
        losses={
            policy: "mean_square_error",
            value: "mean_square_error"
        },
        optimiser=Adam(1e-5),
        batch_size=1,
        kernel_profiling=KERNEL_PROFILING,
        feedback_type="random",
        gamma=gamma,
        td_lambda=td_lambda,
        train_output_bias=False,
        reset_time_between_batches=False,
        entropy_coeff=entropy_coeff,
        entropy_coeff_decay=entropy_decay,
        entropy_coeff_min=entropy_coeff_min,
        dale_rewiring_l1_strength=dale_l1_reg,
        policy_heads=policy,
        value_head=value
    )

    if CHECKPOINT_BOARD_SIZE is not None:
        network.load((CHECKPOINT_BOARD_SIZE,), serialiser)

    compiled_net = compiler.compile(network)
    return compiled_net, network, input_pop, hidden_layers, policy, value


compiled_net, network, input_pop, hidden_layers, policy, value = \
    build_compiled_network(connectivity_type=CONNECTIVITY_TYPE)

# ── Metrics ───────────────────────────────────────────────────────────────────
policy_train_metrics = get_object_mapping(
    "mean_square_error", [policy], Metric, "Metric", default_metrics)
value_train_metrics = get_object_mapping(
    "mean_square_error", [value], Metric, "Metric", default_metrics)

train_callback_list = CallbackList(
    [*set(compiled_net.base_train_callbacks)] + [Checkpoint(serialiser)],
    compiled_network=compiled_net, num_batches=1, num_epochs=1)

all_metrics = {}
all_metrics.update(policy_train_metrics)
all_metrics.update(value_train_metrics)


####################### ENCODE SPIKES ######################
############################################################

def make_rate_coded_spikes(values, base_timestep, input_size, K):
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
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), input_size)
    return preprocess_spikes(np.concatenate(times), np.concatenate(idxs), input_size)


####################### VISUALIZERS #######################
############################################################

def encode_frame(frame, compress=True, ext='.png', quality=80):
    if not compress:
        return frame
    ret, buf = cv2.imencode(ext, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ret else frame


def decode_frame(blob_or_array):
    if isinstance(blob_or_array, bytes):
        arr = np.frombuffer(blob_or_array, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return blob_or_array


def viz_plots_loop(metrics_q: Queue, stop_event: mp.Event):
    """
    Live matplotlib plot with:
      ax1   – total reward + running average (all episodes)
      ax1b  – recent reward window
      ax2   – value & reward trace (best run)
      ax3   – continuous x-axis action signal (best run)  ← NEW
      ax4   – continuous y-axis action signal (best run)  ← NEW
    """
    plt.ion()
    fig, (ax1, ax1b, ax2, ax3, ax4) = plt.subplots(5, 1, figsize=(10, 16))
    plt.subplots_adjust(hspace=0.55)

    # ── ax1: reward history ───────────────────────────────────────────────────
    snake_len_line, = ax1.plot([], [], label='Snake length', color=(0, 0.8, 0.1, 0.5), zorder=1)
    line_reward,    = ax1.plot([], [], label='Total Reward', color=(0, 0.1, 0.8, 0.5), zorder=2)
    avg_line,       = ax1.plot([], [], label='Running Avg',  color=(0.8, 0.05, 0.05, 1.0), zorder=3)
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress'); ax1.legend()

    # ── ax1b: recent window ───────────────────────────────────────────────────
    line_reward_recent, = ax1b.plot([], [], label=f'Reward (last {WINDOW_EPISODES})')
    avg_line_recent,    = ax1b.plot([], [], label=f'Avg (last {WINDOW_EPISODES})')
    ax1b.set_title(f'Recent Training (last {WINDOW_EPISODES} episodes)'); ax1b.legend()

    # ── ax2: value trace (best run) ───────────────────────────────────────────
    value_line,        = ax2.plot([], [], label='Value')
    reward_trace_line, = ax2.plot([], [], label='Reward Trace')
    ax2.set_title('Value (best run)'); ax2.set_xlabel('Frame')
    ax2.set_ylabel('Value'); ax2.legend()


    # ── ax3: horizontal neuron readout (best run) ─────────────────────────────
    h_line, = ax3.plot([], [], label='horizontal neuron', color='steelblue')
    ax3.axhline( ACTION_THRESHOLD, color='red',  linestyle=':', linewidth=0.8, label='+threshold → RIGHT')
    ax3.axhline(0, color='red',  linestyle=':', linewidth=0.8, label='−threshold → LEFT')
    ax3.axhline(0,                 color='grey', linestyle='-', linewidth=0.4)
    ax3.set_title('Horizontal neuron readout (best run) — threshold fires RIGHT / LEFT')
    ax3.set_xlabel('Decision step'); ax3.set_ylabel('readout'); ax3.legend(fontsize=7)

    # ── ax4: vertical neuron readout (best run) ───────────────────────────────
    v_line, = ax4.plot([], [], label='vertical neuron', color='seagreen')
    ax4.axhline( ACTION_THRESHOLD, color='red',  linestyle=':', linewidth=0.8, label='+threshold → UP')
    ax4.axhline(0, color='red',  linestyle=':', linewidth=0.8, label='−threshold → DOWN')
    ax4.axhline(0,                 color='grey', linestyle='-', linewidth=0.4)
    ax4.set_title('Vertical neuron readout (best run) — threshold fires UP / DOWN')
    ax4.set_xlabel('Decision step'); ax4.set_ylabel('readout'); ax4.legend(fontsize=7)

    # local buffers
    ep_list, rewards, avgs, lens = [], [], [], []
    best_values, best_reward_traces = [], []
    best_h, best_v = [], []   # continuous signal histories for best run

    last_plot_time = 0.0
    plot_interval = 0.2

    while not stop_event.is_set():
        try:
            while True:
                metrics = metrics_q.get_nowait()
                ep = metrics.get('ep')
                if ep is not None:
                    ep_list.append(ep)
                    rewards.append(metrics.get('reward', 0.0))
                    avgs.append(metrics.get('running_avg', 0.0))
                    lens.append(metrics.get('snake_len', 0))
                if 'best_values' in metrics:
                    best_values        = metrics['best_values']
                    best_reward_traces = metrics['best_reward_traces']
                if 'best_continuous' in metrics:
                    bc     = metrics['best_continuous']
                    best_h = bc['h']
                    best_v = bc['v']
        except Exception:
            pass

        now = time.time()
        if now - last_plot_time > plot_interval:
            if ep_list:
                line_reward.set_data(ep_list, rewards)
                avg_line.set_data(ep_list, avgs)
                snake_len_line.set_data(ep_list, lens)
                ax1.relim(); ax1.autoscale_view()

            if rewards:
                rr = rewards[-WINDOW_EPISODES:]
                ra = avgs[-WINDOW_EPISODES:]
                axis_r = np.arange(len(rr))
                line_reward_recent.set_data(axis_r, rr)
                avg_line_recent.set_data(axis_r, ra)
                ax1b.relim(); ax1b.autoscale_view()

            if best_values:
                value_line.set_data(range(len(best_values)), best_values)
                reward_trace_line.set_data(range(len(best_reward_traces)), best_reward_traces)
                ax2.relim(); ax2.autoscale_view()

            if best_h:
                t = list(range(len(best_h)))
                h_line.set_data(t, best_h)
                ax3.relim(); ax3.autoscale_view()

            if best_v:
                t = list(range(len(best_v)))
                v_line.set_data(t, best_v)
                ax4.relim(); ax4.autoscale_view()

            plt.pause(0.001)
            last_plot_time = now

        time.sleep(0.03)

    plt.close(fig)


def viz_runs_loop(best_run_q: Queue, random_run_q: Queue, stop_event: mp.Event):
    best_run, random_run = [], []
    window_best   = "Best Run"
    window_random = "Random Run"
    cv2.namedWindow(window_best,   flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_random, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(window_best, 600, 600)
    cv2.resizeWindow(window_random, 600, 600)
    t = 0
    while not stop_event.is_set():
        try:
            while True:
                br = best_run_q.get_nowait()
                if br is None: stop_event.set(); break
                best_run = br
        except Exception: pass
        try:
            while True:
                rr = random_run_q.get_nowait()
                if rr is None: stop_event.set(); break
                random_run = rr
        except Exception: pass

        if best_run:
            f = decode_frame(best_run[t % len(best_run)])
            if f is not None: cv2.imshow(window_best, f)
            if cv2.waitKey(1) == 27: stop_event.set(); break

        if random_run and t % (len(random_run) + 10) < len(random_run):
            f = decode_frame(random_run[t % (len(random_run) + 10)])
            if f is not None: cv2.imshow(window_random, f)
            if cv2.waitKey(1) == 27: stop_event.set(); break

        time.sleep(0.1)
        t += 1

    cv2.destroyWindow(window_best)
    cv2.destroyWindow(window_random)


def start_visualizers():
    manager = Manager()
    metrics_q    = manager.Queue(maxsize=10)
    best_run_q   = manager.Queue(maxsize=2)
    random_run_q = manager.Queue(maxsize=2)
    stop_event   = manager.Event()

    p_plots = mp.Process(target=viz_plots_loop, args=(metrics_q, stop_event), daemon=True)
    p_runs  = mp.Process(target=viz_runs_loop,  args=(best_run_q, random_run_q, stop_event), daemon=True)
    p_plots.start()
    p_runs.start()
    return manager, metrics_q, best_run_q, random_run_q, stop_event, p_plots, p_runs


####################### TRAIN ###########################
#########################################################

def train_snake_agent_with_ipc(episodes=100000,
                               metrics_q: Queue = None,
                               best_run_q: Queue = None,
                               random_run_q: Queue = None,
                               compress_frames=True,
                               compress_quality=80):
    global BOARD_SIZE
    opt_updt = 0

    with compiled_net:
        env = SnakeEnv(size=BOARD_SIZE, visible_range=VISIBLE_RANGE,
                       wait_inc=WAIT_INC, scale=SCALE, inp_shape=INPUT_SHAPE)
        best_reward = -np.inf
        best_run    = []
        running_avg = []
        snake_len_history = []
        smoothing = 0.95
        avg = 0

        last_best_values            = []
        last_best_reward_traces     = []
        last_best_continuous        = None   # dict of lists for x/y signals

        # ── Threshold queue (persistent across episodes) ──────────────────────
        tq = ThresholdQueue(threshold=ACTION_THRESHOLD)

        train_callback_list.on_epoch_begin(0)
        train_callback_list.on_batch_begin(0)

        for ep in range(episodes):

            # ── Board size increase on win ────────────────────────────────────
            if env.won:
                compiled_net.save_connectivity((BOARD_SIZE,), serialiser)
                compiled_net.save((BOARD_SIZE,), serialiser)
                best_reward = -np.inf
                BOARD_SIZE += 1
                print(f"WON! Increasing board size to {BOARD_SIZE}")
                env = SnakeEnv(size=BOARD_SIZE, visible_range=VISIBLE_RANGE,
                               wait_inc=WAIT_INC, scale=SCALE, inp_shape=INPUT_SHAPE)

            # ── Periodic diagnostics ──────────────────────────────────────────
            if ep % 1000 == 0:
                print("/////////////////////////////")
                connections_sum = 0
                print("Actual instantiated connections:")
                for k, v in extract_actual_sparse_connections(compiled_net).items():
                    connections_sum += v
                    print(f"- {k}: {v}")
                print("Total connections:", connections_sum)
                print("/////////////////////////////")

            if (ep + 1) % 1000 == 0:
                best_reward = -np.inf
                compiled_net.save_connectivity((f"{BOARD_SIZE}_mid_completion",), serialiser)
                compiled_net.save((f"{BOARD_SIZE}_mid_completion",), serialiser)

            for m in all_metrics.values():
                m.reset()

            obs   = env.reset()
            tq.reset()   # reset threshold accumulator each episode
            done  = False
            total_reward  = 0
            current_run   = []
            current_values        = []
            current_reward_traces = []

            # ── Continuous signal history for this episode ────────────────────
            ep_h_signal = []   # horizontal neuron readout, logged at each decision step
            ep_v_signal = []   # vertical   neuron readout, logged at each decision step

            reward_trace = 0
            frame = 1
            dir_map = {'left': 0, 'up': 1, 'right': 2, 'down': 3}

            # ── Initial spike input ───────────────────────────────────────────
            spikes = make_rate_coded_spikes(
                obs.reshape(-1), compiled_net.genn_model.timestep, INPUT_SIZE, K=WAIT_INC)
            compiled_net.set_input({input_pop: [spikes]})
            env.wait_count = WAIT_INC

            while not done:
                current_values.append(compiled_net.get_readout(value)[0][0])
                current_reward_traces.append(reward_trace)

                # ── Every timestep: read neuron, push to action queue if threshold crossed ──
                compiled_net.neuron_populations[policy].vars["Action"].pull_from_device()
                values = compiled_net.neuron_populations[policy].vars["Action"].view
                tq.check(values)         # may push discrete actions onto tq.queue
                ep_h_signal.append(tq._values[0])
                ep_v_signal.append(tq._values[1])

                # ── Decision step: consume one queued action ───────────────────
                action_label = None
                if env.wait_count == 0:
                    # Pop next committed action; fall back to current direction
                    action_label = tq.pop(default_action=dir_map[env.direction])

                obs, reward, done = env.step(action_label)
                total_reward  += reward
                reward_trace   = reward_trace * 0.9261 + reward

                if reward != 0:
                    compiled_net.losses[policy].set_var(
                        compiled_net.neuron_populations[policy], "reward", reward)

                if env.wait_count == env.wait_inc:
                    frame_img = obs
                    frame_img = env.img()
                    if compress_frames:
                        current_run.append(encode_frame(frame_img, compress=True,
                                                        quality=compress_quality))
                    else:
                        current_run.append(frame_img.copy())

                    spikes = make_rate_coded_spikes(
                        obs.reshape(-1), compiled_net.genn_model.timestep, INPUT_SIZE, K=WAIT_INC)
                    compiled_net.set_input({input_pop: [spikes]})

                compiled_net.step_time(train_callback_list)
                compiled_net.genn_model.custom_update("GradientLearn")
                for o, custom_updates in compiled_net.optimisers:
                    for c in custom_updates:
                        o.set_step(c, opt_updt := opt_updt + 1)

                frame += 1

            # ── Post-episode decay steps ──────────────────────────────────────
            spikes = make_rate_coded_spikes(
                obs.reshape(-1), compiled_net.genn_model.timestep, INPUT_SIZE, K=WAIT_INC)
            compiled_net.set_input({input_pop: [spikes]})
            for i in range(WAIT_INC):
                current_values.append(compiled_net.get_readout(value)[0][0])
                compiled_net.neuron_populations[policy].vars["Action"].pull_from_device()
                compiled_net.neuron_populations[policy].vars["Action"].pull_from_device()
                values = compiled_net.neuron_populations[policy].vars["Action"].view
                tq.check(values)
                ep_h_signal.append(tq._values[0])
                ep_v_signal.append(tq._values[1])
                current_reward_traces.append(reward_trace)
                
                reward_trace = reward_trace * 0.9261
                compiled_net.step_time(train_callback_list)
                compiled_net.genn_model.custom_update("GradientLearn")
                for o, custom_updates in compiled_net.optimisers:
                    for c in custom_updates:
                        o.set_step(c, opt_updt := opt_updt + 1)

            if dale_l1_reg > 0:
                compiled_net.genn_model.custom_update("DaleRL1")
            compiled_net.genn_model.custom_update("DalePrune")
            compiled_net.genn_model.custom_update("DaleRewire")

            # ── Track best run ────────────────────────────────────────────────
            if total_reward >= best_reward:
                best_reward         = total_reward
                best_run            = list(current_run)
                last_best_values        = list(current_values)
                last_best_reward_traces = list(current_reward_traces)
                last_best_continuous    = {
                    'h': list(ep_h_signal),   # horizontal neuron readout per decision step
                    'v': list(ep_v_signal),   # vertical   neuron readout per decision step
                }

                try:
                    if best_run_q is not None:
                        if best_run_q.full():
                            try: best_run_q.get_nowait()
                            except: pass
                        best_run_q.put_nowait(best_run)
                except Exception as e:
                    print("Best-run enqueue error:", e)

            if random_run_q is not None and ep % 100 == 0:
                try:
                    if random_run_q.full():
                        try: random_run_q.get_nowait()
                        except: pass
                    random_run_q.put_nowait(current_run)
                except Exception as e:
                    print("Random-run enqueue error:", e)

            # ── Metrics ───────────────────────────────────────────────────────
            snake_len_history.append(len(env.snake) - 1)
            if len(snake_len_history) > 100:
                snake_len_history = snake_len_history[-100:]

            avg = total_reward if avg == 0 else smoothing * avg + (1 - smoothing) * total_reward
            running_avg.append(avg)

            if metrics_q is not None:
                metrics_payload = {
                    'ep': ep,
                    'reward': total_reward,
                    'running_avg': avg,
                    'snake_len': len(env.snake) - 1
                }
                if last_best_values:
                    metrics_payload['best_values']        = last_best_values
                    metrics_payload['best_reward_traces'] = last_best_reward_traces
                if last_best_continuous is not None:
                    metrics_payload['best_continuous'] = last_best_continuous

                try:
                    if metrics_q.full():
                        try: metrics_q.get_nowait()
                        except: pass
                    metrics_q.put_nowait(metrics_payload)
                except Exception as e:
                    print("Metrics enqueue error:", e)

            if ep % 10 == 0:
                print(
                    f"Episode {ep+1} | "
                    f"Reward: {' ' if total_reward >= 0 else ''}{total_reward:.2f} | "
                    f"Best: {best_reward:.2f} | "
                    f"Len: {len(env.snake)-1:2d} | "
                    f"Avg-100: {np.mean(snake_len_history):.2f} | "
                    f"Frame: {frame} | "
                    f"Queue depth: {len(tq.queue)} | "
                    f"α={compiled_net.optimisers[0][0].alpha:.8f}"
                )

        # ── Sentinel to stop visualizers ──────────────────────────────────────
        for q in [best_run_q, random_run_q]:
            if q is not None:
                try: q.put_nowait(None)
                except: pass


####################### MAIN ############################
#########################################################

if __name__ == "__main__":
    manager, metrics_q, best_run_q, random_run_q, stop_event, p_plots, p_runs = start_visualizers()

    try:
        train_snake_agent_with_ipc(
            episodes=int(1e10),
            metrics_q=metrics_q,
            best_run_q=best_run_q,
            random_run_q=random_run_q,
            compress_frames=True,
            compress_quality=80
        )
    finally:
        stop_event.set()
        time.sleep(0.2)
        if p_plots.is_alive(): p_plots.terminate()
        if p_runs.is_alive():  p_runs.terminate()