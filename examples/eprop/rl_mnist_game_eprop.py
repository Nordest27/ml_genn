#################### DOOR-KEY MNIST MEMORY RL ENV WITH TRAINING ####################
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
from ml_genn.compilers import EPropCompiler, InferenceCompiler, PolicyTypes
from ml_genn.connectivity import Dense, FixedProbability, ToroidalGaussian2D
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, AdaptiveLeakyIntegrateFire, SpikeInput
from ml_genn.serialisers import Numpy
from ml_genn.optimisers import Adam

from ml_genn.compilers.eprop_compiler import default_params

import numpy as np
import cv2
import mnist
import multiprocessing as mp
from multiprocessing import Manager, Queue

# ---------------- CONFIG ----------------
GRID_SIZE = 4
CELL_SIZE = 60

AGENT_COLOR = (50, 200, 50)
DOOR_COLOR_LOCKED = (120, 120, 120)
DOOR_COLOR_OPEN = (0, 200, 0)
REVEAL_COLOR = (200, 200, 50)
BG_COLOR = (50, 50, 50)

TRAIN = True
mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
labels = mnist.train_labels() if TRAIN else mnist.test_labels()
images = mnist.train_images() if TRAIN else mnist.test_images()

#####################################################################
#                         ENVIRONMENT                               #
#####################################################################

class DoorKeyMNISTMemoryEnv:
    """
    Observation (flattened):
        - Walls / doors:   10x10 = 100
        - Reveal cell:     10x10 = 100
        - MNIST image:     28x28 = 784
        Total: 984 dimensions
    """

    ACTIONS = ['left', 'up', 'right', 'down']

    def __init__(self, wait_inc=5, visible_range=5):
        self.wait_inc = wait_inc
        self.visible_range = visible_range
        self.reset()

    def reset(self):
        self.agent_pos = (
                np.random.randint(2, GRID_SIZE), 
                np.random.randint(0, GRID_SIZE)
            )
        self.current_digit = np.random.randint(0, GRID_SIZE)
        self.current_mnist = np.zeros((28, 28), dtype=np.uint8)

        self.reveal_pos = self.agent_pos
        while self.reveal_pos[0] == self.agent_pos[0] and self.reveal_pos[1] == self.agent_pos[1]:
            self.reveal_pos = (
                np.random.randint(2, GRID_SIZE), 
                np.random.choice([i for i in range(GRID_SIZE) 
                    if i != self.current_digit])
            )

        self.doors = [(0, i) for i in range(GRID_SIZE)]
        self.doors_locked = True
        self.correct_door = None

        self.done = False
        self.steps_since_reveal = 0
        self.wait_count = self.wait_inc
        self.visualizing_digit = False
        self.head_butts = 0
        self.no_move_count = 0
        self.digit_visualizations = 0

        return self.get_observation()

    # ------------------------------------------------------------ #
    #                         STEP                                 #
    # ------------------------------------------------------------ #
    def step(self, action):
        if self.done:
            raise RuntimeError("Episode done. Call reset().")

        # Handle wait period (no movement during wait)
        if self.wait_count > 0:
            self.wait_count -= 1
            return self.get_observation(), 0.0, False

        reward = 0.0
        dy, dx = [(0,-1),(-1,0),(0,1),(1,0),(0,0)][action]
        if (dy, dx) == (0, 0):
            self.no_move_count += 1
            reward -= 0.05
            if self.no_move_count > 20:
                reward -= 1.0
                self.done = True

        ny = self.agent_pos[0] + dy
        nx = self.agent_pos[1] + dx

        self.steps_since_reveal += 1
        # Timeout penalty (too many steps after seeing digit)
        if self.steps_since_reveal > 15:
            # reward -= 1.0
            self.done = True
            return self.get_observation(), reward, self.done

        # out of bounds
        if ny < 0 or ny >= GRID_SIZE or nx < 0 or nx >= GRID_SIZE:
            # reward -= 1.0
            # self.done = True
            return self.get_observation(), reward, self.done

        if (ny, nx) in self.doors and self.doors_locked:
            # self.done = True
            # reward -= 1.0
            return self.get_observation(), reward, self.done

        self.agent_pos = [ny, nx]

        # reveal zone
        if (ny, nx) == self.reveal_pos:
            if self.doors_locked:
                reward += 0.5
                self.steps_since_reveal = 0
                self.doors_locked = False
            if self.digit_visualizations < 3:
                self.digit_visualizations += 1
                self.sample_digit()
            self.visualizing_digit = True
        # door entry
        elif ny==1:
            self.sample_door_digit(nx)
            self.visualizing_digit = True
        else:
            self.current_mnist[:] = 0
            self.visualizing_digit = False

        if (ny, nx) in self.doors and not self.doors_locked:
            if nx == self.correct_door:
                reward += 1.0
            else:
                reward += 0.0
            self.done = True

        self.wait_count = self.wait_inc
        return self.get_observation(), reward, self.done

    # ------------------------------------------------------------ #
    #                         MNIST                                #
    # ------------------------------------------------------------ #
    def sample_digit(self):
        self.correct_door = self.current_digit
        self.doors_locked = False
        self.current_mnist = self.get_mnist_image(self.current_digit)

    def sample_door_digit(self, door_idx):
        self.current_mnist = self.get_mnist_image(door_idx)
    
    def get_mnist_image(self, digit):
        #idx = np.random.choice(np.where(labels == digit)[0])
        idx = np.where(labels == digit)[0][0]
        return images[idx].copy()

    # ------------------------------------------------------------ #
    #                      OBSERVATION                             #
    # ------------------------------------------------------------ #
    def get_observation(self):
        v = self.visible_range
        r = v // 2
        agent_y, agent_x = self.agent_pos

        obs = np.zeros((v, v, 3), dtype=np.float32)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y = agent_y + dy
                x = agent_x + dx
                local_y = dy + r
                local_x = dx + r

                # Check walls (out of bounds)
                if y < 0 or y >= GRID_SIZE or x < 0 or x >= GRID_SIZE:
                    obs[local_y, local_x] = [100, 100, 100]
                    continue

                # Reveal button location
                if (y, x) == self.reveal_pos:
                    obs[local_y, local_x] = REVEAL_COLOR
                
                if (dy, dx) == (0, 0):
                    obs[local_y, local_x] = AGENT_COLOR

                # Doors
                if y == 0:
                    if self.doors_locked or x != self.correct_door:
                        obs[local_y, local_x] = DOOR_COLOR_LOCKED
                    else:
                        obs[local_y, local_x] = DOOR_COLOR_OPEN


        mnist_channel = self.current_mnist.astype(np.float32)

        return (obs / 255.0, mnist_channel / 255.0)

    # ------------------------------------------------------------ #
    #                     VISUALIZATION                            #
    # ------------------------------------------------------------ #
    def img(self, scale=60):
        img = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
        img[:] = BG_COLOR

        for y, x in self.doors:
            if self.doors_locked:
                img[y, x] = DOOR_COLOR_LOCKED
            else:
                if x == self.correct_door:
                    img[y, x] = (0, 255, 0)
                else:
                    img[y, x] = DOOR_COLOR_OPEN

        ry, rx = self.reveal_pos
        img[ry, rx] = REVEAL_COLOR

        ay, ax = self.agent_pos
        img[ay, ax] = AGENT_COLOR

        img = cv2.resize(
            img,
            (GRID_SIZE * scale, GRID_SIZE * scale),
            interpolation=cv2.INTER_NEAREST
        )

        if self.visualizing_digit:
            mnist_size = int(scale*1.5)
            mnist_display = cv2.resize(self.current_mnist, (mnist_size, mnist_size),
                                      interpolation=cv2.INTER_NEAREST)
            mnist_display = cv2.cvtColor(mnist_display, cv2.COLOR_GRAY2BGR)
            
            border_size = GRID_SIZE//2
            mnist_display = cv2.copyMakeBorder(mnist_display, border_size, border_size, 
                                              border_size, border_size,
                                              cv2.BORDER_CONSTANT, value=(255, 255, 0))
            
            h, w = img.shape[:2]
            mh, mw = mnist_display.shape[:2]
            y_offset = max(0, h - mh - GRID_SIZE)
            x_offset = max(0, w - mw - GRID_SIZE)
            
            img[y_offset:y_offset+mh, x_offset:x_offset+mw] = mnist_display

        return img

    def render(self):
        cv2.imshow("Door-Key MNIST", self.img())

        mn = cv2.resize(self.current_mnist, (200, 200),
                        interpolation=cv2.INTER_NEAREST)
        mn = cv2.cvtColor(mn, cv2.COLOR_GRAY2BGR)
        cv2.imshow("MNIST Panel", mn)

        cv2.waitKey(1)

################### DEFINE MODEL ####################
#####################################################
CHECKPOINT_BOARD_SIZE = None

CONNECTIVITY_TYPE = "toroidal"
WINDOW_EPISODES = 100

VISIBLE_RANGE = 5
SCALE = 1

WAIT_INC = 30

INPUT_C = 3

ENV_INPUT_SHAPE = (5, 5, INPUT_C)
MNIST_INPUT_SHAPE = (28, 28, 1)

NORMALIZATION_SHAPE = (10, 10, INPUT_C)

HIDDEN_E_SHAPE = (20, 20, INPUT_C)
HIDDEN_I_SHAPE = (15, 15, INPUT_C)

ENV_INPUT_SIZE = np.prod(ENV_INPUT_SHAPE)
MNIST_INPUT_SIZE = np.prod(MNIST_INPUT_SHAPE)

NUM_HIDDEN_E = np.prod(HIDDEN_E_SHAPE)
NUM_HIDDEN_I = np.prod(HIDDEN_I_SHAPE)

GAUSSIAN_TRACE_POLICY = False

SIGMA_IN = 0.1
SIGMA_H = 0.05

DESIRED_FAN_IN_IN = 100
DESIRED_FAN_IN_H1 = 100
DESIRED_FAN_IN_H2 = 100

FAN_IN_SCALE_IN = 0.25

def compute_p_max(desired_fan_in, sigma, src_shape, dst_shape=None, wrap=True):
    src_h, src_w, src_c = src_shape

    src_cols = np.arange(src_w) / max(src_w - 1, 1)
    src_rows = np.arange(src_h) / max(src_h - 1, 1)

    if dst_shape is not None:
        dst_h, dst_w, _ = dst_shape
        sample_xs = np.arange(dst_w) / max(dst_w - 1, 1)
        sample_ys = np.arange(dst_h) / max(dst_h - 1, 1)
    else:
        sample_xs = np.array([0.5])
        sample_ys = np.array([0.5])

    # sample_ys: (S_y,), sample_xs: (S_x,), src_rows: (src_h,), src_cols: (src_w,)
    dx = np.abs(sample_xs[:, np.newaxis] - src_cols[np.newaxis, :])   # (S_x, src_w)
    dy = np.abs(sample_ys[:, np.newaxis] - src_rows[np.newaxis, :])   # (S_y, src_h)
    if wrap:
        dx = np.minimum(dx, 1.0 - dx)
        dy = np.minimum(dy, 1.0 - dy)

    # d2[sy, sx, src_h, src_w]
    d2 = (dy[:, np.newaxis, :, np.newaxis] ** 2 +   # (S_y, 1,   src_h, 1)
          dx[np.newaxis, :, np.newaxis, :] ** 2)     # (1,   S_x, 1,    src_w)

    gaussian_sums = np.sum(np.exp(-d2 / (2.0 * sigma**2)), axis=(-2, -1))  # (S_y, S_x)
    avg_gaussian_sum = gaussian_sums.mean()

    p_max = desired_fan_in / (avg_gaussian_sum * src_c)

    # Debug
    print(f"  src={src_shape}, dst={dst_shape}, sigma={sigma}")
    print(f"  avg_gaussian_sum={avg_gaussian_sum:.3f}, src_c={src_c}, p_max={p_max:.6f}")
    print(f"  predicted fan_in = p_max * avg_gaussian_sum * src_c = {p_max * avg_gaussian_sum * src_c:.1f}")

    return p_max

# p_max_in = compute_p_max(DESIRED_FAN_IN_IN, SIGMA_IN, INPUT_SHAPE, HIDDEN_E_SHAPE)
# p_max_h1 = compute_p_max(DESIRED_FAN_IN_H1, SIGMA_H, HIDDEN_E_SHAPE)
# p_max_h2 = compute_p_max(DESIRED_FAN_IN_H2, SIGMA_H, HIDDEN_I_SHAPE)


NUM_OUTPUT = 4
if GAUSSIAN_TRACE_POLICY:
    NUM_OUTPUT = 4

CONN_P = {
    "I-H": 0.001,
    "D-O": 0.001,
    "H-H": 0.005,
    #"H-H": np.log(NUM_HIDDEN_1+NUM_HIDDEN_2)/(NUM_HIDDEN_1 + NUM_HIDDEN_2),
    "H-P": 0.1,
    "H-V": 0.1,
    "F": 0.1
}

gamma                  = 0.1  ** (1 / WAIT_INC)
td_lambda              = 0.8   ** (1 / WAIT_INC)

entropy_coeff     = 0.0*1e-4
entropy_decay     = 0.9999 ** (1 / WAIT_INC)
entropy_coeff_min = 0.0*1e-5

dale_l1_reg = 0.0 

serialiser = Numpy("door_key_mnist_checkpoints")

def make_connectivity(
    connectivity_type,
    src_shape,
    desired_fan_in=None,
    fan_in_scale=None,
    p=None,
    sigma=None,
    sign=None,
    mean_scale=0.1,
    sd_scale=0.05
):
    if connectivity_type == "fixed":

        if p is None:
            raise ValueError("Fixed connectivity requires p")
        
        if sign is None:
            sd_scale = 1.0
        
        fan_in = p * np.prod(src_shape)

        mean = (sign or 0) * mean_scale / np.sqrt(fan_in)
        sd = sd_scale / np.sqrt(fan_in)

        return FixedProbability(
            p,
            Normal(mean=mean, sd=sd)
        )

    elif connectivity_type == "toroidal":

        if sigma is None:
            raise ValueError("Toroidal connectivity requires sigma")

        if desired_fan_in is None:
            raise ValueError("Toroidal connectivity requires desired_fan_in")

        # compute p_max automatically
        # p_max = compute_p_max(desired_fan_in, sigma, src_shape, dst_shape)
        fan_in = desired_fan_in

        if sign == -1:
            mean_scale *= 3
        elif sign is None:
            sd_scale = 1.0

        mean = (sign or 0) * mean_scale / np.sqrt(fan_in)
            
        sd = sd_scale / np.sqrt(fan_in)

        return ToroidalGaussian2D(
            sigma=sigma,
            fan_in=desired_fan_in,
            fan_in_scale=fan_in_scale,
            weight=Normal(mean=mean, sd=sd)
        )

    else:
        raise ValueError(f"Unknown connectivity_type: {connectivity_type}")


def build_compiled_network(connectivity_type="toroidal"):
    global dale_l1_reg

    network = Network(default_params)
    hidden_layers = {}

    with network:
        # ================= INPUTS =================
        input_pop_mnist = Population(
            SpikeInput(max_spikes=MNIST_INPUT_SIZE * WAIT_INC),
            MNIST_INPUT_SHAPE
        )

        input_pop_env = Population(
            SpikeInput(max_spikes=ENV_INPUT_SIZE * WAIT_INC),
            ENV_INPUT_SHAPE
        )

        # ================= PREPROCESS =================
        hidden_layers["mnist_down"] = Population(
            AdaptiveLeakyIntegrateFire(
                v_thresh=0.61, tau_mem=10.0,
                tau_refrac=3.0, tau_adapt=300, beta=0.0
            ),
            NORMALIZATION_SHAPE
        )
        
        hidden_layers["env_up"] = Population(
            AdaptiveLeakyIntegrateFire(
                v_thresh=0.61, tau_mem=10.0,
                tau_refrac=3.0, tau_adapt=300, beta=0.0
            ),
            NORMALIZATION_SHAPE
        )

        # ================= E / I =================
        hidden_layers["E"] = Population(
            AdaptiveLeakyIntegrateFire(
                v_thresh=0.61, tau_mem=10.0,
                tau_refrac=3.0, tau_adapt=300, beta=0.17
            ),
            HIDDEN_E_SHAPE
        )

        hidden_layers["I"] = Population(
            AdaptiveLeakyIntegrateFire(
                v_thresh=0.61, tau_mem=10.0,
                tau_refrac=3.0, tau_adapt=300, beta=0.17
            ),
            HIDDEN_I_SHAPE
        )

        # ================= FIELDS =================
        hidden_layers["policy_field"] = Population(
            AdaptiveLeakyIntegrateFire(
                v_thresh=0.61, tau_mem=10.0,
                tau_refrac=3.0, tau_adapt=300
            ),
            HIDDEN_E_SHAPE
        )

        hidden_layers["value_field"] = Population(
            AdaptiveLeakyIntegrateFire(
                v_thresh=0.61, tau_mem=10.0,
                tau_refrac=3.0, tau_adapt=300
            ),
            HIDDEN_E_SHAPE
        )

        # ================= OUTPUT =================
        policy = Population(
            LeakyIntegrate(tau_mem=10.0, readout="var"),
            NUM_OUTPUT
        )

        value = Population(
            LeakyIntegrate(tau_mem=10.0, readout="var"),
            1
        )

        # =========================================================
        # INPUT → PREPROCESS
        # =========================================================
        Connection(
            input_pop_mnist,
            hidden_layers["mnist_down"],
            make_connectivity(
                connectivity_type="toroidal",
                src_shape=MNIST_INPUT_SHAPE,
                sigma=SIGMA_IN,
                desired_fan_in=DESIRED_FAN_IN_IN,
                sign=1
            ),
            exc_inh_sign=1
        )

        Connection(
            input_pop_env,
            hidden_layers["env_up"],
            make_connectivity(
                connectivity_type="toroidal",
                src_shape=ENV_INPUT_SHAPE,
                sigma=SIGMA_IN,
                desired_fan_in=DESIRED_FAN_IN_IN,
                sign=1
            ),
            exc_inh_sign=1
        )

        # =========================================================
        # PREPROCESS → E / I
        # =========================================================
        for src in ["mnist_down", "env_up"]:
            for target in ["E", "I"]:
                Connection(
                    hidden_layers[src],
                    hidden_layers[target],
                    make_connectivity(
                        connectivity_type=connectivity_type,
                        src_shape=NORMALIZATION_SHAPE,
                        sigma=SIGMA_H,
                        desired_fan_in=DESIRED_FAN_IN_IN,
                        fan_in_scale=FAN_IN_SCALE_IN,
                        sign=1
                    ),
                    exc_inh_sign=1
                )

        # =========================================================
        # EXCITATORY
        # =========================================================
        for target in ["E", "I", "policy_field", "value_field"]:
            Connection(
                hidden_layers["E"],
                hidden_layers[target],
                make_connectivity(
                    connectivity_type=connectivity_type,
                    src_shape=HIDDEN_E_SHAPE,
                    sigma=SIGMA_H,
                    desired_fan_in=DESIRED_FAN_IN_H1,
                    sign=1
                ),
                exc_inh_sign=1
            )

        # =========================================================
        # INHIBITORY
        # =========================================================
        for target in ["E", "I", "policy_field", "value_field"]:
            Connection(
                hidden_layers["I"],
                hidden_layers[target],
                make_connectivity(
                    connectivity_type=connectivity_type,
                    src_shape=HIDDEN_I_SHAPE,
                    sigma=SIGMA_H,
                    desired_fan_in=DESIRED_FAN_IN_H2,
                    sign=-1
                ),
                exc_inh_sign=-1
            )

        # =========================================================
        # READOUT
        # =========================================================
        Connection(
            hidden_layers["policy_field"],
            policy,
            make_connectivity(
                connectivity_type="fixed",
                src_shape=HIDDEN_E_SHAPE,
                p=0.5,
                sign=None
            ),
            exc_inh_sign=None
        )

        Connection(
            hidden_layers["value_field"],
            value,
            make_connectivity(
                connectivity_type="fixed",
                src_shape=HIDDEN_E_SHAPE,
                p=0.99999,
                sign=None
            ),
            exc_inh_sign=None
        )

        # =========================================================
        # FEEDBACK
        # =========================================================
        Connection(
            policy, value, Dense(weight=1.0),
            feedback_name="tde_transport"
        )
        for name, layer in hidden_layers.items():
            Connection(
                layer, value, Dense(weight=1.0),
                feedback_name="tde_transport"
            )
            sign = 1 if name == "E" else -1 if name == "I" else None

            for fb_name in [
                "policy_feedback",
                "policy_regularisation",
                "value_feedback",
                "value_regularisation"
            ]:
                Connection(
                    layer,
                    policy if "policy" in fb_name else value,
                    FixedProbability(
                        CONN_P["F"],
                        Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))
                    ),
                    feedback_name=fb_name,
                    exc_inh_sign=sign
                )

    # ================= COMPILER =================
    dale_l1_reg = 0.0

    compiler = EPropCompiler(
        example_timesteps=1,
        losses={
            policy: "sparse_categorical_crossentropy",
            value: "mean_square_error",
        },
        optimiser=Adam(1e-5, soft_grad_clip=10),
        batch_size=1,
        feedback_type="random",
        gamma=gamma,
        td_lambda=td_lambda,
        train_output_bias=False,
        reset_time_between_batches=False,
        entropy_coeff=entropy_coeff,
        entropy_coeff_decay=entropy_decay,
        entropy_coeff_min=entropy_coeff_min,
        dale_rewiring_l1_strength=dale_l1_reg,
        policy_heads={
            policy: PolicyTypes.CATEGORICAL if not GAUSSIAN_TRACE_POLICY else PolicyTypes.GAUSSIAN_TRACE, 
        },
        value_head=value,
    )

    if CHECKPOINT_BOARD_SIZE:
        network.load((CHECKPOINT_BOARD_SIZE,), serialiser)

    compiled_net = compiler.compile(network)

    return (
        compiled_net,
        network,
        input_pop_env,
        input_pop_mnist,
        hidden_layers,
        policy,
        value
    )

compiled_net, network, input_pop_env, input_pop_mnist, hidden_layers, policy, value = \
    build_compiled_network()

train_callback_list = CallbackList(
    [*set(compiled_net.base_train_callbacks)],
    compiled_network=compiled_net,
    num_batches=1,
    num_epochs=1
)

####################### HELPER FUNCTIONS #######################
#################################################################

def make_repeated_spikes(indices, base_timestep, input_size, K=5, period=1):
    """Deterministic repeated spike encoding (used for the agent / wall channel)."""
    indices = np.asarray(indices, dtype=np.int64)
    times   = np.repeat(base_timestep + np.arange(K) * period, len(indices))
    idxs    = np.tile(indices, K)
    return preprocess_spikes(times, idxs, input_size)


def make_rate_coded_spikes(values, base_timestep, input_size, K):
    """
    Rate-coded spike encoding: each neuron fires Bernoulli(v) at each of K timesteps.
    Used for the MNIST channel (pixel intensities as firing probabilities).
    """
    values = np.clip(values, 0.0, 1.0)
    times  = []
    idxs   = []

    assert len(values) == input_size

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


def encode_frame(frame, compress=True, ext='.jpg', quality=80):
    if not compress:
        return frame
    ret, buf = cv2.imencode(ext, frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ret:
        return frame
    return buf.tobytes()


def decode_frame(blob_or_array):
    if isinstance(blob_or_array, bytes):
        arr = np.frombuffer(blob_or_array, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    return blob_or_array

####################### VISUALIZATION PROCESSES #######################
########################################################################

def viz_plots_loop(metrics_q: Queue, stop_event: mp.Event):
    plt.ion()
    fig, (ax1, ax1b, ax2, ax3) = plt.subplots(4, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.5)

    line_reward,  = ax1.plot([], [], label='Total Reward',  color=(0, 0.1, 0.8, 0.5), zorder=2)
    avg_line,     = ax1.plot([], [], label='Running Avg',   color=(0.8, 0.05, 0.05, 1.0), zorder=3)
    success_line, = ax1.plot([], [], label='Success Rate',  color=(0, 0.8, 0.1, 0.5), zorder=1)
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Reward / Success Rate')
    ax1.set_title('Training Progress'); ax1.legend()

    line_reward_recent, = ax1b.plot([], [], label='Reward (last 500)')
    avg_line_recent,    = ax1b.plot([], [], label='Avg (last 500)')
    ax1b.set_title('Recent Training (last 500 episodes)'); ax1b.legend()

    value_line, = ax2.plot([], [], label='Value')
    ax2.set_title('Value (best run)'); ax2.set_xlabel('Frame')
    ax2.set_ylabel('Value')

    prob_img = ax3.imshow(np.zeros((4, 1)), aspect='auto', origin='lower', vmin=0, vmax=1)
    ax3.set_xlabel('Timestep'); ax3.set_ylabel('Action')
    ax3.set_title('Action probs (best run)')
    fig.colorbar(prob_img, ax=ax3, fraction=0.02, pad=0.04)

    ep_list      = []
    rewards      = []
    avgs         = []
    success_rates = []
    best_values  = []
    best_probs   = None

    last_plot_time = 0.0
    plot_interval  = 0.2

    while not stop_event.is_set():
        try:
            while True:
                metrics = metrics_q.get_nowait()
                ep = metrics.get('ep')
                if ep is not None:
                    ep_list.append(ep)
                    rewards.append(metrics.get('reward', 0.0))
                    avgs.append(metrics.get('running_avg', 0.0))
                    success_rates.append(metrics.get('success_rate', 0.0))
                if 'best_values' in metrics:
                    best_values = metrics['best_values']
                if 'best_probs' in metrics:
                    best_probs = metrics['best_probs']
        except Exception:
            pass

        now = time.time()
        if now - last_plot_time > plot_interval:
            if ep_list:
                line_reward.set_data(ep_list, rewards)
                avg_line.set_data(ep_list, avgs)
                success_line.set_data(ep_list, success_rates)
                ax1.relim(); ax1.autoscale_view()

            if rewards:
                recent_rewards = rewards[-WINDOW_EPISODES:]
                recent_avgs    = avgs[-WINDOW_EPISODES:]
                axis_recent    = np.arange(len(recent_rewards))
                line_reward_recent.set_data(axis_recent, recent_rewards)
                avg_line_recent.set_data(axis_recent, recent_avgs)
                ax1b.relim(); ax1b.autoscale_view()

            if best_values:
                value_line.set_data(range(len(best_values)), best_values)
                ax2.relim(); ax2.autoscale_view()

            if best_probs is not None:
                data = np.array(best_probs).T
                n_actions, time_steps = data.shape
                prob_img.set_data(data)
                prob_img.set_extent([0, time_steps, 0, n_actions])
                ax3.set_xlim(0, time_steps); ax3.set_ylim(0, n_actions)
                ax3.set_aspect('auto')

            plt.pause(0.001)
            last_plot_time = now

        time.sleep(0.03)

    plt.close(fig)


def viz_runs_loop(best_run_q: Queue, random_run_q: Queue,
                  stop_event: mp.Event, decompress=True):
    best_run   = []
    random_run = []
    window_best   = "Best Run"
    window_random = "Random Run"

    cv2.namedWindow(window_best,   cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_random, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_best,   600, 600)
    cv2.resizeWindow(window_random, 600, 600)

    t = 0
    while not stop_event.is_set():
        try:
            while True:
                br = best_run_q.get_nowait()
                if br is None:
                    stop_event.set(); break
                best_run = br
        except Exception:
            pass

        try:
            while True:
                rr = random_run_q.get_nowait()
                if rr is None:
                    stop_event.set(); break
                random_run = rr
        except Exception:
            pass

        if best_run:
            if stop_event.is_set(): break
            f_blob = best_run[t % len(best_run)]
            frame  = decode_frame(f_blob) if decompress else f_blob
            if frame is None: continue
            cv2.imshow(window_best, frame)
            if cv2.waitKey(1) == 27:
                stop_event.set(); break

        if random_run and t % (len(random_run) + 10) < len(random_run):
            f_blob = random_run[t % (len(random_run) + 10)]
            if stop_event.is_set(): break
            frame  = decode_frame(f_blob) if decompress else f_blob
            if frame is None: continue
            cv2.imshow(window_random, frame)
            if cv2.waitKey(1) == 27:
                stop_event.set(); break

        time.sleep(0.1)
        t += 1

    cv2.destroyWindow(window_best)
    cv2.destroyWindow(window_random)


def start_visualizers():
    manager      = Manager()
    metrics_q    = manager.Queue(maxsize=10)
    best_run_q   = manager.Queue(maxsize=2)
    random_run_q = manager.Queue(maxsize=2)
    stop_event   = manager.Event()

    p_plots = mp.Process(target=viz_plots_loop,
                         args=(metrics_q, stop_event), daemon=True)
    p_runs  = mp.Process(target=viz_runs_loop,
                         args=(best_run_q, random_run_q, stop_event), daemon=True)

    p_plots.start(); p_runs.start()
    return manager, metrics_q, best_run_q, random_run_q, stop_event, p_plots, p_runs

####################### TRAINING AGENT #######################
##############################################################

def train_door_key_agent(episodes=100000,
                         metrics_q: Queue=None,
                         best_run_q: Queue=None,
                         random_run_q: Queue=None):
    """
    Train the Door-Key MNIST Memory agent using e-prop learning.
    Gradient updates are driven by GradientLearn / DalePrune / DaleRewire
    custom updates, mirroring the snake training loop.
    """
    opt_updt = 0

    with compiled_net:
        env = DoorKeyMNISTMemoryEnv(wait_inc=WAIT_INC, visible_range=VISIBLE_RANGE)
        best_reward    = -np.inf
        best_run       = []
        running_avg    = []
        success_history = []
        smoothing      = 0.95
        avg            = 0

        last_best_values = []
        last_best_probs  = None
        time_since_last_best = 0

        train_callback_list.on_epoch_begin(0)
        train_callback_list.on_batch_begin(0)

        for ep in range(episodes):
            obs  = env.reset()
            done = False
            total_reward   = 0
            current_run    = []
            current_values = []
            current_probs  = []
            frame  = 1
            success = False

            reward_trace = 0.0

            if ep % 10000 == 0:
                compiled_net.save_connectivity((CHECKPOINT_BOARD_SIZE,), serialiser)
                compiled_net.save((CHECKPOINT_BOARD_SIZE,), serialiser)

            # ---- Initial spike encoding ----
            spikes_env = make_rate_coded_spikes(
                obs[0].reshape(-1),
                compiled_net.genn_model.timestep,
                ENV_INPUT_SIZE,
                K=WAIT_INC
            )
            spikes_mnist = make_rate_coded_spikes(
                obs[1].reshape(-1),
                compiled_net.genn_model.timestep,
                MNIST_INPUT_SIZE,
                K=WAIT_INC
            )
            compiled_net.set_input({input_pop_env: [spikes_env],
                                    input_pop_mnist:  [spikes_mnist]})
            env.wait_count = WAIT_INC

            while not done:
                current_values.append(compiled_net.get_readout(value)[0][0])
                action_label = 0

                if env.wait_count == 0:
                    probs = compiled_net.get_readout(policy).flatten()
                    if abs(sum(probs) - 1.0) > 0.0001:
                        print("BAD PROBS", sum(probs))
                    action_label = np.random.choice(NUM_OUTPUT, p=probs)
                    current_probs.append(probs)

                    compiled_net.losses[policy].set_target(
                        compiled_net.neuron_populations[policy],
                        [action_label], policy.shape,
                        compiled_net.genn_model.batch_size,
                        compiled_net.example_timesteps
                    )
                    compiled_net.losses[policy].set_var(
                        compiled_net.neuron_populations[policy], "actionTaken", 1.0
                    )

                obs, reward, done = env.step(action_label)
                total_reward  += reward
                reward_trace   = reward_trace * 0.9261 + reward * 1.0

                if done and reward > 0.5:
                    success = True

                # Pass reward directly to the value head (device-side TD computation)
                if reward != 0:
                    compiled_net.losses[value].set_var(
                        compiled_net.neuron_populations[value], "reward", reward
                    )

                if env.wait_count == env.wait_inc:
                    # Capture frame
                    frame_img = env.img(scale=8)
                    # frame_img = obs[0]
                    current_run.append(frame_img.copy())

                    # Encode next observation
                    spikes_env = make_rate_coded_spikes(
                        obs[0].reshape(-1),
                        compiled_net.genn_model.timestep,
                        ENV_INPUT_SIZE,
                        K=WAIT_INC
                    )
                    spikes_mnist = make_rate_coded_spikes(
                        obs[1].reshape(-1),
                        compiled_net.genn_model.timestep,
                        MNIST_INPUT_SIZE,
                        K=WAIT_INC
                    )
                    compiled_net.set_input({input_pop_env: [spikes_env],
                                            input_pop_mnist:  [spikes_mnist]})

                compiled_net.step_time(train_callback_list)

                # if env.wait_count == env.wait_inc:
                compiled_net.genn_model.custom_update("GradientLearn")
                for o, custom_updates in compiled_net.optimisers:
                    for c in custom_updates:
                        o.set_step(c, opt_updt := opt_updt + 1)

                frame += 1

            # ---- Episode tail: drain reward trace ----
            spikes_env = make_rate_coded_spikes(
                obs[0].reshape(-1),
                compiled_net.genn_model.timestep,
                ENV_INPUT_SIZE,
                K=WAIT_INC
            )
            spikes_mnist = make_rate_coded_spikes(
                obs[1].reshape(-1),
                compiled_net.genn_model.timestep,
                MNIST_INPUT_SIZE,
                K=WAIT_INC
            )
            compiled_net.set_input({input_pop_env: [spikes_env],
                                    input_pop_mnist:  [spikes_mnist]})

            for _ in range(WAIT_INC):
                reward_trace = reward_trace * 0.9261
                current_values.append(compiled_net.get_readout(value)[0][0])

                compiled_net.step_time(train_callback_list)

                compiled_net.genn_model.custom_update("GradientLearn")
                for o, custom_updates in compiled_net.optimisers:
                    for c in custom_updates:
                        o.set_step(c, opt_updt := opt_updt + 1)
            """
            compiled_net.step_time(train_callback_list)
            compiled_net.genn_model.custom_update("GradientLearn")
            for o, custom_updates in compiled_net.optimisers:
                for c in custom_updates:
                    o.set_step(c, opt_updt := opt_updt + 1)
            """
            # Dale's law rewiring (no-op when dale_l1_reg == 0)
            if dale_l1_reg > 0:
                compiled_net.genn_model.custom_update("DaleRL1")
            compiled_net.genn_model.custom_update("DalePrune")
            compiled_net.genn_model.custom_update("DaleRewire")

            # ---- Update best run ----
            time_since_last_best += 1
            if total_reward == best_reward and time_since_last_best > 100:
                best_reward -= 1
                time_since_last_best = 0

            if total_reward > best_reward and len(current_probs) > 0:
                best_reward          = total_reward
                best_run             = list(current_run)
                last_best_values     = list(current_values)
                last_best_probs      = list(current_probs)
                time_since_last_best = 0

                if best_run_q is not None:
                    try:
                        if best_run_q.full():
                            try: best_run_q.get_nowait()
                            except Exception: pass
                        best_run_q.put_nowait(best_run)
                    except Exception as e:
                        print("Best-run enqueue error:", e)

            # Random run snapshot every 100 episodes
            if random_run_q is not None and ep % 100 == 0:
                try:
                    if random_run_q.full():
                        try: random_run_q.get_nowait()
                        except: pass
                    random_run_q.put_nowait(current_run)
                except Exception as e:
                    print("Random-run enqueue error:", e)

            # ---- Metrics ----
            success_history.append(1.0 if success else 0.0)
            if len(success_history) > 100:
                success_history = success_history[-100:]

            avg = total_reward if avg == 0 else (smoothing * avg + (1 - smoothing) * total_reward)
            running_avg.append(avg)

            if metrics_q is not None:
                metrics = {
                    'ep':           ep,
                    'reward':       total_reward,
                    'running_avg':  avg,
                    'success_rate': np.mean(success_history),
                }
                if last_best_values:
                    metrics['best_values'] = last_best_values
                if last_best_probs is not None:
                    metrics['best_probs'] = last_best_probs
                try:
                    if metrics_q.full():
                        try: metrics_q.get_nowait()
                        except: pass
                    metrics_q.put_nowait(metrics)
                except Exception as e:
                    print("Metrics enqueue error:", e)

            print(
                f"Episode {ep+1:5d} - "
                f"Reward: {' ' if total_reward >= 0 else ''}{total_reward:+.3f} "
                f"- Best: {best_reward:+.3f} "
                f"- Success rate (last 100): {np.mean(success_history):.2%} "
                f"- Frames: {frame:3d} "
                f"- Alpha: {compiled_net.optimisers[0][0].alpha:.8f}"
            )

    # Signal visualizers to stop
    for q in (best_run_q, random_run_q):
        if q is not None:
            try: q.put_nowait(None)
            except: pass

####################################################################
#                              MAIN                                #
####################################################################

if __name__ == "__main__":
    manager, metrics_q, best_run_q, random_run_q, stop_event, p_plots, p_runs = \
        start_visualizers()

    try:
        train_door_key_agent(
            episodes=int(1e10),
            metrics_q=metrics_q,
            best_run_q=best_run_q,
            random_run_q=random_run_q
        )
    finally:
        stop_event.set()
        time.sleep(0.2)
        if p_plots.is_alive(): p_plots.terminate()
        if p_runs.is_alive():  p_runs.terminate()