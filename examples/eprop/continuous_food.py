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
from ml_genn.compilers import EPropCompiler, InferenceCompiler, PolicyTypes
from ml_genn.connectivity import Dense, FixedProbability, Conv2D, ToroidalGaussian2D
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, AdaptiveLeakyIntegrateFire, SpikeInput
from ml_genn.serialisers import Numpy
from ml_genn.optimisers import Adam
from scipy.ndimage import gaussian_filter

from ml_genn.compilers.eprop_compiler import default_params
from collections import defaultdict

def extract_actual_sparse_connections(compiled_net):
    """
    Extract the actual number of instantiated connections per connection group.
    Returns a dict: {connection_name: n_connections}
    """

    connection_counts = {}

    for c, genn_pop in compiled_net.connection_populations.items():

        # Only meaningful for sparse connectivity
        if genn_pop.matrix_type & SynapseMatrixConnectivity.SPARSE:

            # Make sure connectivity is on host
            genn_pop.pull_connectivity_from_device()

            pre_inds = genn_pop.get_sparse_pre_inds()

            n_conn = len(pre_inds)

            connection_counts[c] = n_conn

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

    def __init__(self, size=1.0, visible_range=5, scale=1, wait_inc=5, inp_shape=(5,5,3)):

        self.world_size = size
        self.visible_range = visible_range
        self.scale = scale
        self.wait_inc = wait_inc
        self.inp_shape = inp_shape

        self.agent_radius = 0.08
        self.food_radius = 0.08
        self.max_speed = 0.5

        self.reset()

    # ---------------------------------------------------------

    def reset(self):

        self.agent_pos = np.random.rand(2) * self.world_size
        self.food_pos = np.random.rand(2) * self.world_size
        self.velocity = np.zeros(2)

        self.done = False
        self.wait_count = self.wait_inc
        self.steps = 0

        self.prev_dist = np.linalg.norm(self.agent_pos - self.food_pos)

        return self.get_local_img_observation()

    # ---------------------------------------------------------
    def step(self, action_vec):

        if self.done:
            raise Exception("reset needed")
        
        self.velocity = np.array(list(map(lambda x: np.sign(x)*min(abs(x), self.max_speed), self.velocity*0.5 + action_vec)))

        self.agent_pos += self.velocity
        """
        # ----- border collision
        if (
            self.agent_pos[0] < 0.0 or
            self.agent_pos[0] > self.world_size or
            self.agent_pos[1] < 0.0 or
            self.agent_pos[1] > self.world_size
        ):
            reward = -1.0
            self.done = True
            return self.get_local_img_observation(), reward, self.done
        """
        self.agent_pos = np.clip(self.agent_pos, 0, self.world_size)

        dist = np.linalg.norm(self.agent_pos - self.food_pos)

        reward =  0.0
        if dist < (self.agent_radius + self.food_radius):
            reward += 1.0
            self.food_pos = np.random.rand(2) * self.world_size

        self.prev_dist = dist

        self.steps += 1
        if self.steps > 100:
            self.done = True

        return self.get_local_img_observation(), reward, self.done

    # ---------------------------------------------------------
    def get_local_img_observation(self):

        img = self.local_img(scale=self.scale)

        return cv2.resize(
            img,
            (self.inp_shape[0], self.inp_shape[1]),
            interpolation=cv2.INTER_LINEAR
        )

    # ---------------------------------------------------------
    def img(self, scale=20):

        H = W = 100

        img = np.zeros((H, W, 3), dtype=np.float32)

        ys = np.linspace(0, self.world_size, H)
        xs = np.linspace(0, self.world_size, W)

        Y, X = np.meshgrid(ys, xs, indexing="ij")

        # ----- agent (green)
        d2 = (X - self.agent_pos[0])**2 + (Y - self.agent_pos[1])**2
        agent_blob = np.exp(-d2/(2*self.agent_radius**2))

        # ----- apple (red)
        d2f = (X - self.food_pos[0])**2 + (Y - self.food_pos[1])**2
        food_blob = np.exp(-d2f/(2*self.food_radius**2))

        img[...,1] = agent_blob
        img[...,2] = food_blob

        # ----- border frame (white)
        bw = 3
        img[:bw,:,:] = 0.2
        img[-bw:,:,:] = 0.2
        img[:,:bw,:] = 0.2
        img[:,-bw:,:] = 0.2

        img = (img * 255).astype(np.uint8)

        return cv2.resize(
            img,
            (H*scale, W*scale),
            interpolation=cv2.INTER_NEAREST
        )
    
    def local_img(self, scale=10):
        v = self.visible_range
        S = v * scale  # resolution

        img = np.zeros((S, S, 3), dtype=np.float32)

        yy, xx = np.mgrid[0:S, 0:S]

        # scale velocity so max shift ~ half a cell
        vel_scale = (S / self.visible_range) * 0.5
        shift_x = self.velocity[0] * vel_scale  # right positive
        shift_y = self.velocity[1] * vel_scale  # up positive

        # ===== agent position in pixel space (shifted by velocity)
        agent_px = S // 2 - shift_x
        agent_py = S // 2 - shift_y

        # ===== map pixels → relative to agent
        rel_x = (xx - agent_px) / S * self.visible_range
        rel_y = (yy - agent_py) / S * self.visible_range

        world_x = self.agent_pos[0] + rel_x
        world_y = self.agent_pos[1] + rel_y

        # ===== WALLS (finite outline)
        thickness_px = 1.0
        thickness = thickness_px * (self.visible_range / S)

        inside_y = (world_y >= 0.0) & (world_y <= self.world_size)
        inside_x = (world_x >= 0.0) & (world_x <= self.world_size)

        left_wall   = (world_x < 0.0) & (world_x > -thickness) & inside_y
        right_wall  = (world_x > self.world_size) & (world_x < self.world_size + thickness) & inside_y
        bottom_wall = (world_y < 0.0) & (world_y > -thickness) & inside_x
        top_wall    = (world_y > self.world_size) & (world_y < self.world_size + thickness) & inside_x

        wall_mask = left_wall | right_wall | bottom_wall | top_wall
        img[wall_mask] = [0.4, 0.4, 0.4]

        # ===== AGENT (continuous blob)
        px = xx - agent_px
        py = yy - agent_py
        agent_sigma = scale * 0.5
        d2_agent = px**2 + py**2
        agent = np.exp(-d2_agent / (2 * agent_sigma**2))

        # ===== FOOD (continuous)
        fx = (self.food_pos[0] - self.agent_pos[0]) / self.visible_range * S
        fy = (self.food_pos[1] - self.agent_pos[1]) / self.visible_range * S
        d2_food = (px - fx)**2 + (py - fy)**2
        food_sigma = scale * 0.5
        food = np.exp(-d2_food / (2 * food_sigma**2))

        # ===== compose
        img[..., 1] += agent * 0.8   # green
        img[..., 2] += food * 0.9    # red

        img = np.clip(img, 0.0, 1.0)
        return (img * 255).astype(np.uint8)
################### DEFINE MODEL ####################
#####################################################
def expected_toroidal_connections(
    src_size,
    dst_size,
    sigma,
    desired_fan_in=None,
    p_max=None
):
    """
    Computes expected total number of toroidal connections.

    Either desired_fan_in OR p_max must be provided.
    """

    if desired_fan_in is None and p_max is None:
        raise ValueError("Provide either desired_fan_in or p_max")

    if desired_fan_in is None:
        # derive fan-in from p_max
        desired_fan_in = src_size * 2.0 * np.pi * sigma**2 * p_max

    # total connections = fan_in per neuron × number of postsyn neurons
    return dst_size * desired_fan_in


CONNECTIVITY_TYPE = "toroidal"
WINDOW_EPISODES = 100
BOARD_SIZE = 4

VISIBLE_RANGE = 10
SCALE = 5

WAIT_INC = 30

INPUT_C = 3

INPUT_SHAPE = (50, 50, INPUT_C)
# DOWNSAMPLE_SHAPE = (100, 100, INPUT_C)
HIDDEN_E_SHAPE = (30, 30, INPUT_C)
HIDDEN_I_SHAPE = (25, 25, INPUT_C)
INPUT_SIZE = np.prod(INPUT_SHAPE)
NUM_HIDDEN_E = np.prod(HIDDEN_E_SHAPE)
NUM_HIDDEN_I = np.prod(HIDDEN_I_SHAPE)
GAUSSIAN_TRACE_POLICY = True

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


print("Expected random connections:")
expected_conns = 0
print("- I-H:", aux_conns := INPUT_SIZE * (NUM_HIDDEN_E + NUM_HIDDEN_I) * CONN_P["I-H"])
expected_conns += aux_conns * int(CONNECTIVITY_TYPE=="fixed")
print("- H-H:", aux_conns := ((NUM_HIDDEN_E + NUM_HIDDEN_I)**2) * CONN_P["H-H"])
expected_conns += aux_conns * int(CONNECTIVITY_TYPE=="fixed")
print("- H-P:", aux_conns := (NUM_HIDDEN_E + NUM_HIDDEN_I) * NUM_OUTPUT * CONN_P["H-P"])
expected_conns += aux_conns
print("- H-V:", aux_conns := (NUM_HIDDEN_E + NUM_HIDDEN_I) * CONN_P["H-V"])
expected_conns += aux_conns

print("Expected toroidal connections:")
print("- I-H:",
      aux_conns := expected_toroidal_connections(
          INPUT_SIZE,
          NUM_HIDDEN_E + NUM_HIDDEN_I,
          SIGMA_IN,
          desired_fan_in=DESIRED_FAN_IN_IN
      ))
expected_conns += aux_conns * int(CONNECTIVITY_TYPE=="toroidal")

print("- H-H:",
      aux_conns := expected_toroidal_connections(
          NUM_HIDDEN_E + NUM_HIDDEN_I,
          NUM_HIDDEN_E + NUM_HIDDEN_I,
          SIGMA_H,
          desired_fan_in=DESIRED_FAN_IN_H1
      ))
expected_conns += aux_conns * int(CONNECTIVITY_TYPE=="toroidal")

TRAIN = True

CHECKPOINT_BOARD_SIZE = None # "4_mid_completion"
if CHECKPOINT_BOARD_SIZE is not None:
    CONNECTIVITY_TYPE = "fixed"
KERNEL_PROFILING = False


gamma = 0.1** (1/WAIT_INC)
td_lambda = 0.1** (1/WAIT_INC)
td_error_trace_discount = 0.001**(1/WAIT_INC)

entropy_coeff = 0.0 #1e-4
entropy_decay = 0.99999 ** (1/WAIT_INC)
entropy_coeff_min = 0.0 #1e-6

dale_l1_reg = 0.0

serialiser = Numpy("snake_checkpoints")
network = Network(default_params)
hidden_layers = {}

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


def build_compiled_network(connectivity_type="fixed"):
    global dale_l1_reg
    network = Network(default_params)
    hidden_layers = {}

    with network:

        # ================= POPULATIONS =================

        input_pop = Population(
            SpikeInput(max_spikes=INPUT_SIZE * WAIT_INC), INPUT_SHAPE
        )

        hidden_layers["E"] = Population(
            AdaptiveLeakyIntegrateFire(
                v_thresh=0.61,
                tau_mem=10.0,
                tau_refrac=3.0,
                tau_adapt=300,
            ),
            HIDDEN_E_SHAPE
        )

        hidden_layers["I"] = Population(
            AdaptiveLeakyIntegrateFire(
                v_thresh=0.61,
                tau_mem=10.0,
                tau_refrac=3.0,
                tau_adapt=300,
            ),
            HIDDEN_I_SHAPE
        )

        value = Population(
            LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"),
            1
        )
        policy = Population(
            LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"),
            NUM_OUTPUT
        )

        # ================= INPUT → HIDDEN =================
        for layer, prob, c_type in [
            (hidden_layers["I"], CONN_P["I-H"], connectivity_type),
            (hidden_layers["E"], CONN_P["I-H"], connectivity_type),
            # (policy, CONN_P["D-O"], "fixed"),
            # (value, CONN_P["D-O"], "fixed"),
        ]:
            Connection(
                input_pop,
                layer,
                make_connectivity(
                    connectivity_type=c_type,
                    src_shape=INPUT_SHAPE,
                    p=prob,
                    sigma=SIGMA_IN,
                    desired_fan_in=DESIRED_FAN_IN_IN,
                    fan_in_scale=FAN_IN_SCALE_IN,
                    sign=1
                ),
                exc_inh_sign=1
            )

        # ================= EXCITATORY =================
        for layer, prob, c_type, sign in [
            (hidden_layers["I"], CONN_P["H-H"], connectivity_type, 1),
            (hidden_layers["E"], CONN_P["H-H"], connectivity_type, 1),
            (policy, CONN_P["H-P"], "fixed", 1),
            (value, CONN_P["H-V"], "fixed", 1),
        ]:
            Connection(
                hidden_layers["E"],
                layer,
                make_connectivity(
                    connectivity_type=c_type,
                    src_shape=HIDDEN_E_SHAPE,
                    p=prob,
                    sigma=SIGMA_H,
                    desired_fan_in=DESIRED_FAN_IN_H1,
                    sign=sign
                ),
                exc_inh_sign=sign
            )

        # ================= INHIBITORY =================
        for layer, prob, c_type, sign in [
            (hidden_layers["I"], CONN_P["H-H"], connectivity_type, -1),
            (hidden_layers["E"], CONN_P["H-H"], connectivity_type, -1),
            (policy, CONN_P["H-P"], "fixed", -1),
            (value, CONN_P["H-V"], "fixed", -1),
        ]:
            Connection(
                hidden_layers["I"],
                layer,
                make_connectivity(
                    connectivity_type=c_type,
                    src_shape=HIDDEN_I_SHAPE,
                    p=prob,
                    sigma=SIGMA_H,
                    desired_fan_in=DESIRED_FAN_IN_H2,
                    sign=sign
                ),
                exc_inh_sign=sign
            )

        # ================= FEEDBACK CONNECTIONS =================
        Connection(
            policy, value, Dense(weight=1.0),
            feedback_name="tde_transport"
        )
        for hidden_layer in hidden_layers.values():
            Connection(
                hidden_layer, value, Dense(weight=1.0),
                feedback_name="tde_transport"
            )
            sign = None
            if hidden_layer == hidden_layers["E"]:
                sign = 1
            elif hidden_layer == hidden_layers["I"]:
                sign = -1

            Connection(
                hidden_layer, policy,
                FixedProbability(CONN_P["F"], Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
                feedback_name="policy_feedback",
                exc_inh_sign=sign
            )
            Connection(
                hidden_layer, policy,
                FixedProbability(CONN_P["F"], Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
                feedback_name="policy_regularisation",
                exc_inh_sign=sign
            )
            Connection(
                hidden_layer, value,
                FixedProbability(CONN_P["F"], Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
                feedback_name="value_feedback",
                exc_inh_sign=sign
            )
            Connection(
                hidden_layer, value,
                FixedProbability(CONN_P["F"], Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
                feedback_name="value_regularisation",
                exc_inh_sign=sign
            )
            

    # ================= COMPILER =================
    dale_l1_reg = 0.0001/np.sqrt(DESIRED_FAN_IN_IN)
    if CONNECTIVITY_TYPE == "fixed":
        dale_l1_reg = 0.01/np.sqrt(max(INPUT_SIZE, NUM_HIDDEN_E))
    dale_l1_reg = 0
    print("L1 reg strength:", dale_l1_reg)
    compiler = EPropCompiler(
        example_timesteps=1,
        losses={
            policy: "sparse_categorical_crossentropy" 
                if not GAUSSIAN_TRACE_POLICY else 
                "mean_square_error",
            value: "mean_square_error"
        },
        optimiser=Adam(1e-4, soft_grad_clip=10), 
        # optimiser=Adam(1e-4, clamp_grad=(-5.0, 5.0)),
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
        policy_heads={
            policy: PolicyTypes.CATEGORICAL if not GAUSSIAN_TRACE_POLICY else PolicyTypes.GAUSSIAN_TRACE, 
        },
        value_head=value
    )

    if CHECKPOINT_BOARD_SIZE is not None:
        network.load((CHECKPOINT_BOARD_SIZE,), serialiser)

    compiled_net = compiler.compile(network)

    return compiled_net, network, input_pop, hidden_layers, policy, value

compiled_net, network, input_pop, hidden_layers, policy, value = \
    build_compiled_network(connectivity_type=CONNECTIVITY_TYPE)

train_callback_list = CallbackList(
    [*set(compiled_net.base_train_callbacks)] + [Checkpoint(serialiser)],
    compiled_network=compiled_net,
    num_batches=1, 
    num_epochs=1
)

all_metrics = {}

####################### TRAIN #######################
#####################################################

def make_repeated_spikes(
    indices,
    base_timestep,
    input_size,
    K,
    period=1,
    spike_prob=0.5
):
    """
    Probabilistic spike encoding.
    Each active input neuron fires with probability spike_prob at each timestep.
    """
    if len(indices) == 0 or K <= 0:
        return preprocess_spikes(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            input_size
        )
    indices = np.asarray(indices, dtype=np.int64)

    times = []
    idxs = []

    for k in range(K):
        t = base_timestep + k * period

        # Bernoulli sampling
        mask = np.random.rand(len(indices)) < spike_prob
        if not np.any(mask):
            continue

        fired = indices[mask]
        times.append(np.full(len(fired), t, dtype=np.int64))
        idxs.append(fired)

    if len(times) == 0:
        return preprocess_spikes(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            input_size
        )

    times = np.concatenate(times)
    idxs = np.concatenate(idxs)

    return preprocess_spikes(times, idxs, input_size)

def make_single_step_rate_spikes(values, timestep, input_size):
    values = np.clip(values, 0.0, 1.0)

    mask = np.random.rand(len(values)) < values

    if not np.any(mask):
        return preprocess_spikes(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            input_size
        )

    idxs = np.nonzero(mask)[0]
    times = np.full(len(idxs), timestep, dtype=np.int64)

    return preprocess_spikes(times, idxs.astype(np.int64), input_size)

# --- Helper: optionally compress frames before sending to reduce IPC size ---
def encode_frame(frame, compress=True, ext='.png', quality=80):
    if not compress:
        return frame
    ret, buf = cv2.imencode(ext, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ret:
        return frame
    return buf.tobytes()

def decode_frame(blob_or_array):
    if isinstance(blob_or_array, bytes):
        arr = np.frombuffer(blob_or_array, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    return blob_or_array

# --- Visualization process: matplotlib live charts ---
def viz_plots_loop(metrics_q: Queue, stop_event: mp.Event):
    plt.ion()
    fig, (ax1, ax1b, ax2, ax3) = plt.subplots(4, 1, figsize=(10, 12))

    plt.subplots_adjust(hspace=0.5)
    # initial empty lines
    snake_len_line, = ax1.plot([], [], label='Snake length', color=(0, 0.8, 0.1, 0.5), zorder=1)
    line_reward,    = ax1.plot([], [], label='Total Reward', color=(0, 0.1, 0.8, 0.5), zorder=2)
    avg_line,       = ax1.plot([], [], label='Running Avg',  color=(0.8, 0.05, 0.05, 1.0), zorder=3)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress')
    ax1.legend()

    line_reward_recent, = ax1b.plot([], [], label=f'Reward (last {WINDOW_EPISODES})')
    avg_line_recent,    = ax1b.plot([], [], label=f'Avg (last {WINDOW_EPISODES})')
    ax1b.set_title(f'Recent Training (last {WINDOW_EPISODES} episodes)')
    ax1b.legend()


    value_line, = ax2.plot([], [], label='Value')
    reward_trace_line, = ax2.plot([], [], label='Reward Trace')
    # value_function_line, = ax2.plot([], [], label='Value Function')

    ax2.set_title('Value (best run)')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Value')
    ax2.legend()

    prob_img = ax3.imshow(np.zeros((4,1)), aspect='auto', origin='lower', vmin=0, vmax=1)
    ax3.set_xlabel('Timestep'); ax3.set_ylabel('Action'); ax3.set_title('Action probs (best run)')
    fig.colorbar(prob_img, ax=ax3, fraction=0.02, pad=0.04)

    # local buffers
    ep_list = []
    rewards = []
    avgs = []
    lens = []
    best_values = []
    best_reward_traces = []
    best_value_function_values = []
    best_probs = None

    last_plot_time = 0.0
    plot_interval = 0.2  # seconds

    while not stop_event.is_set():
        try:
            # drain queue (non-blocking) to keep latest metrics
            while True:
                metrics = metrics_q.get_nowait()
                # expected fields: {'ep': int, 'reward': float, 'running_avg': float, 'snake_len': int,
                #                   'best_values': [..] (optional), 'best_probs': 2D array (optional)}
                ep = metrics.get('ep')
                if ep is not None:
                    ep_list.append(ep)
                    rewards.append(metrics.get('reward', 0.0))
                    avgs.append(metrics.get('running_avg', 0.0))
                    lens.append(metrics.get('snake_len', 0))
                if 'best_values' in metrics:
                    best_values = metrics['best_values']
                    best_reward_traces = metrics['best_reward_traces']
                    best_value_function_values = metrics['best_value_function_values']
                if 'best_probs' in metrics:
                    best_probs = metrics['best_probs']
                
        except Exception:
            pass

        now = time.time()
        if now - last_plot_time > plot_interval:
            if len(ep_list) > 0:
                axis = ep_list
                line_reward.set_data(axis, rewards)
                avg_line.set_data(axis, avgs)
                snake_len_line.set_data(axis, lens)
                ax1.relim(); ax1.autoscale_view()

            if len(rewards) > 0:
                recent_rewards = rewards[-WINDOW_EPISODES:]
                recent_avgs = avgs[-WINDOW_EPISODES:]

                axis_recent = np.arange(len(recent_rewards))
                line_reward_recent.set_data(axis_recent, recent_rewards)
                avg_line_recent.set_data(axis_recent, recent_avgs)

                ax1b.relim()
                ax1b.autoscale_view()

            if best_values:
                value_line.set_data(range(len(best_values)), best_values)
                reward_trace_line.set_data(range(len(best_reward_traces)), best_reward_traces)
                # value_function_line.set_data(range(len(best_value_function_values)), best_value_function_values)
                ax2.relim(); ax2.autoscale_view()

            if best_probs is not None:
                data = np.array(best_probs).T                     # shape: (n_actions, time_steps)
                n_actions, time_steps = data.shape

                prob_img.set_data(data)

                # IMPORTANT: force full stretching just like original code
                prob_img.set_extent([0, time_steps, 0, n_actions])

                ax3.set_xlim(0, time_steps)
                ax3.set_ylim(0, n_actions)
                ax3.set_aspect('auto')

            plt.pause(0.001)
            last_plot_time = now

        time.sleep(0.03)  # give CPU a break

    plt.close(fig)

# --- Visualization process: show best + random runs using OpenCV ---
def viz_runs_loop(best_run_q: Queue, random_run_q: Queue, stop_event: mp.Event, decompress=True):
    # best run stored as encoded frames (bytes) or arrays; we'll decode on display
    best_run = []
    random_run = []
    window_best = "Best Run"
    window_random = "Random Run"

    cv2.namedWindow(window_best, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_random, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(window_best, 600, 600)
    cv2.resizeWindow(window_random, 600, 600)

    t = 0
    while not stop_event.is_set():
        try:
            while True:
                # latest best run (replace)
                br = best_run_q.get_nowait()
                if br is None:
                    stop_event.set()
                    break
                # br is expected to be list of frames (encoded or arrays)
                best_run = br
        except Exception:
            pass

        try:
            while True:
                rr = random_run_q.get_nowait()
                if rr is None:
                    stop_event.set()
                    break
                random_run = rr
        except Exception:
            pass

        # Play best run in one window, cycling
        if best_run:
            if stop_event.is_set():
                break
            f_blob = best_run[t%len(best_run)]
            frame = decode_frame(f_blob) if decompress else f_blob
            if frame is None:
                continue
            cv2.imshow(window_best, frame)
            key = cv2.waitKey(1)  # adjust speed here
            if key == 27:  # Esc to exit
                stop_event.set()
                break

        # Play random run (if provided) once (or cycle)
        if random_run and t%(len(random_run)+10) < len(random_run):
            f_blob = random_run[t%(len(random_run)+10)]
            if stop_event.is_set():
                break
            frame = decode_frame(f_blob) if decompress else f_blob
            if frame is None:
                continue
            cv2.imshow(window_random, frame)
            key = cv2.waitKey(1)
            if key == 27:
                stop_event.set()
                break

        time.sleep(0.1)
        t += 1

    cv2.destroyWindow(window_best)
    cv2.destroyWindow(window_random)

# --- Wiring: start processes and pass queues to trainer ---
def start_visualizers():
    manager = Manager()
    metrics_q = manager.Queue(maxsize=10)
    best_run_q = manager.Queue(maxsize=2)
    random_run_q = manager.Queue(maxsize=2)
    stop_event = manager.Event()

    p_plots = mp.Process(target=viz_plots_loop, args=(metrics_q, stop_event), daemon=True)
    p_runs = mp.Process(target=viz_runs_loop, args=(best_run_q, random_run_q, stop_event), daemon=True)

    p_plots.start()
    p_runs.start()
    return manager, metrics_q, best_run_q, random_run_q, stop_event, p_plots, p_runs

def decode_population_vector(p):
    vx = p[2] - p[0]
    vy = p[1] - p[3]

    vec = np.array([vx, vy], dtype=np.float32)

    n = np.linalg.norm(vec)
    if n < 1e-6:
        return np.zeros(2, dtype=np.float32)

    return vec / n
# --- Modify your train_snake_agent to send updates instead of internal plotting ---
# Replace plt.ion() + figure creation in train_snake_agent with nothing and send updates to queues.
# I will show a skeleton wrapper around your train loop:
def train_snake_agent_with_ipc(episodes=1e10,
                               metrics_q: Queue=None,
                               best_run_q: Queue=None,
                               random_run_q: Queue=None,
                               compress_frames=True,
                               compress_quality=80):
    """
    This function is essentially your train_snake_agent() with the plotting removed.
    Instead it sends metrics and best_run frames via the provided queues.
    """
    global BOARD_SIZE
    opt_updt = 0
    with compiled_net:
        env = SnakeEnv(size=BOARD_SIZE, visible_range=VISIBLE_RANGE, wait_inc=WAIT_INC, scale=SCALE, inp_shape=INPUT_SHAPE)
        best_reward = -np.inf
        best_run = []
        running_avg = []
        snake_len_history = []
        smoothing = 0.95
        avg = 0

        # keep last best values and probs for plot process
        last_best_values = []
        last_best_reward_traces = []
        last_best_value_function_values = []
        last_best_probs = None

        train_callback_list.on_epoch_begin(0)
        train_callback_list.on_batch_begin(0)

        for ep in range(episodes):
            if (ep) % 1000 == 0:
                print("/////////////////////////////")
                connections_sum = 0
                print("Actual instantiated connections:")
                for k, v in extract_actual_sparse_connections(compiled_net).items():
                    connections_sum += v
                    print(f"- {k}: {v}")
                print("Total    connections:", connections_sum)
                print("Expected connections:", int(expected_conns))
                print("Difference          :", int(connections_sum-expected_conns))
                print("")
                print("Fanin statistics:")
                for k, v in extract_fanin_statistics(compiled_net).items():
                    print(f"- {k}: {v}")
                print("/////////////////////////////")
            if (ep+1) % 1000 == 0:
                best_reward = -np.inf
                compiled_net.save_connectivity((f"{BOARD_SIZE}_mid_completion",), serialiser)
                compiled_net.save((f"{BOARD_SIZE}_mid_completion",), serialiser)

            for m in all_metrics.values():
                m.reset()
            obs = env.reset()
            done = False
            total_reward = 0
            current_run = []
            current_values = []
            current_reward_traces = []
            current_value_function_values = []
            current_probs = []
            # total_td = 0
            frame = 1
            # value_target = 0

            """
            indices = obs.nonzero()[0]
            spikes = make_repeated_spikes(
                indices,
                compiled_net.genn_model.timestep,
                INPUT_SIZE,
                K=WAIT_INC,
                period=1
            )
            """

            env.wait_count = WAIT_INC
            reward_trace = 0
            # td_error_trace = 0

            while not done:
                # ---- encode observation
                current_values.append(compiled_net.get_readout(value)[0][0])
                current_reward_traces.append(reward_trace)
                
                obs_norm = obs.astype(np.float32) / 255.0

                spikes = make_single_step_rate_spikes(
                    obs_norm.reshape(-1),
                    compiled_net.genn_model.timestep,
                    INPUT_SIZE
                )

                compiled_net.set_input({input_pop: [spikes]})

                compiled_net.step_time(train_callback_list)

                # ---- read policy continuously
                policy_out = compiled_net.get_readout(policy).flatten()
                # compiled_net.neuron_populations[policy].vars["Action"].pull_from_device()
                # policy_out = compiled_net.neuron_populations[policy].vars["Action"].view
                action_vec = decode_population_vector(policy_out)

                current_probs.append((policy_out.copy()-policy_out.min())/((policy_out-policy_out.min()).sum()))

                # ---- env physics step
                obs, reward, done = env.step(action_vec)

                total_reward += reward

                reward_trace = reward_trace * 0.5 + reward

                if reward != 0:
                    compiled_net.losses[value].set_var(
                        compiled_net.neuron_populations[value],
                        "reward",
                        reward
                    )
                    
                # ---- capture frame
                frame_img = obs # env.local_img(scale=5)
                encoded = encode_frame(frame_img, compress=True)
                current_run.append(encoded)

                frame += 1
            
            for i in range(WAIT_INC):
                reward_trace = reward_trace * 0.5
                compiled_net.step_time(train_callback_list)
                
            compiled_net.genn_model.custom_update("GradientLearn")

            for o, custom_updates in compiled_net.optimisers:
                for c in custom_updates:
                    o.set_step(c, opt_updt := opt_updt + 1) 

            if dale_l1_reg > 0:
                compiled_net.genn_model.custom_update("DaleRL1")

            compiled_net.genn_model.custom_update("DalePrune")
            compiled_net.genn_model.custom_update("DaleRewire")


            # if compiled_net.optimisers[0][0].alpha > 1e-5 and np.mean(snake_len_history) > 2:
            #     compiled_net.optimisers[0][0].alpha = 1e-5

            # Update if new best run (send best run to viz process)
            if total_reward >= best_reward and len(current_probs) > 0:
                best_reward = total_reward
                best_run = [img for img in current_run]  # frames already possibly encoded
                last_best_values = list(current_values)
                last_best_reward_traces = list(current_reward_traces)
                last_best_value_function_values = list(current_reward_traces)
                for i in range(len(last_best_value_function_values)-1, 0, -1):
                    last_best_value_function_values[i-1] += last_best_value_function_values[i]*gamma
                last_best_probs = list(current_probs)

                # send best run into queue (non-blocking put — if queue full, replace oldest)
                try:
                    if best_run_q is not None:
                        # make a tiny container with values + probs for plotting + frames
                        # we'll send frames via best_run_q
                        if best_run_q.full():
                            try:
                                _ = best_run_q.get_nowait()  # drop oldest
                            except Exception:
                                pass
                        best_run_q.put_nowait(best_run)
                except Exception as e:
                    print("Best-run enqueue error:", e)

            # occasional random visualization sample (send a few frames)
            if random_run_q is not None and (ep % 100 == 0):
                # sample a short random chunk from current_run
                try:
                    if random_run_q.full():
                        try:
                            _ = random_run_q.get_nowait()
                        except:
                            pass
                    random_run_q.put_nowait(current_run)
                except Exception as e:
                    print("Random-run enqueue error:", e)

            snake_len_history.append(0)
            if len(snake_len_history) > 100:
                snake_len_history = snake_len_history[-100:]

            if avg == 0:
                avg = total_reward
            else:
                avg = smoothing * avg + (1 - smoothing) * total_reward
            running_avg.append(avg)

            # Send metrics for plotting
            if metrics_q is not None:
                metrics = {
                    'ep': ep,
                    'reward': total_reward,
                    'running_avg': avg,
                    'snake_len': 0 
                }
                if last_best_values:
                    metrics['best_values'] = last_best_values
                    metrics['best_reward_traces'] = last_best_reward_traces
                    metrics['best_value_function_values'] = last_best_value_function_values
                if last_best_probs is not None:
                    metrics['best_probs'] = last_best_probs
                try:
                    if metrics_q.full():
                        try:
                            metrics_q.get_nowait()
                        except:
                            pass
                    metrics_q.put_nowait(metrics)
                except Exception as e:
                    print("Metrics enqueue error:", e)

            # logging
            if ep %10 == 0:
                print(
                    f"Episode {ep+1} - "
                    f"Total reward: {' ' if total_reward >= 0 else ''}{total_reward:.2f} "
                    f"- Best reward: {best_reward:.2f} "
                    f"- Snake len: {0:2d} "
                    f"- Snake len avg (last 100): {np.mean(snake_len_history):.2f} "
                    f"- Frame death: {frame} "
                    f"- Alpha: {compiled_net.optimisers[0][0].alpha:.8f}"
                )

            # optional checkpoint / early stop etc.

        # After training finished, send sentinel None to visualizers so they stop cleanly
        if best_run_q is not None:
            try:
                best_run_q.put_nowait(None)
            except:
                pass
        if random_run_q is not None:
            try:
                random_run_q.put_nowait(None)
            except:
                pass

# --- Example of launching everything in __main__ ---
if __name__ == "__main__":
    # start visualizers
    manager, metrics_q, best_run_q, random_run_q, stop_event, p_plots, p_runs = start_visualizers()

    # run trainer in main process (keeps compiled_net / GPU access local)
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
        # signal visualizers to stop
        stop_event.set()
        time.sleep(0.2)
        if p_plots.is_alive():
            p_plots.terminate()
        if p_runs.is_alive():
            p_runs.terminate()