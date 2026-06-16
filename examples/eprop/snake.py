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

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import os 
import csv

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
    def __init__(self, size=28, visible_range=5, scale=2, wait_inc=5, inp_shape=(5,5,3)):
        assert visible_range % 2 == 1, "visible_range must be odd"
        self.size = size
        self.visible_range = visible_range
        self.scale = scale
        self.wait_inc = wait_inc
        self.won = False
        self.inp_shape = inp_shape
        self.reset()

    def reset(self):
        self.snake = [
            #     [
            #     (self.size // 2 - 1, self.size // 2),
            #     (self.size // 2 + 1, self.size // 2),
            #     (self.size // 2, self.size // 2 + 1),
            #     (self.size // 2, self.size // 2 - 1),
            # ][random.randint(0, 3)],
            # (self.size // 2 - 1, self.size // 2),
            (self.size // 2, self.size // 2)
        ]

        self.dir_idx = 1
    
        self.target_angle = self.dir_idx * 90
        self.current_angle = self.target_angle

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
            assert len(self.apples) == 0
            return
        #elif max(np.ceil(np.log10(len(empty_cells))), 1) > len(self.apples):
         #   self.apples.extend(random.sample(empty_cells, k=int(max(np.ceil(np.log10(len(empty_cells))), 1) - len(self.apples))))
        self.apples = [random.choice(empty_cells)]

    def step(self, action):

        if self.done:
            raise Exception("Environment needs reset. Call env.reset().")
        
        if self.wait_count > 0:
            self.wait_count -= 1
            reward = 0.0
            return self.get_local_img_observation(), reward/100, self.done

        dirs = ['left', 'up', 'right', 'down']
        # self.dir_idx = (self.dir_idx + (action-1)) % 4
        # new_dir = dirs[self.dir_idx]
        new_dir = dirs[action]
        
        reward = - 0.0*90/self.size * ((self.steps_since_last_apple+1) % self.size == 0)

        if (self.direction == 'up' and new_dir == 'down') or \
           (self.direction == 'down' and new_dir == 'up') or \
           (self.direction == 'left' and new_dir == 'right') or \
           (self.direction == 'right' and new_dir == 'left'):
        #     reward -= 10
            new_dir = self.direction
            # reward -= 100
            # self.done = True
            # return self.get_observation(), reward/100, True 

        self.direction = new_dir
        self.dir_idx = dirs.index(self.direction)
        self.target_angle = ((self.dir_idx-1)%4) * 90
        # self.current_angle += (self.target_angle - self.current_angle) / self.wait_inc
        self.current_angle = self.target_angle

        head_y, head_x = self.snake[0]
        if self.direction == 'up':
            head_y -= 1
        elif self.direction == 'down':
            head_y += 1
        elif self.direction == 'left':
            head_x -= 1
        elif self.direction == 'right':
            head_x += 1

        # if head_y < 0:
        #     head_y = self.size-1
        # elif head_y >= self.size:
        #     head_y = 0
        # elif head_x < 0:
        #     head_x = self.size-1
        # elif head_x >= self.size:
        #     head_x = 0
            
        new_head = (head_y, head_x)
        apple_found = None
        for apple in self.apples:
            if new_head == apple:
                apple_found = apple
                break

        if ((head_y < 0 or head_y >= self.size or
            head_x < 0 or head_x >= self.size or
            new_head in self.snake[:-1])
        ):
            reward -= 100
            self.done = True
            return self.get_local_img_observation(), reward/100, True

        self.snake.insert(0, new_head)

        if apple_found is not None:
            self.apples.remove(apple_found)
            self.spawn_apples()
            reward += 100.0
            if len(self.apples) == 0:
                self.done = True
                # reward += 500.0
                self.won = True
            if len(self.snake) > 0.75 * self.size**2 and self.size > 5:
                self.won = True
            self.steps_since_last_apple = 0
        else:
            self.snake.pop()
            self.steps_since_last_apple += 1
        
        if self.steps_since_last_apple > self.size**2:
            self.done = True
            reward -= 100.0
            return self.get_local_img_observation(), reward/100, self.done

        self.wait_count = self.wait_inc

        return self.get_local_img_observation(), reward/100, self.done

    def get_observation(self):
        """
        Returns a (4, visible_range, visible_range) tensor:
        0: snake body
        1: head
        2: apple
        3: walls
        """
        v = self.visible_range
        r = v // 2
        head_y, head_x = self.snake[0]

        obs = np.zeros((v, v, 3), dtype=np.float32)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y = head_y + dy
                x = head_x + dx
                local_y = dy + r
                local_x = dx + r

                # check walls
                if y == -1 or y == self.size + 1 or x == -1 or x == self.size + 1:
                    obs[local_y, local_x, 0] = 1.0
                    continue

                # body
                if (y, x) in self.snake:
                    obs[local_y, local_x, 1] = 1.0

                # # head
                # if (y, x) == (head_y, head_x):
                #     obs[3, local_y, local_x] = 1.0

                # apple
                if (y, x) in self.apples:
                    obs[local_y, local_x, 2] = 1.0

        return obs.flatten()
    
    def get_local_img_observation(self):
        img = self.local_img(scale=self.scale)
        return cv2.resize(img, (self.inp_shape[0], self.inp_shape[1]), interpolation=cv2.INTER_NEAREST) / 255

    # def get_local_img_observation(self):
    #     img = self.local_img(scale=self.scale)

    #     # center = (img.shape[1] // 2, img.shape[0] // 2)
    #     # M = cv2.getRotationMatrix2D(center, self.current_angle, 1.0)
    #     # img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        
    #     if self.dir_idx == 0:      # facing left → rotate 90° CW
    #         img = np.rot90(img, k=3)
    #     elif self.dir_idx == 2:    # facing right → rotate 90° CCW
    #         img = np.rot90(img, k=1)
    #     elif self.dir_idx == 3:    # facing down → rotate 180°
    #         img = np.rot90(img, k=2)

    #     return cv2.resize(
    #         img,
    #         (self.inp_shape[0], self.inp_shape[1]),
    #         interpolation=cv2.INTER_NEAREST
    #     ) / 255.0

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
        head_y, head_x = self.snake[0]

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y = head_y + dy
                x = head_x + dx

                # walls
                if y < 0 or y >= self.size or x < 0 or x >= self.size:
                    continue


                img[y, x] = [0,0,0]

                # body
                if (y, x) in self.snake[1:]:
                    img[y, x] = [0, 255, 0]

                # head
                if (y, x) == (head_y, head_x):
                    img[y, x] = [0, 155, 0]

                # apple
                for apple in self.apples:
                    if (y, x) == apple:
                        img[y, x] = [0, 0, 255]
        
        return cv2.resize(img, (self.size * scale, self.size * scale), interpolation=cv2.INTER_NEAREST)

    def local_img(self, scale=10):
        """
        Returns a local RGB image centered on the snake's head,
        matching the visible_range used in get_observation().
        Channels:
            Green  = snake body
            Dark green = snake head
            Red    = apple
            Gray   = walls
        """
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

                # single-cell outline walls
                if (
                    (x == -1 and -1 <= y < self.size+1) or
                    (x == self.size and -1 <= y < self.size+1) or
                    (y == -1 and -1 <= x < self.size+1) or
                    (y == self.size and -1 <= x < self.size+1)
                ):
                    img[local_y, local_x] = [100, 100, 100]  # gray
                    continue
                
                # body
                if (y, x) in self.snake[1:]:
                    img[local_y, local_x] = [0, 255, 0]

                # head
                if (y, x) == (head_y, head_x):
                    img[local_y, local_x] = [0, 155, 0]

                # appleobs
                for apple in self.apples:
                    if (y, x) == apple:
                        img[local_y, local_x] = [0, 0, 255]

        # upscale for visualization
        return cv2.resize(img, (v * scale, v * scale), interpolation=cv2.INTER_NEAREST)

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
BOARD_SIZE = 5

VISIBLE_RANGE = 5
SCALE = 4

WAIT_INC = 30

INPUT_C = 3

INPUT_SHAPE = (20, 20, INPUT_C)
DOWNSAMPLE_SHAPE = (30, 30, INPUT_C)
UNIFIED_SHAPE = (30, 30, INPUT_C)
HIDDEN_E_SHAPE = (20, 20, INPUT_C)
HIDDEN_I_SHAPE = (15, 15, INPUT_C)
INPUT_SIZE = np.prod(INPUT_SHAPE)
NUM_HIDDEN_E = np.prod(HIDDEN_E_SHAPE)
NUM_HIDDEN_I = np.prod(HIDDEN_I_SHAPE)
GAUSSIAN_TRACE_POLICY = False

SIGMA_IN = 0.1
SIGMA_H = 0.05

DESIRED_FAN_IN_IN = 300
DESIRED_FAN_IN_H1 = 300
DESIRED_FAN_IN_H2 = 300

# FAN_IN_SCALE_IN = 0.25

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
NUM_OUTPUT_CHART = NUM_OUTPUT
if GAUSSIAN_TRACE_POLICY:
    NUM_OUTPUT = 1
    NUM_OUTPUT_CHART = 3

CONN_P = {
    "I-H": 0.001,
    "D-O": 0.001,
    "H-H": 0.005,
    #"H-H": np.log(NUM_HIDDEN_1+NUM_HIDDEN_2)/(NUM_HIDDEN_1 + NUM_HIDDEN_2),
    "H-P": 0.1,
    "H-V": 0.1,
    "F": 1.0
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

CHECKPOINT_BOARD_SIZE = None # "7"
if CHECKPOINT_BOARD_SIZE is not None:
    CONNECTIVITY_TYPE = "fixed"
KERNEL_PROFILING = False


reward_decay = 0.1 ** (1/WAIT_INC)
gamma = 0.5 ** (1/WAIT_INC)
td_lambda = 0.8 ** (1/WAIT_INC)
td_error_trace_discount = 0.001**(1/WAIT_INC)

entropy_coeff = 1e-0
entropy_decay = 0.99999 ** (1/WAIT_INC)
entropy_coeff_min = 0.0 * 1e-7

dale_l1_reg = 0.0

serialiser = Numpy("snake_checkpoints_unified")
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

@dataclass
class EILayerConfig:
    """Configuration for a single EI layer."""
    e_shape: Tuple[int, ...]
    i_shape: Tuple[int, ...]
    
    # Neuron params
    v_thresh: float = 0.61
    tau_mem: float = 10.0
    tau_refrac: float = 3.0
    tau_adapt: float = 300.0
    beta: float = 0.0174
    
    # Connectivity
    connectivity_type: str = "toroidal"   # "toroidal" | "fixed"
    sigma: float = 0.05
    desired_fan_in_ee: int = 100
    desired_fan_in_ei: int = 100
    desired_fan_in_ie: int = 100
    desired_fan_in_ii: int = 100
    p_ee: float = 0.005  # used only if connectivity_type == "fixed"
    p_ei: float = 0.005
    p_ie: float = 0.005
    p_ii: float = 0.005

    # Weight init scales
    mean_scale: float = 0.1
    sd_scale: float = 0.05


class EILayer:
    """
    A single Excitatory-Inhibitory layer.
    
    Creates E and I populations and wires all four internal
    connections (E→E, E→I, I→E, I→I) using Dale's law signs.
    
    External connections (input→layer, layer→output) are handled
    by connect_input() / connect_output(), keeping the EI layer
    self-contained but composable.
    """

    def __init__(self, cfg: EILayerConfig, name: str = ""):
        self.cfg = cfg
        self.name = name
        self.e: Optional[Population] = None
        self.i: Optional[Population] = None
        self._internal_connections: list = []

    # ------------------------------------------------------------------
    # Build populations + internal wiring (call inside `with network:`)
    # ------------------------------------------------------------------
    def build(self):
        cfg = self.cfg
        suffix = f"_{self.name}" if self.name else ""

        neuron_kwargs = dict(
            v_thresh=cfg.v_thresh,
            tau_mem=cfg.tau_mem,
            tau_refrac=cfg.tau_refrac,
            tau_adapt=cfg.tau_adapt,
            beta=cfg.beta,
        )

        self.e = Population(AdaptiveLeakyIntegrateFire(**neuron_kwargs), cfg.e_shape)
        self.i = Population(AdaptiveLeakyIntegrateFire(**neuron_kwargs), cfg.i_shape)

        # Wire all four internal connections
        internal = [
            # (pre,    post,   src_shape,    fan_in,              p,          sign)
            (self.e, self.e, cfg.e_shape, cfg.desired_fan_in_ee, cfg.p_ee,  +1),
            (self.e, self.i, cfg.e_shape, cfg.desired_fan_in_ei, cfg.p_ei,  +1),
            (self.i, self.e, cfg.i_shape, cfg.desired_fan_in_ie, cfg.p_ie,  -1),
            (self.i, self.i, cfg.i_shape, cfg.desired_fan_in_ii, cfg.p_ii,  -1),
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

        return self  # allow chaining: layer = EILayer(cfg).build()

    # ------------------------------------------------------------------
    # External connectivity helpers
    # ------------------------------------------------------------------
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
        """
        Connect an external source population into both E and I
        populations of this layer (always excitatory input).
        """
        cfg = self.cfg
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

    def connect_feedback(
        self,
        output_pop: Population,
        feedback_name: str,
        p: float = 0.5,
        n_output: int = 4,
    ):
        """
        Wire both E and I populations as feedback sources to an
        output head, respecting Dale's law.
        """
        for hidden, sign in ((self.e, +1), (self.i, -1)):
            Connection(
                hidden, output_pop,
                FixedProbability(p, Normal(sd=1.0 / np.sqrt(n_output))),
                feedback_name=feedback_name,
                exc_inh_sign=sign,
            )
    
    def populations(self):
        """Return (e, i) tuple — useful for iterating over all hidden pops."""
        return self.e, self.i


def build_compiled_network(connectivity_type="fixed"):
    global dale_l1_reg
    network = Network(default_params)
    hidden_layers = {}

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
        input_pop = Population(SpikeInput(max_spikes=INPUT_SIZE * WAIT_INC), INPUT_SHAPE)

        ei_layers = []
        for i in range(1):  # increase to stack more layers
            ei_layers.append(EILayer(ei_cfg, name=f"L{i+1}").build())

        policy_field = Population(
            AdaptiveLeakyIntegrateFire(v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=300),
            HIDDEN_I_SHAPE
        )
        value_field = Population(
            AdaptiveLeakyIntegrateFire(v_thresh=0.61, tau_mem=10.0, tau_refrac=3.0, tau_adapt=300),
            HIDDEN_I_SHAPE
        )

        # policy_field = EILayer(ei_cfg, name=f"policy_field").build()
        # value_field = EILayer(ei_cfg, name=f"value_field").build()

        policy = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), NUM_OUTPUT)
        value  = Population(LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"), 1)

        # Input → first EI layer (excitatory only, using full connectivity_type)
        ei_layers[0].connect_from(
            input_pop, INPUT_SHAPE,
            connectivity_type=connectivity_type,
            desired_fan_in=DESIRED_FAN_IN_IN,
            sigma=SIGMA_IN,
            # fan_in_scale=FAN_IN_SCALE_IN,
        )

        # Stack EI layers
        for i in range(len(ei_layers) - 1):
            ei_layers[i].connect_to_next(ei_layers[i+1])

        # Last EI layer → field layers
        ei_layers[-1].connect_to_field(policy_field, p=CONN_P["H-H"])
        ei_layers[-1].connect_to_field(value_field,  p=CONN_P["H-H"])

        # Field layers → output heads (forward + feedback)
        for field, head, feedback_name in (
            (policy_field, policy, "policy_feedback"),
            (value_field,  value,  "value_feedback"),
        ):
            # for pop in field.populations():
            Connection(
                field, head,
                make_connectivity("fixed", src_shape=HIDDEN_I_SHAPE, p=0.99999, sign=None),
                exc_inh_sign=None
            )
            Connection(
                field, head,
                FixedProbability(0.99999, Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
                feedback_name=feedback_name,
                exc_inh_sign=None
            )
           
        # for pop in ei_layers[-1].populations():
        #     Connection(
        #         pop, policy,
        #         make_connectivity("fixed", src_shape=HIDDEN_I_SHAPE, p=CONN_P["H-P"], sign=None),
        #         exc_inh_sign=None
        #     )
        #     Connection(
        #         pop, value,
        #         make_connectivity("fixed", src_shape=HIDDEN_I_SHAPE, p=CONN_P["H-V"], sign=None),
        #         exc_inh_sign=None
        #     )

        # tde_transport from policy, all EI pops, and both fields
        Connection(policy, value, Dense(weight=1.0), feedback_name="tde_transport")
        for layer in ei_layers:
            for pop in layer.populations():
                Connection(pop, value, Dense(weight=1.0), feedback_name="tde_transport")
        for field in (policy_field, value_field):
            # for pop in field.populations():
            Connection(field, value, Dense(weight=1.0), feedback_name="tde_transport")

        # policy/value feedback from EI layers
        for layer in ei_layers:
            for pop in layer.populations():
                Connection(
                    pop, policy,
                    FixedProbability(CONN_P["F"], Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
                    feedback_name="policy_feedback",
                    exc_inh_sign=None
                )
                Connection(
                    pop, value,
                    FixedProbability(CONN_P["F"], Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
                    feedback_name="value_feedback",
                    exc_inh_sign=None
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
            policy: "mean_square_error" # "sparse_categorical_crossentropy" 
                if not GAUSSIAN_TRACE_POLICY else 
                "mean_square_error",
            value: "mean_square_error"
        },
        optimiser=Adam(8e-5), #, soft_grad_clip=10), 
        # optimiser=Adam(1e-4, clamp_grad=(-10.0, 10.0)),
        # c_reg=1e-2,
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
        dale_rewiring_l1_strength=dale_l1_reg,
        policy_heads={
            policy: PolicyTypes.GENERIC 
            if not GAUSSIAN_TRACE_POLICY 
            else PolicyTypes.GAUSSIAN_TRACE, 
        },
        value_head=value
    )

    if CHECKPOINT_BOARD_SIZE is not None:
        network.load((CHECKPOINT_BOARD_SIZE,), serialiser)

    compiled_net = compiler.compile(network)

    # return compiled_net, network, input_pop, hidden_layers, policy, value
    return compiled_net, network, input_pop, {i: l for i, l in enumerate(ei_layers[0].populations())}, policy, value

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

def make_poisson_spikes(
    values,    
    base_timestep,
    input_size,
    K,
    bg_noise=0.0   
):
    values = np.clip(values + bg_noise, 0.0, 1.0) * 0.3
    # print(values)
    # for i in range(INPUT_SHAPE[0]):
    #     print()
    #     for _ in range(3):
    #         for j in range(INPUT_SHAPE[1]):
    #                 symbol = ""
    #                 if values[i*INPUT_SHAPE[1]*INPUT_SHAPE[2] + j* INPUT_SHAPE[2] + 0] > 0.1:
    #                     symbol += "r"
    #                 else:
    #                     symbol += "-"
    #                 if values[i*INPUT_SHAPE[1]*INPUT_SHAPE[2] + j* INPUT_SHAPE[2] + 1] > 0.1:
    #                     symbol += "g"
    #                 else:
    #                     symbol += "-"
    #                 if values[i*INPUT_SHAPE[1]*INPUT_SHAPE[2] + j* INPUT_SHAPE[2] + 2] > 0.1:
    #                     symbol += "b"
    #                 else:
    #                     symbol += "-"
    #                 print(symbol, end="  ")
    #         print()
    # print()

    times = []
    idxs = []

    for i, v in enumerate(values):
        if v <= 0:
            continue

        # Sample K Bernoulli trials with probability = intensity
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

    prob_img = ax3.imshow(np.zeros((NUM_OUTPUT_CHART,1)), aspect='auto', origin='lower', vmin=0, vmax=1)
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
        while not metrics_q.empty():
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

        plt.pause(0.1)
        time.sleep(0.1)  # give CPU a break

    plt.close(fig)

# --- Visualization process: show best + random runs using OpenCV ---
def viz_runs_loop(
        best_run_q: Queue, 
        random_run_q: Queue, 
        stop_event: mp.Event, 
        decompress=True):
    # best run stored as encoded frames (bytes) or arrays; we'll decode on display
    best_run = []
    random_run = []
    window_best = "Best Run"
    window_random = "Random Run"

    cv2.namedWindow(window_best)
    cv2.namedWindow(window_random)
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


def viz_sigma_loop(sigma_q: Queue, stop_event: mp.Event, shape_e, shape_i):
    plt.ion()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # 1 — Mean log sigma over time
    sigma_mean_e_line, = ax1.plot([], [], label='E layer', color='orange')
    sigma_mean_i_line, = ax1.plot([], [], label='I layer', color='blue')
    ax1.set_title('mean log sigma over training\nlog(-1)→σ=0.37  log(-2)→σ=0.14  log(-7)→σ=0.001')
    ax1.set_ylabel('log sigma')
    ax1.legend(fontsize=8)

    # 2 — Current sigma distribution histogram
    ax2.set_title('sigma distribution — current')
    ax2.set_xlabel('sigma')

    # 3 — Log sigma heatmap over neurons and time (E layer)
    sigma_hist_img_e = ax3.imshow(np.zeros((shape_e[0]*shape_e[1], 1)), 
                                   aspect='auto', cmap='RdBu_r', vmin=-8, vmax=1)
    ax3.set_title('log sigma per neuron over time — E')
    ax3.set_xlabel('episode sample')
    ax3.set_ylabel('neuron index')
    fig.colorbar(sigma_hist_img_e, ax=ax3, fraction=0.02, pad=0.04)

    # 4 — 2D spatial heatmap of current sigma (E layer only, I layer smaller)
    heatmap_img = ax4.imshow(np.zeros(shape_e[:2]), aspect='auto',
                          cmap='RdBu_r', vmin=-8, vmax=0)
    
    ax4.set_title('log sigma spatial map — E (current)')
    fig.colorbar(heatmap_img, ax=ax4, fraction=0.02, pad=0.04)

    # local buffers
    sigma_mean_e_history = []
    sigma_mean_i_history = []
    sigma_dist_e_history = []

    while not stop_event.is_set():
        try:
            while True:
                data = sigma_q.get_nowait()
                log_sigma_e = data['E']
                log_sigma_i = data['I']

                # update buffers
                sigma_mean_e_history.append(log_sigma_e.mean())
                sigma_mean_i_history.append(log_sigma_i.mean())
                sigma_dist_e_history.append(log_sigma_e.copy())

                # 1 — mean over time
                sigma_mean_e_line.set_data(range(len(sigma_mean_e_history)), sigma_mean_e_history)
                sigma_mean_i_line.set_data(range(len(sigma_mean_i_history)), sigma_mean_i_history)
                ax1.relim(); ax1.autoscale_view()

                # 2 — current distribution
                ax2.cla()
                ax2.hist(np.exp(log_sigma_e), bins=40, alpha=0.6, 
                         color='orange', label=f'E mean={np.exp(log_sigma_e).mean():.3f}')
                ax2.hist(np.exp(log_sigma_i), bins=40, alpha=0.6,
                         color='blue', label=f'I mean={np.exp(log_sigma_i).mean():.3f}')
                ax2.set_title('sigma distribution — current')
                ax2.set_xlabel('sigma')
                ax2.legend(fontsize=8)

                # 3 — heatmap over neurons and time
                e_arr = np.array(sigma_dist_e_history).T  # (N_e, T)
                sigma_hist_img_e.set_data(e_arr)
                sigma_hist_img_e.set_extent([0, e_arr.shape[1], 0, e_arr.shape[0]])
                ax3.set_xlim(0, e_arr.shape[1])
                ax3.set_ylim(0, e_arr.shape[0])

                # 4 — spatial map
                heatmap_img.set_data(log_sigma_e.reshape(shape_e).mean(axis=2))
                # After set_data:
                vmin = log_sigma_e.min()
                vmax = log_sigma_e.max()
                heatmap_img.set_clim(vmin, vmax)
                sigma_hist_img_e.set_clim(vmin, vmax)
                ax4.set_title(f'log sigma spatial map — E  mean={log_sigma_e.mean():.3f}  std={log_sigma_e.std():.3f}')
                ax4.set_title(f'log sigma spatial map — E  std={log_sigma_e.std():.3f}')

        except Exception:
            pass

        plt.pause(0.03)
        time.sleep(0.03)

    plt.close(fig)


# --- Wiring: start processes and pass queues to trainer ---
def start_visualizers():
    manager = Manager()
    metrics_q = manager.Queue(maxsize=1000)
    best_run_q = manager.Queue(maxsize=2)
    random_run_q = manager.Queue(maxsize=2)
    sigma_q = manager.Queue(maxsize=2)
    stop_event = manager.Event()

    p_plots = mp.Process(target=viz_plots_loop, args=(metrics_q, stop_event), daemon=True)
    p_runs = mp.Process(target=viz_runs_loop, args=(best_run_q, random_run_q, stop_event), daemon=True)
    p_sigma = mp.Process(target=viz_sigma_loop, args=(sigma_q, stop_event, HIDDEN_E_SHAPE, HIDDEN_I_SHAPE), daemon=True)
    # p_sigma = None


    p_plots.start()
    p_runs.start()
    p_sigma.start()
    return manager, metrics_q, best_run_q, random_run_q, sigma_q, stop_event, p_plots, p_runs, p_sigma


# CSV_OUTPUT = "outputs/weight-back-dist-eprop-fields-hidden-layers-1-lambda-099.csv"
# CSV_OUTPUT = "outputs/rand-eprop-with-noise-little-dist-weight.csv"
# CSV_OUTPUT = "outputs/rand-eprop-fields-hidden-layers-5.csv"
# CSV_OUTPUT = "outputs/weight-dist-eprop-fields-hidden-layers-5.csv"
CSV_OUTPUT = "outputs/experiment_1.csv"
os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)

# ALWAYS reset file for a new run
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
        "frequency"
    ])

# --- Modify your train_snake_agent to send updates instead of internal plotting ---
# Replace plt.ion() + figure creation in train_snake_agent with nothing and send updates to queues.
# I will show a skeleton wrapper around your train loop:
def train_snake_agent_with_ipc(episodes=10000,
                               metrics_q: Queue=None,
                               best_run_q: Queue=None,
                               random_run_q: Queue=None,
                               sigma_q: Queue=None,
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

        action = np.zeros(NUM_OUTPUT)
        
        # for hidden_layer in hidden_layers.values():
        #     compiled_net.neuron_populations[hidden_layer].vars["Beta"].pull_from_device()
        #     betas = compiled_net.neuron_populations[hidden_layer].vars["Beta"].view
        #     compiled_net.neuron_populations[hidden_layer].vars["Beta"].view[:] = betas * (0.2 > np.random.uniform(0.0, 1.0, betas.shape))
        #     compiled_net.neuron_populations[hidden_layer].vars["Beta"].push_to_device()
        
        v_avg = 0
        v_reg_loss_avg = 0
        freq_avg = 0
        gamma_disc_reward = 0

        for ep in range(episodes):
            # if env.won:
            #     compiled_net.save_connectivity((BOARD_SIZE,), serialiser)
            #     compiled_net.save((BOARD_SIZE,), serialiser)
            #     best_reward = -np.inf
            #     BOARD_SIZE += 1
            #     print(f"WON! Increasing board size to {BOARD_SIZE}")
            #     env = SnakeEnv(size=BOARD_SIZE, visible_range=VISIBLE_RANGE, wait_inc=WAIT_INC, scale=SCALE, inp_shape=INPUT_SHAPE)
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
                # print("Sleeping...")
                # for _ in range(1000):
                #     spikes = make_rate_coded_spikes(
                #         np.zeros(INPUT_SIZE),
                #         compiled_net.genn_model.timestep,
                #         INPUT_SIZE,
                #         K=WAIT_INC,
                #         bg_noise=0.01                  
                #     )
                #     compiled_net.set_input({input_pop: [spikes]})
                #     for _ in range(WAIT_INC):
                #         compiled_net.step_time(train_callback_list)
                #     compiled_net.genn_model.custom_update("GradientLearn")
                #     for o, custom_updates in compiled_net.optimisers:
                #         for c in custom_updates:
                #             o.set_step(c, opt_updt := opt_updt+1)

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
            ep_frames = 0
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
            spikes = make_poisson_spikes(
                obs.reshape(-1),
                compiled_net.genn_model.timestep,
                INPUT_SIZE,
                K=WAIT_INC                       # tune this
            )
            compiled_net.set_input({input_pop: [spikes]})
            # compiled_net.step_time(train_callback_list)
            # previous_value_estimate = compiled_net.get_readout(value)[0][0]
            env.wait_count = WAIT_INC
            reward_trace = 0
            td_error_sum_abs = 0

            while not done:
                gamma_disc_reward *= gamma
                action_label = 0
                current_values.append(compiled_net.get_readout(value)[0].mean())
                current_reward_traces.append(reward_trace)
        
                if env.wait_count == 0:
                    # compiled_net.neuron_populations[hidden_layers["E"]].vars["ISynSigmaEps"].pull_from_device()
                    # print(sum(compiled_net.neuron_populations[hidden_layers["E"]].vars["ISynSigmaEps"].view), sum(np.exp(compiled_net.neuron_populations[hidden_layers["E"]].vars["ISynSigmaEps"].view)) )
                    # print("----------------")
                    if GAUSSIAN_TRACE_POLICY:
                        # action = compiled_net.get_readout(policy).flatten()
                        compiled_net.neuron_populations[policy].vars["TanhOut"].pull_from_device()
                        action_scalar = compiled_net.neuron_populations[policy].vars["TanhOut"].view[0]

                        x = np.clip(action_scalar, -1.0, 1.0)
                        p_left  = max(0.0, -x)
                        p_right = max(0.0,  x)
                        p_fwd   = 1.0 - abs(x)

                        probs = np.array([p_left, p_fwd, p_right])
                        probs = probs / probs.sum()

                        # map to your action space
                        # assuming: 0=left, 1=forward, 2=right
                        action_label = np.random.choice(3, p=probs)

                        # relative action → absolute direction
                        # 0 = left turn, 1 = forward, 2 = right turn
                        dir_idx = env.dir_idx
                        if action_label == 0:   # left
                            dir_idx = (dir_idx - 1) % 4
                        elif action_label == 2: # right
                            dir_idx = (dir_idx + 1) % 4
                        
                        # forward = no change
                        action_label = dir_idx
                    else:
                        logits = compiled_net.get_readout(policy).flatten()
                        shifted_logits = logits - logits.max()
                        exp_logs = np.exp(shifted_logits)
                        probs = exp_logs / (exp_logs.sum() + 1e-8)
                        
                        # if abs(sum(probs) - 1.0) > 0.0001:
                        #     raise ValueError(f"BAD PROBS {probs}")
                        action_label = np.random.choice(NUM_OUTPUT, p=probs)

                        y_true = np.zeros(NUM_OUTPUT)
                        y_true[action_label] = 1.0
                        PG = (probs - y_true)
                        
                        # log_p = np.log(probs + 1e-8)
                        # entropy = -np.sum(probs * log_p)

                        # entropy_grad_logits = -probs * (log_p + entropy)

                        # E = entropy_coeff * entropy_grad_logits

                        # Write into staging vars — sim code will move to PG/E next timestep
                        compiled_net.neuron_populations[policy].vars["pre_PG"].view[:] = PG.astype(np.float32)
                        compiled_net.neuron_populations[policy].push_var_to_device("pre_PG")

                        # compiled_net.neuron_populations[policy].vars["pre_PRew"].view[:] = -E.astype(np.float32)
                        # compiled_net.neuron_populations[policy].push_var_to_device("pre_PRew")

                        # compiled_net.neuron_populations[policy].vars["pre_E"].view[:] = E.astype(np.float32)
                        # compiled_net.neuron_populations[policy].push_var_to_device("pre_E")

                        # GPU SOFTMAX VERSION
                        # compiled_net.losses[policy].set_target(
                        #     compiled_net.neuron_populations[policy],
                        #     [action_label], policy.shape, 
                        #     compiled_net.genn_model.batch_size,
                        #     compiled_net.example_timesteps
                        # )
                        # compiled_net.losses[policy].set_var(
                        #     compiled_net.neuron_populations[policy], "actionTaken", 1.0
                        # )
                        
                    current_probs.append(probs)

                obs, reward, done = env.step(action_label)
                total_reward += reward
                gamma_disc_reward += reward
                reward_trace = reward_trace * reward_decay + reward * 1.0
                if reward != 0:
                    compiled_net.losses[value].set_var(
                        compiled_net.neuron_populations[value], "reward", reward * 1.0
                )
                
                # frame_img = (obs*255).astype(int)
                # if compress_frames:
                #     encoded = encode_frame(frame_img, compress=True, quality=compress_quality)
                #     current_run.append(encoded)
                # else:
                #     current_run.append(frame_img.copy())
                if env.wait_count == env.wait_inc:
                    frame_img = env.img(scale=25) 
                    # frame_img = (obs*255).astype(int)
                    if compress_frames:
                        encoded = encode_frame(frame_img, compress=True, quality=compress_quality)
                        current_run.append(encoded)
                    else:
                        current_run.append(frame_img.copy())
                    spikes = make_poisson_spikes(
                        obs.reshape(-1),
                        compiled_net.genn_model.timestep,
                        INPUT_SIZE,
                        K=WAIT_INC                   # tune this
                    )
                    compiled_net.set_input({input_pop: [spikes]})
                # spikes = make_rate_coded_spikes(
                #     obs.reshape(-1),
                #     compiled_net.genn_model.timestep,
                #     INPUT_SIZE,
                #     K=1
                # )
                # compiled_net.set_input({input_pop: [spikes]})

                compiled_net.step_time(train_callback_list)

                if env.wait_count == env.wait_inc:
                    for conn_pop in list(compiled_net.connection_populations.values())[::-1]:
                        try:
                            conn_pop.post_vars["FAvg"].pull_from_device()
                            f = conn_pop.post_vars["FAvg"].view
                        except:
                            f = 0
                            pass
                    # print("----------------------------")
                    # conn_pop.vars["SynSig"].pull_from_device()
                    # print(np.mean(conn_pop.vars["SynSig"].values))
                    
                    # conn_pop.vars["RLNoiseTrace"].pull_from_device()
                    # print(sum(abs(conn_pop.vars["RLNoiseTrace"].values)))
                    # for smth in dir(conn_pop):
                    #     print(smth)
                    freq_avg = freq_avg*0.999 + 0.001*np.mean(abs(f))

                    compiled_net.neuron_populations[list(hidden_layers.values())[0]].vars["V"].pull_from_device()
                    compiled_net.neuron_populations[list(hidden_layers.values())[0]].vars["A"].pull_from_device()
                    compiled_net.neuron_populations[list(hidden_layers.values())[0]].vars["Beta"].pull_from_device()

                    v = compiled_net.neuron_populations[list(hidden_layers.values())[0]].vars["V"].view
                    A = compiled_net.neuron_populations[list(hidden_layers.values())[0]].vars["A"].view
                    beta = compiled_net.neuron_populations[list(hidden_layers.values())[0]].vars["Beta"].view

                    # print("Betas mean:", np.mean(beta))

                    v_avg = v_avg*0.999 + 0.001*np.mean(abs(v))
                    v_reg_loss_avg = v_reg_loss_avg*0.999 + 0.001*np.mean(abs(
                        np.maximum( v - (0.61 + beta * A), 0.0) +
                        np.maximum(-v - (0.61 + beta * A), 0.0)
                    ))
                    if np.isnan(v_avg):
                        raise ValueError("Nan errors found in voltages") 

                    compiled_net.genn_model.custom_update("GradientLearn")
                    for o, custom_updates in compiled_net.optimisers:
                        for c in custom_updates:
                            o.set_step(c, opt_updt := opt_updt+1)
                
                # compiled_net.genn_model.custom_update("GradientLearn")
                # for o, custom_updates in compiled_net.optimisers:
                #     for c in custom_updates:
                #         o.set_step(c, opt_updt := opt_updt+1)
                
                """
                value_estimate = compiled_net.get_readout(value)[0][0]
                value_target = reward_trace + gamma * value_estimate

                td_error = value_target - previous_value_estimate
                
                previous_value_estimate = value_estimate
                
                compiled_net.losses[policy].set_var(
                    compiled_net.neuron_populations[policy], "tdError", td_error
                )
                compiled_net.losses[value].set_var(
                    compiled_net.neuron_populations[value], "tdError", td_error
                )
                for hidden_layer in list(hidden_layers.values()):
                    compiled_net.neuron_populations[hidden_layer].vars["TdE"].view[:] = td_error
                    compiled_net.neuron_populations[hidden_layer].vars["TdE"].push_to_device()
            
                compiled_net.neuron_populations[value].vars["tdError"].pull_from_device()
                compiled_net.neuron_populations[value].vars["E"].pull_from_device() 
                if abs(td_error - compiled_net.neuron_populations[value].vars["tdError"].view[0]) > 0.001:
                    print("Host", td_error)
                    print("Dev", compiled_net.neuron_populations[value].vars["tdError"].view[0])
                    raise
                if abs(td_error - compiled_net.neuron_populations[value].vars["E"].view[0]) > 0.001:
                    print("Host", td_error)
                    print("Dev", compiled_net.neuron_populations[value].vars["E"].view[0])
                    raise
                
                compiled_net.neuron_populations[policy].vars["tdError"].pull_from_device()
                compiled_net.neuron_populations[policy].vars["TdE"].pull_from_device() 
                if abs(td_error - compiled_net.neuron_populations[policy].vars["tdError"].view[0]) > 0.001:
                    print("Host", td_error)
                    print("Dev", compiled_net.neuron_populations[policy].vars["tdError"].view[0])
                    raise "tdError"
                if abs(td_error - compiled_net.neuron_populations[policy].vars["TdE"].view[0]) > 0.001:
                    print("Host", td_error)
                    print("Dev", compiled_net.neuron_populations[policy].vars["TdE"].view[0])
                    print("-----------------TdE----------------")
                # td_error_trace = 0

                compiled_net.neuron_populations[list(hidden_layers.values())[0]].vars["TdE"].pull_from_device() 
                if abs(td_error - compiled_net.neuron_populations[list(hidden_layers.values())[0]].vars["TdE"].view[0]) > 0.001:
                    print("Host", td_error)
                    print("Dev", compiled_net.neuron_populations[list(hidden_layers.values())[0]].vars["TdE"].view[0])
                    print("--------------Hid E TdE----------------")
                """
                 
                compiled_net.neuron_populations[value].vars["E"].pull_from_device() 
                td_error_sum_abs += abs(compiled_net.neuron_populations[value].vars["E"].view[0])
                ep_frames += 1

            spikes = make_poisson_spikes(
                obs.reshape(-1),                                 # flattened RGB values
                compiled_net.genn_model.timestep,
                INPUT_SIZE,
                K=WAIT_INC                   # tune this
            )
            compiled_net.set_input({input_pop: [spikes]})
            for i in range(WAIT_INC):
                gamma_disc_reward *= gamma
                if i % WAIT_INC == 0:
                    if GAUSSIAN_TRACE_POLICY:
                        # action = compiled_net.get_readout(policy).flatten()
                        compiled_net.neuron_populations[policy].vars["TanhOut"].pull_from_device()
                        action_scalar = compiled_net.neuron_populations[policy].vars["TanhOut"].view[0]

                        x = np.clip(action_scalar, -1.0, 1.0)

                        p_left  = max(0.0, -x)
                        p_right = max(0.0,  x)
                        p_fwd   = 1.0 - abs(x)

                        probs = np.array([p_left, p_fwd, p_right])
                        probs = probs / probs.sum()
                    else:
                        logits = compiled_net.get_readout(policy).flatten()
                        probs = np.exp((logits-logits.max())) / (np.exp((logits-logits.max())).sum() + 1e-8)
                    current_probs.append(probs)
                current_values.append(compiled_net.get_readout(value)[0].mean())
                current_reward_traces.append(reward_trace)
                
                reward_trace = reward_trace * reward_decay
                compiled_net.step_time(train_callback_list)
                
                # compiled_net.genn_model.custom_update("GradientLearn")
                # for o, custom_updates in compiled_net.optimisers:
                #     for c in custom_updates:
                #         o.set_step(c, opt_updt := opt_updt+1)
            
            compiled_net.genn_model.custom_update("GradientLearn")
            for o, custom_updates in compiled_net.optimisers:
                for c in custom_updates:
                    o.set_step(c, opt_updt := opt_updt+1)

            # if dale_l1_reg > 0:
            #     compiled_net.genn_model.custom_update("DaleRL1")

            # compiled_net.genn_model.custom_update("DalePrune")
            # compiled_net.genn_model.custom_update("DaleRewire")

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

            if sigma_q is not None and (ep % 100 == 0):
                try:
                    compiled_net.neuron_populations[list(hidden_layers.values())[0]].vars["LogSigma"].pull_from_device()
                    compiled_net.neuron_populations[list(hidden_layers.values())[1]].vars["LogSigma"].pull_from_device()
                    for conn_pop in list(compiled_net.connection_populations.values()):
                        if "LogSynSig" in conn_pop.vars:
                            break
                    conn_pop.vars["LogSynSig"].pull_from_device()
                    # print(np.mean(conn_pop.vars["SynSig"].values))
                    sigma_data = {
                        # 'E': compiled_net.neuron_populations[list(hidden_layers.values())[0]].vars["LogSigma"].view.copy(),
                        'E': conn_pop.vars["LogSynSig"].values - 5,
                        # 'I': compiled_net.neuron_populations[list(hidden_layers.values())[1]].vars["LogSigma"].view.copy(),
                        'I': conn_pop.vars["LogSynSig"].values - 5 
                    }
                    if sigma_q.full():
                        try: sigma_q.get_nowait()
                        except: pass
                    sigma_q.put_nowait(sigma_data)
                except Exception as e:
                    print("Sigma enqueue error:", e)

            snake_len_history.append(len(env.snake)-1)
            if len(snake_len_history) > 100:
                snake_len_history = snake_len_history[-100:]

            if avg == 0:
                avg = total_reward
            else:
                avg = smoothing * avg + (1 - smoothing) * total_reward
            running_avg.append(avg)

            with open(CSV_OUTPUT, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ep,
                    total_reward,
                    ep_frames,
                    td_error_sum_abs / ep_frames,
                    WAIT_INC * total_reward / ep_frames,
                    v_avg,
                    v_reg_loss_avg,
                    freq_avg
                ])
            
            # Send metrics for plotting
            if metrics_q is not None:
                metrics = {
                    'ep': ep,
                    'reward': total_reward,
                    'running_avg': avg,
                    'snake_len': len(env.snake)-1
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
                    metrics_q.put(metrics)
                except Exception as e:
                    print("Metrics enqueue error:", e)
            
            # logging
            if ep %10 == 0:
                print(
                    f"Episode {ep+1} - "
                    f"Total reward: {' ' if total_reward >= 0 else ''}{total_reward:.2f} "
                    f"- Best reward: {best_reward:.2f} "
                    f"- Snake len: {len(env.snake)-1:2d} "
                    f"- Snake len avg (last 100): {np.mean(snake_len_history):.2f} "
                    f"- Voltage running avg: {v_avg:.2f} "
                    f"- Voltage reg loss running avg: {v_reg_loss_avg:.2f} "
                    f"- Frequency running avg: {1000 * freq_avg:.2f} "
                    f"- Frame death: {ep_frames} "
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

if __name__ == "__main__":
    manager, metrics_q, best_run_q, random_run_q, sigma_q, stop_event, p_plots, p_runs, p_sigma = start_visualizers()

    try:
        train_snake_agent_with_ipc(
            episodes=int(10000),
            metrics_q=metrics_q,
            best_run_q=best_run_q,
            random_run_q=random_run_q,
            sigma_q=sigma_q,
            compress_frames=True,
            compress_quality=80
        )
    finally:
        stop_event.set()
        time.sleep(0.2)
        if p_plots.is_alive(): p_plots.terminate()
        if p_runs.is_alive(): p_runs.terminate()
        if p_sigma.is_alive(): p_sigma.terminate()