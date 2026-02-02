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
from ml_genn.compilers import EPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense, FixedProbability
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
GRID_SIZE = 10
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
        self.agent_pos = [9, 5]

        self.current_digit = np.random.randint(0, 10)
        self.current_mnist = np.zeros((28, 28), dtype=np.uint8)

        self.reveal_pos = (
            np.random.randint(4, GRID_SIZE), 
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

        ny = self.agent_pos[0] + dy
        nx = self.agent_pos[1] + dx

        self.steps_since_reveal += 1

        # out of bounds
        if ny < 0 or ny >= GRID_SIZE or nx < 0 or nx >= GRID_SIZE:
            self.wait_count = self.wait_inc
            self.done = True
            return self.get_observation(), -1.0, self.done

        if (ny, nx) in self.doors and self.doors_locked:
            self.wait_count = self.wait_inc
            self.done = True
            reward -= 1.0
            # self.head_butts += 1
            # if self.head_butts > 10:
            #     self.done = True
            #     reward -= 0.9
            return self.get_observation(), reward, self.done

        self.agent_pos = [ny, nx]

        # reveal zone
        if (ny, nx) == self.reveal_pos:
            if self.doors_locked:
                reward += 1.0
                self.steps_since_reveal = 0
            self.sample_digit()
            self.visualizing_digit = True
        else:
            # MNIST disappears immediately when leaving reveal
            self.current_mnist[:] = 0
            self.visualizing_digit = False

        # door entry
        if ny==1:
            self.sample_door_digit(nx)
            self.visualizing_digit = True
        elif ny == 2:
            self.current_mnist[:] = 0
            self.visualizing_digit = False

        if (ny, nx) in self.doors and not self.doors_locked:
            if nx == self.correct_door:
                reward += 5.0
            else:
                reward += 1.0
            self.done = True

        # Timeout penalty (too many steps after seeing digit)
        if self.steps_since_reveal > GRID_SIZE*GRID_SIZE//2:
            reward -= 1.0
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
        idx = np.random.choice(np.where(labels == digit)[0])
        return images[idx].copy()

    # ------------------------------------------------------------ #
    #                      OBSERVATION                             #
    # ------------------------------------------------------------ #
    def get_observation(self):
        """
        Returns a local view around the agent:
        - Channel 0: walls (grid boundaries)
        - Channel 1: reveal button
        - Channel 2: MNIST image (always present as separate 28x28)
        
        Total: (3, visible_range, visible_range) + (28, 28) flattened
        """
        v = self.visible_range
        r = v // 2
        agent_y, agent_x = self.agent_pos

        # Local grid observation (3 channels)
        obs = np.zeros((3, v, v), dtype=np.float32)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y = agent_y + dy
                x = agent_x + dx
                local_y = dy + r
                local_x = dx + r

                # Check walls (out of bounds)
                if y < 0 or y >= GRID_SIZE or x < 0 or x >= GRID_SIZE:
                    obs[0, local_y, local_x] = 1.0
                    continue

                # Reveal button location
                if (y, x) == self.reveal_pos:
                    obs[1, local_y, local_x] = 1.0

        # MNIST image channel (always full 28x28, separate from local view)
        mnist_channel = (self.current_mnist.astype(np.float32) / 255.0 > 0.5).astype(np.int32)

        # Concatenate: local view (3 * v * v) + MNIST (28 * 28)
        return np.concatenate([
            obs.flatten(),
            mnist_channel.flatten()
        ])

    # ------------------------------------------------------------ #
    #                     VISUALIZATION                            #
    # ------------------------------------------------------------ #
    def img(self, scale=60):
        img = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
        img[:] = BG_COLOR

        # Draw doors with highlighting for correct door when unlocked
        for y, x in self.doors:
            if self.doors_locked:
                img[y, x] = DOOR_COLOR_LOCKED
            else:
                # Highlight the correct door
                if x == self.correct_door:
                    img[y, x] = (0, 255, 0)  # Bright green for correct door
                else:
                    img[y, x] = DOOR_COLOR_OPEN

        ry, rx = self.reveal_pos
        img[ry, rx] = REVEAL_COLOR

        ay, ax = self.agent_pos
        img[ay, ax] = AGENT_COLOR

        # Resize the grid
        img = cv2.resize(
            img,
            (GRID_SIZE * scale, GRID_SIZE * scale),
            interpolation=cv2.INTER_NEAREST
        )

        # Add MNIST digit overlay if doors are unlocked (digit has been seen)
        if self.visualizing_digit:
            # Resize MNIST to fit in corner (e.g., 150x150 pixels)
            mnist_size = int(scale * 2.5)  # 2.5 cells worth of space
            mnist_display = cv2.resize(self.current_mnist, (mnist_size, mnist_size),
                                      interpolation=cv2.INTER_NEAREST)
            mnist_display = cv2.cvtColor(mnist_display, cv2.COLOR_GRAY2BGR)
            
            # Add border around MNIST
            border_size = 5
            mnist_display = cv2.copyMakeBorder(mnist_display, border_size, border_size, 
                                              border_size, border_size,
                                              cv2.BORDER_CONSTANT, value=(255, 255, 0))
            
            # Place in bottom-right corner
            h, w = img.shape[:2]
            mh, mw = mnist_display.shape[:2]
            y_offset = h - mh - 10
            x_offset = w - mw - 10
            
            # Overlay the MNIST with slight transparency effect
            img[y_offset:y_offset+mh, x_offset:x_offset+mw] = mnist_display

        return img

    def local_img(self, scale=40):
        """
        Returns a local RGB image centered on the agent,
        matching the visible_range used in get_observation().
        Channels:
            Gray   = walls (out of bounds)
            Yellow = reveal button
            Green  = agent (center)
        """
        v = self.visible_range
        r = v // 2
        agent_y, agent_x = self.agent_pos

        img = np.zeros((v, v, 3), dtype=np.uint8)
        img[:] = BG_COLOR  # default background

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y = agent_y + dy
                x = agent_x + dx
                local_y = dy + r
                local_x = dx + r

                # Walls (out of bounds)
                if y < 0 or y >= GRID_SIZE or x < 0 or x >= GRID_SIZE:
                    img[local_y, local_x] = (100, 100, 100)  # gray
                    continue

                # Reveal button
                if (y, x) == self.reveal_pos:
                    img[local_y, local_x] = REVEAL_COLOR  # yellow

                # Agent position (center)
                if (y, x) == tuple(self.agent_pos):
                    img[local_y, local_x] = AGENT_COLOR  # green

        # Upscale for visualization
        return cv2.resize(img, (v * scale, v * scale), interpolation=cv2.INTER_NEAREST)


    def render(self):
        cv2.imshow("Door-Key MNIST", self.img())

        mn = cv2.resize(self.current_mnist, (200, 200),
                        interpolation=cv2.INTER_NEAREST)
        mn = cv2.cvtColor(mn, cv2.COLOR_GRAY2BGR)
        cv2.imshow("MNIST Panel", mn)

        cv2.waitKey(1)

################### DEFINE MODEL ####################
#####################################################
WINDOW_EPISODES = 100
WAIT_INC = 10

INPUT_SIZE = GRID_SIZE * GRID_SIZE * 2 + 28 * 28  # walls + reveal + MNIST = 984
NUM_HIDDEN_1 = 1024
NUM_OUTPUT = 5  # 4 actions + wait
CONN_P = {
    "I-H": 0.1,
    "H-H": np.log(NUM_HIDDEN_1)/NUM_HIDDEN_1,
    # "H-H": 0.25,
    "H-P": 0.1,
    "H-V": 0.1
}

KERNEL_PROFILING = False

gamma = 0.99 ** (1/WAIT_INC)
td_lambda = 0.01 ** (1/WAIT_INC)
td_error_trace_discount = 0.001**(1/WAIT_INC)

entropy_coeff = 1e-3
entropy_decay = 0.9999 ** (1/WAIT_INC)
entropy_min = 1e-5

serialiser = Numpy("door_key_mnist_checkpoints")
network = Network(default_params)
with network:
    # Populations
    input_pop = Population(SpikeInput(max_spikes=INPUT_SIZE * WAIT_INC), INPUT_SIZE)
    hidden_1 = Population(AdaptiveLeakyIntegrateFire(v_thresh=0.61, tau_mem=30.0,
                                           tau_refrac=3.0, tau_adapt=1000),
                        NUM_HIDDEN_1)
    policy = Population(LeakyIntegrate(tau_mem=10.0, readout="var"),
                        NUM_OUTPUT)

    value = Population(LeakyIntegrate(tau_mem=10.0, readout="var"),
                        1)
    
    # Connections
    Connection(input_pop,  hidden_1, 
        FixedProbability(CONN_P['I-H'], (Normal(sd=1.0 / np.sqrt(CONN_P['I-H'] * INPUT_SIZE)))))
    Connection(hidden_1, hidden_1, 
        FixedProbability(CONN_P['H-H'], (Normal(sd=1.0 / np.sqrt(CONN_P['H-H'] * NUM_HIDDEN_1)))))
    
    Connection(hidden_1, policy, 
        FixedProbability(CONN_P['H-P'], (Normal(sd=1.0 / np.sqrt(CONN_P['H-P'] * NUM_HIDDEN_1)))))
    Connection(hidden_1, value, 
        FixedProbability(CONN_P['H-V'], (Normal(sd=1.0 / np.sqrt(CONN_P['H-V'] * NUM_HIDDEN_1)))))
    
    Connection(hidden_1, policy, Dense(Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))), feedback_name="policy_feedback")
    Connection(hidden_1, policy, Dense(Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))), feedback_name="policy_regularisation")
    Connection(hidden_1, value, Dense(Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))), feedback_name="value_feedback")
    Connection(hidden_1, value, Dense(Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))), feedback_name="value_regularisation")


max_example_timesteps = 1
compiler = EPropCompiler(
    example_timesteps=max_example_timesteps,
    losses={policy: "sparse_categorical_crossentropy",
            value: "mean_square_error"},
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
    entropy_min=entropy_min
)

compiled_net = compiler.compile(network)

# Build metrics for training
policy_train_metrics = get_object_mapping(
    "sparse_categorical_accuracy", [policy], Metric, 
    "Metric", default_metrics)

value_train_metrics = get_object_mapping(
    "mean_square_error", [value], Metric, 
    "Metric", default_metrics)

train_callback_list = CallbackList(
    [*set(compiled_net.base_train_callbacks)],
    compiled_network=compiled_net,
    num_batches=1, 
    num_epochs=1
)

all_metrics = {}
all_metrics.update(policy_train_metrics)
all_metrics.update(value_train_metrics)

####################### HELPER FUNCTIONS #######################
#################################################################

def make_repeated_spikes(indices, base_timestep, input_size, K=5, period=1):
    indices = np.asarray(indices, dtype=np.int64)
    times = np.repeat(base_timestep + np.arange(K) * period, len(indices))
    idxs = np.tile(indices, K)

    return preprocess_spikes(times, idxs, input_size)

def encode_frame(frame, compress=True, ext='.jpg', quality=80):
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

####################### VISUALIZATION PROCESSES #######################
########################################################################

def viz_plots_loop(metrics_q: Queue, stop_event: mp.Event):
    plt.ion()
    fig, (ax1, ax1b, ax2, ax3) = plt.subplots(4, 1, figsize=(10, 12))

    plt.subplots_adjust(hspace=0.5)
    
    # Main training plot
    line_reward, = ax1.plot([], [], label='Total Reward', color=(0, 0.1, 0.8, 0.5), zorder=2)
    avg_line, = ax1.plot([], [], label='Running Avg', color=(0.8, 0.05, 0.05, 1.0), zorder=3)
    success_line, = ax1.plot([], [], label='Success Rate', color=(0, 0.8, 0.1, 0.5), zorder=1)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward / Success Rate')
    ax1.set_title('Training Progress')
    ax1.legend()

    # Recent episodes plot
    line_reward_recent, = ax1b.plot([], [], label='Reward (last 500)')
    avg_line_recent, = ax1b.plot([], [], label='Avg (last 500)')
    ax1b.set_title('Recent Training (last 500 episodes)')
    ax1b.legend()

    # Value plot
    value_line, = ax2.plot([], [], label='Value')
    ax2.set_title('Value (best run)')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Value')

    # Action probabilities plot
    prob_img = ax3.imshow(np.zeros((4,1)), aspect='auto', origin='lower', vmin=0, vmax=1)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Action')
    ax3.set_title('Action probs (best run)')
    fig.colorbar(prob_img, ax=ax3, fraction=0.02, pad=0.04)

    # Local buffers
    ep_list = []
    rewards = []
    avgs = []
    success_rates = []
    best_values = []
    best_probs = None

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
                    success_rates.append(metrics.get('success_rate', 0.0))
                if 'best_values' in metrics:
                    best_values = metrics['best_values']
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
                success_line.set_data(axis, success_rates)
                ax1.relim()
                ax1.autoscale_view()

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
                ax2.relim()
                ax2.autoscale_view()

            if best_probs is not None:
                data = np.array(best_probs).T
                n_actions, time_steps = data.shape

                prob_img.set_data(data)
                prob_img.set_extent([0, time_steps, 0, n_actions])

                ax3.set_xlim(0, time_steps)
                ax3.set_ylim(0, n_actions)
                ax3.set_aspect('auto')

            plt.pause(0.001)
            last_plot_time = now

        time.sleep(0.03)

    plt.close(fig)

def viz_runs_loop(best_run_q: Queue, random_run_q: Queue, stop_event: mp.Event, decompress=True):
    best_run = []
    random_run = []
    window_best = "Best Run"
    window_random = "Random Run"

    cv2.namedWindow(window_best, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_random, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_best, 600, 600)
    cv2.resizeWindow(window_random, 600, 600)

    t = 0
    while not stop_event.is_set():
        try:
            while True:
                br = best_run_q.get_nowait()
                if br is None:
                    stop_event.set()
                    break
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

        if best_run:
            if stop_event.is_set():
                break
            f_blob = best_run[t % len(best_run)]
            frame = decode_frame(f_blob) if decompress else f_blob
            if frame is None:
                continue
            cv2.imshow(window_best, frame)
            key = cv2.waitKey(1)
            if key == 27:
                stop_event.set()
                break

        if random_run and t % (len(random_run) + 10) < len(random_run):
            f_blob = random_run[t % (len(random_run) + 10)]
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

####################### TRAINING AGENT #######################
##############################################################

def train_door_key_agent(episodes=100000,
                         metrics_q: Queue=None,
                         best_run_q: Queue=None,
                         random_run_q: Queue=None,
                         compress_frames=True,
                         compress_quality=80):
    """
    Train the Door-Key MNIST Memory agent using e-prop learning.
    """
    with compiled_net:
        env = DoorKeyMNISTMemoryEnv(wait_inc=WAIT_INC)
        best_reward = -np.inf
        best_run = []
        running_avg = []
        success_history = []
        smoothing = 0.95
        avg = 0

        last_best_values = []
        last_best_probs = None

        train_callback_list.on_batch_begin(0)
        time_since_last_best = 0

        for ep in range(episodes):
            for m in all_metrics.values():
                m.reset()
            
            obs = env.reset()
            done = False
            total_reward = 0
            current_run = []
            current_values = []
            current_probs = []
            frame = 1
            value_target = 0
            success = False

            # Initial spike encoding
            indices = obs.nonzero()[0]
            spikes = make_repeated_spikes(
                indices,
                compiled_net.genn_model.timestep,
                INPUT_SIZE,
                K=WAIT_INC,
                period=1
            )
            compiled_net.set_input({input_pop: [spikes]})
            compiled_net.step_time(train_callback_list)
            previous_value_estimate = compiled_net.get_readout(value)[0][0]
            env.wait_count = WAIT_INC
            reward_trace = 0
            td_error_trace = 0

            while not done:
                action_label = 0
                current_values.append(previous_value_estimate)
                
                if env.wait_count == 0:
                    probs = compiled_net.get_readout(policy).flatten()
                    if abs(sum(probs) - 1.0) > 0.0001:
                        print("BAD PROBS", sum(probs))
                    action_label = np.random.choice(5, p=probs)
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
                total_reward += reward
                reward_trace = reward_trace * 0.0 + reward * 1.0

                # Track success (agent chose correct door)
                if done and reward > 5:
                    success = True

                if env.wait_count == env.wait_inc:
                    # Capture frame
                    frame_img = env.img(scale=8)
                    if compress_frames:
                        encoded = encode_frame(frame_img, compress=True, quality=compress_quality)
                        current_run.append(encoded)
                    else:
                        current_run.append(frame_img.copy())
                    
                    train_callback_list.on_batch_end(0, all_metrics)

                    # Encode next observation
                    indices = obs.nonzero()[0]
                    spikes = make_repeated_spikes(
                        indices,
                        compiled_net.genn_model.timestep,
                        INPUT_SIZE,
                        K=WAIT_INC,
                        period=1
                    )
                    compiled_net.set_input({input_pop: [spikes]})

                compiled_net.step_time(train_callback_list)
                value_estimate = compiled_net.get_readout(value)[0][0]
                value_target = reward_trace + gamma * value_estimate

                td_error = value_target - previous_value_estimate
                td_error_trace = 0 * td_error_trace_discount * td_error_trace + td_error
                
                previous_value_estimate = value_estimate

                compiled_net.losses[value].set_var(
                    compiled_net.neuron_populations[value], "tdError", td_error_trace
                )
                compiled_net.losses[policy].set_var(
                    compiled_net.neuron_populations[policy], "tdError", td_error_trace
                )
                compiled_net.neuron_populations[hidden_1].vars["TdE"].view[:] = td_error_trace
                compiled_net.neuron_populations[hidden_1].vars["TdE"].push_to_device()
                td_error_trace = 0

                frame += 1
            
            # Flash agent to signal end of episode
            indices = np.arange(28*28) + (INPUT_SIZE - 28*28)
            spikes = make_repeated_spikes(
                indices,
                compiled_net.genn_model.timestep,
                INPUT_SIZE,
                K=WAIT_INC//2,
                period=1
            )
            compiled_net.set_input({input_pop: [spikes]})
            for _ in range(WAIT_INC):
                reward_trace = reward_trace * 0.5
                compiled_net.step_time(train_callback_list)
                value_estimate = compiled_net.get_readout(value)[0][0]
                value_target = reward_trace + gamma * value_estimate

                td_error = value_target - previous_value_estimate
                td_error_trace = 0 * td_error_trace_discount * td_error_trace + td_error
                
                previous_value_estimate = value_estimate

                compiled_net.losses[value].set_var(
                    compiled_net.neuron_populations[value], "tdError", td_error_trace
                )
                compiled_net.losses[policy].set_var(
                    compiled_net.neuron_populations[policy], "tdError", td_error_trace
                )
                compiled_net.neuron_populations[hidden_1].vars["TdE"].view[:] = td_error_trace
                compiled_net.neuron_populations[hidden_1].vars["TdE"].push_to_device()
                td_error_trace = 0

            # End of episode
            train_callback_list.on_batch_end(0, all_metrics)
            for o, custom_updates in compiled_net.optimisers:
                for c in custom_updates:
                    o.set_step(c, ep)
            
            # Update best run
            time_since_last_best += 1
            if total_reward == best_reward and time_since_last_best > 100:
                best_reward -= 1
                time_since_last_best = 0
                
            if total_reward > best_reward and len(current_probs) > 0:
                best_reward = total_reward
                best_run = [img for img in current_run]
                last_best_values = list(current_values)
                last_best_probs = list(current_probs)

                if best_run_q is not None:
                    try:
                        if best_run_q.full():
                            try:
                                _ = best_run_q.get_nowait()
                            except Exception:
                                pass
                        best_run_q.put_nowait(best_run)
                    except Exception as e:
                        print("Best-run enqueue error:", e)

            # Random run visualization
            if random_run_q is not None and (ep % 100 == 0):
                try:
                    if random_run_q.full():
                        try:
                            _ = random_run_q.get_nowait()
                        except:
                            pass
                    random_run_q.put_nowait(current_run)
                except Exception as e:
                    print("Random-run enqueue error:", e)

            # Track success rate
            success_history.append(1.0 if success else 0.0)
            if len(success_history) > 100:
                success_history = success_history[-100:]

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
                    'success_rate': np.mean(success_history)
                }
                if last_best_values:
                    metrics['best_values'] = last_best_values
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

            # Logging
            print(
                f"Episode {ep+1:5d} - "
                f"Reward: {' ' if total_reward >= 0 else ''}{total_reward:+.3f} "
                f"- Best: {best_reward:+.3f} "
                f"- Success rate (last 100): {np.mean(success_history):.2%} "
                f"- Frames: {frame:3d} "
                f"- Alpha: {compiled_net.optimisers[0][0].alpha:.8f}"
            )

        # Signal visualizers to stop
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

##############################################################
#                     RANDOM AGENT                           #
##############################################################

def run_random_agent(episodes=1000, delay=0.05):
    env = DoorKeyMNISTMemoryEnv(wait_inc=0)

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = random.randint(0, 3)
            obs, reward, done = env.step(action)
            total_reward += reward

            env.render()
            time.sleep(delay)

        print(f"Episode {ep:4d} | Total reward: {total_reward:+.2f}")

####################################################################
#                              MAIN                                #
####################################################################

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "random":
        # Run random agent for testing
        run_random_agent(episodes=1e10, delay=0.03)
        cv2.destroyAllWindows()
    else:
        # Train agent with visualization
        manager, metrics_q, best_run_q, random_run_q, stop_event, p_plots, p_runs = start_visualizers()

        try:
            train_door_key_agent(
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
            if p_plots.is_alive():
                p_plots.terminate()
            if p_runs.is_alive():
                p_runs.terminate()