##################### SNAKE ENV #####################
#####################################################
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
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, AdaptiveLeakyIntegrateFire, SpikeInput
from ml_genn.serialisers import Numpy
from ml_genn.optimisers import Adam


from ml_genn.compilers.eprop_compiler import default_params


class SnakeEnv:
    def __init__(self, size=28, visible_range=7, wait_inc=5):
        assert visible_range % 2 == 1, "visible_range must be odd"
        self.size = size
        self.visible_range = visible_range
        self.wait_inc = wait_inc
        self.won = False
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
        self.direction = 'up'

        self.apples = []
        self.spawn_apples()
        self.done = False
        self.steps_since_last_apple = 0
        self.wait_count = self.wait_inc
        return self.get_observation()

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
            return self.get_observation(), reward/100, self.done


        dirs = ['left', 'up', 'right', 'down']
        # self.dir_idx = (self.dir_idx + (action-1)) % 4
        # new_dir = dirs[self.dir_idx]
        new_dir = dirs[action]
        
        reward = - 90/self.size * (self.steps_since_last_apple % self.size == 0)
        # reward = 0.0

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
            return self.get_observation(), reward/100, True

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
            # reward -= 50.0
            return self.get_observation(), reward/100, self.done

        self.wait_count = self.wait_inc

        return self.get_observation(), reward/100, self.done

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

        obs = np.zeros((3, v, v), dtype=np.float32)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y = head_y + dy
                x = head_x + dx
                local_y = dy + r
                local_x = dx + r

                # check walls
                if y < 0 or y >= self.size or x < 0 or x >= self.size:
                    obs[2, local_y, local_x] = 1.0
                    continue

                # body
                if (y, x) in self.snake:
                    obs[0, local_y, local_x] = 1.0

                # # head
                # if (y, x) == (head_y, head_x):
                #     obs[3, local_y, local_x] = 1.0

                # apple
                for apple in self.apples:
                    if (y, x) == apple:
                        obs[1, local_y, local_x] = 1.0

        return obs.flatten()

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

                # walls
                if y < 0 or y >= self.size or x < 0 or x >= self.size:
                    img[local_y, local_x] = [100, 100, 100]  # gray
                    continue

                # body
                if (y, x) in self.snake[1:]:
                    img[local_y, local_x] = [0, 255, 0]

                # head
                if (y, x) == (head_y, head_x):
                    img[local_y, local_x] = [0, 155, 0]

                # apple
                for apple in self.apples:
                    if (y, x) == apple:
                        img[local_y, local_x] = [0, 0, 255]

        # upscale for visualization
        return cv2.resize(img, (v * scale, v * scale), interpolation=cv2.INTER_NEAREST)

    def render_cv2(self, scale=10):
        img = self.img()
        cv2.imshow("Snake", img)
        cv2.waitKey(1)

################### DEFINE MODEL ####################
#####################################################
WINDOW_EPISODES = 100
BOARD_SIZE = 6
VISIBLE_RANGE = 5

WAIT_INC = 30

INPUT_SIZE = 3 * VISIBLE_RANGE**2
NUM_HIDDEN_1 = 2048
NUM_OUTPUT = 4
CONN_P = {
    "I-H": 0.25,
    #"H-H": 0.25,
    "H-H": np.log(NUM_HIDDEN_1)/NUM_HIDDEN_1,
    "H-P": 0.25,
    "H-V": 0.25
}
TRAIN = True
CHECKPOINT_BOARD_SIZE = 5

KERNEL_PROFILING = False


gamma = 0.99 ** (1/WAIT_INC)
td_lambda = 0.8 ** (1/WAIT_INC)
td_error_trace_discount = 0.001**(1/WAIT_INC)

entropy_coeff = 1e-4
entropy_decay = 0.9999 ** (1/WAIT_INC)
entropy_min = 1e-5

serialiser = Numpy("snake_checkpoints")
network = Network(default_params)
with network:
    # Populations
    input_pop = Population(SpikeInput(max_spikes=INPUT_SIZE * WAIT_INC // 2), INPUT_SIZE)
    hidden_1 = Population(AdaptiveLeakyIntegrateFire(v_thresh=0.61, tau_mem=10.0,
                                           tau_refrac=3.0, tau_adapt=100),
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


if CHECKPOINT_BOARD_SIZE:
    network.load((CHECKPOINT_BOARD_SIZE,), serialiser)

max_example_timesteps = 1
compiler = EPropCompiler(
    example_timesteps=max_example_timesteps,
    losses={policy: "sparse_categorical_crossentropy",
            value: "mean_square_error"},
    optimiser=Adam(1e-4),
    batch_size=1,
    kernel_profiling=KERNEL_PROFILING,
    feedback_type="random",
    gamma=gamma,          # 0.99
    td_lambda=td_lambda,        # or whatever you want
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

####################### TRAIN #######################
#####################################################

def make_repeated_spikes(indices, base_timestep, input_size, K=5, period=1):
    if len(indices) == 0 or K <= 0:
        return preprocess_spikes([], [], input_size)

    indices = np.asarray(indices, dtype=np.int64)

    # Build times and indices
    times = np.repeat(base_timestep + np.arange(K) * period, len(indices))
    idxs = np.tile(indices, K)

    # Convert back to Python lists because preprocess_spikes expects list-like
    return preprocess_spikes(times, idxs, input_size)

# --- Helper: optionally compress frames before sending to reduce IPC size ---
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


    value_line, = ax2.plot([], [], label='Value'); ax2.set_title('Value (best run)')
    ax2.set_xlabel('Frame'); ax2.set_ylabel('Value')

    prob_img = ax3.imshow(np.zeros((4,1)), aspect='auto', origin='lower', vmin=0, vmax=1)
    ax3.set_xlabel('Timestep'); ax3.set_ylabel('Action'); ax3.set_title('Action probs (best run)')
    fig.colorbar(prob_img, ax=ax3, fraction=0.02, pad=0.04)

    # local buffers
    ep_list = []
    rewards = []
    avgs = []
    lens = []
    best_values = []
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

    cv2.namedWindow(window_best, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_random, cv2.WINDOW_NORMAL)
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

# --- Modify your train_snake_agent to send updates instead of internal plotting ---
# Replace plt.ion() + figure creation in train_snake_agent with nothing and send updates to queues.
# I will show a skeleton wrapper around your train loop:
def train_snake_agent_with_ipc(episodes=100000,
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
    with compiled_net:
        env = SnakeEnv(size=BOARD_SIZE, visible_range=VISIBLE_RANGE, wait_inc=WAIT_INC)
        best_reward = -np.inf
        best_run = []
        running_avg = []
        snake_len_history = []
        smoothing = 0.95
        avg = 0

        # keep last best values and probs for plot process
        last_best_values = []
        last_best_probs = None

        train_callback_list.on_batch_begin(0)
        for ep in range(episodes):
            if env.won:
                compiled_net.save_connectivity((BOARD_SIZE,), serialiser)
                compiled_net.save((BOARD_SIZE,), serialiser)
                best_reward = -np.inf
                BOARD_SIZE += 1
                print(f"WON! Increasing board size to {BOARD_SIZE}")
                env = SnakeEnv(size=BOARD_SIZE, visible_range=VISIBLE_RANGE, wait_inc=WAIT_INC)
            if ep % 1000 == 0:
                compiled_net.save_connectivity((BOARD_SIZE,"mid-complition"), serialiser)
                compiled_net.save((BOARD_SIZE, "mid-completion"), serialiser)

            for m in all_metrics.values():
                m.reset()
            obs = env.reset()
            done = False
            total_reward = 0
            current_run = []
            current_values = []
            current_probs = []
            total_td = 0
            frame = 1
            value_target = 0

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
                    action_label = np.random.choice(4, p=probs)
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
                reward_trace = reward_trace*0.5 + reward*0.5

                if env.wait_count == env.wait_inc:
                    # capture frame scaled down for IPC
                    frame_img = env.img(scale=8)  # smaller scale to reduce size
                    if compress_frames:
                        encoded = encode_frame(frame_img, compress=True, quality=compress_quality)
                        current_run.append(encoded)
                    else:
                        current_run.append(frame_img.copy())
                    train_callback_list.on_batch_end(0, all_metrics)
                    # for o, custom_updates in compiled_net.optimisers:
                    #     for c in custom_updates:
                    #         o.set_step(c, ep)

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
                
                total_td += td_error
                previous_value_estimate = value_estimate

                # if reward != 0:
                compiled_net.losses[value].set_var(
                    compiled_net.neuron_populations[value], "tdError", td_error_trace
                )
                compiled_net.losses[policy].set_var(
                    compiled_net.neuron_populations[policy], "tdError", td_error_trace
                )
                compiled_net.neuron_populations[hidden_1].vars["TdE"].view[:] = td_error_trace
                compiled_net.neuron_populations[hidden_1].vars["TdE"].push_to_device()
                td_error_trace = 0

                # train_callback_list.on_batch_end(0, all_metrics)
                # for o, custom_updates in compiled_net.optimisers:
                #     for c in custom_updates:
                #         o.set_step(c, ep)

                frame += 1
            
            for _ in range(WAIT_INC):
                reward_trace = reward_trace*0.5
                compiled_net.step_time(train_callback_list)
                value_estimate = compiled_net.get_readout(value)[0][0]
                value_target = reward_trace + gamma * value_estimate

                td_error = value_target - previous_value_estimate
                td_error_trace = 0 * td_error_trace_discount * td_error_trace + td_error
                
                total_td += td_error
                previous_value_estimate = value_estimate

                # if reward != 0 or True:
                compiled_net.losses[value].set_var(
                    compiled_net.neuron_populations[value], "tdError", td_error_trace
                )
                compiled_net.losses[policy].set_var(
                    compiled_net.neuron_populations[policy], "tdError", td_error_trace
                )
                compiled_net.neuron_populations[hidden_1].vars["TdE"].view[:] = td_error_trace
                compiled_net.neuron_populations[hidden_1].vars["TdE"].push_to_device()
                
                # train_callback_list.on_batch_end(0, all_metrics)
                # for o, custom_updates in compiled_net.optimisers:
                #     for c in custom_updates:
                #         o.set_step(c, ep)
                td_error_trace = 0

            # end of episode bookkeeping
            train_callback_list.on_batch_end(0, all_metrics)
            for o, custom_updates in compiled_net.optimisers:
                for c in custom_updates:
                    o.set_step(c, ep)
            
            # if compiled_net.optimisers[0][0].alpha > 1e-5 and np.mean(snake_len_history) > 2:
            #     compiled_net.optimisers[0][0].alpha = 1e-5

            # Update if new best run (send best run to viz process)
            if total_reward >= best_reward and len(current_probs) > 0:
                best_reward = total_reward
                best_run = [img for img in current_run]  # frames already possibly encoded
                last_best_values = list(current_values)
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

            snake_len_history.append(len(env.snake)-1)
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
                    'snake_len': len(env.snake)-1
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

            # logging
            print(
                f"Episode {ep+1} - "
                f"Total reward: {' ' if total_reward >= 0 else ''}{total_reward:.2f} "
                f"- Best reward: {best_reward:.2f} "
                f"- Snake len: {len(env.snake)-1:2d} "
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