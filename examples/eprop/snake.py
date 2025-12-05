##################### SNAKE ENV #####################
#####################################################
from ftplib import all_errors
from ml_genn.metrics.metric import Metric
from ml_genn.metrics import default_metrics
from ml_genn.callbacks.custom_update import CustomUpdateOnBatchEnd
from ml_genn.utils.module import get_object_mapping
from ml_genn.utils.data import preprocess_spikes
from ml_genn.utils.callback_list import CallbackList
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib.pyplot as plt
import mnist
from line_profiler import profile

from ml_genn import InputLayer, Layer, Network, Population, Connection
from ml_genn.callbacks import Checkpoint
from ml_genn.compilers import EPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, AdaptiveLeakyIntegrateFire, SpikeInput
from ml_genn.serialisers import Numpy
from ml_genn.optimisers import Adam

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                log_latency_encode_data)

from ml_genn.compilers.eprop_compiler import default_params


class SnakeEnv:
    def __init__(self, size=28, visible_range=7, wait_inc=5):
        assert visible_range % 2 == 1, "visible_range must be odd"
        self.size = size
        self.visible_range = visible_range
        self.wait_inc = wait_inc
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
        
        reward = -0.1 #-max(0.5 * (self.steps_since_last_apple-1) / self.size, 5)

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
                reward += 500.0
            self.steps_since_last_apple = 0
        else:
            self.snake.pop()
            self.steps_since_last_apple += 1
        
        if self.steps_since_last_apple > self.size**2:
            self.done = True
            # reward -= 100.0
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
BOARD_SIZE = 15
VISIBLE_RANGE = 5

WAIT_INC = 30

INPUT_SIZE = 3 * VISIBLE_RANGE**2
NUM_HIDDEN_1 = 4096
NUM_OUTPUT = 4
CONN_P = {
    "I-H": 0.1,
    # "H-H": 0.01,
    "H-H": np.log(NUM_HIDDEN_1)/NUM_HIDDEN_1,
    "H-P": 0.1,
    "H-V": 0.1
}
TRAIN = True

KERNEL_PROFILING = False

gamma = 0.99 ** (1/WAIT_INC)
td_lambda = 0.8 ** (1/WAIT_INC)
td_error_trace_discount = 0.001**(1/WAIT_INC)

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
    
    # Random feedback matrices (COMMENT THESE LINES TO COMPARE WITH SYMMETRIC EPROP)
    Connection(hidden_1, policy, Dense(Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))), is_feedback=True)
    Connection(hidden_1, value, Dense(Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))), is_feedback=True)

max_example_timesteps = 1
compiler = EPropCompiler(
    example_timesteps=max_example_timesteps,
    losses={policy: "sparse_categorical_crossentropy",
            value: "mean_square_error"},
    optimiser=Adam(1e-5),
    batch_size=1,
    kernel_profiling=KERNEL_PROFILING,
    feedback_type="random",
    gamma=gamma,          # 0.99
    td_lambda=td_lambda,        # or whatever you want
    train_output_bias=False,
    reset_time_between_batches=False
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


with compiled_net:

    @profile
    def train_snake_agent(episodes=100000):
        env = SnakeEnv(size=BOARD_SIZE, visible_range=VISIBLE_RANGE, wait_inc=WAIT_INC)
        
        best_reward = -np.inf
        best_run = []
        reward_history = []
        running_avg = []
        snake_len_history = []

        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
        plt.subplots_adjust(hspace=0.5)

        # --- 1. Training reward subplot ---
        snake_len, = ax1.plot([], [], label='Snake length', color='green')
        line, = ax1.plot([], [], label='Total Reward', color='blue')
        avg_line, = ax1.plot([], [], label='Running Avg', color='red')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress')
        ax1.legend()

        # --- 2. Value output subplot ---
        value_line, = ax2.plot([], [], color='green')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Value Output')
        ax2.set_title('Best Run Value Outputs')

        # --- 3. Action probabilities heatmap subplot ---
        prob_img = ax3.imshow(
            np.zeros((3, 1)),
            aspect='auto',
            cmap='viridis',
            origin='lower',
            vmin=0.0,
            vmax=1.0,
            extent=[0, 1, 0, 3]  # <<-- explicit extent (x from 0→1, y from 0→n_actions)
        )
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Action')
        ax3.set_title('Best Run Action Probabilities')
        fig.colorbar(prob_img, ax=ax3, fraction=0.02, pad=0.04)
        smoothing = 0.95
        avg = 0


        train_callback_list.on_batch_begin(0)
        for ep in range(episodes):            
            # Reset time to 0 if desired
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
            rewards_gotten = 0
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
                reward_trace = reward_trace*0.0 + reward

                if env.wait_count == env.wait_inc:
                    current_run.append(env.img(scale=20))
                
                    train_callback_list.on_batch_end(0, all_metrics)
                    for o, custom_updates in compiled_net.optimisers:
                        # Set step on all custom updates
                        for c in custom_updates:
                            o.set_step(c, ep)

                    # train_callback_list.on_batch_begin(0)
                    
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
                td_error_trace = td_error_trace_discount * td_error_trace + td_error

                total_td += td_error
                
                previous_value_estimate = value_estimate
                    
                if reward != 0:
                    rewards_gotten += 1
                    # compiled_net.losses[value].set_target(
                    #     compiled_net.neuron_populations[value], 
                    #     [[[value_target]]], value.shape,
                    #     compiled_net.genn_model.batch_size,
                    #     compiled_net.example_timesteps
                    # )
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
            
            for _ in range(1):
                compiled_net.step_time(train_callback_list)

                # value_estimate = compiled_net.get_readout(value)[0][0]
                
                # value_target = reward_trace + gamma * value_estimate
                # td_error = value_target - previous_value_estimate

                # total_td += td_error
                
                # previous_value_estimate = value_estimate
                
                # compiled_net.losses[value].set_td_error(
                #     compiled_net.neuron_populations[value], td_error
                # )
                # compiled_net.losses[policy].set_td_error(
                #     compiled_net.neuron_populations[policy], td_error
                # )
                # compiled_net.neuron_populations[hidden_1].vars["TdE"].view[:] = td_error
                # compiled_net.neuron_populations[hidden_1].vars["TdE"].push_to_device()
            
            train_callback_list.on_batch_end(0, all_metrics)
            if np.mean(snake_len_history) > 7.5 and compiled_net.optimisers[0][0].alpha != 1e-5:
                compiled_net.optimisers[0][0].alpha = 1e-5
                # compiled_net.optimisers[0][0].alpha = (
                #    max(1e-5, 5e-4 / rewards_gotten) 
                #    # max(1e-5, compiled_net.optimisers[0][0].alpha*0.9999)
                # )

            for o, custom_updates in compiled_net.optimisers:
                # Set step on all custom updates
                for c in custom_updates:
                    o.set_step(c, ep)

            # --- Update if new best run ---
            if total_reward >= best_reward and len(current_probs) > 0:
                best_reward = total_reward
                best_run = [img for img in current_run]

                # Update value plot
                value_line.set_data(range(len(current_values)), current_values)
                ax2.relim()
                ax2.autoscale_view()
                ax2.set_title(f'Best Run Value Outputs (Reward = {total_reward:.2f})')

                # --- Update heatmap plot with correct extent ---
                time_steps = len(current_probs)
                prob_img.set_data(np.array(current_probs).T)
                prob_img.set_extent([0, time_steps, 0, 4])  # <<-- fix horizontal scaling
                ax3.set_xlim(0, time_steps)
                ax3.set_ylim(0, 4)
                ax3.set_aspect('auto')
                ax3.set_title(f'Best Run Action Probabilities (Reward = {total_reward:.2f})')
                plt.pause(0.01)

            if (ep) % 1000 == 0 and best_run:
                print(f"Replaying best run so far (reward = {best_reward:.2f})...")
                # agent.update_parameters()
                best_reward = -np.inf
                for img in best_run:
                    cv2.imshow("Snake", img)
                    cv2.waitKey(1)
                    time.sleep(0.1)

            reward_history.append(total_reward)
            snake_len_history.append(len(env.snake)-1)

            print(f"Episode {ep+1} - "
            f"Total reward: {' ' if total_reward >= 0 else ''}{total_reward:.2f}"
            # f" - Td Error: {total_td/frame:.3f}"
            f" - Snake len: {len(env.snake)-1:2d}"
            f" - Snake len avg (last 300): {np.mean(snake_len_history):.2f}"
            f" - Frame death: {frame}"
            f" - Alpha: {compiled_net.optimisers[0][0].alpha:.8f}")

            if avg == 0:
                avg = total_reward
            else:
                avg = smoothing * avg + (1 - smoothing) * total_reward
            running_avg.append(avg)

            if ep % 100 == 0:
                snake_len_history = snake_len_history[-200:]
                reward_history = reward_history[-200:]
                running_avg = running_avg[-200:]
                axis = range(ep - len(reward_history), ep)
                snake_len.set_data(axis, snake_len_history)
                line.set_data(axis, reward_history)
                avg_line.set_data(axis, running_avg)
                ax1.relim()
                ax1.autoscale_view()
                plt.pause(0.01)

        plt.ioff()
        plt.show()
        cv2.destroyAllWindows()


    if __name__ == "__main__":
        train_snake_agent()
