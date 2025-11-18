####################### SNAKE ENV #######################
#########################################################
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import time

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
            self.done = True
        elif max(np.ceil(np.log10(len(empty_cells))), 1) > len(self.apples):
            self.apples.extend(random.sample(empty_cells, k=int(max(np.ceil(np.log10(len(empty_cells))), 1) - len(self.apples))))
        #self.apple = empty_cells[15%len(empty_cells)]

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
        
        reward = 0.0
        if (self.direction == 'up' and new_dir == 'down') or \
           (self.direction == 'down' and new_dir == 'up') or \
           (self.direction == 'left' and new_dir == 'right') or \
           (self.direction == 'right' and new_dir == 'left'):
            reward -= 10
            new_dir = self.direction

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
                reward += 500.0
            self.steps_since_last_apple = 0
        else:
            self.snake.pop()
            self.steps_since_last_apple += 1

        if self.steps_since_last_apple > self.size**2:
            self.done = True
            reward -= 100.0
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

####################### TRAIN #######################
#####################################################


BOARD_SIZE = 5
VISIBLE_RANGE = 5
INPUT_SIZE = 3 * VISIBLE_RANGE**2

def train_snake_agent(episodes=100000):
    env = SnakeEnv(size=BOARD_SIZE, visible_range=VISIBLE_RANGE, wait_inc=7)
    
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

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        current_run = []
        current_values = []
        current_probs = []
        total_td = 0
        frame = 0

        env.wait_count = 16
        while not done:
            action_label = 0
            if env.wait_count == 0:       
                current_probs.append(np.random.rand(4))
                current_values.append(np.random.rand(1))

                action_label = np.random.choice([0, 1, 2, 3])

            obs, reward, done = env.step(action_label)
            total_reward += reward

            current_run.append(env.img(scale=20))

            frame += 1
       
        # --- Update if new best run ---
        if total_reward >= best_reward:
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

        if (ep) % 10000 == 0 and best_run:
            print(f"Replaying best run so far (reward = {best_reward:.2f})...")
            # agent.update_parameters()
            best_reward = -np.inf
            for img in best_run:
                cv2.imshow("Snake", img)
                cv2.waitKey(1)
                time.sleep(0.01)

        print(f"Episode {ep+1} - Total reward: {total_reward:.2f} - Td Error: {total_td/frame:.3f} - Frame death: {frame}")

        reward_history.append(total_reward)
        snake_len_history.append(len(env.snake)-1)
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
