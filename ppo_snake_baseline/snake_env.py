
from re import S
import numpy as np
import random
import time
import cv2

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
