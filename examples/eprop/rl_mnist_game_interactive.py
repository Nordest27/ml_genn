#################### DOOR-KEY MNIST ENV ####################
import numpy as np
import cv2
import mnist
import random
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                log_latency_encode_data)
# ---------------- CONFIG ----------------
GRID_SIZE = 10
CELL_SIZE = 60
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

AGENT_COLOR = (50, 200, 50)
DOOR_COLOR_LOCKED = (120, 120, 120)
DOOR_COLOR_OPEN = (0, 200, 0)
REVEAL_COLOR = (200, 200, 50)
WALL_COLOR = (30, 30, 30)
BG_COLOR = (240, 240, 240)

FONT = cv2.FONT_HERSHEY_SIMPLEX

TRAIN = True

mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
labels = mnist.train_labels() if TRAIN else mnist.test_labels()
images = mnist.train_images() if TRAIN else mnist.test_images()
spikes = log_latency_encode_data(images, 20.0, 51)

# ---------------- ENV ----------------
class DoorKeyMNISTEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.agent_pos = [9, 5]
        self.reveal_pos = (9, 0)
        self.current_digit = np.random.randint(0, 10)
        self.doors = [(0, i) for i in range(10)]
        self.doors_locked = True
        self.correct_door = None
        self.done = False
        self.result = None
        self.agent_looking_at_digit = False

    def step(self, action):
        if self.done:
            return

        dy, dx = 0, 0
        if action == 'up': dy = -1
        elif action == 'down': dy = 1
        elif action == 'left': dx = -1
        elif action == 'right': dx = 1

        ny = self.agent_pos[0] + dy
        nx = self.agent_pos[1] + dx

        if ny < 0 or ny >= GRID_SIZE or nx < 0 or nx >= GRID_SIZE:
            return

        # doors block movement if locked
        if (ny, nx) in self.doors and self.doors_locked:
            return

        self.agent_pos = [ny, nx]

        # reveal zone
        self.agent_looking_at_digit = False
        if (ny, nx) == self.reveal_pos:
            self.sample_digit()
            self.agent_looking_at_digit = True

        # entering a door
        if (ny, nx) in self.doors and not self.doors_locked:
            door_idx = nx
            if door_idx == self.correct_door:
                self.done = True
                self.result = "WIN"
            else:
                self.done = True
                self.result = "LOSE"

    def sample_digit(self):
        self.correct_door = self.current_digit
        self.doors_locked = False

        self.current_mnist_image = self.get_mnist_image(self.current_digit)

    # ---------------- RENDERING ----------------
    def render_mnist_panel(self, scale=6):
        """
        Shows a separate window with the current 28x28 MNIST image.
        Black if no digit has been revealed yet.
        """
        if not self.agent_looking_at_digit:
            img = np.zeros((28, 28), dtype=np.uint8)
        else:
            img = self.current_mnist_image.copy()

        img = cv2.resize(img, (28 * scale, 28 * scale),
                        interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.imshow("MNIST Panel", img)

    def render(self):
        img = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
        img[:] = BG_COLOR

        # grid lines
        for i in range(GRID_SIZE + 1):
            cv2.line(img, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE), WALL_COLOR, 1)
            cv2.line(img, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE), WALL_COLOR, 1)

        # reveal zone
        ry, rx = self.reveal_pos
        self.draw_cell(img, ry, rx, REVEAL_COLOR)
        cv2.putText(img, "REVEAL", (rx * CELL_SIZE + 5, ry * CELL_SIZE + 35),
                    FONT, 0.4, (0,0,0), 1)

        # doors
        for (y, x) in self.doors:
            color = DOOR_COLOR_LOCKED if self.doors_locked else DOOR_COLOR_OPEN
            self.draw_cell(img, y, x, color)
            cv2.putText(img, str(x), (x * CELL_SIZE + 20, y * CELL_SIZE + 40),
                        FONT, 0.7, (0,0,0), 2)

        # agent
        ay, ax = self.agent_pos
        self.draw_cell(img, ay, ax, AGENT_COLOR)

        # status text
        status = "Locked" if self.doors_locked else "Unlocked"
        cv2.putText(img, f"Doors: {status}", (10, WINDOW_SIZE - 10),
                    FONT, 0.6, (0,0,0), 2)

        if self.done:
            cv2.putText(img, self.result, (WINDOW_SIZE//2 - 60, WINDOW_SIZE//2),
                        FONT, 1.2, (0,0,255), 3)
        cv2.imshow("Door-Key MNIST", img)

    def draw_cell(self, img, y, x, color):
        y0 = y * CELL_SIZE
        x0 = x * CELL_SIZE
        img[y0:y0+CELL_SIZE, x0:x0+CELL_SIZE] = color

    def get_mnist_image(self, digit):
        indices = np.where(labels == digit)[0]
        idx = np.random.choice(indices)
        return images[idx]


# ---------------- MAIN LOOP ----------------
def main():
    env = DoorKeyMNISTEnv()
    cv2.namedWindow("Door-Key MNIST")

    while True:
        env.render()
        env.render_mnist_panel()
        
        key = cv2.waitKey(50) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('w'):
            env.step('up')
        elif key == ord('s'):
            env.step('down')
        elif key == ord('a'):
            env.step('left')
        elif key == ord('d'):
            env.step('right')
        elif key == ord('r'):
            env.reset()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
