"""Play the arena task (snake_switch.ArenaEnv) yourself: coins and enemies in the dark, a player light and a gaze light.

    python arena_play.py [--level 0.0] [--render bumps|pixels] [--fps 60] [--scale 16] [--seed 0]

Keys (discrete: a held key pushes its axis fully):
    W A S D       move the body
    arrow keys    the gaze joystick: the gaze light swings that way, and with the stick pushed (any arrow held) the
                  body fires toward the gaze, as the agent does (gaze = aim)
    space         pause / resume
    + / -         curriculum level up / down by 0.1 (fewer coins, more and faster enemies, smaller lights)
    r             a new world
    Esc / q       quit
The window shows the world from above (left, everything visible) and the agent's own view (right: what the network
receives, upscaled), with the segment's coins, kills and hits and the running score. Uses only Tkinter.
"""
import argparse
import time
import tkinter as tk

import numpy as np

import snake_switch

MOVE = {"w": (1, 1), "s": (1, -1), "a": (0, -1), "d": (0, 1)}
LOOK = {"Up": (1, 1), "Down": (1, -1), "Left": (0, -1), "Right": (0, 1)}


def ppm(img):
    """An RGB uint8 array as PPM bytes for tk.PhotoImage."""
    h, w, _ = img.shape
    return b"P6 %d %d 255 " % (w, h) + np.ascontiguousarray(img, dtype=np.uint8).tobytes()


class Player:
    def __init__(self, a):
        self.a = a
        self.root = tk.Tk()
        self.root.title("arena: WASD move, arrows look / shoot, space pause, +/- level, r reset")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.hud = tk.Label(self.root, font=("monospace", 11), justify="left", anchor="w")
        self.hud.pack(fill="x")
        self.held, self.release_at = set(), {}
        self.paused, self.score, self.frames = False, 0.0, 0
        self.level = float(a.level)
        self.new_world()
        self.root.bind("<KeyPress>", self.press)
        self.root.bind("<KeyRelease>", self.release)

    def new_world(self):
        cur = {"step": 0.0, "up": 10 ** 9, "down": -10 ** 9}        # the level stays where the player puts it
        self.env = snake_switch.ArenaEnv(0, (20, 20, 3), seed=self.a.seed, render=self.a.render, curriculum=cur,
                                         segment_steps=10 ** 9)
        self.env.level = self.level
        self.env._apply_level()
        self.env.reset()
        self.score = 0.0

    def press(self, ev):
        k = ev.keysym if ev.keysym in LOOK else ev.keysym.lower()
        self.release_at.pop(k, None)                               # X11 auto-repeat: a release then a press
        if k in MOVE or k in LOOK:
            self.held.add(k)
        elif k == "space":
            self.paused = not self.paused
        elif k in ("plus", "equal", "kp_add", "minus", "kp_subtract"):
            self.level = float(np.clip(self.level + (0.1 if k in ("plus", "equal", "kp_add") else -0.1), 0, 1))
            self.env.level = self.level
            self.env._apply_level()
        elif k == "r":
            self.new_world()
        elif k in ("escape", "q"):
            self.root.destroy()

    def release(self, ev):
        k = ev.keysym if ev.keysym in LOOK else ev.keysym.lower()
        self.release_at[k] = time.time()                           # confirmed in the loop if no press follows

    def action(self):
        now = time.time()
        for k, t in list(self.release_at.items()):
            if now - t > 0.03:
                self.held.discard(k)
                del self.release_at[k]
        act = np.zeros(4)
        for k in self.held:
            if k in MOVE:
                axis, sign = MOVE[k]; act[axis] += sign
            elif k in LOOK:
                axis, sign = LOOK[k]; act[2 + axis] += sign
        return np.clip(act, -1, 1)

    def frame(self):
        """World view (drawn at 80 px, scaled up) | the agent's 20 x 20 view (scaled up)."""
        sc = self.a.scale
        view = np.kron((np.clip(self.env.get_local_img_observation(), 0, 1) * 255).astype(np.uint8),
                       np.ones((sc, sc, 1), np.uint8))
        h = view.shape[0]
        k = max(1, h // 80)
        world = np.kron(self.env.world_img(h // k), np.ones((k, k, 1), np.uint8))
        world = np.pad(world, ((0, h - world.shape[0]), (0, h - world.shape[1]), (0, 0)))
        return np.concatenate([world, np.full((h, 6, 3), 255, np.uint8), view], axis=1)

    def tick(self):
        t0 = time.time()
        if not self.paused:
            _, r, _ = self.env.step(self.action())
            self.score += r
            self.frames += 1
        img = self.frame()
        self.photo = tk.PhotoImage(data=ppm(img), format="PPM")
        self.label.configure(image=self.photo)
        e = self.env
        self.hud.configure(text=f"score {self.score:+.0f}   coins {e.n_got}  kills {e.kills}  hits {e.hits}  "
                                f"shots {e.n_shots}   level {e.level:.1f}  coins {len(e.coins)}  enemies "
                                f"{len(e.enemies)}  light {e.light:.2f}" + ("   [paused]" if self.paused else ""))
        if self.a.smoke and self.frames >= self.a.smoke:
            self.root.destroy()
            return
        delay = max(1, int(1000 / self.a.fps - 1000 * (time.time() - t0)))
        self.root.after(delay, self.tick)

    def run(self):
        self.root.after(10, self.tick)
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", type=float, default=0.0); ap.add_argument("--render", default="bumps")
    ap.add_argument("--fps", type=int, default=60); ap.add_argument("--scale", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0, help=argparse.SUPPRESS)  # tests: quit after N frames
    Player(ap.parse_args()).run()


if __name__ == "__main__":
    main()
