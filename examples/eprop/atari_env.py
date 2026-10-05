##################### ATARI ENV WRAPPER #####################
###############################################################
# Drop-in replacement for ConnectFourEnv / SnakeEnv.
# Wraps a Gymnasium/ALE Atari game behind the same interface
# the EProp / mlGeNN training loop expects:
#
#   - reset()          -> obs
#   - step(action)     -> (obs, reward, done)
#     SnakeEnv/ConnectFourEnv use: hold the last observation for
#     `wait_inc` timesteps so the SNN has time to integrate spikes
#     before a new action is sampled)
#   - legal_mask()      -> bool array (all actions legal for Atari)
#   - render()          -> upscaled RGB frame for the visualizer
#
# Simplest possible version: single grayscale frame in, no frame
# stacking. Any notion of velocity/motion has to come from the SNN's
# own recurrent state (membrane potential, adaptation, recurrent
# E/I connectivity) rather than being pre-baked into the input —
# see conversation notes if you want to add stacking back later.
###############################################################

import numpy as np
import cv2
import gymnasium as gym
import ale_py


gym.register_envs(ale_py)

def sigmoid_contrast(img, gain=10.0, cutoff=0.5):
    """
    S-curve contrast boost. Pushes values above `cutoff` toward 1.0 and
    values below it toward 0.0, with `gain` controlling how sharp the
    transition is (higher gain = closer to a hard threshold, lower gain
    = gentler curve, gain=0 is a no-op / identity).

    img:    array of values in [0, 1] (e.g. already min-max stretched).
    gain:   steepness of the curve. ~5-10 is a mild-to-moderate boost;
            higher (20+) starts to approximate binarization.
    cutoff: midpoint of the curve — pixels at this value stay ~unchanged,
            pixels above rise, pixels below fall. 0.5 is a sensible
            default once img is already stretched to [0, 1].

    Returns an array in [0, 1], same shape as img.
    """
    raw = 1.0 / (1.0 + np.exp(gain * (cutoff - img)))

    # rescale so the curve's own min/max hit exactly 0 and 1 —
    # otherwise a sigmoid never quite reaches the endpoints
    lo = 1.0 / (1.0 + np.exp(gain * cutoff))
    hi = 1.0 / (1.0 + np.exp(gain * (cutoff - 1.0)))
    return (raw - lo) / (hi - lo)

class AtariEnv:
    def __init__(self,
                 game_id="ALE/Pong-v5",
                 inp_shape=(20, 20, 1),
                 frame_skip=4,
                 terminal_on_life_loss=True,
                 render_scale=4,
                 clip_rewards=True):
        """
        inp_shape: (H, W, 1) — single grayscale frame, no stacking.
        wait_inc:  timesteps the SNN gets to "think" per env step,
                   same role as WAIT_INC in snake.py / connect_four.
        """
        self.game_id = game_id
        self.inp_shape = inp_shape
        self.render_scale = render_scale
        self.clip_rewards = clip_rewards

        base_env = gym.make(game_id, frameskip=1, render_mode="rgb_array")
        self.env = gym.wrappers.AtariPreprocessing(
            base_env,
            frame_skip=frame_skip,
            grayscale_obs=True,
            screen_size=84,
            terminal_on_life_loss=terminal_on_life_loss,
            scale_obs=False,
        )

        self.n_actions = self.env.action_space.n

        self.done = False
        self.steps_taken = 0          # real env steps this episode
        self._last_raw_frame = None
        self._last_processed = None

        self.reset()

    # ------------------------------------------------------------
    def reset(self):
        obs, info = self.env.reset()

        self.done = False
        self.steps_taken = 0
        self._last_raw_frame = obs
        self._last_processed = self._process(obs)

        return self._last_processed

    # ------------------------------------------------------------
    def step(self, action):
        if self.done:
            raise Exception("Environment needs reset. Call env.reset().")

        obs, reward, terminated, truncated, info = self.env.step(int(action))
        self.done = bool(terminated or truncated)
        self.steps_taken += 1

        if self.clip_rewards:
            reward = float(np.clip(reward, -1.0, 1.0))

        self._last_raw_frame = obs
        self._last_processed = self._process(obs)

        return self._last_processed, reward, self.done

    # ------------------------------------------------------------
    def legal_mask(self):
        # Atari action spaces have no illegal moves — kept for parity
        # with ConnectFourEnv/softmax_masked so the training loop can
        # stay identical.
        return np.ones(self.n_actions, dtype=bool)

    # ------------------------------------------------------------
    def _process(self, obs):
        """Downsample the single grayscale frame down to inp_shape using
        a max-pool style reduction, then contrast-stretch to [0, 1].

        Plain area-averaging (cv2.INTER_AREA) blends a tiny fast-moving
        object (e.g. the Pong ball, often just 1-2px at 84x84) into the
        surrounding background block, so it can fade out or vanish
        entirely once downsampled to something like 20x20. Taking the
        local max instead (grayscale dilation) keeps the brightest pixel
        in each neighborhood, so small bright objects survive even when
        the source/target size ratio isn't a clean integer factor.
        """
        h, w, _ = self.inp_shape
        src_h, src_w = obs.shape

        # kernel sized to roughly match the downsampling ratio, so each
        # output pixel's neighborhood covers the source pixels it will
        # effectively replace
        k_h = max(1, int(round(src_h / h)))
        k_w = max(1, int(round(src_w / w)))
        kernel = np.ones((k_h, k_w), dtype=np.uint8)

        dilated = cv2.dilate(obs, kernel)  # local max filter
        resized = cv2.resize(dilated, (w, h), interpolation=cv2.INTER_NEAREST)
        resized = resized.astype(np.float32)

        lo, hi = resized.min(), resized.max()
        if hi > lo:
            stretched = (resized - lo) / (hi - lo)
        else:
            stretched = np.zeros_like(resized)

        return sigmoid_contrast(stretched[:, :, None])

    # ------------------------------------------------------------
    def render(self, mode="agent"):
        """
        Frame for the visualizer window.

        mode="agent" (default): what the SNN actually receives — the
            downsampled inp_shape image, upscaled with nearest-neighbor
            so you see the true blocky resolution rather than something
            implying detail the network was never given.
        mode="raw": the full 84x84 preprocessed game frame, for context
            only — NOT what the agent sees. Useful side-by-side with
            "agent" to sanity-check how much visual information the
            current resolution is throwing away.
        """
        if mode == "raw":
            frame = self._last_raw_frame
            if frame is None:
                frame = np.zeros((84, 84), dtype=np.uint8)
        else:
            if self._last_processed is None:
                h, w, _ = self.inp_shape
                frame = np.zeros((h, w), dtype=np.uint8)
            else:
                # _last_processed is float32 in [0,1], shape (h, w, 1)
                frame = (self._last_processed[:, :, 0] * 255.0).astype(np.uint8)

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        return cv2.resize(
            frame, (w * self.render_scale, h * self.render_scale),
            interpolation=cv2.INTER_NEAREST
        )

    def close(self):
        self.env.close()