##################### PONG ENV #####################
######################################################
# Hardcoded Pong implementation (no ALE/gym), mirroring SnakeEnv's
# structure and API exactly: reset() -> obs, step(action) -> (obs,
# reward, done), plus img()/local_img() for rendering.
#
# The whole point of this file: the observation is a LOCAL window
# clamped (never wrapped) to the player paddle's position, the same
# way SnakeEnv.get_local_img_observation() clamps cam_y/cam_x to
# [0, size - visible_range] instead of ever wrapping around the
# torus. Here the "camera" only has one degree of freedom (paddle y),
# since Pong's paddle moves vertically only.
#
# Court geometry, in "table units" (not pixels -- pixels are only
# used for rendering / the img() methods, exactly like SnakeEnv keeps
# a grid model separate from its cv2 image model):
#
#   y in [0, COURT_H)   -- vertical position, 0 = top wall
#   x in [0, COURT_W)   -- horizontal position, 0 = left wall
#
# Player paddle sits at a fixed x column (PADDLE_X_PLAYER), the
# opponent (simple tracking AI) at the opposite column
# (PADDLE_X_OPPONENT). Both paddles have height PADDLE_H (in table
# units) and move up/down only.
######################################################

import numpy as np
import cv2
import random

# ─── Court / paddle / ball geometry (table units) ────────────────
COURT_H = 84
COURT_W = 84

PADDLE_H = 10
PADDLE_MARGIN = 6          # distance from the side wall to the paddle column
PADDLE_X_PLAYER   = PADDLE_MARGIN
PADDLE_X_OPPONENT = COURT_W - 1 - PADDLE_MARGIN

BALL_SIZE = 2

PADDLE_SPEED   = 4          # table units / step, per action
OPPONENT_SPEED = 2          # deliberately a bit slower/imperfect than the player
BALL_SPEED_X   = 3
BALL_SPEED_Y_RANGE = (1, 5)

MAX_SCORE = 21              # rally cap; episode also ends on this
MAX_STEPS_WITHOUT_SCORE = 2000


class PongEnv:
    """
    Hardcoded Pong. API mirrors SnakeEnv:

        env = PongEnv(visible_range=21, scale=4, wait_inc=0,
                       inp_shape=(21, 21, 1))
        obs = env.reset()
        obs, reward, done = env.step(action)   # action: 0=down, 1=stay, 2=up

    Observation is a LOCAL grayscale crop of the court, `visible_range`
    table-units tall and wide, CENTERED ON THE PLAYER PADDLE's y
    position -- clamped to stay inside the court, never wrapped. This
    is exactly SnakeEnv.get_local_img_observation()'s camera policy,
    just with the "head" being the paddle center instead of the snake
    head, and only one moving axis (y) since the paddle is pinned to
    a fixed x column.
    """

    def __init__(self, visible_range=21, scale=4, wait_inc=0,
                 inp_shape=(21, 21, 1)):
        assert visible_range % 2 == 1, "visible_range must be odd"
        self.visible_range = visible_range
        self.scale = scale
        self.wait_inc = wait_inc
        self.inp_shape = inp_shape
        self.n_actions = 3   # 0=down, 1=stay, 2=up (ordinal, matches ACTION_MAP style)
        self.steps_taken = 0
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.player_y   = COURT_H / 2.0 - PADDLE_H / 2.0
        self.opponent_y = COURT_H / 2.0 - PADDLE_H / 2.0

        self._serve(towards_player=random.random() < 0.5)

        self.player_score    = 0
        self.opponent_score  = 0
        self.done             = False
        self.steps_since_score = 0
        self.steps_taken      = 0

        return self.get_local_img_observation()

    def _serve(self, towards_player):
        self.ball_x = COURT_W / 2.0
        self.ball_y = COURT_H / 2.0
        self.ball_vx = -abs(BALL_SPEED_X) if towards_player else abs(BALL_SPEED_X)
        self.ball_vy = random.choice([-1, 1]) * random.uniform(*BALL_SPEED_Y_RANGE)

    # ------------------------------------------------------------------
    def step(self, action):
        if self.done:
            raise Exception("Environment needs reset. Call env.reset().")

        # ── move player paddle ────────────────────────────────────
        # action: 0=down, 1=stay, 2=up  (matches ACTION_MAP ordinal scheme)
        if action == 0:
            self.player_y += PADDLE_SPEED
        elif action == 2:
            self.player_y -= PADDLE_SPEED
        self.player_y = float(np.clip(self.player_y, 0, COURT_H - PADDLE_H))

        # ── simple tracking opponent ──────────────────────────────
        opp_center = self.opponent_y + PADDLE_H / 2.0
        if opp_center < self.ball_y - 2:
            self.opponent_y += OPPONENT_SPEED
        elif opp_center > self.ball_y + 2:
            self.opponent_y -= OPPONENT_SPEED
        self.opponent_y = float(np.clip(self.opponent_y, 0, COURT_H - PADDLE_H))

        # ── move ball ──────────────────────────────────────────────
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # bounce off top/bottom walls (clamp, no wrap)
        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y >= COURT_H - 1:
            self.ball_y = COURT_H - 1
            self.ball_vy = -abs(self.ball_vy)

        reward = 0.0

        # bounce off player paddle
        if (self.ball_vx < 0 and
                self.ball_x <= PADDLE_X_PLAYER + 1 and
                self.ball_x >= PADDLE_X_PLAYER - 1 and
                self.player_y - 1 <= self.ball_y <= self.player_y + PADDLE_H + 1):
            self.ball_x = PADDLE_X_PLAYER + 1
            self.ball_vx = abs(self.ball_vx)
            offset = (self.ball_y - (self.player_y + PADDLE_H / 2.0)) / (PADDLE_H / 2.0)
            self.ball_vy = float(np.clip(offset * 3.0, -3.0, 3.0))

        # bounce off opponent paddle
        elif (self.ball_vx > 0 and
                self.ball_x >= PADDLE_X_OPPONENT - 1 and
                self.ball_x <= PADDLE_X_OPPONENT + 1 and
                self.opponent_y - 1 <= self.ball_y <= self.opponent_y + PADDLE_H + 1):
            self.ball_x = PADDLE_X_OPPONENT - 1
            self.ball_vx = -abs(self.ball_vx)
            offset = (self.ball_y - (self.opponent_y + PADDLE_H / 2.0)) / (PADDLE_H / 2.0)
            self.ball_vy = float(np.clip(offset * 3.0, -3.0, 3.0))

        # ── scoring ────────────────────────────────────────────────
        if self.ball_x < 0:
            self.opponent_score += 1
            reward -= 1.0
            self.steps_since_score = 0
            self._serve(towards_player=True)
        elif self.ball_x > COURT_W - 1:
            self.player_score += 1
            reward += 1.0
            self.steps_since_score = 0
            self._serve(towards_player=True)
        else:
            self.steps_since_score += 1

        if (self.player_score >= MAX_SCORE or
                self.opponent_score >= MAX_SCORE or
                self.steps_since_score > MAX_STEPS_WITHOUT_SCORE):
            self.done = True

        self.steps_taken += 1

        return self.get_local_img_observation(), reward, self.done

    # ------------------------------------------------------------------
    # Locality-preserving observation: crop centered on the player
    # paddle, clamped to stay inside the court (never wraps), exactly
    # SnakeEnv.get_local_img_observation()'s camera policy.
    # ------------------------------------------------------------------
    def get_local_img_observation(self):
        img = self.img(scale=1)   # raw table-unit-resolution crop
        return cv2.resize(
            img, (self.inp_shape[0], self.inp_shape[1]),
            interpolation=cv2.INTER_NEAREST
        ).reshape(self.inp_shape) / 255.0

    def local_img(self, scale=4):
        """
        Local observation camera.

        X:
            Fixed relative to the player paddle.

        Y:
            Follows the player paddle, clamped to the court.

        Ball:
            - 1.0 when inside the observation.
            - 0.5 projected onto the observation edge when outside.
        """
        v = self.visible_range
        r = v // 2

        # --------------------------------------------------------------
        # Camera
        # --------------------------------------------------------------

        # Player paddle stays near the left side of the observation.
        paddle_view_x = 3

        cam_x = int(round(PADDLE_X_PLAYER)) - paddle_view_x

        # Camera follows paddle vertically.
        paddle_center_y = self.player_y + PADDLE_H / 2.0
        cam_y = int(round(paddle_center_y)) - r

        # Clamp camera to court.
        cam_x = max(0, min(cam_x, COURT_W - v))
        cam_y = max(0, min(cam_y, COURT_H - v))

        # --------------------------------------------------------------
        # Render
        # --------------------------------------------------------------

        img = np.zeros((v, v, 1), dtype=np.uint8)

        for ly in range(v):
            y = cam_y + ly

            for lx in range(v):
                x = cam_x + lx

                # Court walls
                if y == 0 or y == COURT_H - 1:
                    img[ly, lx, 0] = 60

                # Player paddle
                if (
                    x == PADDLE_X_PLAYER
                    and self.player_y <= y < self.player_y + PADDLE_H
                ):
                    img[ly, lx, 0] = 255

                # Opponent paddle
                if (
                    x == PADDLE_X_OPPONENT
                    and self.opponent_y <= y < self.opponent_y + PADDLE_H
                ):
                    img[ly, lx, 0] = 255

        # --------------------------------------------------------------
        # Ball
        # --------------------------------------------------------------

        # Ball position in local-observation coordinates.
        ball_lx = self.ball_x - cam_x
        ball_ly = self.ball_y - cam_y

        # Is the ball inside the observation?
        ball_in_view = (
            0 <= ball_lx < v
            and 0 <= ball_ly < v
        )

        if ball_in_view:
            # Normal ball = 1.0 after normalization.
            bx0 = max(0, int(np.floor(ball_lx - BALL_SIZE / 2.0)))
            bx1 = min(v, int(np.ceil(ball_lx + BALL_SIZE / 2.0 + 1)))

            by0 = max(0, int(np.floor(ball_ly - BALL_SIZE / 2.0)))
            by1 = min(v, int(np.ceil(ball_ly + BALL_SIZE / 2.0 + 1)))

            img[by0:by1, bx0:bx1, 0] = 255

        else:
            # ----------------------------------------------------------
            # Project off-screen ball onto observation boundary.
            #
            # We find the intersection of the ray:
            #
            #     center_of_view -> ball
            #
            # with the observation rectangle.
            # ----------------------------------------------------------

            cx = (v - 1) / 2.0
            cy = (v - 1) / 2.0

            dx = ball_lx - cx
            dy = ball_ly - cy

            if dx != 0 or dy != 0:
                # Find how far we need to travel along the ray to
                # reach one of the four edges.
                tx = float("inf")
                ty = float("inf")

                if dx > 0:
                    tx = (v - 1 - cx) / dx
                elif dx < 0:
                    tx = (0 - cx) / dx

                if dy > 0:
                    ty = (v - 1 - cy) / dy
                elif dy < 0:
                    ty = (0 - cy) / dy

                t = min(tx, ty)

                proj_x = cx + dx * t
                proj_y = cy + dy * t

                # Keep the indicator safely on the observation boundary.
                px = int(round(np.clip(proj_x, 0, v - 1)))
                py = int(round(np.clip(proj_y, 0, v - 1)))

                # Off-screen ball indicator = 0.5.
                indicator_radius = 1

                x0 = max(0, px - indicator_radius)
                x1 = min(v, px + indicator_radius + 1)
                y0 = max(0, py - indicator_radius)
                y1 = min(v, py + indicator_radius + 1)

                img[y0:y1, x0:x1, 0] = 128

        return cv2.resize(
            img,
            (v * scale, v * scale),
            interpolation=cv2.INTER_NEAREST
        )
    
    def img(self, scale=4):
        """Full-court (non-local) render, for the 'best run' visualizer feed."""
        img = np.zeros((COURT_H, COURT_W, 1), dtype=np.uint8)
        img[0, :] = 60
        img[-1, :] = 60

        py0, py1 = int(self.player_y), int(self.player_y + PADDLE_H)
        img[py0:py1, PADDLE_X_PLAYER] = 255

        oy0, oy1 = int(self.opponent_y), int(self.opponent_y + PADDLE_H)
        img[oy0:oy1, PADDLE_X_OPPONENT] = 255

        bx, by = int(self.ball_x), int(self.ball_y)
        y0, y1 = max(0, by - 1), min(COURT_H, by + 2)
        x0, x1 = max(0, bx - 1), min(COURT_W, bx + 2)
        img[y0:y1, x0:x1] = 255

        return cv2.resize(img, (COURT_W * scale, COURT_H * scale),
                          interpolation=cv2.INTER_NEAREST)

    def render(self):
        return self.img(scale=self.scale)

    def close(self):
        pass
