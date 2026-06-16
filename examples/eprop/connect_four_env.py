##################### CONNECT FOUR ENV #####################
#############################################################

import numpy as np
import cv2
import random
from performance_visualizer import PerformanceVisualizer

# ─── Board & display constants ───────────────────────────────
BOARD_ROWS   = 5
BOARD_COLS   = 6
WAIT_INC     = 100          # "thinking" timesteps per move
PIXEL_SCALE  = 40          # pixels per cell in the observation image
NUM_ACTIONS  = BOARD_COLS  # one action per column

# Agent occupies channel 0 (blue), opponent channel 1 (red), empty = black
AGENT_COLOR    = np.array([  0, 100, 220], dtype=np.uint8)   # blue
OPPONENT_COLOR = np.array([220,  30,  30], dtype=np.uint8)   # red
EMPTY_COLOR    = np.array([ 20,  20,  20], dtype=np.uint8)   # near-black


class ConnectFourEnv:
    """
    Connect-4 environment compatible with the SNN training loop.

    Observation : (BOARD_ROWS, BOARD_COLS, 3) uint8 image / 255
                  – same array the agent sees, rendered as pixel art.
    Actions     : integer in [0, BOARD_COLS)  →  column to drop a piece.
    Opponent    : controlled by `opponent` parameter — see below.

    opponent : str
        "random"        — picks a random legal column every turn.
        "opportunistic" — wins if possible, blocks agent if needed, else random.
    """

    OPPONENT_MODES = ("random", "opportunistic")

    def __init__(self, rows=BOARD_ROWS, cols=BOARD_COLS, wait_inc=WAIT_INC,
                scale=PIXEL_SCALE, obs_scale=1, opponent="opportunistic"):
        assert opponent in self.OPPONENT_MODES, \
            f"opponent must be one of {self.OPPONENT_MODES}, got '{opponent}'"
        self.rows     = rows
        self.cols     = cols
        self.wait_inc = wait_inc
        self.scale    = scale
        self.opponent = opponent
        self.obs_scale = obs_scale
        self.reset()

    # ── public API ──────────────────────────────────────────────

    def reset(self):
        # board[r][c] : 0 = empty, 1 = agent, -1 = opponent
        self.board      = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.done       = False
        self.winner     = None          # None | 'agent' | 'opponent' | 'draw'
        self.wait_count = self.wait_inc
        self.moves      = 0
        return self._get_obs()

    def step(self, action: int):
        """
        Mimics SnakeEnv.step():
          • returns (obs, reward/100, done)
          • during wait_count > 0 returns the same obs with reward=0

        Reward shaping
        --------------
        Terminal rewards (win/loss) are kept at ±1.0 after /100.
        In addition, every non-terminal move gives a *dense* shaped reward
        based on the change in "threat score" — the maximum consecutive-piece
        run the agent has on the board.  This gives the agent a local,
        immediate signal for moves that are making progress toward 4-in-a-row,
        so positive credit assignment is as easy as negative.

        shaped_reward = threat_scale * (new_max_run - old_max_run)
                      - threat_scale * (opp_new_max_run - opp_old_max_run)

        This is zero-sum w.r.t. the opponent so it doesn't inflate rewards,
        and it fires on the *same timestep* as the causative action.
        """
        if self.done:
            raise RuntimeError("Call reset() before stepping again.")

        # ── waiting phase ──────────────────────────────────────
        if self.wait_count > 0:
            self.wait_count -= 1
            return self._get_obs(), 0.0, False

        reward = 0.0

        # snapshot threat scores before the move
        agent_run_before = self._max_threat(player=1)
        opp_run_before   = self._max_threat(player=-1)

        # ── agent move ─────────────────────────────────────────
        row = self._drop(action, player=1)
        if row is None:                          # illegal column – penalise
            reward = -100.0
            self.done = True
            return self._get_obs(), reward / 100, True

        self.moves += 1

        if self._check_win(row, action, player=1):
            reward    = 100.0
            self.done = True
            self.winner = 'agent'
            return self._get_obs(), reward / 100, True

        if self._board_full():
            self.done   = True
            self.winner = 'draw'
            return self._get_obs(), reward / 100, True

        # ── opponent move ──────────────────────────────────────
        opp_col = self._pick_opponent_col()
        opp_row = self._drop(opp_col, player=-1)

        if self._check_win(opp_row, opp_col, player=-1):
            reward    = -100.0
            self.done  = True
            self.winner = 'opponent'
            return self._get_obs(), reward / 100, True

        if self._board_full():
            self.done   = True
            self.winner = 'draw'
            return self._get_obs(), reward / 100, True

        # ── dense shaping reward ───────────────────────────────
        # reward progress toward 4-in-a-row; penalise opponent's progress
        # agent_run_after = self._max_threat(player=1)
        # opp_run_after   = self._max_threat(player=-1)

        # THREAT_SCALE = 10.0   # in raw units; will be /100 at return
        # reward += THREAT_SCALE * (agent_run_after - agent_run_before)
        # reward -= THREAT_SCALE * (opp_run_after   - opp_run_before)

        # ── alive, reset wait counter ──────────────────────────
        self.wait_count = self.wait_inc
        return self._get_obs(), reward / 100, False

    def legal_mask(self):
        """Boolean mask of playable columns (for masking logits)."""
        return np.array([self.board[0, c] == 0 for c in range(self.cols)], dtype=bool)

    # ── rendering ───────────────────────────────────────────────
    def _get_obs(self):
        img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c] == 1:
                    img[r, c] = AGENT_COLOR
                elif self.board[r, c] == -1:
                    img[r, c] = OPPONENT_COLOR
                else:
                    img[r, c] = EMPTY_COLOR
        if self.obs_scale > 1:
            img = cv2.resize(img,
                            (self.cols * self.obs_scale, self.rows * self.obs_scale),
                            interpolation=cv2.INTER_NEAREST)
        return img.astype(np.float32) / 255.0
    
    def render(self):
        """Returns a scaled-up uint8 BGR image suitable for display/recording."""
        obs = self._get_obs()
        img = (obs * 255).astype(np.uint8)
        h = self.rows  * self.scale
        w = self.cols  * self.scale
        big = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

        # draw thin grid lines
        for r in range(self.rows + 1):
            y = r * self.scale
            cv2.line(big, (0, y), (w, y), (50, 50, 50), 1)
        for c in range(self.cols + 1):
            x = c * self.scale
            cv2.line(big, (x, 0), (x, h), (50, 50, 50), 1)

        return big

    # ── internals ───────────────────────────────────────────────

    def _max_threat(self, player: int) -> int:
        """
        Return the length of the longest consecutive run of `player`'s pieces
        along any direction (horizontal, vertical, diagonal).  Empty cells are
        *not* counted — this measures actual pieces only, so it goes 0→1→2→3
        as the agent builds toward 4-in-a-row.
        """
        best = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c] != player:
                    continue
                for dr, dc in directions:
                    run = 1
                    nr, nc = r + dr, c + dc
                    while (0 <= nr < self.rows and 0 <= nc < self.cols
                           and self.board[nr, nc] == player):
                        run += 1
                        nr += dr
                        nc += dc
                    best = max(best, run)
        return best

    def _pick_opponent_col(self) -> int:
        if self.opponent == "random":
            return self._opponent_random()
        else:
            return self._opponent_opportunistic()

    def _opponent_random(self) -> int:
        return random.choice(self._legal_cols())

    def _opponent_opportunistic(self) -> int:
        """Win if possible → block agent → random."""
        legal = self._legal_cols()
        for col in legal:
            if self._wins_in(col, -1):
                return col
        for col in legal:
            if self._wins_in(col, 1):
                return col
        return random.choice(legal)

    def _wins_in(self, col: int, player: int) -> bool:
        """Return True if dropping `player`'s piece in `col` wins immediately."""
        row = self._peek_row(col)
        if row is None:
            return False
        self.board[row, col] = player
        result = self._check_win(row, col, player)
        self.board[row, col] = 0
        return result

    def _peek_row(self, col: int):
        """Return the row a piece would land in without placing it, or None if full."""
        if self.board[0, col] != 0:
            return None
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, col] == 0:
                return r
        return None

    def _legal_cols(self):
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def _drop(self, col: int, player: int):
        """Drop piece in column; return row index or None if full."""
        if self.board[0, col] != 0:
            return None
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, col] == 0:
                self.board[r, col] = player
                return r
        return None
    
    def _check_win(self, row: int, col: int, player: int) -> bool:
        def count_in_direction(dr, dc):
            """Count consecutive pieces from (row, col) in one direction."""
            count = 0
            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            return count

        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            total = 1  # the placed piece itself
            total += count_in_direction(dr, dc)    # forward
            total += count_in_direction(-dr, -dc)  # backward
            if total >= 4:
                return True
        return False

    def _board_full(self):
        return not self._legal_cols()


# ─── Standalone demo (no SNN – random agent) ─────────────────

if __name__ == "__main__":
    """
    Quick visual sanity-check: two random agents play N games.
    Uses only OpenCV, no GeNN needed.
    """
    import time

    DEMO_GAMES = 5
    env = ConnectFourEnv(scale=80)

    results = {"agent": 0, "opponent": 0, "draw": 0}

    for g in range(DEMO_GAMES):
        obs  = env.reset()
        done = False
        t    = 0
        while not done:
            frame = env.render()
            cv2.imshow("Connect Four", frame)
            cv2.waitKey(100)

            if env.wait_count == 0:
                legal = env.legal_mask()
                action = random.choice(np.where(legal)[0].tolist())
            else:
                action = 0

            obs, reward, done = env.step(action)
            t += 1

        frame = env.render()
        cv2.imshow("Connect Four", frame)
        cv2.waitKey(800)

        results[env.winner or "draw"] += 1
        print(f"Game {g+1}: winner={env.winner} | moves={env.moves}")

    cv2.destroyAllWindows()
    print("\nResults:", results)