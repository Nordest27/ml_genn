##################### TIC TAC TOE ENV #####################
#############################################################

import numpy as np
import cv2
import random
from performance_visualizer import PerformanceVisualizer

# ─── Board & display constants ───────────────────────────────
BOARD_ROWS   = 3
BOARD_COLS   = 3
WAIT_INC     = 100          # "thinking" timesteps per move (SAME for both players)
PIXEL_SCALE  = 40           # pixels per cell in the RENDER image
CELL_SCALE   = 4            # pixels per cell in the OBSERVATION image (4x4 -> 12x12 total)
NUM_ACTIONS  = BOARD_ROWS * BOARD_COLS  # one action per cell (flattened row-major)

# Agent occupies channel 0 (blue), opponent channel 1 (red), empty = black
AGENT_COLOR    = np.array([ 50,  100, 255], dtype=np.uint8)   # blue
OPPONENT_COLOR = np.array([255,   30,   0], dtype=np.uint8)   # red
EMPTY_COLOR    = np.array([   0,   0,   0], dtype=np.uint8)   # near-black


class TicTacToeEnv:
    """
    Tic-Tac-Toe environment compatible with the SNN training loop.

    Observation : (BOARD_ROWS * obs_scale, BOARD_COLS * obs_scale, 3) uint8 image / 255
                  – same array the agent sees, rendered as pixel art.
                  Default obs_scale=4 gives a 12x12x3 observation for the 3x3 board.
    Actions     : integer in [0, BOARD_ROWS*BOARD_COLS)  →  flattened cell index
                  (row = action // cols, col = action % cols).
                  Only consulted on timesteps where it is the AGENT's turn to
                  actually place a piece (see `awaiting_agent_action`); ignored
                  otherwise (including during any "thinking" wait phase, and
                  during the opponent's own thinking/move resolution).
    Opponent    : controlled by `opponent` parameter — see below.

    opponent : str
        "random" — picks a random legal cell every turn.
        "medium" — wins if possible, blocks agent if needed, else random.

    Turn / timing model
    --------------------
    Both players get an identical `wait_inc`-timestep "thinking" phase before
    each of their moves — the agent no longer gets to move the instant
    `wait_count` reaches zero while the opponent is placed "for free" in the
    same call. Every single move (agent's or opponent's) is preceded by its
    own wait phase, and `step()` advances the game by exactly one timestep,
    regardless of whose turn it is:

      - While waiting for the CURRENT mover's think-time to elapse, `step()`
        decrements `wait_count` and returns unchanged.
      - When `wait_count` hits 0 and it's the agent's turn, `step()` consumes
        the passed-in `action` and places the agent's piece.
      - When `wait_count` hits 0 and it's the opponent's turn, `step()`
        ignores `action`, has the opponent's policy pick a cell, and places
        the opponent's piece.
      - After any move, if the game isn't over, the turn flips and
        `wait_count` resets to `wait_inc` for the new mover.

    `awaiting_agent_action` tells the caller (e.g. the training loop) whether
    the *next* call to `step()` will actually use the `action` argument to
    place the agent's piece — this replaces the old `wait_count == 0` check,
    since `wait_count == 0` can now also correspond to an opponent-move tick.

    first_player : who moves first each episode.
        "agent"    — agent always moves first (old default behaviour).
        "opponent" — opponent always moves first.
        "random"   — coin flip each `reset()` (new default).
        1 / -1     — same as "agent" / "opponent", provided as ints for
                     convenience if you want to alternate first-player
                     yourself between episodes.
    """
    OPPONENT_MODES = ("random", "medium", "minimax")
    FIRST_PLAYER_MODES = ("agent", "opponent", "random")
    

    def __init__(self, rows=BOARD_ROWS, cols=BOARD_COLS, wait_inc=WAIT_INC,
                scale=PIXEL_SCALE, obs_scale=CELL_SCALE, opponent="medium",
                first_player="random"):
        
        self.all_oponents = opponent=="all"
        opponent = opponent if opponent != "all" else self.OPPONENT_MODES[0]
        assert opponent in self.OPPONENT_MODES, \
            f"opponent must be one of {self.OPPONENT_MODES}, got '{opponent}'"
        self._validate_first_player(first_player)
        self.rows         = rows
        self.cols         = cols
        self.wait_inc     = wait_inc
        self.scale        = scale
        self.opponent     = opponent
        self.obs_scale    = obs_scale
        self.first_player = first_player
        self.reset()

    # ── public API ──────────────────────────────────────────────

    def reset(self):
        # board[r][c] : 0 = empty, 1 = agent, -1 = opponent
        self.board      = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.done       = False
        self.winner     = None          # None | 'agent' | 'opponent' | 'draw'
        self.moves      = 0

        self.current_player = self._resolve_first_player()
        self.wait_count = self.wait_inc

        if self.all_oponents:
            self.opponent = np.random.choice(self.OPPONENT_MODES, p=[0.1, 0.1, 0.8])

        # Opening-move shortcut: on an empty 3x3 board every cell is
        # provably fine with perfect subsequent play, so bake in a
        # uniformly random first move for whichever player goes first,
        # rather than consulting the agent's policy or the opponent's
        # search on a position with no information to act on. This also
        # means the agent never sees `awaiting_agent_action == True` on
        # an empty board — no policy readout/PG is ever computed for it.
        opening_row, opening_col = random.choice(self._legal_cells())
        self._place(opening_row, opening_col, player=self.current_player)
        self.moves += 1
        self.current_player *= -1
        self.wait_count = self.wait_inc

        return self._get_obs()

    def step(self, action: int):
        """
        Advances the game by exactly one timestep.

          • While the current mover's `wait_count > 0`: decrement it and
            return the same obs with reward=0, done=False (identical
            behaviour whether it's the agent's or the opponent's turn).
          • Once `wait_count == 0`:
              - if it's the agent's turn, `action` is placed on the board.
              - if it's the opponent's turn, `action` is ignored and the
                opponent's own policy (random/medium) picks the cell.
            After a move, win/draw is checked; if the game continues, the
            turn flips to the other player and `wait_count` resets to
            `wait_inc` so that player gets an equal think-time before its
            own move.

        Returns (obs, reward/100, done), same convention as before:
        terminal rewards are ±1.0 / 0 after the /100 scaling, illegal
        agent moves are penalised the same way as previously.
        """
        if self.done:
            raise RuntimeError("Call reset() before stepping again.")

        # ── waiting phase (same wait length for either player) ──
        if self.wait_count > 0:
            self.wait_count -= 1
            return self._get_obs(), 0.0, False

        reward = 0.0

        if self.current_player == 1:
            # ── agent's move ─────────────────────────────────────
            row, col = self._decode(action)
            if not self._place(row, col, player=1):     # illegal cell – penalise
                reward = -100.0
                self.done = True
                return self._get_obs(), reward / 100, True

            self.moves += 1

            if self._check_win(row, col, player=1):
                reward      = 100.0
                self.done   = True
                self.winner = 'agent'
                return self._get_obs(), reward / 100, True

            if self._board_full():
                self.done   = True
                self.winner = 'draw'
                return self._get_obs(), reward / 100, True

            self.current_player = -1

        else:
            # ── opponent's move (action arg ignored) ─────────────
            opp_row, opp_col = self._pick_opponent_cell()
            self._place(opp_row, opp_col, player=-1)

            if self._check_win(opp_row, opp_col, player=-1):
                reward      = -100.0
                self.done   = True
                self.winner = 'opponent'
                return self._get_obs(), reward / 100, True

            if self._board_full():
                self.done   = True
                self.winner = 'draw'
                return self._get_obs(), reward / 100, True

            self.current_player = 1

        # ── alive: reset wait counter for the next mover ────────
        self.wait_count = self.wait_inc
        return self._get_obs(), reward / 100, False

    @property
    def awaiting_agent_action(self) -> bool:
        """
        True exactly when the NEXT call to step() will consume `action` to
        place the agent's piece (i.e. wait_count is 0 and it's the agent's
        turn). False during any wait phase and during opponent turns.
        Replaces the old `env.wait_count == 0` check used by callers, which
        is no longer sufficient on its own since wait_count also reaches 0
        on the opponent's turn.
        """
        return (not self.done) and self.wait_count == 0 and self.current_player == 1

    def legal_mask(self):
        """Boolean mask of playable cells (flattened, for masking logits)."""
        return (self.board.flatten() == 0)

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
        h = img.shape[0] * (self.scale // self.obs_scale if self.obs_scale else self.scale)
        w = img.shape[1] * (self.scale // self.obs_scale if self.obs_scale else self.scale)
        # fall back to a simple full-cell scale if the divide doesn't work out cleanly
        h = self.rows * self.scale
        w = self.cols * self.scale
        big = cv2.resize((self._board_to_img()).astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_NEAREST)

        # draw thin grid lines
        for r in range(self.rows + 1):
            y = r * self.scale
            cv2.line(big, (0, y), (w, y), (50, 50, 50), 2)
        for c in range(self.cols + 1):
            x = c * self.scale
            cv2.line(big, (x, 0), (x, h), (50, 50, 50), 2)

        return big

    def _board_to_img(self):
        img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c] == 1:
                    img[r, c] = AGENT_COLOR
                elif self.board[r, c] == -1:
                    img[r, c] = OPPONENT_COLOR
                else:
                    img[r, c] = EMPTY_COLOR
        return img

    # ── internals ───────────────────────────────────────────────

    def _validate_first_player(self, first_player):
        if first_player in (1, -1):
            return
        if first_player not in self.FIRST_PLAYER_MODES:
            raise ValueError(
                f"first_player must be one of {self.FIRST_PLAYER_MODES} or "
                f"1 / -1, got {first_player!r}"
            )

    def _resolve_first_player(self) -> int:
        fp = self.first_player
        if fp == 1 or fp == "agent":
            return 1
        if fp == -1 or fp == "opponent":
            return -1
        if fp == "random":
            return random.choice((1, -1))
        raise ValueError(f"Unhandled first_player value: {fp!r}")

    def _decode(self, action: int):
        return action // self.cols, action % self.cols

    def _encode(self, row: int, col: int):
        return row * self.cols + col

    def _pick_opponent_cell(self):
        if self.opponent == "random":
            return self._opponent_random()
        elif self.opponent == "minimax":
            return self._opponent_minimax()
        else:
            return self._opponent_medium()
        
    def _opponent_random(self):
        return random.choice(self._legal_cells())

    def _opponent_medium(self):
        """Win if possible → block agent → maybe create a 3-in-a-row threat → random."""
        legal = self._legal_cells()

        # 1. Win now if possible
        for (r, c) in legal:
            if self._wins_in(r, c, -1):
                return (r, c)

        # 2. Block agent's immediate win
        for (r, c) in legal:
            if self._wins_in(r, c, 1):
                return (r, c)

        # 3. With 50% probability, prefer a move that creates a threat
        #    (a winning cell available next turn) if one exists.
        # if random.random() < 0.5 or True:
        #     threat_moves = []
        #     for (r, c) in legal:
        #         self.board[r, c] = -1
        #         has_threat = any(
        #             self._wins_in(r2, c2, -1) for (r2, c2) in self._legal_cells()
        #         )
        #         self.board[r, c] = 0
        #         if has_threat:
        #             threat_moves.append((r, c))

        #     if threat_moves:
        #         return random.choice(threat_moves)

        # 4. Fallback: random among all legal cells
        return random.choice(legal)

    def _opponent_minimax(self):
        """Perfect play via negamax with alpha-beta pruning.
        Returns a (row, col) chosen uniformly at random among all moves
        that tie for the optimal score. Root-level moves are evaluated
        without pruning against each other (only alpha=-inf, beta=+inf
        passed in), so every tied-optimal move is fairly considered —
        pruning still happens inside each move's own recursive search.

        Opening-move shortcut: on an empty 3x3 board every one of the 9
        cells is provably drawing-or-better with perfect subsequent play
        (no first move can lose against perfect play), so there's nothing
        to search for on move 1 — just pick uniformly among all legal
        cells. This is also by far negamax's most expensive call otherwise
        (nothing yet narrows the tree), so skipping it there is what
        actually saves the time. Every later move (board no longer empty)
        still goes through the real search exactly as before."""
        if not np.any(self.board):
            return random.choice(self._legal_cells())

        best_score = -float('inf')
        best_moves = []
        legal = self._legal_cells()

        for (r, c) in legal:
            self.board[r, c] = -1
            score = -self._negamax(player=1, depth=1, alpha=-float('inf'), beta=float('inf'))
            self.board[r, c] = 0

            if score > best_score:
                best_score = score
                best_moves = [(r, c)]
            elif score == best_score:
                best_moves.append((r, c))

        return random.choice(best_moves)

    def _negamax(self, player: int, depth: int, alpha: float, beta: float) -> float:
        """
        Standard negamax with alpha-beta pruning over the tic-tac-toe game tree.
        Score is from the perspective of `player` (1 = agent, -1 = opponent):
            +N  = win for `player`, found in N plies  (prefer faster wins)
            -N  = loss for `player`, found in N plies (prefer slower losses)
            0  = draw
        `depth` counts plies already played in this simulated line, used only
        to bias faster wins / slower losses (irrelevant for a 3x3 board's size
        but harmless and cheap to keep).
        """
        # Check terminal state resulting from the last move (player's opponent moved last)
        last_player = -player

        # Instead of relying on tracking, just re-derive terminal status generically:
        if self._someone_has_won(last_player):
            return -(10 - depth)  # bad for `player`: the other side just won
        if self._board_full():
            return 0

        best = -float('inf')
        for (r, c) in self._legal_cells():
            self.board[r, c] = player
            score = -self._negamax(-player, depth + 1, -beta, -alpha)
            self.board[r, c] = 0
            if score > best:
                best = score
            alpha = max(alpha, score)
            if alpha >= beta:
                break  # prune
        return best

    def _someone_has_won(self, player: int) -> bool:
        """Check the whole board for a win by `player` (no last-move coordinate needed)."""
        lines = []
        b = self.board
        for i in range(self.rows):
            lines.append(b[i, :])                      # rows
        for j in range(self.cols):
            lines.append(b[:, j])                       # cols
        if self.rows == self.cols:
            lines.append(np.diag(b))                    # main diagonal
            lines.append(np.diag(np.fliplr(b)))          # anti-diagonal
        return any(np.all(line == player) for line in lines)

    def _wins_in(self, row: int, col: int, player: int) -> bool:
        """Return True if placing `player`'s piece at (row, col) wins immediately."""
        if self.board[row, col] != 0:
            return False
        self.board[row, col] = player
        result = self._check_win(row, col, player)
        self.board[row, col] = 0
        return result

    def _legal_cells(self):
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.board[r, c] == 0]

    def _place(self, row: int, col: int, player: int) -> bool:
        """Place piece at (row, col); return False if occupied/out of bounds."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if self.board[row, col] != 0:
            return False
        self.board[row, col] = player
        return True

    def _check_win(self, row: int, col: int, player: int) -> bool:
        def count_in_direction(dr, dc):
            count = 0
            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            return count

        win_len = self.rows  # 3-in-a-row for a 3x3 board
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            total = 1  # the placed piece itself
            total += count_in_direction(dr, dc)
            total += count_in_direction(-dr, -dc)
            if total >= win_len:
                return True
        return False

    def _board_full(self):
        return not self._legal_cells()


# ─── Standalone demo (no SNN – random agent) ─────────────────

if __name__ == "__main__":
    """
    Quick visual sanity-check: two random agents play N games.
    Uses only OpenCV, no GeNN needed.
    """
    import time

    DEMO_GAMES = 5
    env = TicTacToeEnv(scale=120, first_player="random")

    results = {"agent": 0, "opponent": 0, "draw": 0}

    for g in range(DEMO_GAMES):
        obs  = env.reset()
        done = False
        t    = 0
        while not done:
            frame = env.render()
            cv2.imshow("Tic Tac Toe", frame)
            cv2.waitKey(100)

            if env.awaiting_agent_action:
                legal = env.legal_mask()
                action = random.choice(np.where(legal)[0].tolist())
            else:
                action = 0

            obs, reward, done = env.step(action)
            t += 1

        frame = env.render()
        cv2.imshow("Tic Tac Toe", frame)
        cv2.waitKey(800)

        results[env.winner or "draw"] += 1
        print(f"Game {g+1}: winner={env.winner} | moves={env.moves} | first_player={env.first_player}")

    cv2.destroyAllWindows()
    print("\nResults:", results)