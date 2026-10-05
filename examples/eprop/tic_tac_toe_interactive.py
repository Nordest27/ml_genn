"""
##################### TIC TAC TOE — PLAY VS CHECKPOINT #####################
##############################################################################
# Interactive GUI: play against the trained SNN agent loaded from a
# checkpoint (mirrors tic_tac_toe_snn.py's build/loop) — either as a human
# via mouse clicks, or by pitting the checkpoint against one of the env's
# built-in bot policies (random / medium / minimax) to watch/benchmark it.
#
# UPDATED for the new TicTacToeEnv turn model:
#   env.step(action) now resolves exactly ONE move per call, whichever
#   player's turn it currently is (env.current_player), and BOTH players
#   get an identical wait_inc "thinking" phase before their move — there is
#   no longer a combined "SNN moves, then opponent moves for free" call.
#   Use env.awaiting_agent_action to know when step() is about to consume a
#   real action for the SNN; every other step() call (during either
#   player's wait phase, or on the opponent's actual move) ignores the
#   `action` argument.
#
#   Either player can now go first each game (env.first_player, default
#   "random"), instead of the SNN always moving first.
#
#   When OPPONENT="human", the human is supplied via a monkey-patched
#   _pick_opponent_cell, called on its own, at the natural point in the
#   turn sequence when it's the opponent's turn — no more queuing a click
#   ahead of time to be consumed inside someone else's step() call. When
#   OPPONENT is one of the env's built-in policies instead, no patch is
#   applied and TicTacToeEnv's own _pick_opponent_cell drives that turn.
#
# ASSUMPTIONS THAT MATCH THE REAL ENV:
#   - Board layout: action = row*cols + col, row-major.
#   - obs is float32 in [0,1], render() returns uint8 BGR grid.
#   - wait_count > 0 causes step() to ignore `action` entirely and just
#     decrement, returning the same obs unchanged — true for BOTH players
#     now, not just the agent.
#   - legal_mask() is a flat boolean array of length rows*cols.
#   - env.awaiting_agent_action is True only when the NEXT step() call will
#     consume `action` to place the SNN's piece.
#
# CHECKPOINT LOADING: mirrors the snake script's working pattern —
# network.load(...) must happen BEFORE compiler.compile(network), not
# after. tic_tac_toe_train.build_compiled_network() already does this
# internally, gated on its module-level CHECKPOINT_NAME global, so this
# script sets that global prior to calling build_compiled_network()
# rather than calling network.load() a second time post-compile.
##############################################################################
"""

import os
import sys
import numpy as np
import cv2
import matplotlib
from ml_genn.serialisers import Numpy
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from ml_genn.utils.callback_list import CallbackList

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tic_tac_toe_train as ttt_snn

# ─── Config ────────────────────────────────────────────────────────────
CHECKPOINT_NAME = os.environ.get("TTT_CHECKPOINT", "ttt_ep1000")
FIRST_PLAYER = os.environ.get("TTT_FIRST_PLAYER", "random")  # "agent" | "opponent" | "random"

# "human" plays via mouse clicks (original behaviour). Any other value
# ("random" | "medium" | "minimax") is passed straight through to
# TicTacToeEnv's built-in opponent policies instead — lets you benchmark
# the checkpoint against a specific bot without clicking through games
# yourself.
OPPONENT = os.environ.get("TTT_OPPONENT", "human")  # "human" | "random" | "medium" | "minimax"

# Who plays the "agent" seat (board value 1). "snn" (default) loads and
# uses the trained checkpoint, same as before. Any other value
# ("random" | "medium" | "minimax") swaps in that hardcoded policy for the
# agent seat instead, with no checkpoint/network involved at all — this is
# what lets you play yourself (as OPPONENT) against a hardcoded bot
# directly, e.g. TTT_AGENT=minimax TTT_OPPONENT=human.
AGENT = os.environ.get("TTT_AGENT", "snn")  # "snn" | "random" | "medium" | "minimax"

BOT_PAUSE_MS = int(os.environ.get("TTT_BOT_PAUSE_MS", "400"))  # end-of-game pause for bot-vs-bot

# CRITICAL: build_compiled_network() only calls network.load(...) if the
# module-level CHECKPOINT_NAME global is set *before* it's called — and it
# does so BEFORE compiler.compile(network), which is the only point at
# which loading actually works (mirrors the snake script's working
# pattern: network.load() happens pre-compile, inside/around
# build_compiled_network, never on an already-compiled net). Setting the
# global here, then importing the rest by reference, keeps this correct.
ttt_snn.CHECKPOINT_NAME = CHECKPOINT_NAME

build_compiled_network = ttt_snn.build_compiled_network
softmax_masked = ttt_snn.softmax_masked
obs_to_poisson_rate = ttt_snn.obs_to_poisson_rate
BOARD_ROWS = ttt_snn.BOARD_ROWS
BOARD_COLS = ttt_snn.BOARD_COLS
WAIT_INC = ttt_snn.WAIT_INC
PIXEL_SCALE = ttt_snn.PIXEL_SCALE
OBS_SCALE = ttt_snn.OBS_SCALE
NUM_ACTIONS = ttt_snn.NUM_ACTIONS
INPUT_SIZE = ttt_snn.INPUT_SIZE
CONNECTIVITY_TYPE = "fixed"

serialiser = Numpy(f"ttt_checkpoints")
ttt_snn.serialiser = serialiser

from tic_tac_toe_env import TicTacToeEnv
WINDOW_NAME = "Tic Tac Toe vs SNN"
# "agent" (board value 1) is the SNN. "opponent" (board value -1) is either
# a human (via the HumanMoveSource monkey-patch on _pick_opponent_cell) or
# one of the env's own built-in bot policies, selected by OPPONENT above.
# Who actually moves first each game is decided by env.first_player.


class HumanMoveSource:
    """
    Replaces env._pick_opponent_cell for the duration of the game.
    Blocks (busy-waits, servicing the cv2 window) until the human clicks
    a legal cell, then returns (row, col). Called by env.step() at the
    natural point where it's the opponent's (human's) turn to move — i.e.
    after the human's own wait_inc think-phase has already elapsed, same
    as the SNN.
    """
    def __init__(self, env, window_name):
        self.env = env
        self.window_name = window_name
        self._clicked_cell = None

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        col = x // self.env.scale
        row = y // self.env.scale
        if 0 <= row < self.env.rows and 0 <= col < self.env.cols:
            self._clicked_cell = (int(row), int(col))

    def wait_for_click(self):
        """Poll the cv2 window until a legal cell is clicked."""
        self._clicked_cell = None
        while True:
            frame = self.env.render()
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                raise KeyboardInterrupt
            if self._clicked_cell is not None:
                r, c = self._clicked_cell
                if self.env.board[r, c] == 0:
                    return (r, c)
                # illegal (occupied) cell clicked — ignore and keep waiting
                self._clicked_cell = None

    def __call__(self):
        """Drop-in replacement for env._pick_opponent_cell(self)."""
        return self.wait_for_click()


class AgentBotMoveSource:
    """
    Drop-in stand-in for the SNN when AGENT is a hardcoded policy instead
    of "snn". Reuses TicTacToeEnv's own bot logic (_opponent_random /
    _opponent_medium / _opponent_minimax) but for the agent seat (player
    1) — those methods are written generically off `self.board` /
    `self.rows` / `self.cols`, they don't hardcode which player they're
    picking for, so calling them against `env` while it's the agent's
    turn works unmodified and picks a move exactly as good as if that
    policy were the opponent.
    """
    def __init__(self, env, kind):
        assert kind in ("random", "medium", "minimax")
        self.env = env
        self.kind = kind

    def pick_cell(self):
        if self.kind == "random":
            return self.env._opponent_random()
        elif self.kind == "medium":
            return self.env._opponent_medium()
        else:
            return self.env._opponent_minimax()

    def pick_action(self):
        r, c = self.pick_cell()
        return self.env._encode(r, c)


class ValuePlotter:
    """
    Live matplotlib plot of the SNN's value-head readout over the course
    of a game, sampled every simulated timestep (so it also shows how the
    estimate evolves within a single wait_inc think-phase, not just once
    per move).
    """
    def __init__(self, window_title="Value function"):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 3))
        self.fig.canvas.manager.set_window_title(window_title)
        self.line, = self.ax.plot([], [], color="tab:blue", lw=1.5)
        self.move_lines = []  # vertical markers at each real move boundary
        self.ax.set_xlabel("timestep")
        self.ax.set_ylabel("value estimate")
        self.ax.set_title("Agent value function this game")
        self.ax.grid(True, alpha=0.3)
        self.values = []
        self.move_ticks = []  # x-positions where a real move happened
        self.fig.tight_layout()
        self.fig.show()

    def reset(self):
        self.values = []
        self.move_ticks = []
        for vl in self.move_lines:
            vl.remove()
        self.move_lines = []
        self.line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self._flush()

    def record(self, value):
        self.values.append(float(value))

    def mark_move(self):
        """Call right after a real move (agent's or opponent's) is placed."""
        self.move_ticks.append(len(self.values) - 1)

    def _flush(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def refresh(self):
        xs = np.arange(len(self.values))
        self.line.set_data(xs, self.values)

        # (re)draw move-boundary markers
        for vl in self.move_lines:
            vl.remove()
        self.move_lines = []
        for t in self.move_ticks:
            vl = self.ax.axvline(t, color="tab:red", alpha=0.4, lw=1, linestyle="--")
            self.move_lines.append(vl)

        if self.values:
            self.ax.set_xlim(0, max(len(self.values), 10))
            lo, hi = min(self.values), max(self.values)
            pad = max(0.05, (hi - lo) * 0.1)
            self.ax.set_ylim(lo - pad, hi + pad)
        self._flush()

    def close(self):
        plt.close(self.fig)


class ProbHeatmapPlotter:
    """
    Live matplotlib heatmap of the SNN's policy-head move probabilities,
    reshaped to the board's (rows, cols) layout. Updated once per SNN
    decision (i.e. each time get_snn_action() runs), so it shows what the
    agent was "considering" at the moment it moved.

    Illegal cells (already occupied) are masked out to NaN so they render
    as blank/transparent rather than a misleading zero probability.
    """
    def __init__(self, rows, cols, window_title="Move probabilities"):
        self.rows = rows
        self.cols = cols
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.fig.canvas.manager.set_window_title(window_title)
        blank = np.full((rows, cols), np.nan)
        self.im = self.ax.imshow(blank, cmap="viridis", vmin=0, vmax=1)
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
        self.cbar.set_label("probability")
        self.ax.set_xticks(range(cols))
        self.ax.set_yticks(range(rows))
        self.ax.set_title("Agent move probabilities")

        # Text annotations, one per cell, updated in place each refresh.
        self.texts = [
            [self.ax.text(c, r, "", ha="center", va="center", color="white", fontsize=10)
             for c in range(cols)]
            for r in range(rows)
        ]
        self.fig.tight_layout()
        self.fig.show()

    def reset(self):
        blank = np.full((self.rows, self.cols), np.nan)
        self.im.set_data(blank)
        for row in self.texts:
            for t in row:
                t.set_text("")
        self._flush()

    def update(self, probs, mask):
        """
        probs: flat array of length rows*cols (softmax over legal moves).
        mask: flat boolean array of length rows*cols, True = legal.
        """
        grid = np.asarray(probs, dtype=float).reshape(self.rows, self.cols).copy()
        mask_grid = np.asarray(mask, dtype=bool).reshape(self.rows, self.cols)
        grid[~mask_grid] = np.nan

        self.im.set_data(grid)
        vmax = np.nanmax(grid) if np.any(mask_grid) else 1.0
        self.im.set_clim(vmin=0, vmax=max(vmax, 1e-6))

        for r in range(self.rows):
            for c in range(self.cols):
                if mask_grid[r, c]:
                    self.texts[r][c].set_text(f"{grid[r, c]:.2f}")
                else:
                    self.texts[r][c].set_text("")
        self._flush()

    def _flush(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)


def get_snn_action(compiled_net, policy, env, obs, prob_plotter=None, deterministic=True):
    logits = compiled_net.get_readout(policy).flatten()
    mask = env.legal_mask()
    probs = softmax_masked(logits, mask)

    if prob_plotter is not None:
        prob_plotter.update(probs, mask)

    if deterministic:
        action = int(np.argmax(probs))
    else:
        action = int(np.random.choice(NUM_ACTIONS, p=probs))

    return action


def run_think_phase(compiled_net, train_callback_list, input_pop, value, env, obs,
                     window_name, value_plotter=None):
    """
    Steps through exactly one player's wait_inc "thinking" phase (whoever
    env.current_player currently is), keeping the window responsive and
    feeding/holding spikes for the current obs throughout. Returns the
    (obs, done) state after the wait phase — action is irrelevant here
    since wait_count > 0 always ignores it.

    Both players get this same treatment now, so the SNN's think-time and
    the opponent's think-time (i.e. the window staying open before the
    opponent is allowed/expected to move) are visually and temporally
    symmetric, whether the opponent is a human or a bot policy.

    If `value_plotter` is given, the value-head readout is sampled and
    plotted at every simulated timestep, including during the opponent's
    think-phase (the SNN keeps "watching" and evaluating the board while
    it waits for the opponent to move).
    """
    compiled_net.set_input({input_pop: obs_to_poisson_rate(obs)})
    done = False
    while env.wait_count > 0 and not done:
        frame = env.render()
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt
        compiled_net.step_time(train_callback_list)
        if value_plotter is not None:
            v = compiled_net.get_readout(value)[0].mean()
            value_plotter.record(v)
            value_plotter.refresh()
        obs, reward, done = env.step(0)  # action ignored while waiting

    return obs, done


def main():
    is_human = (OPPONENT == "human")
    agent_is_snn = (AGENT == "snn")

    # Only build/load the SNN when it's actually going to play a seat —
    # skip it entirely for e.g. TTT_AGENT=minimax TTT_OPPONENT=human,
    # where no network is needed at all.
    if agent_is_snn:
        compiled_net, network, input_pop, hidden_layers, policy, value = \
            build_compiled_network(connectivity_type=CONNECTIVITY_TYPE)
    else:
        compiled_net = None
        agent_bot = AgentBotMoveSource(None, AGENT)  # env attached below
    env = TicTacToeEnv(
        rows=BOARD_ROWS, cols=BOARD_COLS,
        wait_inc=WAIT_INC, scale=PIXEL_SCALE, obs_scale=OBS_SCALE,
        # "random" placeholder when human (gets monkey-patched over below);
        # otherwise this is the real opponent policy used for the game,
        # handled natively by TicTacToeEnv._pick_opponent_cell.
        opponent="random" if is_human else OPPONENT,
        first_player=FIRST_PLAYER,
    )

    if not agent_is_snn:
        agent_bot.env = env

    cv2.namedWindow(WINDOW_NAME)

    if is_human:
        human = HumanMoveSource(env, WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, human.on_mouse)
        # Monkey-patch: env.step() calls self._pick_opponent_cell() with no
        # args (it's a bound method), so replace the instance attribute.
        env._pick_opponent_cell = human
    else:
        # No mouse interaction needed for bot-vs-bot — window is still shown
        # so you can watch the game, but clicks are ignored. Setting a
        # no-op callback avoids leaving a stale one from a previous run.
        cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)

    # Simple running tally, printed after each game, handy when watching
    # several games back to back to gauge a policy (SNN or hardcoded).
    results = {"agent": 0, "opponent": 0, "draw": 0}

    # The value/probability plots only mean anything when the SNN is
    # actually playing a seat — skip them entirely for bot-vs-human or
    # bot-vs-bot runs so a hardcoded-agent game doesn't pop up blank,
    # meaningless matplotlib windows.
    value_plotter = ValuePlotter() if agent_is_snn else None
    prob_plotter = ProbHeatmapPlotter(BOARD_ROWS, BOARD_COLS) if agent_is_snn else None

    def play_all_games():
        while True:  # play repeatedly until 'q'
            obs = env.reset()
            done = False
            if agent_is_snn:
                value_plotter.reset()
                prob_plotter.reset()

            while not done:
                if agent_is_snn:
                    # Whoever's turn it is now waits out an identical
                    # wait_inc think-phase before moving (keeps the SNN's
                    # spiking state advancing and the window responsive).
                    obs, done = run_think_phase(
                        compiled_net, train_callback_list, input_pop, value,
                        env, obs, WINDOW_NAME, value_plotter=value_plotter,
                    )
                    if done:
                        break
                else:
                    # No SNN in play for this seat: still service the
                    # window (so a human opponent can click, and 'q' still
                    # works) without stepping any network.
                    frame = env.render()
                    cv2.imshow(WINDOW_NAME, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        raise KeyboardInterrupt
                    while env.wait_count > 0 and not done:
                        obs, reward, done = env.step(0)
                        frame = env.render()
                        cv2.imshow(WINDOW_NAME, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            raise KeyboardInterrupt
                    if done:
                        break

                if env.awaiting_agent_action:
                    if agent_is_snn:
                        action = get_snn_action(
                            compiled_net, policy, env, obs,
                            prob_plotter=prob_plotter, deterministic=False,
                        )
                    else:
                        # Hardcoded agent seat: pick a cell with the same
                        # policy TicTacToeEnv would use for an opponent,
                        # just applied to the agent's turn instead.
                        action = agent_bot.pick_action()
                    obs, reward, done = env.step(action)
                else:
                    # Opponent's turn: human click (via monkey-patch) or
                    # the env's own built-in policy — action ignored
                    # either way.
                    obs, reward, done = env.step(0)

                if agent_is_snn:
                    value_plotter.mark_move()

            if agent_is_snn:
                value_plotter.refresh()
            frame = env.render()
            cv2.imshow(WINDOW_NAME, frame)

            results[env.winner or "draw"] += 1
            print(f"winner={env.winner} | moves={env.moves} | "
                  f"agent={AGENT} | opponent={OPPONENT} | totals={results}")

            # Bot-vs-bot: brief pause so you can still watch each game
            # without needing to click through; any game with a human
            # keeps the longer pause to register the final board state.
            cv2.waitKey(1500 if is_human else BOT_PAUSE_MS)

    try:
        if agent_is_snn:
            train_callback_list = CallbackList(
                [*set(compiled_net.base_train_callbacks)],
                compiled_network=compiled_net,
                num_batches=1,
                num_epochs=1,
            )
            with compiled_net:
                train_callback_list.on_epoch_begin(0)
                train_callback_list.on_batch_begin(0)
                play_all_games()
        else:
            play_all_games()
    except KeyboardInterrupt:
        pass

    if agent_is_snn:
        value_plotter.close()
        prob_plotter.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()