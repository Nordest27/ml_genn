##################### TIC TAC TOE — TWO-AGENT SNN TRAINING (MULTIPROCESS) ######
##################################################################################
# Two INDEPENDENT SNNs, each with its own compiled mlGeNN network, trained
# against each other. Split across TWO WORKER PROCESSES (one per agent)
# because mlGeNN's CompiledNetwork.__enter__ raises:
#
#   RuntimeError: Nested compiled networks are not currently supported
#
# — i.e. only one compiled network can be "active" (inside its `with`
# block) per process at a time, presumably due to GeNN backend global
# state. The previous single-process version
# (tic_tac_toe_snn_2agent.py) cannot work for this reason; this script
# replaces it.
#
# ARCHITECTURE:
#   - COORDINATOR (main process): owns the one TicTacToeEnv, drives
#     env.step(), and is the single source of truth for the board. It does
#     NOT touch GeNN/mlGeNN directly at all.
#   - WORKER A / WORKER B (separate processes): each builds+compiles its
#     own network via tic_tac_toe_train.build_compiled_network() with its
#     own CHECKPOINT_NAME, and lives entirely inside its own
#     `with compiled_net:` block for the process's lifetime. Each worker
#     also owns its own PerformanceVisualizer window (matplotlib windows
#     don't cross process boundaries anyway, so this is required, not just
#     requested).
#   - IPC: one multiprocessing.Pipe per worker. Every simulated timestep,
#     the coordinator sends both workers a TICK message (current obs,
#     legal mask, whose turn it conceptually is, wait_count/wait_inc,
#     done-flag) and blocks until BOTH reply — this preserves the original
#     semantics where BOTH networks advance/observe every timestep
#     (including during the other side's think-phase), not just the mover.
#     A worker's reply carries a real `action` only when the coordinator's
#     TICK told it "it's your move now"; otherwise action is None and the
#     reply is just an acknowledgement that step_time() ran.
#
# TURN MODEL / ASSUMPTIONS (same as the single-process version, restated
# because they still apply and still couldn't be verified against
# tic_tac_toe_env.py's actual source):
#   - env.step(action) resolves exactly ONE move per call, for whichever
#     player env.current_player currently is; action is ignored unless
#     wait_count == 0 and it's a real move.
#   - env.legal_mask(), env.current_player, env.wait_count, env.wait_inc,
#     env.awaiting_agent_action all behave as documented in
#     tic_tac_toe_train.py / the play script.
#   - env.awaiting_agent_action is True only for player 1's real-move
#     ticks (the original single-learner "agent" side). Player -1's real
#     move is instead obtained via the env's opponent-callback hook
#     (_pick_opponent_cell), same trick used for the human in the play
#     script and for Agent B in the single-process version. If your actual
#     env exposes a symmetric per-player "awaiting" check instead, only
#     the coordinator's per-tick branch (see `# ROUTING` below) needs
#     to change.
#   - env's reward is assumed to be from player 1's perspective (as in the
#     single-agent script); Agent B (player -1) receives the negated
#     reward. Adjust in the coordinator's reward-broadcast if your env
#     returns per-player rewards directly.
##################################################################################

import os
import sys
import csv
import random
import numpy as np
import multiprocessing as mp
from time import sleep

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Shared constants (safe to import in coordinator; no GeNN build here) ──
import tic_tac_toe_train as ttt_snn
BOARD_ROWS   = ttt_snn.BOARD_ROWS
BOARD_COLS   = ttt_snn.BOARD_COLS
WAIT_INC     = ttt_snn.WAIT_INC
PIXEL_SCALE  = ttt_snn.PIXEL_SCALE
OBS_SCALE    = ttt_snn.OBS_SCALE
NUM_ACTIONS  = ttt_snn.NUM_ACTIONS
CONNECTIVITY_TYPE = "fixed"
FIRST_PLAYER = os.environ.get("TTT_FIRST_PLAYER", "random")

CHECKPOINT_NAME_A = os.environ.get("TTT_CKPT_A", "ttt_2agent_A")
CHECKPOINT_NAME_B = os.environ.get("TTT_CKPT_B", "ttt_2agent_B")

CSV_OUTPUT = "outputs/ttt_2agent_experiment.csv"


# ════════════════════════════════════════════════════════════════════════
# WORKER PROCESS — everything GeNN-related lives inside here, entirely
# within one `with compiled_net:` block for the process's whole life.
# ════════════════════════════════════════════════════════════════════════

def agent_worker_main(name, player_id, checkpoint_name, conn, episodes):
    """
    Entry point for a worker process. Builds+compiles this agent's own
    network, opens its own PerformanceVisualizer, then services TICK /
    EPISODE_START / EPISODE_END / CHECKPOINT / SHUTDOWN messages from the
    coordinator over `conn` until told to stop.
    """
    if "B" in name:
        sleep(15)
    # Re-import inside the child process (required on some platforms /
    # start methods, and keeps each process's module-level
    # CHECKPOINT_NAME global independent).
    import tic_tac_toe_train as ttt
    from ml_genn.serialisers import Numpy
    from ml_genn.utils.callback_list import CallbackList
    from performance_visualizer import PerformanceVisualizer

    ttt.CHECKPOINT_NAME = checkpoint_name
    serialiser = Numpy(f"ttt_checkpoints_2agents")
    ttt.serialiser = serialiser

    (compiled_net, network, input_pop, hidden_layers, policy, value) = \
        ttt.build_compiled_network(connectivity_type=CONNECTIVITY_TYPE)

    train_callback_list = CallbackList(
        [*set(compiled_net.base_train_callbacks)],
        compiled_network=compiled_net,
        num_batches=1,
        num_epochs=1,
    )

    visualizer = PerformanceVisualizer(window=100)

    reward_decay = ttt.reward_decay
    gamma        = ttt.gamma
    softmax_masked      = ttt.softmax_masked
    obs_to_poisson_rate = ttt.obs_to_poisson_rate
    compute_returns     = ttt.compute_returns

    opt_updt = 0
    v_avg = 0.0
    v_reg_loss_avg = 0.0
    freq_avg = 0.0

    # Per-episode state, reset on EPISODE_START.
    total_reward = 0.0
    reward_trace = 0.0
    current_values = []
    current_raw_r = []
    current_probs = []
    ep_frames = 0
    last_wait_count = None

    def reset_episode_state():
        nonlocal total_reward, reward_trace, current_values, current_raw_r
        nonlocal current_probs, ep_frames, last_wait_count
        total_reward = 0.0
        reward_trace = 0.0
        current_values = []
        current_raw_r = []
        current_probs = []
        ep_frames = 0
        last_wait_count = None

    def egocentric_obs(obs):
        """
        The env always renders ABSOLUTE colors: board value 1 -> blue
        (AGENT_COLOR, channel 2 high), value -1 -> red (OPPONENT_COLOR,
        channel 0 high) -- see TicTacToeEnv._get_obs(). This is fixed
        regardless of whose turn it is or which physical player an agent
        is. Agent A is physically player 1, so raw obs already matches
        "my pieces = blue" -- the same convention the original
        single-agent script was built around. Agent B is physically
        player -1 and would otherwise see itself as red / the opponent as
        blue: backwards relative to that convention, with no signal in
        the input itself indicating "you are the other side" (only the
        negated reward would hint at it, indirectly and slowly).

        For player_id == -1, explicitly swap channel 0 and channel 2 so
        this agent's own pieces always render in "its" blue, regardless
        of which physical side it's playing this episode. No-op for
        player_id == 1. (A full obs[..., ::-1] reversal is NOT equivalent
        here since AGENT_COLOR/OPPONENT_COLOR don't share the same green
        channel value -- swap channels 0 and 2 explicitly instead.)
        """
        if player_id != -1:
            return obs
        swapped = obs.copy()
        swapped[..., 0] = obs[..., 2]
        swapped[..., 2] = obs[..., 0]
        return swapped

    def select_action(obs, mask):
        logits = compiled_net.get_readout(policy).flatten()
        probs = softmax_masked(logits, mask)
        action_label = np.random.choice(NUM_ACTIONS, p=probs)

        y_true = np.zeros(NUM_ACTIONS)
        y_true[action_label] = 1.0
        PG = probs - y_true

        compiled_net.neuron_populations[policy].vars["pre_PG"].view[:] = PG.astype(np.float32)
        compiled_net.neuron_populations[policy].push_var_to_device("pre_PG")

        current_probs.append(probs)
        return int(action_label)

    def observe_reward(reward):
        nonlocal total_reward, reward_trace
        total_reward += reward
        reward_trace = reward_trace * reward_decay + reward
        current_raw_r.append(reward)
        if reward != 0:
            compiled_net.losses[value].set_var(
                compiled_net.neuron_populations[value], "reward", reward
            )

    def maybe_start_new_think_phase(wait_count, wait_inc, current_player, obs, done):
        nonlocal ep_frames, last_wait_count, v_avg, v_reg_loss_avg, freq_avg
        started_new_think_phase = (
            (not done)
            and wait_count == wait_inc
            and last_wait_count != wait_inc
            and current_player == player_id
        )
        if started_new_think_phase:
            ep_frames += 1
            f = 0
            for conn_pop in list(compiled_net.connection_populations.values())[::-1]:
                try:
                    conn_pop.post_vars["FAvg"].pull_from_device()
                    f = conn_pop.post_vars["FAvg"].view
                except Exception:
                    pass
            freq_avg = np.mean(np.abs(f))

            first_hidden = list(hidden_layers.values())[0]
            compiled_net.neuron_populations[first_hidden].vars["V"].pull_from_device()
            compiled_net.neuron_populations[first_hidden].vars["A"].pull_from_device()
            compiled_net.neuron_populations[first_hidden].vars["Beta"].pull_from_device()

            v_view    = compiled_net.neuron_populations[first_hidden].vars["V"].view
            A_view    = compiled_net.neuron_populations[first_hidden].vars["A"].view
            beta_view = compiled_net.neuron_populations[first_hidden].vars["Beta"].view

            v_avg = v_avg * 0.999 + 0.001 * np.mean(abs(v_view))
            v_reg_loss_avg = v_reg_loss_avg * 0.999 + 0.001 * np.mean(abs(
                np.maximum( v_view - (0.61 + beta_view * A_view), 0.0) +
                np.maximum(-v_view - (0.61 + beta_view * A_view), 0.0)
            ))

            compiled_net.set_input({input_pop: obs_to_poisson_rate(obs)})

        if not done:
            last_wait_count = wait_count

    def step_time():
        nonlocal opt_updt
        compiled_net.step_time(train_callback_list)
        compiled_net.genn_model.custom_update("GradientLearn")
        for o, custom_updates in compiled_net.optimisers:
            for c in custom_updates:
                opt_updt += 1
                o.set_step(c, opt_updt)

    def drain_terminal(obs):
        nonlocal reward_trace
        compiled_net.set_input({input_pop: obs_to_poisson_rate(obs)})
        for _ in range(WAIT_INC):
            current_values.append(compiled_net.get_readout(value)[0].mean())
            reward_trace = reward_trace * reward_decay
            current_raw_r.append(0.0)
            step_time()

    with compiled_net:
        train_callback_list.on_epoch_begin(0)
        train_callback_list.on_batch_begin(0)

        while True:
            msg = conn.recv()
            kind = msg["kind"]

            if kind == "SHUTDOWN":
                break

            elif kind == "EPISODE_START":
                reset_episode_state()
                last_wait_count = msg["wait_count"]
                compiled_net.set_input({input_pop: obs_to_poisson_rate(egocentric_obs(msg["obs"]))})
                conn.send({"kind": "ACK"})

            elif kind == "TICK":
                obs = egocentric_obs(msg["obs"])

                # Record value/trace every tick, same as single-process version.
                current_values.append(compiled_net.get_readout(value)[0].mean())

                action = None
                if msg["want_action"]:
                    action = select_action(obs, msg["mask"])

                if msg["reward"] != 0.0:
                    observe_reward(msg["reward"])

                maybe_start_new_think_phase(
                    msg["wait_count"], msg["wait_inc"], msg["current_player"],
                    obs, msg["done"],
                )

                step_time()

                conn.send({"kind": "TICK_ACK", "action": action})

            elif kind == "EPISODE_END":
                drain_terminal(egocentric_obs(msg["obs"]))
                if current_probs:
                    G = compute_returns(current_raw_r, gamma)
                    visualizer.push_metrics(
                        reward=total_reward,
                        values=current_values,
                        reward_trace=np.array(G),
                        probs=current_probs,
                    )
                else:
                    visualizer.push_metrics(reward=total_reward)

                conn.send({
                    "kind": "EPISODE_SUMMARY",
                    "total_reward": total_reward,
                    "ep_frames": ep_frames,
                    "v_avg": v_avg,
                    "freq_avg": freq_avg,
                })

            elif kind == "CHECKPOINT":
                compiled_net.save_connectivity((msg["tag"],), serialiser)
                compiled_net.save((msg["tag"],), serialiser)
                conn.send({"kind": "ACK"})

    visualizer.close()


# ════════════════════════════════════════════════════════════════════════
# COORDINATOR — owns the env, drives the game loop, talks to both workers.
# No GeNN/mlGeNN imports here.
# ════════════════════════════════════════════════════════════════════════

def run_coordinator(episodes=int(1e10)):
    from tic_tac_toe_env import TicTacToeEnv

    os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
    if os.path.exists(CSV_OUTPUT):
        os.remove(CSV_OUTPUT)
    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "reward_A", "reward_B", "ep_steps",
            "voltage_A", "voltage_B", "freq_A", "freq_B",
            "first_player", "winner",
        ])

    conn_a_coord, conn_a_worker = mp.Pipe()
    conn_b_coord, conn_b_worker = mp.Pipe()

    proc_a = mp.Process(
        target=agent_worker_main,
        args=("A", 1, CHECKPOINT_NAME_A, conn_a_worker, episodes),
        daemon=False,
    )
    proc_b = mp.Process(
        target=agent_worker_main,
        args=("B", -1, CHECKPOINT_NAME_B, conn_b_worker, episodes),
        daemon=False,
    )
    proc_a.start()
    proc_b.start()

    env = TicTacToeEnv(
        rows=BOARD_ROWS, cols=BOARD_COLS,
        wait_inc=WAIT_INC, scale=PIXEL_SCALE, obs_scale=OBS_SCALE,
        opponent="random",  # placeholder; _pick_opponent_cell patched below
        first_player=FIRST_PLAYER,
    )

    # Player -1's real move is requested from Worker B over the pipe, then
    # returned synchronously to the env — mirrors the human/AgentB
    # callback pattern from the earlier scripts, just IPC-backed. The
    # actual TICK/step_time() call for Worker B on this same timestep has
    # ALREADY happened by the time this is invoked (see ROUTING below):
    # this callback only extracts the action Worker B already computed.
    pending_action_b = {"action": None}

    def pick_opponent_cell():
        action = pending_action_b["action"]
        row, col = divmod(action, env.cols)
        return (row, col)

    env._pick_opponent_cell = pick_opponent_cell

    avg_a = 0.0
    avg_b = 0.0
    smoothing = 0.95

    try:
        for ep in range(episodes):
            obs = env.reset()
            done = False
            first_player_this_ep = env.current_player

            conn_a_coord.send({"kind": "EPISODE_START", "obs": obs, "wait_count": env.wait_count})
            conn_b_coord.send({"kind": "EPISODE_START", "obs": obs, "wait_count": env.wait_count})
            conn_a_coord.recv()
            conn_b_coord.recv()

            last_reward = 0.0

            while not done:
                # ROUTING: figure out, for THIS tick, whether a real move
                # is expected from A, from B, or neither (still thinking).
                want_a = (not done) and env.wait_count == 0 and env.current_player == 1
                want_b = (not done) and env.wait_count == 0 and env.current_player == -1

                tick_common = dict(
                    kind="TICK",
                    obs=obs, mask=env.legal_mask(),
                    wait_count=env.wait_count, wait_inc=env.wait_inc,
                    current_player=env.current_player, done=done,
                )

                # Reward from the PREVIOUS env.step() is broadcast now, at
                # the start of the tick both nets advance through —
                # assumed to be from player 1's (Agent A's) perspective;
                # Agent B gets the negation. Adjust here if your env
                # instead returns per-player rewards.
                conn_a_coord.send({**tick_common, "want_action": want_a, "reward": last_reward})
                conn_b_coord.send({**tick_common, "want_action": want_b, "reward": -last_reward})

                reply_a = conn_a_coord.recv()
                reply_b = conn_b_coord.recv()

                if want_a:
                    action_to_step = reply_a["action"]
                elif want_b:
                    pending_action_b["action"] = reply_b["action"]
                    action_to_step = 0  # ignored by env.step(); B's move comes via callback
                else:
                    legal = np.where(env.legal_mask())[0]
                    action_to_step = int(random.choice(legal)) if len(legal) else 0

                obs, reward, done = env.step(action_to_step)
                last_reward = reward

            # Final reward tick: deliver the terminal reward from the last
            # env.step() to both workers before EPISODE_END, since the
            # game-loop's own TICK sends only ever carry the *previous*
            # reward (rewards lag by one tick by construction above).
            final_tick = dict(
                kind="TICK", obs=obs, mask=env.legal_mask(),
                wait_count=env.wait_count, wait_inc=env.wait_inc,
                current_player=env.current_player, done=True,
                want_action=False,
            )
            conn_a_coord.send({**final_tick, "reward": last_reward})
            conn_b_coord.send({**final_tick, "reward": -last_reward})

            conn_a_coord.recv()
            conn_b_coord.recv()

            conn_a_coord.send({"kind": "EPISODE_END", "obs": obs})
            conn_b_coord.send({"kind": "EPISODE_END", "obs": obs})
            summary_a = conn_a_coord.recv()
            summary_b = conn_b_coord.recv()

            if (ep + 1) % 1000 == 0:
                conn_a_coord.send({"kind": "CHECKPOINT", "tag": f"ttt_2agent_A_ep{ep+1}"})
                conn_b_coord.send({"kind": "CHECKPOINT", "tag": f"ttt_2agent_B_ep{ep+1}"})
                conn_a_coord.recv()
                conn_b_coord.recv()
                print(f"  [checkpoints saved at ep {ep+1}]")

            ra, rb = summary_a["total_reward"], summary_b["total_reward"]
            avg_a = smoothing * avg_a + (1 - smoothing) * ra if avg_a else ra
            avg_b = smoothing * avg_b + (1 - smoothing) * rb if avg_b else rb

            ep_steps = max(summary_a["ep_frames"], summary_b["ep_frames"], 1)
            with open(CSV_OUTPUT, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ep, ra, rb, ep_steps,
                    summary_a["v_avg"], summary_b["v_avg"],
                    summary_a["freq_avg"], summary_b["freq_avg"],
                    "A" if first_player_this_ep == 1 else "B",
                    env.winner,
                ])

            if ep % 10 == 0:
                print(
                    f"Ep {ep+1:6d} | "
                    f"A reward {ra:+7.2f} (avg {avg_a:+7.2f}) | "
                    f"B reward {rb:+7.2f} (avg {avg_b:+7.2f}) | "
                    f"winner {env.winner} | moves {env.moves} | "
                    f"first {'A' if first_player_this_ep == 1 else 'B'}"
                )
    finally:
        conn_a_coord.send({"kind": "SHUTDOWN"})
        conn_b_coord.send({"kind": "SHUTDOWN"})
        proc_a.join(timeout=10)
        proc_b.join(timeout=10)
        if proc_a.is_alive():
            proc_a.terminate()
        if proc_b.is_alive():
            proc_b.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_coordinator(episodes=int(1e10))