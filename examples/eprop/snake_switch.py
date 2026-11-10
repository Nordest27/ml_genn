"""Snake with performance-triggered task switches, and a monitor of frozen ("dead") hidden neurons.

SwitchingEnv wraps SnakeEnv. Whenever the agent's reward per environment step over the last `window` moves
reaches `criterion`, the task is mangled: the input channels are permuted, the actions are remapped and/or
the view is flipped. Every mangle is a bijection, so each new task is exactly as learnable as the first one,
but the learned input/output mapping is wrong after it. A rule that keeps its plasticity relearns quickly and
reaches more switches within a fixed budget; the number of switches is the score. The sequence of mangles is
fixed by `seed`, so every rule sees the same sequence.

NeuronMonitor samples the hidden ALIF populations once per environment step and, every `report_every`
moves, writes per population: the fraction of plasticity-dead neurons (mean pseudo-derivative psi below
`psi_eps`, so their synapses cannot change), split into silent and still-firing ones, the mean psi, and the
fraction of the previous report's dead neurons that recovered.
"""
import csv
import os
from collections import deque

import numpy as np
from pygenn import SynapseMatrixConnectivity

MANGLES = ("channels", "actions", "flip")


class SwitchingEnv:
    def __init__(self, env, criterion=0.08, window=20000, mangles=("channels", "actions"), seed=0,
                 log_path=None, wait_inc=30):
        unknown = set(mangles) - set(MANGLES)
        if unknown:
            raise ValueError(f"unknown mangles {sorted(unknown)}; available: {MANGLES}")
        self.env, self.criterion, self.window, self.mangles = env, criterion, int(window), tuple(mangles)
        self.rng = np.random.default_rng(seed)
        self.n_channels = env.inp_shape[2]
        self.n_actions = 4
        self.chan = np.arange(self.n_channels)
        self.act = np.arange(self.n_actions)
        self.flip = (False, False)
        self.switches = 0
        self.seen = {self._key(self.chan, self.act, self.flip)}   # mappings used so far (never repeated)
        self.moves = 0                     # environment moves (excluding the waiting steps)
        self.moves_at_switch = 0
        self.recent = deque(maxlen=self.window)
        self.recent_sum = 0.0
        self.log_path, self.wait_inc = log_path, wait_inc
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(["switch", "moves", "moves_since_last", "channels", "actions", "flip"])

    # ---- delegation: the training loop reads env.wait_count, env.dir_idx, env.img(...), ...
    def __getattr__(self, name):
        return getattr(self.env, name)

    def __setattr__(self, name, value):
        if name in ("wait_count",):
            setattr(self.env, name, value)
        else:
            object.__setattr__(self, name, value)

    def _mangle(self, obs):
        obs = obs[:, :, self.chan]
        if self.flip[0]:
            obs = obs[::-1]
        if self.flip[1]:
            obs = obs[:, ::-1]
        return np.ascontiguousarray(obs)

    def reset(self):
        return self._mangle(self.env.reset())

    def step(self, action):
        moving = self.env.wait_count == 0
        obs, reward, done = self.env.step(int(self.act[action]))
        if moving or done:
            self.moves += 1
            if len(self.recent) == self.window:
                self.recent_sum -= self.recent[0]
            self.recent.append(reward)
            self.recent_sum += reward
            if self.ready():
                self._switch()
        return self._mangle(obs), reward, done

    def env_probs(self, probs):
        """Policy probabilities in the environment's action order (index = real direction), for plots."""
        out = np.zeros_like(np.asarray(probs, dtype=float))
        out[self.act] = probs
        return out

    def agent_view(self):
        """The observation the network receives now (mangled), as a uint8 image."""
        return (np.clip(self._mangle(self.env.get_local_img_observation()), 0, 1) * 255).astype(np.uint8)

    def img(self, scale=10):
        """Frame for the viewers: the real board, what the agent sees, and the current task in a header."""
        import cv2
        board = (self.env.board_img if hasattr(self.env, "board_img") else self.env.img)(scale=scale)
        h = board.shape[0]
        view = cv2.resize(self.agent_view(), (h, h), interpolation=cv2.INTER_NEAREST)
        sep = np.full((h, 4, 3), 255, np.uint8)
        frame = np.concatenate([board, sep, view], axis=1)
        header = np.zeros((34, frame.shape[1], 3), np.uint8)
        flip = "".join(("V" if self.flip[0] else "") + ("H" if self.flip[1] else "")) or "-"
        cv2.putText(header, f"task {self.switches}   board | agent sees", (4, 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(header, f"ch {''.join(map(str, self.chan))} act {''.join(map(str, self.act))} flip {flip}",
                    (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1, cv2.LINE_AA)
        return np.concatenate([header, frame], axis=0)

    def rate(self):
        """Reward per environment move over the window (moves since the last switch only)."""
        return self.recent_sum / max(len(self.recent), 1)

    def ready(self):
        return len(self.recent) == self.window and self.rate() >= self.criterion

    @staticmethod
    def _key(chan, act, flip):
        return (tuple(int(c) for c in chan), tuple(int(a) for a in act), tuple(flip))

    def _draw(self):
        chan = self.rng.permutation(self.n_channels) if "channels" in self.mangles else self.chan
        act = self.rng.permutation(self.n_actions) if "actions" in self.mangles else self.act
        flip = (tuple(bool(x) for x in self.rng.integers(0, 2, 2)) if "flip" in self.mangles else self.flip)
        return chan, act, flip

    def _switch(self):
        """A uniformly random mapping the agent has not seen before (once all are used, any other than the
        current one), so improvements across tasks cannot come from remembering an earlier mapping."""
        for attempt in range(10000):
            chan, act, flip = self._draw()
            key = self._key(chan, act, flip)
            if key not in self.seen or (attempt > 5000 and key != self._key(self.chan, self.act, self.flip)):
                break
        self.chan, self.act, self.flip = np.asarray(chan), np.asarray(act), flip
        self.seen.add(key)
        self.switches += 1
        since = self.moves - self.moves_at_switch
        self.moves_at_switch = self.moves
        self.recent.clear()
        self.recent_sum = 0.0
        print(f"[switch {self.switches}] after {since} moves: channels {self.chan.tolist()}, "
              f"actions {self.act.tolist()}, flip {self.flip}", flush=True)
        if self.log_path:
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([self.switches, self.moves, since, " ".join(map(str, self.chan)),
                                        " ".join(map(str, self.act)), "".join("1" if x else "0" for x in self.flip)])


class ReplayEnv:
    """Replays a probe recording (ProbeRecorder with its input frames and board state) instead of playing: every move
    the network is given the recorded agent's frame, whatever its own actions, so another network (e.g. the untrained
    reservoir) can be probed on exactly the states a trained agent visited. The board state behind each frame (body,
    apple, heading) is set on the wrapped SnakeEnv for the probe's task variables and the viewers.

    Episodes follow the recording's episode index. A recorded sample is the frame after a move; the first sample of
    each episode is presented at reset (its own move was not recorded), the following ones after each move.
    `exhausted` turns True when the last recorded episode ends (snake.py then stops)."""
    DIRS = ["left", "up", "right", "down"]

    def __init__(self, env, probe, window="last"):
        z = np.load(probe, allow_pickle=False)
        if f"{window}_inp" not in z.files or z[f"{window}_inp"].size == 0:
            raise ValueError(f"replay: {probe} has no input frames for window {window!r}")
        if int(z["board"]) != env.size:
            raise ValueError(f"replay: recorded board {int(z['board'])} != this board {env.size}")
        self.env = env
        self.frames, self.body, self.apple = z[f"{window}_inp"], z[f"{window}_body"], z[f"{window}_apple"]
        self.heading = z[f"{window}_y"][:, 6].astype(int)
        ep = z[f"{window}_extra"][:, 4]
        starts = np.r_[0, np.where(ep[1:] != ep[:-1])[0] + 1]
        self.episodes = list(zip(starts, np.r_[starts[1:], len(ep)]))
        self.shape = env.get_local_img_observation().shape
        if int(np.prod(self.shape)) != self.frames.shape[1]:
            raise ValueError(f"replay: recorded frames have {self.frames.shape[1]} values, this input {self.shape}")
        self.k, self.i, self.end = 0, 0, 0
        self.wait_inc, self.wait_count = env.wait_inc, env.wait_inc
        self.done, self.exhausted = False, False

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _frame(self, i):
        return (self.frames[i].astype(np.float32) / 255.0).reshape(self.shape)

    def _set_state(self, i):
        self.env.snake = [tuple(int(v) for v in b) for b in self.body[i] if b[0] >= 0]
        self.env.apples = [tuple(int(v) for v in self.apple[i])] if self.apple[i][0] >= 0 else []
        self.env.direction = self.DIRS[self.heading[i]]

    def reset(self):
        self.i, self.end = self.episodes[self.k]
        self.k += 1
        self._set_state(self.i)
        self.wait_count, self.done = self.wait_inc, False
        return self._frame(self.i)

    def step(self, action):
        if self.wait_count > 0:
            self.wait_count -= 1
            return self._frame(self.i), 0.0, False
        self.i += 1
        if self.i >= self.end:                  # the recorded episode ended here
            self.done = True
            self.exhausted = self.k >= len(self.episodes)
            return self._frame(self.end - 1), 0.0, True
        self._set_state(self.i)
        self.wait_count = self.wait_inc
        return self._frame(self.i), 0.0, False


class CuedForagingEnv:
    """Cued foraging: two apples on the board, type A and type B, each drawn in its own input channel (channels 2 and
    3 of a 4-channel observation; 0 and 1 as in Snake). A cue says which type is good; it is re-drawn at random at
    every spawn and exposed as `cue` (0: A is good, 1: B is good) for the training script to feed to the network's
    cue population. Eating the good apple works as in Snake (+1, the snake grows); eating the bad one gives
    `bad_reward` and no growth. Either way both apples respawn with a new cue. The value of heading for an apple is
    cue x type, a contextual XOR that a linear readout of the frame and the cue cannot represent.

    Apples never spawn on a cell next to the head, so no move can be forced onto one. From a snake length (cells,
    head included) of `max_bad_length` on, only the good apple spawns. Starvation (no good apple for size^2 moves)
    is the wrapped Snake's rule; a bad apple does not reset it. fixed_cue=0 or 1 keeps the cue constant (always the same
    type good): the control in which no conjunction with the cue is needed."""

    def __init__(self, env, bad_reward=-1.0, max_bad_length=10, seed=None, fixed_cue=None):
        self.env, self.bad_reward, self.max_bad_length = env, float(bad_reward), int(max_bad_length)
        self.fixed_cue = None if fixed_cue is None else int(fixed_cue)
        self.rng = np.random.default_rng(seed)
        self.cue, self.bad, self._good = 0, None, None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def __setattr__(self, name, value):
        if name in ("wait_count",):
            setattr(self.env, name, value)
        else:
            object.__setattr__(self, name, value)

    def _spawn(self):
        snake, size = self.env.snake, self.env.size
        hy, hx = snake[0]
        near = {(hy + dy, hx + dx) for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))}
        free = [(i, j) for i in range(size) for j in range(size) if (i, j) not in snake]
        safe = [c for c in free if c not in near] or free
        if not safe:                                       # the board is full
            self.env.apples, self.bad, self._good = [], None, None
            return
        self._good = safe.pop(int(self.rng.integers(len(safe))))
        self.env.apples = [self._good]
        self.bad = (safe[int(self.rng.integers(len(safe)))]
                    if len(snake) < self.max_bad_length and safe else None)
        self.cue = int(self.rng.integers(2)) if self.fixed_cue is None else self.fixed_cue

    def reset(self):
        self.env.reset()
        self._spawn()
        return self.get_local_img_observation()

    def step(self, action):
        moving = self.env.wait_count == 0
        _, reward, done = self.env.step(action)
        if moving and not done:
            if self.bad is not None and self.env.snake[0] == self.bad:
                reward += self.bad_reward
                self._spawn()
            elif list(self.env.apples) != ([self._good] if self._good is not None else []):
                self._spawn()                              # the good apple was eaten (Snake respawned its own)
        return self.get_local_img_observation(), reward, done

    def get_local_img_observation(self):
        good = list(self.env.apples)
        try:
            g = self.env.get_local_img_observation()
            self.env.apples = [self.bad] if self.bad is not None else []
            b = self.env.get_local_img_observation()
        finally:
            self.env.apples = good
        a_map, b_map = (g[..., 2], b[..., 2]) if self.cue == 0 else (b[..., 2], g[..., 2])
        return np.stack([g[..., 0], g[..., 1], a_map, b_map], axis=-1).astype(np.float32)

    def img(self, scale=10):
        """The board with both apples (the good one in Snake's apple colour, red; the bad one blue)."""
        board = self.env.img(scale=scale)
        if self.bad is not None:
            y, x = self.bad
            board[y * scale:(y + 1) * scale, x * scale:(x + 1) * scale] = (255, 0, 0)       # BGR: blue
        return board


class MemoryEnv:
    """Snake with a disappearing apple: a new apple is visible to the agent for `visible_moves` moves, then it is
    removed from the agent's observation (it stays on the board and can still be eaten). The view is centred on the
    head, so the agent has to remember where the apple was and track how its own moves shift it. Fewer visible moves
    = more memory needed. Composes with SwitchingEnv (wrap this one first)."""

    def __init__(self, env, visible_moves=3):
        self.env, self.visible_moves = env, int(visible_moves)
        self.moves_since_spawn = 0
        self._apples = list(env.apples)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def __setattr__(self, name, value):
        if name in ("wait_count",):
            setattr(self.env, name, value)
        else:
            object.__setattr__(self, name, value)

    @property
    def hidden(self):
        return self.moves_since_spawn >= self.visible_moves

    def get_local_img_observation(self):
        if not self.hidden:
            return self.env.get_local_img_observation()
        saved = self.env.apples
        self.env.apples = []
        try:
            return self.env.get_local_img_observation()
        finally:
            self.env.apples = saved

    def _track(self, moved):
        if list(self.env.apples) != self._apples:           # new apple (eaten and respawned, or reset)
            self._apples = list(self.env.apples)
            self.moves_since_spawn = 0
        elif moved:
            self.moves_since_spawn += 1

    def reset(self):
        self.env.reset()
        self._apples = None
        self._track(False)
        return self.get_local_img_observation()

    def step(self, action):
        moving = self.env.wait_count == 0
        _, reward, done = self.env.step(action)
        self._track(moving)
        return self.get_local_img_observation(), reward, done

    def board_img(self, scale=10):
        return self.env.img(scale=scale)

    def img(self, scale=10):
        """The real board (apple drawn), what the agent sees, and whether the apple is hidden."""
        import cv2
        board = self.env.img(scale=scale)
        h = board.shape[0]
        view = cv2.resize((np.clip(self.get_local_img_observation(), 0, 1) * 255).astype(np.uint8), (h, h),
                          interpolation=cv2.INTER_NEAREST)
        frame = np.concatenate([board, np.full((h, 4, 3), 255, np.uint8), view], axis=1)
        header = np.zeros((20, frame.shape[1], 3), np.uint8)
        cv2.putText(header, f"apple {'hidden' if self.hidden else 'visible'} ({self.moves_since_spawn} moves)",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return np.concatenate([header, frame], axis=0)


def pseudo_derivative(v, a, beta, v_thresh, refrac):
    """psi of the ALIF e-prop rule (zero during the refractory period)."""
    psi = (1.0 / v_thresh) * 0.3 * np.maximum(0.0, 1.0 - np.abs((v - (v_thresh + beta * a)) / v_thresh))
    return np.where(refrac > 0, 0.0, psi)


class NeuronMonitor:
    def __init__(self, compiled_net, populations, labels, log_path, report_every=10000, psi_eps=0.01,
                 silent_hz=0.5, dt_ms=1.0, weights=True, score_factors=False):
        """populations: hidden ALIF ml_genn Populations; labels: their names in the log.
        weights: also log, per connection group into a hidden population, how far its weights have moved from
        their initial values (||W - W0|| / ||W0||), to see which layers learn when.
        score_factors: also log, per such group, the size of the adaptation part of e-prop's presynaptic factor
        ZFilter - Beta epsA at the report step: RMS(Beta epsA) / RMS(ZFilter) and the fraction of synapses
        (weighted by |ZFilter|) where the factor is negative."""
        self.net, self.pops, self.labels = compiled_net, list(populations), list(labels)
        self.report_every, self.psi_eps, self.silent_hz, self.dt_ms = int(report_every), psi_eps, silent_hz, dt_ms
        self.log_path = log_path
        self.psi_sum = [np.zeros(int(np.prod(p.shape))) for p in self.pops]
        self.spikes = [np.zeros(int(np.prod(p.shape))) for p in self.pops]
        self.vloss = [0.0 for _ in self.pops]
        self.v_sum = [np.zeros(int(np.prod(p.shape))) for p in self.pops]
        self.dead_zone = [0.0 for _ in self.pops]
        self.prev_spike_count = [None] * len(self.pops)
        self.samples, self.moves = 0, 0
        self.prev_dead = [None] * len(self.pops)
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        name_of = {p: l for p, l in zip(self.pops, self.labels)}
        self.weight_groups = []
        self.score_groups = []
        if score_factors:
            for c, sg in compiled_net.connection_populations.items():
                if not c.is_feedback and c.target() in name_of and "epsilonA" in sg.vars:
                    src = name_of.get(c.source(), c.source().name)
                    self.score_groups.append((f"{src}->{name_of[c.target()]}", sg, c.target()))
        if weights:
            for c, sg in compiled_net.connection_populations.items():
                if c.is_feedback or c.target() not in name_of:
                    continue
                sg.vars["g"].pull_from_device()
                w0 = np.asarray(sg.vars["g"].values, dtype=np.float64).copy()
                src = name_of.get(c.source(), c.source().name)
                self.weight_groups.append((f"{src}->{name_of[c.target()]}", sg, w0, np.linalg.norm(w0) + 1e-12,
                                           getattr(c, "exc_inh_sign", None)))
        cols = ["timestep", "moves", "switch"]
        for l in self.labels:
            cols += [f"{l}_dead", f"{l}_dead_silent", f"{l}_dead_firing", f"{l}_mean_psi", f"{l}_hz",
                     f"{l}_recovered", f"{l}_vloss", f"{l}_v_mean", f"{l}_v_p10", f"{l}_v_p90", f"{l}_hyper",
                     f"{l}_deadzone", f"{l}_n"]
        cols += [f"dW|{name}" for name, *_ in self.weight_groups]
        # Dale's law is not enforced during continuous training: fraction of each sign-constrained group's synapses
        # whose weight has crossed to the other sign (E source: g < 0; I source: g > 0)
        cols += [f"flip|{name}" for name, *_, sign in self.weight_groups if sign is not None]
        # net sign of each sign-constrained group, sign * sum(g) / sum|g|: 1 = every synapse keeps its population's
        # sign, 0 = the group's excitation and inhibition cancel, < 0 = the group acts with the opposite sign
        cols += [f"net|{name}" for name, *_, sign in self.weight_groups if sign is not None]
        cols += [f"epsA|{name}" for name, *_ in self.score_groups]
        cols += [f"piNeg|{name}" for name, *_ in self.score_groups]
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(cols)

    def _zfilter_group(self, sg):
        """GeNN fuses identical presynaptic updates: only one synapse group per source owns the ZFilter array."""
        for other in [sg] + list(self.net.connection_populations.values()):
            if other.src is sg.src and "ZFilter" in other.pre_vars and other.pre_vars["ZFilter"]._array is not None:
                return other
        raise RuntimeError(f"no synapse group owns ZFilter for {sg.name}")

    def _vars(self, pop, names):
        npop = self.net.neuron_populations[pop]
        out = []
        for n in names:
            npop.vars[n].pull_from_device()
            out.append(np.asarray(npop.vars[n].view).ravel().astype(np.float64))
        return out

    def sample(self, timestep, switch=0):
        """Call once per environment move."""
        for i, p in enumerate(self.pops):
            v, a, refrac = self._vars(p, ("V", "A", "RefracTime"))
            self.psi_sum[i] += pseudo_derivative(v, a, p.neuron.beta, p.neuron.v_thresh, refrac)
            # the voltage-regularisation loss snake.py logs for the first core population, here per population
            thr = p.neuron.v_thresh + p.neuron.beta * a
            self.vloss[i] += float(np.mean(np.maximum(v - thr, 0.0) + np.maximum(-v - thr, 0.0)))
            self.v_sum[i] += v
            # dead zone: below the range where psi > 0 (V < Beta A) but above the voltage regulariser's band
            # (V > -(Vthresh + Beta A)), not refractory: no psi-gated term and no vreg gradient acts there
            ba = p.neuron.beta * a
            self.dead_zone[i] += float(np.mean((v < ba) & (v > -(p.neuron.v_thresh + ba)) & (refrac <= 0)))
            # a neuron fired since the last sample if it is (or was recently) refractory
            self.spikes[i] += refrac > 0
        self.samples += 1
        self.moves += 1
        if self.moves % self.report_every == 0:
            self.report(timestep, switch)

    def report(self, timestep, switch):
        row = [int(timestep), self.moves, switch]
        for i, p in enumerate(self.pops):
            psi = self.psi_sum[i] / max(self.samples, 1)
            active = self.spikes[i] / max(self.samples, 1)        # fraction of samples in refractory
            refrac_ms = float(np.mean(np.atleast_1d(p.neuron.tau_refrac or 1.0)))
            hz = 1000.0 * active / max(refrac_ms, self.dt_ms)       # refractory fraction / refractory time
            dead = psi < self.psi_eps
            silent = dead & (hz < self.silent_hz)
            rec = (float(np.mean(~dead[self.prev_dead[i]])) if self.prev_dead[i] is not None
                   and self.prev_dead[i].any() else float("nan"))
            # per-neuron mean membrane potential: population mean, 10th / 90th percentile across neurons, and the
            # fraction held far below threshold (mean V < -5 thresholds: hyperpolarised by strong inhibition)
            v_mean = self.v_sum[i] / max(self.samples, 1)
            row += [dead.mean(), silent.mean(), (dead & ~silent).mean(), psi.mean(), hz.mean(), rec,
                    self.vloss[i] / max(self.samples, 1), float(v_mean.mean()), float(np.percentile(v_mean, 10)),
                    float(np.percentile(v_mean, 90)), float(np.mean(v_mean < -5.0 * p.neuron.v_thresh)),
                    self.dead_zone[i] / max(self.samples, 1), int(np.prod(p.shape))]
            self.dead_zone[i] = 0.0
            self.vloss[i] = 0.0
            self.v_sum[i][:] = 0
            self.prev_dead[i] = dead
            self.psi_sum[i][:] = 0
            self.spikes[i][:] = 0
        flips, nets = [], []
        for name, sg, w0, n0, sign in self.weight_groups:
            sg.vars["g"].pull_from_device()
            g = np.asarray(sg.vars["g"].values, dtype=np.float64)
            row.append(float(np.linalg.norm(g - w0) / n0))
            if sign is not None:
                flips.append(float(np.mean(sign * g < 0)))
                nets.append(float(sign * g.sum() / (np.abs(g).sum() + 1e-12)))
        row += flips + nets
        ratios, negs = [], []
        for name, sg, tgt in self.score_groups:
            sg.vars["epsilonA"].pull_from_device()
            zsg = self._zfilter_group(sg)
            zsg.pre_vars["ZFilter"].pull_from_device()
            eps = np.asarray(sg.vars["epsilonA"].values, dtype=np.float64)
            if sg.matrix_type & SynapseMatrixConnectivity.SPARSE:
                pre, post = np.asarray(sg.get_sparse_pre_inds()), np.asarray(sg.get_sparse_post_inds())
            else:
                pre = np.repeat(np.arange(sg.src.num_neurons), sg.trg.num_neurons)
                post = np.tile(np.arange(sg.trg.num_neurons), sg.src.num_neurons)
            zf = np.asarray(zsg.pre_vars["ZFilter"].view, dtype=np.float64).ravel()[pre]
            beta = np.broadcast_to(np.asarray(tgt.neuron.beta, dtype=np.float64).ravel(), (sg.trg.num_neurons,))[post]
            adapt = beta * eps
            ratios.append(float(np.sqrt(np.mean(adapt ** 2) / (np.mean(zf ** 2) + 1e-30))))
            negs.append(float(np.sum(np.abs(zf) * (zf - adapt < 0)) / (np.sum(np.abs(zf)) + 1e-30)))
        row += ratios + negs
        self.samples = 0
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{x:.4g}" if isinstance(x, float) else x for x in row])
        return row


def entropy_error(probs, coeff):
    """Error signal E for the policy readout that raises the policy's entropy.

    The readout descends along ZFilter * E (E plays the role of dLoss/dlogit), so E is the gradient of -coeff * H
    with respect to the logits: d(-H)/dz_k = p_k (log p_k + H), with H = -sum p log p."""
    p = np.asarray(probs, dtype=np.float64)
    logp = np.log(np.clip(p, 1e-8, 1.0))
    h = -(p * logp).sum()
    return (coeff * p * (logp + h)).astype(np.float32)


class PolicyMonitor:
    """Per-episode log of the policy: mean entropy (nats; uniform over 4 actions = 1.386), mean max probability,
    the effective entropy coefficient, the running |TD error| and the size of the entropy term relative to the
    reward-driven policy-gradient term, ||E|| / (|delta| * ||PG||): ~0.01 = negligible, ~1 = as strong."""
    COLUMNS = ["episode", "timestep", "switch", "decisions", "entropy", "max_prob", "entropy_coeff", "abs_td",
               "entropy_vs_pg"]

    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(self.COLUMNS)
        self._reset()

    def _reset(self):
        self.n, self.h, self.maxp, self.coeff, self.e_norm, self.pg_norm, self.abs_td = 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def decision(self, probs, pg, coeff=0.0, e=None, abs_td=0.0):
        p = np.asarray(probs, dtype=np.float64)
        self.n += 1
        self.h += float(-(p * np.log(np.clip(p, 1e-8, 1.0))).sum())
        self.maxp += float(p.max())
        self.coeff += coeff
        self.abs_td += abs_td
        if e is not None:
            self.e_norm += float(np.linalg.norm(e))
        self.pg_norm += abs_td * float(np.linalg.norm(pg))

    def end_episode(self, episode, timestep, switch=0):
        if self.n == 0:
            return None
        ratio = self.e_norm / self.pg_norm if self.pg_norm > 0 else float("nan")
        row = [episode, int(timestep), switch, self.n, self.h / self.n, self.maxp / self.n, self.coeff / self.n,
               self.abs_td / self.n, ratio]
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{x:.5g}" if isinstance(x, float) else x for x in row])
        self._reset()
        return row


class ProbeRecorder:
    """Records, at every move, the hidden populations' membrane potentials and refractory state together with the
    task variables (apple relative to the head, danger in each direction, heading). Keeps the first `moves` and the
    last `moves` samples of the run, for linear-probe analysis of what the hidden layer represents.

    Timing: a sample is taken one simulation step after the NEXT frame has been presented, so the state mostly
    reflects the frame before. Each sample also stores the frame the network has just been given (`inp`, uint8) and
    extra variables (`extra`: tail relative to the head, snake length, previous heading, episode index), so an
    analysis can pair the state at sample t with the frame and variables of sample t-1 of the same episode
    (layer_probe_tests.py). adapt=True also stores every neuron's adaptation variable A (float16; large for big
    networks), the slow state that carries memory across frames."""
    DIRS = {"left": (0, -1), "up": (-1, 0), "right": (0, 1), "down": (1, 0)}
    EXTRA = ["tail_dy", "tail_dx", "length", "prev_heading", "episode"]

    def __init__(self, compiled_net, populations, labels, moves=5000, inputs=True, adapt=False):
        from collections import deque
        self.net, self.pops, self.labels, self.moves = compiled_net, list(populations), list(labels), int(moves)
        self.inputs, self.adapt = bool(inputs), bool(adapt)
        self.first, self.last = [], deque(maxlen=self.moves)
        self._prev_heading, self._episode = -1, None
        self.board = 0

    @staticmethod
    def task_variables(env):
        """Board-level variables, read through any wrappers (they delegate attribute access)."""
        hy, hx = env.snake[0]
        ay, ax = env.apples[0] if env.apples else (hy, hx)
        danger = []
        for dy, dx in ProbeRecorder.DIRS.values():
            y, x = hy + dy, hx + dx
            danger.append(float(y < 0 or y >= env.size or x < 0 or x >= env.size or (y, x) in env.snake[:-1]))
        heading = list(ProbeRecorder.DIRS).index(env.direction)
        return np.array([ay - hy, ax - hx, *danger, heading], dtype=np.float32)

    def sample(self, env, obs=None, episode=None):
        feats, adapt = [], []
        for p in self.pops:
            npop = self.net.neuron_populations[p]
            for v in ("V", "RefracTime") + (("A",) if self.adapt else ()):
                npop.vars[v].pull_from_device()
            feats.append(np.asarray(npop.vars["V"].view, dtype=np.float32).ravel().copy())
            feats.append((np.asarray(npop.vars["RefracTime"].view).ravel() > 0).astype(np.float32))
            if self.adapt:
                adapt.append(np.asarray(npop.vars["A"].view).ravel().astype(np.float16))
        self.board = env.size
        y = self.task_variables(env)
        if episode != self._episode:                       # a new episode: no previous heading
            self._prev_heading, self._episode = -1, episode
        hy, hx = env.snake[0]
        ty, tx = env.snake[-1]
        extra = np.array([ty - hy, tx - hx, len(env.snake), self._prev_heading,
                          -1 if episode is None else episode], dtype=np.float32)
        self._prev_heading = int(y[6])
        inp = None
        if self.inputs and obs is not None:
            inp = (np.clip(np.asarray(obs, dtype=np.float32), 0.0, 1.0) * 255).astype(np.uint8).ravel()
        # board state (body head-first, padded with -1; apple) for labels computed offline, e.g. snake_strategy safety
        body = np.full((env.size * env.size, 2), -1, np.int8)
        body[:len(env.snake)] = np.asarray(env.snake, np.int8)
        apple = np.asarray(env.apples[0] if env.apples else (-1, -1), np.int8)
        row = (np.concatenate(feats), y, extra, inp, np.concatenate(adapt) if adapt else None, body, apple)
        if len(self.first) < self.moves:
            self.first.append(row)
        self.last.append(row)

    def dump(self, path):
        def stack(rows, i, dtype, width):
            if not rows or rows[0][i] is None:
                return np.zeros((0, width), dtype)
            return np.stack([r[i] for r in rows])
        out = {}
        for name, rows in (("first", self.first), ("last", list(self.last))):
            out[f"{name}_x"] = stack(rows, 0, np.float32, 0)
            out[f"{name}_y"] = stack(rows, 1, np.float32, 7)
            out[f"{name}_extra"] = stack(rows, 2, np.float32, len(self.EXTRA))
            out[f"{name}_inp"] = stack(rows, 3, np.uint8, 0)
            if self.adapt:
                out[f"{name}_a"] = stack(rows, 4, np.float16, 0)
            out[f"{name}_body"] = stack(rows, 5, np.int8, 0)
            out[f"{name}_apple"] = stack(rows, 6, np.int8, 2)
        sizes = [int(np.prod(p.shape)) for p in self.pops]
        shapes = np.array([tuple(p.shape) + (1,) * (3 - len(p.shape)) for p in self.pops], dtype=np.int64)
        np.savez_compressed(path, **out, sizes=np.array(sizes), shapes=shapes, labels=np.array(self.labels),
                            variables=np.array(["apple_dy", "apple_dx", "danger_left", "danger_up", "danger_right",
                                                "danger_down", "heading"]), extra_names=np.array(self.EXTRA),
                            board=np.array(self.board))


class SelectionEnv:
    """Candidate selection: a go / no-go categorical decision inside the RL loop ("does this instance have what we
    want?"), the simplest test of learning a choice without a defined policy.

    An episode presents up to `n_candidates` candidates, one per move. A candidate has `features` binary features
    (each present with probability p), drawn as a fixed random pixel pattern in its own region of the frame (region j:
    quadrant j for 4 features; channel j mod 3), with `noise` pixel flips. A candidate is good iff all `required`
    features are present (default: features 0 and 1, a conjunction). At the end of each candidate's move the agent
    decides: action 0 commits (good: `good_reward`, bad: `bad_reward`, and the episode ends), any other action skips
    to the next candidate. Skipping them all ends the episode with `none_reward`. Exposes the Snake interface the
    training loop uses (wait_count / wait_inc, snake, img); `snake` grows by one on a good commit, so the logged
    length is 1 for a correct choice.

    `switch_moves`: every that many decisions the required features change to the next set in `required_after`
    (cycling through [required] + required_after), so the correct action flips for some candidate types (`switches`
    counts the changes). `ctype` is the shown candidate's type, sum_j present_j 2^j."""

    def __init__(self, wait_inc, inp_shape, n_candidates=8, features=4, required=(0, 1), p=0.5, good_reward=1.0,
                 bad_reward=-1.0, none_reward=0.0, noise=0.05, seed=None, switch_moves=None, required_after=((2, 3),)):
        self.wait_inc, self.wait_count = int(wait_inc), int(wait_inc)
        self.inp_shape = tuple(inp_shape)
        self.n_candidates, self.features, self.p, self.noise = int(n_candidates), int(features), float(p), float(noise)
        self.required = [int(j) for j in required]
        if not self.required or max(self.required) >= self.features:
            raise ValueError("selection: required must name features 0 .. features - 1")
        self.good_reward, self.bad_reward, self.none_reward = float(good_reward), float(bad_reward), float(none_reward)
        self.rng = np.random.default_rng(seed)
        self.required_sets = [list(self.required)] + [[int(j) for j in r] for r in required_after]
        if any(not r or max(r) >= self.features for r in self.required_sets):
            raise ValueError("selection: required_after must name features 0 .. features - 1")
        self.switch_moves = None if switch_moves is None else int(switch_moves)
        self.moves, self.switches = 0, 0
        h, w, c = self.inp_shape
        side = int(np.ceil(np.sqrt(self.features)))
        rh, rw = h // side, w // side
        pattern_rng = np.random.default_rng(12345)        # the feature patterns are fixed across runs
        self.patterns = np.zeros((self.features, h, w, c))
        for j in range(self.features):
            y0, x0 = (j // side) * rh, (j % side) * rw
            self.patterns[j, y0:y0 + rh, x0:x0 + rw, j % c] = pattern_rng.random((rh, rw)) < 0.5
        self.snake, self.done, self.index, self.present, self.obs = [None], False, 0, None, None
        self.won = False

    def _draw(self):
        self.present = self.rng.random(self.features) < self.p
        frame = np.clip(self.patterns[self.present].sum(axis=0), 0.0, 1.0)
        flips = self.rng.random(frame.shape) < self.noise
        self.obs = np.where(flips, 1.0 - frame, frame)

    @property
    def ctype(self):
        return int((self.present * (2 ** np.arange(self.features))).sum())

    @property
    def good(self):
        return bool(self.present[self.required].all())

    def reset(self):
        self.snake, self.done, self.index = [None], False, 0
        self.wait_count = self.wait_inc
        self._draw()
        return self.obs

    def get_local_img_observation(self):
        return self.obs

    def step(self, action):
        if self.done:
            raise Exception("Environment needs reset. Call env.reset().")
        if self.wait_count > 0:
            self.wait_count -= 1
            return self.obs, 0.0, False
        self.wait_count = self.wait_inc
        good = self.good                                   # judged by the rule in force when the candidate was shown
        self.moves += 1
        if self.switch_moves and self.moves % self.switch_moves == 0:
            self.switches += 1
            self.required = self.required_sets[self.switches % len(self.required_sets)]
        if int(action) == 0:                               # commit to the candidate shown during the last move
            self.done = True
            if good:
                self.snake.append(None)
            return self.obs, self.good_reward if good else self.bad_reward, True
        self.index += 1                                    # skip
        if self.index >= self.n_candidates:
            self.done = True
            return self.obs, self.none_reward, True
        self._draw()
        return self.obs, 0.0, False

    def img(self, scale=10):
        return np.kron((self.obs * 255).astype(np.uint8), np.ones((scale, scale, 1), dtype=np.uint8))


class ReachEnv:
    """Continuous reaching: a point agent moves with a 2D velocity toward a target (the apple) in the unit square.

    The view is egocentric: the agent sits at the centre of the frame and the frame spans +-1 world units, so the
    whole square is always visible. Channel 0: the agent (centre), channel 1: the target (a small blob at its offset),
    channel 2 (with `walls`): everything outside the square. Each move the agent's velocity is the action (a 2D vector,
    clipped to unit length) times `speed`; the reward is `progress` x (distance before - distance after), plus
    `reach_reward` when the target is within `radius` (the episode ends; `snake` grows by one), and the episode
    ends without a bonus after `max_moves`. The optimal action in every state is the unit vector to the target, so
    `last_cos` (the cosine between the action taken and that direction) measures per-state correctness. `ctype`
    bins the state into 16 types for the move log: 8 direction sectors x 2 distance bins (near: < `near`).
    Exposes the Snake interface the training loop uses (wait_count / wait_inc, snake, img)."""

    def __init__(self, wait_inc, inp_shape, speed=0.08, radius=0.08, progress=10.0, reach_reward=1.0, max_moves=40,
                 near=0.3, noise=0.0, blob=1, walls=False, seed=None):
        self.wait_inc, self.wait_count = int(wait_inc), int(wait_inc)
        self.inp_shape = tuple(inp_shape)
        self.speed, self.radius, self.progress, self.reach_reward = float(speed), float(radius), float(progress), \
            float(reach_reward)
        self.max_moves, self.near, self.noise, self.blob = int(max_moves), float(near), float(noise), int(blob)
        self.walls = bool(walls)
        self.rng = np.random.default_rng(seed)
        self.switches = 0
        self.snake, self.done, self.moves = [None], False, 0
        self.pos = self.target = None
        self.last_cos, self.last_speed = float("nan"), 0.0
        self._obs = None

    # ---- state
    @property
    def offset(self):
        return self.target - self.pos

    @property
    def ctype(self):
        d = self.offset
        sector = int(np.floor((np.arctan2(d[1], d[0]) + np.pi) / (2 * np.pi) * 8)) % 8
        return sector + 8 * int(np.linalg.norm(d) >= self.near)

    @property
    def good(self):
        """1 if the last action pointed within 45 degrees of the target (for the move log)."""
        return bool(self.last_cos > np.cos(np.pi / 4))

    def move_info(self):
        dist = float("nan") if self.pos is None else float(np.linalg.norm(self.offset))
        return {"cos": self.last_cos, "speed": self.last_speed, "dist": dist}

    def _render(self):
        h, w, c = self.inp_shape
        img = np.zeros((h, w, c))
        cy, cx = h // 2, w // 2
        img[cy, cx, 0] = 1.0
        ys, xs = np.mgrid[0:h, 0:w]
        wy, wx = self.pos[1] + (ys - cy) * 2.0 / h, self.pos[0] + (xs - cx) * 2.0 / w
        if self.walls:
            img[..., 2] = ((wx < 0) | (wx > 1) | (wy < 0) | (wy > 1)).astype(float)
        d = self.offset
        ty, tx = int(round(cy + d[1] * h / 2.0)), int(round(cx + d[0] * w / 2.0))
        b = self.blob
        img[max(0, ty - b):ty + b + 1, max(0, tx - b):tx + b + 1, 1] = 1.0
        if self.noise > 0:
            flips = self.rng.random(img.shape) < self.noise
            img = np.where(flips, 1.0 - img, img)
        self._obs = img

    def reset(self):
        self.snake, self.done, self.moves = [None], False, 0
        self.wait_count = self.wait_inc
        self.pos = self.rng.uniform(0.1, 0.9, 2)
        while True:
            self.target = self.rng.uniform(0.1, 0.9, 2)
            if np.linalg.norm(self.offset) > 2 * self.radius:
                break
        self.last_cos, self.last_speed = float("nan"), 0.0
        self._render()
        return self._obs

    def get_local_img_observation(self):
        return self._obs

    def step(self, action):
        if self.done:
            raise Exception("Environment needs reset. Call env.reset().")
        if self.wait_count > 0:
            self.wait_count -= 1
            return self._obs, 0.0, False
        self.wait_count = self.wait_inc
        a = np.asarray(action, dtype=np.float64).ravel()[:2]
        n = np.linalg.norm(a)
        if n > 1:
            a = a / n
        d0 = np.linalg.norm(self.offset)
        self.last_cos = float(a @ self.offset / (np.linalg.norm(a) * d0)) if np.linalg.norm(a) > 0 else 0.0
        self.last_speed = float(np.linalg.norm(a))
        self.pos = np.clip(self.pos + self.speed * a, 0.0, 1.0)
        self.moves += 1
        d1 = np.linalg.norm(self.offset)
        reward = self.progress * (d0 - d1)
        if d1 < self.radius:
            self.snake.append(None)
            self.done = True
            reward += self.reach_reward
        elif self.moves >= self.max_moves:
            self.done = True
        self._render()
        return self._obs, reward, self.done

    def img(self, scale=10):
        return np.kron((self._obs * 255).astype(np.uint8), np.ones((scale, scale, 1), dtype=np.uint8))
