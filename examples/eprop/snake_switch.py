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
                 silent_hz=0.5, dt_ms=1.0, weights=True, score_factors=False, ip_max_bias=None):
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
        # intrinsic plasticity: per population with a bias, its mean and 10th / 90th percentile across neurons and
        # the fractions of neurons at the upper / lower limit (|bias| >= 99 % of max_bias)
        self.ip_max = ip_max_bias
        self.ip_pops = [(l, compiled_net.neuron_populations[p]) for l, p in zip(self.labels, self.pops)
                        if "IPBias" in compiled_net.neuron_populations[p].vars]
        for l, _ in self.ip_pops:
            cols += [f"ip|{l}_mean", f"ip|{l}_p10", f"ip|{l}_p90", f"ip|{l}_at_max", f"ip|{l}_at_min"]
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
        for _, npop in self.ip_pops:
            npop.vars["IPBias"].pull_from_device()
            b = np.asarray(npop.vars["IPBias"].view, dtype=np.float64).ravel()
            m = self.ip_max if self.ip_max else max(float(np.abs(b).max()), 1e-12)
            row += [float(b.mean()), float(np.percentile(b, 10)), float(np.percentile(b, 90)),
                    float(np.mean(b >= 0.99 * m)), float(np.mean(b <= -0.99 * m))]
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
    reward-driven policy-gradient term, ||E|| / (|delta| * ||PG||): ~0.01 = negligible, ~1 = as strong; with a
    headless decoder, the mean decision margin (the winning pool's statistic minus the runner-up's; NaN otherwise)."""
    COLUMNS = ["episode", "timestep", "switch", "decisions", "entropy", "max_prob", "entropy_coeff", "abs_td",
               "entropy_vs_pg", "margin"]

    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(self.COLUMNS)
        self._reset()

    def _reset(self):
        self.n, self.h, self.maxp, self.coeff, self.e_norm, self.pg_norm, self.abs_td = 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.margin, self.n_margin = 0.0, 0

    def decision(self, probs, pg, coeff=0.0, e=None, abs_td=0.0, margin=None):
        p = np.asarray(probs, dtype=np.float64)
        if margin is not None:
            self.margin += float(margin)
            self.n_margin += 1
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
               self.abs_td / self.n, ratio, self.margin / self.n_margin if self.n_margin else float("nan")]
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
    counts the changes). `ctype` is the shown candidate's type, sum_j present_j 2^j.

    `commit_types` / `skip_types` (lists of candidate types) switch to the contingency table used to test whether a
    rewarded candidate action is reinforced: a commit on a type in `commit_types` or a skip on a type in `skip_types`
    gets `table_reward` (negative: a punishment), everything else 0; no decision ends the episode (it ends after
    `n_candidates`), so the two actions differ only in their reward. `good` is then "the type is in commit_types"."""

    def __init__(self, wait_inc, inp_shape, n_candidates=8, features=4, required=(0, 1), p=0.5, good_reward=1.0,
                 bad_reward=-1.0, none_reward=0.0, noise=0.05, seed=None, switch_moves=None, required_after=((2, 3),),
                 commit_types=None, skip_types=None, table_reward=1.0):
        self.table = None if commit_types is None and skip_types is None else (
            {int(t) for t in (commit_types or ())}, {int(t) for t in (skip_types or ())})
        self.table_reward = float(table_reward)
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
        if self.table is not None:
            return self.ctype in self.table[0]
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
        if self.table is not None:                         # contingency table: reward by (type, action), no early end
            hit = self.ctype in self.table[0 if int(action) == 0 else 1]
            self.index += 1
            self.done = self.index >= self.n_candidates
            if not self.done:
                self._draw()
            return self.obs, self.table_reward if hit else 0.0, self.done
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
    ends without a bonus after `max_moves`. `wall_penalty`: a move that would leave the square (the position is
    clipped to it) costs this much; with `wall_end` the episode also ends there (like Snake's walls). The optimal action in every state is the unit vector to the target, so
    `last_cos` (the cosine between the action taken and that direction) measures per-state correctness. `ctype`
    bins the state into 16 types for the move log: 8 direction sectors x 2 distance bins (near: < `near`).
    Exposes the Snake interface the training loop uses (wait_count / wait_inc, snake, img)."""

    def __init__(self, wait_inc, inp_shape, speed=0.08, radius=0.08, progress=10.0, reach_reward=1.0, max_moves=40,
                 near=0.3, noise=0.0, blob=1, walls=False, seed=None, wall_penalty=0.0, wall_end=False):
        self.wait_inc, self.wait_count = int(wait_inc), int(wait_inc)
        self.inp_shape = tuple(inp_shape)
        self.speed, self.radius, self.progress, self.reach_reward = float(speed), float(radius), float(progress), \
            float(reach_reward)
        self.max_moves, self.near, self.noise, self.blob = int(max_moves), float(near), float(noise), int(blob)
        self.walls = bool(walls)
        self.wall_penalty, self.wall_end = float(wall_penalty), bool(wall_end)
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
        new = self.pos + self.speed * a
        hit = bool(np.any(new < 0.0) or np.any(new > 1.0))
        self.pos = np.clip(new, 0.0, 1.0)
        self.moves += 1
        d1 = np.linalg.norm(self.offset)
        reward = self.progress * (d0 - d1)
        if hit and self.wall_penalty != 0.0:
            reward -= self.wall_penalty
        if hit and self.wall_end:
            self.done = True
        elif d1 < self.radius:
            self.snake.append(None)
            self.done = True
            reward += self.reach_reward
        elif self.moves >= self.max_moves:
            self.done = True
        self._render()
        return self._obs, reward, self.done

    def img(self, scale=10):
        return np.kron((self._obs * 255).astype(np.uint8), np.ones((scale, scale, 1), dtype=np.uint8))


class ArmEnv:
    """A two-link planar arm reaching for a target, driven by two joint motors, controlled every simulation step.

    The shoulder sits at the centre of the workspace (world coordinates [-1, 1]^2); link lengths `l1`, `l2` (default
    0.45 each, so the hand reaches up to 0.9). Joint angles: theta1 (shoulder, absolute) in [-pi, pi], theta2 (elbow,
    relative to the upper arm) in [-`elbow_limit`, `elbow_limit`]; both clipped at their limits. Every step the action is
    the two joint velocities in [-1, 1] (clipped), times `joint_speed` radians: there is no decision window, and the
    joint speed sets how much time the network has per unit of movement (a slower arm = more thinking time). Targets
    are reachable by construction (the hand position of random joint angles within the limits, at least `min_reach`
    from the shoulder and more than 2 `radius` from the hand). Reward every step: `progress` x (hand-target distance
    before - after), plus `reach_reward` when the hand is within `radius` (the episode ends; `snake` grows by one); the
    episode ends without a bonus after `max_steps` steps. `wait_inc` only sets the cadence of the training loop's
    per-move bookkeeping (logs, frames, monitors): `wait_count` cycles but never holds the arm.
    View (top-down, allocentric, the workspace fills the frame): channel 0 the arm (both links, `width` cells either
    side of the line), channel 1 the target (a square blob of half-size `blob`), channel 2 the hand (half-size
    `hand_blob`). The defaults light up ~80 input pixels on a 20 x 20 frame (a thin arm and small blobs gave ~20, too
    little drive for networks tuned on denser inputs). `noise`: pixel flip probability.
    Move log (once per bookkeeping window): `ctype` = 8 direction sectors (hand -> target) x 2 distance bins (near:
    < `near`); `good` = over the last window the hand moved within 45 degrees of the target direction; move_info: cosine
    and length of the hand's displacement over that window, distance.
    `curriculum`: None or {"start": 0.0, "step": 0.1, "threshold": 0.5, "window": 100, "dist0": 0.2,
    "radius0": 0.25}: a difficulty level in [0, 1] sets the reach radius (radius0 + level * (radius - radius0)) and
    how far beyond it new targets may start (hand-target distance in (1.2 r, 1.2 r + dist0 + level * (full reach -
    dist0)], r the current radius); the level
    rises by `step` whenever the reach rate over the last `window` episodes reaches `threshold` (then the window
    restarts). Every learner sees the same rule; `level` is logged in move_info and in recorded episodes.
    `continuous`: the arm is never reset (Gym Reacher style): a reached target is replaced by a new one from wherever
    the hand is, and so is a target not reached within `target_steps`; `done` comes only every `segment_steps` steps,
    for logging (reset() then keeps the pose and the target), so one "episode" is a segment, len(snake) - 1 its
    number of reaches. With `curriculum`, the level then follows a staircase per target: + `up` after a reach within
    the time limit, - `down` after a timeout (the curriculum's window / threshold are not used).
    `reward_mode`: "progress" (`progress` x distance decrease) or "sign" (`sign_reward` x sign of the distance
    decrease: every step that brings the hand closer earns the same, as in Chadderdon et al. 2012), plus
    `reach_reward` on a reach; "reach": only `reach_reward` on a reach (the critic then has to learn proximity).
    `kinematics`: "arm" (the two-link arm above) or "point" (a point hand on an open plane: the action is its
    velocity, +x/-x and +y/-y read from the same four pools, `joint_speed` world units per step; no inverse
    kinematics); "polar" (a telescoping arm from the shoulder: joint 0 turns it, `joint_speed` radians per step, joint
    1 extends or retracts it, `joint_speed` x (`l1` + `l2`) / 2 world units per step, so both move the hand alike at
    mid-reach; length clipped to [`min_reach` / 2, `l1` + `l2`]; the inverse kinematics is two independent 1-D
    problems: turn toward the target's angle, extend toward its distance). `view`: "allo" (the workspace, shoulder at the centre) or "ego" (point only: the hand at the centre
    of the frame, the target at its offset, so the target's place in the image is the direction to move; targets
    start within the visible range, `point_reach` at the hardest level).
    `view` "joint" (no image): population codes on a joint-space map, joint 0 along the columns (an angle: periodic),
    joint 1 along the rows (polar: the length over [`min_reach` / 2, `l1` + `l2`]; arm: the elbow over its limits),
    each a Gaussian bump of `joint_sigma` cells: channel 0 the current pose, channel 1 the target's pose (arm: the
    elbow solution nearer the current pose), channel 2 the error (target pose - current pose, angle wrapped, the
    map's centre = no error), so a bump's place in channel 2 says directly which way each motor should turn.
    `view` "body": proprioception plus the workspace, all as Gaussian bumps of `joint_sigma` cells: channel 0 the
    current pose on the joint-space map (as "joint"), channel 1 the arm's moving joints in the workspace (the hand;
    for the two-link arm also the elbow, at `elbow_amp`), channel 2 the target in the workspace (shoulder at the
    centre, the frame spans [-1, 1]^2): reaching is the hand bump meeting the target bump, at the same cells.
    `view` "hand": a camera on the hand, turned with the arm so the shoulder always lies straight below the hand
    (a rotation of the whole scene about the shoulder leaves the image unchanged), the frame spanning +-`hand_span`
    world units (default l1 + l2; farther points sit at the edge, in their direction), Gaussian bumps of
    `joint_sigma` cells: channel 0 the shoulder, channel 1 the elbow (two-link arm) or, for the polar arm (no
    elbow), the hand itself at the centre, channel 2 the target. For the polar arm the motor map is then fixed:
    extending moves everything down the frame (the target appears lower), turning moves the target sideways.
    `target_time`: None (the fixed `target_steps` limit) or {"slack": 4.0, "slack_hard": 2.0, "min_steps": 20}:
    continuous mode, each target's time limit is slack x the fewest steps that can reach it (all motors at full speed
    toward the target's pose, see `ideal_steps`) + `min_steps`, the slack going from `slack` at level 0 to
    `slack_hard` at level 1, so the limit tracks the movement each target actually needs.
    `progress_fade`: None, or the curriculum level at which the progress reward has faded out (it is weighted by
    max(0, 1 - level / progress_fade)): dense guidance while the learner is weak, the clean reach-only reward once it
    reaches the harder levels.
    `episode_per_target` (continuous mode): every reach or timeout ends an episode for the learning signals (done,
    a terminal state), but reset() keeps the arm's pose and only draws the next target: the body never resets,
    credit is per target.
    `record_every`: every that many episodes (0: never) the episode's trajectory (joint angles, elbow and hand positions,
    target, reward per step) is saved to `record_dir`/ep<index>.npz for replays (arm_replay.py).
    Exposes the Snake interface the training loop uses (wait_count / wait_inc, snake, img)."""

    def __init__(self, wait_inc, inp_shape, l1=0.45, l2=0.45, joint_speed=0.005, elbow_limit=2.6, radius=0.1,
                 progress=10.0, reach_reward=1.0, max_steps=1200, near=0.3, min_reach=0.2, noise=0.0, blob=2, width=1,
                 hand_blob=1, seed=None, record_every=0, record_dir="outputs/arm_episodes", curriculum=None,
                 continuous=False, target_steps=300, segment_steps=3000, reward_mode="progress", sign_reward=0.05,
                 episode_per_target=False, progress_fade=None, kinematics="arm", view="allo", point_reach=0.85,
                 target_time=None, joint_sigma=2.0, elbow_amp=0.5,
                 hand_span=None):
        self.wait_inc, self.wait_count = int(wait_inc), int(wait_inc)
        self.inp_shape = tuple(inp_shape)
        self.l1, self.l2, self.joint_speed = float(l1), float(l2), float(joint_speed)
        self.elbow_limit, self.radius, self.progress = float(elbow_limit), float(radius), float(progress)
        self.reach_reward, self.max_steps, self.near = float(reach_reward), int(max_steps), float(near)
        self.min_reach, self.noise, self.blob = float(min_reach), float(noise), int(blob)
        self.width, self.hand_blob = int(width), int(hand_blob)
        self.record_every, self.record_dir = int(record_every), str(record_dir)
        self.curriculum = None if curriculum is None else {"start": 0.0, "step": 0.1, "threshold": 0.5, "window": 100,
                                                           "dist0": 0.2, "radius0": 0.25, **dict(curriculum)}
        self.level = 1.0 if self.curriculum is None else float(self.curriculum["start"])
        self._hits = []
        if reward_mode not in ("progress", "sign", "reach"):
            raise ValueError("arm reward_mode must be 'progress', 'sign' or 'reach'")
        self.episode_per_target = bool(episode_per_target)
        if kinematics not in ("arm", "point", "polar") or view not in ("allo", "ego", "joint", "body", "hand") or \
                (view == "ego" and kinematics != "point") or (view in ("joint", "body", "hand") and kinematics == "point"):
            raise ValueError("arm: kinematics 'arm', 'point' or 'polar', view 'allo', 'ego' (point only), 'joint' or "
                             "'body' / 'hand' (arm / polar)")
        self.kinematics, self.view, self.point_reach = kinematics, view, float(point_reach)
        self.joint_sigma, self.elbow_amp = float(joint_sigma), float(elbow_amp)
        self.hand_span = float(l1) + float(l2) if hand_span is None else float(hand_span)
        self.progress_fade = None if progress_fade is None else float(progress_fade)
        if self.progress_fade is not None and (self.progress_fade <= 0 or curriculum is None or reward_mode != "progress"):
            raise ValueError("arm progress_fade: > 0, with a curriculum and reward_mode 'progress'")
        if self.episode_per_target and not continuous:
            raise ValueError("arm episode_per_target needs continuous")
        self.continuous, self.target_steps, self.segment_steps = bool(continuous), int(target_steps), int(segment_steps)
        self.reward_mode, self.sign_reward = reward_mode, float(sign_reward)
        if self.curriculum is not None:
            self.curriculum = {"up": 0.05, "down": 0.05, **self.curriculum}
        self._started, self._tsteps, self.reaches, self.timeouts = False, 0, 0, 0
        self.target_time = None if target_time is None else {"slack": 4.0, "slack_hard": 2.0, "min_steps": 20,
                                                             **dict(target_time)}
        if self.target_time is not None and (not continuous or min(self.target_time["slack"],
                                                                   self.target_time["slack_hard"]) < 1.0):
            raise ValueError("arm target_time: continuous mode, slacks >= 1")
        self._tlimit = self.target_steps
        self.episode, self._rec = -1, None
        if not (0 < self.elbow_limit <= np.pi and self.l1 > 0 and self.l2 > 0 and self.l1 + self.l2 <= 1.0):
            raise ValueError("arm: links > 0 with l1 + l2 <= 1 (the frame), 0 < elbow_limit <= pi")
        if not self.min_reach < self.l1 + self.l2 - 2 * self.radius:
            raise ValueError("arm: min_reach leaves no room for targets")
        self.rng = np.random.default_rng(seed)
        self.switches = 0
        self.snake, self.done, self.steps = [None], False, 0
        self.theta = np.zeros(2)
        self._win = None
        self.target = None
        self.last_cos, self.last_speed = float("nan"), 0.0
        self._obs = None

    # ---- kinematics
    def elbow(self, theta=None):
        t = self.theta if theta is None else theta
        if self.kinematics == "point":                  # point mode: theta holds the hand's position
            return np.array(t, dtype=np.float64)
        if self.kinematics == "polar":                  # one telescoping link: the "elbow" is the hand
            return self.hand(t)
        return self.l1 * np.array([np.cos(t[0]), np.sin(t[0])])

    def hand(self, theta=None):
        t = self.theta if theta is None else theta
        if self.kinematics == "point":
            return np.array(t, dtype=np.float64)
        if self.kinematics == "polar":                  # theta = (angle, length)
            return t[1] * np.array([np.cos(t[0]), np.sin(t[0])])
        return self.elbow(t) + self.l2 * np.array([np.cos(t[0] + t[1]), np.sin(t[0] + t[1])])

    @property
    def cur_radius(self):
        if self.curriculum is None:
            return self.radius
        return self.curriculum["radius0"] + self.level * (self.radius - self.curriculum["radius0"])

    @property
    def cur_max_dist(self):
        full = self.point_reach if self.kinematics == "point" else 2.0 * (self.l1 + self.l2)
        if self.curriculum is None:
            return full
        return 1.2 * self.cur_radius + self.curriculum["dist0"] + self.level * (full - self.curriculum["dist0"])

    def _new_target(self):
        rad, dmax = self.cur_radius, self.cur_max_dist
        dmin = 2 * rad if self.curriculum is None else 1.2 * rad
        if self.kinematics == "point":                  # an offset from the hand: any direction, distance in the band
            dmax = max(dmax, dmin + 1e-3) if self.curriculum is not None else self.point_reach
            ang = self.rng.uniform(0, 2 * np.pi); dist = self.rng.uniform(dmin, dmax)
            self.target = self.hand() + dist * np.array([np.cos(ang), np.sin(ang)])
            self._tsteps = 0
            self._set_limit()
            return
        for tries in range(100000):
            self.target = self.hand(self._random_theta())
            d = np.linalg.norm(self.offset)
            if np.linalg.norm(self.target) >= self.min_reach and dmin < d <= dmax:
                break
        self._tsteps = 0
        self._set_limit()

    def ideal_steps(self, target=None):
        """The fewest steps that bring the hand to `target` (default: the current one): every motor at full speed
        toward the target's pose (the motors move at once, so the slowest one sets the time; for the two-link arm the
        nearer of the two elbow solutions within the elbow limit)."""
        t = self.target if target is None else np.asarray(target, dtype=np.float64)
        v = self.joint_speed
        if self.kinematics == "point":
            return float(np.max(np.abs(t - self.hand())) / v)
        d = self.pose_error(t)
        if self.kinematics == "polar":
            return float(max(abs(d[0]) / v, abs(d[1]) / (v * (self.l1 + self.l2) / 2)))
        return float(np.max(np.abs(d)) / v)

    def target_pose(self, target=None):
        """The joint angles that put the hand on `target` (default: the current one): polar (angle, length); arm the
        elbow solution within the limits nearest the current pose (the slowest joint's distance)."""
        t = self.target if target is None else np.asarray(target, dtype=np.float64)
        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
        if self.kinematics == "polar":
            return np.array([np.arctan2(t[1], t[0]), np.clip(np.linalg.norm(t), self.min_reach / 2, self.l1 + self.l2)])
        c2 = np.clip((float(t @ t) - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2), -1.0, 1.0)
        best, pose = np.inf, None
        for e in (np.arccos(c2), -np.arccos(c2)):
            if abs(e) > self.elbow_limit + 1e-9:
                continue
            s1 = wrap(np.arctan2(t[1], t[0]) - np.arctan2(self.l2 * np.sin(e), self.l1 + self.l2 * np.cos(e)))
            cost = max(abs(wrap(s1 - self.theta[0])), abs(e - self.theta[1]))
            if cost < best:
                best, pose = cost, np.array([s1, e])
        return pose

    def pose_error(self, target=None):
        """target_pose - the current pose, joint 0 wrapped to [-pi, pi]."""
        d = self.target_pose(target) - self.theta
        d[0] = (d[0] + np.pi) % (2 * np.pi) - np.pi
        return d

    def _set_limit(self):
        if self.target_time is None:
            self._tlimit = self.target_steps
            return
        c = self.target_time
        slack = c["slack"] + self.level * (c["slack_hard"] - c["slack"])
        self._tlimit = int(np.ceil(slack * self.ideal_steps() + c["min_steps"]))

    def _random_theta(self):
        if self.kinematics == "point":
            return np.zeros(2)
        if self.kinematics == "polar":                  # uniform over the reachable annulus
            lo, hi = self.min_reach, self.l1 + self.l2
            return np.array([self.rng.uniform(-np.pi, np.pi), np.sqrt(self.rng.uniform(lo * lo, hi * hi))])
        return np.array([self.rng.uniform(-np.pi, np.pi), self.rng.uniform(-self.elbow_limit, self.elbow_limit)])

    def _clip(self, theta):
        if self.kinematics == "point":                  # an open plane: nothing to clip
            return np.array(theta, dtype=np.float64)
        if self.kinematics == "polar":
            return np.array([(theta[0] + np.pi) % (2 * np.pi) - np.pi,
                             np.clip(theta[1], self.min_reach / 2, self.l1 + self.l2)])
        return np.array([(theta[0] + np.pi) % (2 * np.pi) - np.pi,           # the shoulder turns freely (wraps)
                         np.clip(theta[1], -self.elbow_limit, self.elbow_limit)])

    # ---- state
    @property
    def offset(self):
        return self.target - self.hand()

    @property
    def ctype(self):
        d = self.offset
        sector = int(np.floor((np.arctan2(d[1], d[0]) + np.pi) / (2 * np.pi) * 8)) % 8
        return sector + 8 * int(np.linalg.norm(d) >= self.near)

    @property
    def good(self):
        return bool(self.last_cos > np.cos(np.pi / 4))

    def move_info(self):
        dist = float("nan") if self.target is None else float(np.linalg.norm(self.offset))
        return {"cos": self.last_cos, "speed": self.last_speed, "dist": dist, "level": self.level}

    # ---- view
    def _cell(self, p):
        h, w, _ = self.inp_shape
        return (int(np.clip(np.floor((1.0 - p[1]) / 2.0 * h), 0, h - 1)),
                int(np.clip(np.floor((p[0] + 1.0) / 2.0 * w), 0, w - 1)))

    def _render(self):
        h, w, c = self.inp_shape
        if self.view in ("joint", "body", "hand"):
            self._obs = {"joint": self._joint_view, "body": self._body_view, "hand": self._hand_view}[self.view]()
            if self.noise > 0:
                self._obs = np.where(self.rng.random(self._obs.shape) < self.noise, 1.0 - self._obs, self._obs)
            return
        if self.view == "ego":                          # the hand at the centre, the target at its offset
            img = np.zeros((h, w, c))
            hb, bl = self.hand_blob, self.blob
            y, x = self._cell(np.zeros(2))
            img[max(0, y - hb):y + hb + 1, max(0, x - hb):x + hb + 1, 0] = 1.0
            ty, tx = self._cell(np.clip(self.offset, -0.999, 0.999))
            img[max(0, ty - bl):ty + bl + 1, max(0, tx - bl):tx + bl + 1, 1] = 1.0
            if self.noise > 0:
                img = np.where(self.rng.random(img.shape) < self.noise, 1.0 - img, img)
            self._obs = img
            return
        img = np.zeros((h, w, c))
        e, hd = self.elbow(), self.hand()
        n = 4 * max(h, w)
        wd = self.width
        for a, b in ((np.zeros(2), e), (e, hd)):
            for s in np.linspace(0.0, 1.0, n):
                y, x = self._cell(a + s * (b - a))
                img[max(0, y - wd):y + wd + 1, max(0, x - wd):x + wd + 1, 0] = 1.0
        if c > 2:
            y, x = self._cell(hd); hb = self.hand_blob
            img[max(0, y - hb):y + hb + 1, max(0, x - hb):x + hb + 1, 2] = 1.0
        ty, tx = self._cell(self.target)
        bl = self.blob
        img[max(0, ty - bl):ty + bl + 1, max(0, tx - bl):tx + bl + 1, 1] = 1.0
        if self.noise > 0:
            flips = self.rng.random(img.shape) < self.noise
            img = np.where(flips, 1.0 - img, img)
        self._obs = img

    def _joint_unit(self, pose):
        """A pose in map units: joint 0 / pi in [-1, 1) (periodic), joint 1 scaled to [-1, 1]."""
        if self.kinematics == "polar":
            lo, hi = self.min_reach / 2, self.l1 + self.l2
        else:
            lo, hi = -self.elbow_limit, self.elbow_limit
        return np.array([pose[0] / np.pi, 2 * (pose[1] - lo) / (hi - lo) - 1])

    def _bump(self, u, periodic=True):
        """A Gaussian bump at map units u (columns: u[0], periodic; rows: u[1], +1 at the top)."""
        h, w, _ = self.inp_shape
        cx, cy = (u[0] + 1) / 2 * w - 0.5, (1 - u[1]) / 2 * h - 0.5
        dx = np.arange(w) - cx
        if periodic:
            dx = (dx + w / 2) % w - w / 2
        dy = np.arange(h) - cy
        s2 = 2 * self.joint_sigma ** 2
        return np.exp(-dy[:, None] ** 2 / s2) * np.exp(-dx[None, :] ** 2 / s2)

    def _joint_view(self):
        h, w, c = self.inp_shape
        img = np.zeros((h, w, c))
        pose, tpose = self.theta, self.target_pose()
        img[..., 0] = self._bump(self._joint_unit(pose))
        if c > 1:
            img[..., 1] = self._bump(self._joint_unit(tpose))
        if c > 2:
            e = self._joint_unit(tpose) - self._joint_unit(pose)
            e[0] = (e[0] + 1) % 2 - 1                    # wrapped angle error, in [-1, 1)
            img[..., 2] = self._bump(np.array([e[0], np.clip(e[1] / 2, -1, 1)]))
        return img

    def hand_frame(self, p):
        """World point(s) p in the hand camera's frame, in units of hand_span: the hand at the origin, the shoulder
        straight below (on the -y axis)."""
        hd = self.hand()
        v = -hd                                          # hand -> shoulder
        rot = -np.pi / 2 - np.arctan2(v[1], v[0])        # turn it to point down
        c, s = np.cos(rot), np.sin(rot)
        d = np.asarray(p, dtype=np.float64) - hd
        return np.stack([c * d[..., 0] - s * d[..., 1], s * d[..., 0] + c * d[..., 1]], -1) / self.hand_span

    def _edge(self, u):
        m = np.max(np.abs(u))
        return u if m <= 0.999 else u * 0.999 / m       # beyond the frame: at the edge, in its direction

    def _hand_view(self):
        h, w, c = self.inp_shape
        img = np.zeros((h, w, c))
        img[..., 0] = self._bump(self._edge(self.hand_frame(np.zeros(2))), periodic=False)
        if c > 1:
            p = self.elbow() if self.kinematics == "arm" else self.hand()
            img[..., 1] = self._bump(self._edge(self.hand_frame(p)), periodic=False)
        if c > 2:
            img[..., 2] = self._bump(self._edge(self.hand_frame(self.target)), periodic=False)
        return img

    def _body_view(self):
        h, w, c = self.inp_shape
        img = np.zeros((h, w, c))
        img[..., 0] = self._bump(self._joint_unit(self.theta))
        if c > 1:
            img[..., 1] = self._bump(self.hand(), periodic=False)
            if self.kinematics == "arm":
                img[..., 1] = np.maximum(img[..., 1], self.elbow_amp * self._bump(self.elbow(), periodic=False))
        if c > 2:
            img[..., 2] = self._bump(self.target, periodic=False)
        return img

    def reset(self):
        if self.curriculum is not None and self.episode >= 0 and not self.continuous:   # the episode that ended
            self._hits.append(len(self.snake) > 1)
            c = self.curriculum
            if len(self._hits) >= c["window"] and np.mean(self._hits[-c["window"]:]) >= c["threshold"]:
                self.level = min(1.0, self.level + c["step"]); self._hits = []
        self.snake, self.done, self.steps = [None], False, 0
        self.wait_count = self.wait_inc
        self.episode += 1
        if self.continuous and self._started:        # a new logging segment: the arm and its target carry on
            if self.episode_per_target:              # ... or a new target episode: the pose carries on
                self._new_target()
            self.last_cos, self.last_speed = float("nan"), 0.0
            self._win = (self.hand(), self.offset)
            self._rec = ({"theta": [self.theta.copy()], "reward": [0.0], "target": [self.target.copy()],
                          "rad": [self.cur_radius]}
                         if self.record_every > 0 and self.episode % self.record_every == 0 else None)
            self._render()
            return self._obs
        self._started = True
        self.theta = self._random_theta()
        self._new_target()
        self.last_cos, self.last_speed = float("nan"), 0.0
        self._win = (self.hand(), self.offset)
        self._rec = ({"theta": [self.theta.copy()], "reward": [0.0], "target": [self.target.copy()],
                          "rad": [self.cur_radius]}
                     if self.record_every > 0 and self.episode % self.record_every == 0 else None)
        self._render()
        return self._obs

    def _save_record(self):
        import os
        th = np.array(self._rec["theta"])
        el = np.stack([self.elbow(t) for t in th]); hd = np.stack([self.hand(t) for t in th])
        os.makedirs(self.record_dir, exist_ok=True)
        np.savez_compressed(os.path.join(self.record_dir, f"ep{self.episode:06d}.npz"), theta=th, elbow=el, hand=hd,
                            target=self.target, reward=np.array(self._rec["reward"]), reached=len(self.snake) > 1,
                            level=self.level, radius_now=self.cur_radius, kinematics=self.kinematics,
                            targets=np.array(self._rec["target"]), radii=np.array(self._rec["rad"]),
                            episode=self.episode, l1=self.l1, l2=self.l2, radius=self.radius)
        self._rec = None

    def get_local_img_observation(self):
        return self._obs

    def step(self, action):
        if self.done:
            raise Exception("Environment needs reset. Call env.reset().")
        boundary = self.wait_count == 0               # the training loop's per-move bookkeeping point
        self.wait_count = self.wait_inc if boundary else self.wait_count - 1
        a = np.clip(np.asarray(action, dtype=np.float64).ravel()[:2], -1.0, 1.0)
        n0 = np.linalg.norm(self.offset)
        if self.kinematics == "polar":                # the extension joint moves in world units
            a = a * np.array([1.0, (self.l1 + self.l2) / 2])
        self.theta = self._clip(self.theta + self.joint_speed * a)
        self.steps += 1
        d1 = np.linalg.norm(self.offset)
        if boundary:                                  # the move log's measures: the hand's displacement over the window
            h0, o0 = self._win
            dh = self.hand() - h0
            nd, no = np.linalg.norm(dh), np.linalg.norm(o0)
            self.last_cos = float(dh @ o0 / (nd * no)) if nd > 1e-12 and no > 1e-12 else 0.0
            self.last_speed = float(nd)
            self._win = (self.hand(), self.offset)
        if self.reward_mode == "sign":
            reward = self.sign_reward * float(np.sign(n0 - d1))
        elif self.reward_mode == "reach":
            reward = 0.0
        else:
            w = 1.0 if self.progress_fade is None else max(0.0, 1.0 - self.level / self.progress_fade)
            reward = w * self.progress * (n0 - d1)
        self._tsteps += 1
        if d1 < self.cur_radius:
            self.snake.append(None)
            reward += self.reach_reward
            self.reaches += 1
            if self.continuous:
                if self.curriculum is not None:
                    self.level = min(1.0, self.level + self.curriculum["up"])
                if self.episode_per_target:
                    self.done = True                 # the next target is drawn by reset(), from this pose
                else:
                    self._new_target()
            else:
                self.done = True
        elif self.continuous and self._tsteps >= self._tlimit:
            self.timeouts += 1
            if self.curriculum is not None:
                self.level = max(0.0, self.level - self.curriculum["down"])
            if self.episode_per_target:
                self.done = True
            else:
                self._new_target()
        elif not self.continuous and self.steps >= self.max_steps:
            self.done = True
        if self.continuous and self.steps >= self.segment_steps:
            self.done = True
        if self._rec is not None:
            self._rec["theta"].append(self.theta.copy()); self._rec["reward"].append(float(reward))
            self._rec["target"].append(self.target.copy()); self._rec["rad"].append(self.cur_radius)
            if self.done:
                self._save_record()
        self._render()
        return self._obs, reward, self.done

    def img(self, scale=10):
        return np.kron((self._obs * 255).astype(np.uint8), np.ones((scale, scale, 1), dtype=np.uint8))


def joint_velocities(stats, gain):
    """Agonist-antagonist decoder for the arm: pools (2j, 2j+1) are joint j's flexor and extensor; each joint's velocity
    is gain * (flexor - extensor) / (flexor + extensor) of the pools' statistics (0 when both are silent), clipped to
    [-1, 1]."""
    s = np.asarray(stats, dtype=np.float64).reshape(-1, 2)
    tot = s.sum(1)
    w = np.where(tot > 0, (s[:, 0] - s[:, 1]) / np.where(tot > 0, tot, 1.0), 0.0)
    return np.clip(float(gain) * w, -1.0, 1.0)


class CoinGazeEnv:
    """Embodied coin collection in the dark, with a player light and a movable gaze light. A point body moves on a
    bounded square world ([-`world`, `world`]^2, clipped at the walls); `n_coins` coins lie at random places;
    touching one (within `coin_radius`) gives `coin_reward` and it reappears at a random place.
    The agent sees a frame centred on its body (never rotated), spanning +-(`reach` + the starting light radius)
    world units, in which only lit places show anything: the player light, a disc of radius `light` around the body,
    and the gaze light, a disc of the same radius around the gaze point. The gaze is a joystick: the gaze command
    sets the gaze point's offset from the body, offset -> `reach` x command per axis (smoothed: each step it moves
    `gaze_smooth` of the way there), so releasing it brings the gaze back onto the body; the reach never changes.
    Every step the action is 4 velocities in [-1, 1]: the body's x, y (`body_speed` per step) and the gaze's x, y.
    `render` "image" (default, a picture as in Snake): channel 0 the body (a square of half-size `blob` cells at the
    centre), channel 1 the lit coins (squares), channel 2 the lit floor (lit and inside the world: the discs show
    where the agent looks, and the floor ends at the walls); "bumps": the same as Gaussian bumps of `sigma` cells
    (channel 0 also a bump of `gaze_amp` at the gaze point).
    `curriculum`: None or {"n0": 8, "n1": 2, "light0": 0.3, "light1": 0.1, "mode": "staircase", "step": 0.05,
    "up": 10, "down": 2, "steps": 1e6}: a level in [0, 1] sets the number of coins (round(n0 + level (n1 - n0))) and
    the radius of both lights (light0 + level (light1 - light0)): sparser coins and smaller lights as it rises, the
    gaze's reach unchanged; "staircase": after each segment the level rises by `step` if it collected >= `up` coins
    and falls by `step` if <= `down`; "time": level = steps so far / `steps`.
    `done` only every `segment_steps` steps, for logging (the world carries on), len(snake) - 1 the coins collected
    in the segment. `frame`: the frame's half-width (default reach + the largest light radius; set it to compare a
    no-gaze control, reach 0, at the same scale). Move log: ctype = 8 directions to the nearest coin; good = the body moved within 45 degrees of
    it over the last window; move_info: cos, speed, dist, lit (a coin lit), gaze (offset length), light, level.
    img(): the whole world from above (floor, lit discs, coins, body) next to the agent's view, for replays."""

    n_axes = 4

    def __init__(self, wait_inc, inp_shape, world=1.0, n_coins=8, body_speed=0.02, reach=0.6, light=0.3,
                 gaze_smooth=0.3, coin_radius=0.06, coin_reward=1.0, render="image", blob=1, sigma=1.5, gaze_amp=0.5,
                 segment_steps=3000, noise=0.0, seed=None, task="coins", curriculum=None, frame=None,
                 record_every=0, record_dir=None):                     # recording: not supported (ignored)
        self.wait_inc, self.wait_count = int(wait_inc), int(wait_inc)
        self.inp_shape = tuple(inp_shape)
        self.world, self.n_coins, self.body_speed = float(world), int(n_coins), float(body_speed)
        self.reach, self.light, self.gaze_smooth = float(reach), float(light), float(gaze_smooth)
        self.coin_radius, self.coin_reward = float(coin_radius), float(coin_reward)
        self.render_mode, self.blob, self.sigma, self.gaze_amp = render, int(blob), float(sigma), float(gaze_amp)
        self.segment_steps, self.noise = int(segment_steps), float(noise)
        if render not in ("image", "bumps") or self.n_coins < 1 or self.light <= 0 or self.reach < 0 \
                or not 0 < self.gaze_smooth <= 1:
            raise ValueError("coins: render 'image' | 'bumps', n_coins >= 1, light > 0, reach >= 0, "
                             "0 < gaze_smooth <= 1")
        self.curriculum = None if curriculum is None else {"n0": 8, "n1": 2, "light0": 0.3, "light1": 0.1,
                                                           "mode": "staircase", "step": 0.05, "up": 10, "down": 2,
                                                           "steps": 1e6, **dict(curriculum)}
        c = self.curriculum
        if c is not None and (c["mode"] not in ("staircase", "time") or min(c["n0"], c["n1"]) < 1
                              or min(c["light0"], c["light1"]) <= 0):
            raise ValueError("coins curriculum: mode 'staircase' | 'time', n0, n1 >= 1, light0, light1 > 0")
        self.half = (self.reach + (self.light if c is None else max(c["light0"], c["light1"]))   # the frame
                     if frame is None else float(frame))
        self.rng = np.random.default_rng(seed)
        self.switches, self.level, self.total_steps = 0, 0.0, 0
        self.snake, self.done, self.steps, self.episode = [None], False, 0, -1
        self.body, self.gaze_off = np.zeros(2), np.zeros(2)
        self.coins = None
        self.collected = 0
        self.last_cos, self.last_speed = float("nan"), 0.0
        self._win, self._obs = None, None
        if c is not None:
            self._apply_level()

    # ---- curriculum
    def _apply_level(self):
        c = self.curriculum
        self.light = c["light0"] + self.level * (c["light1"] - c["light0"])
        n = int(round(c["n0"] + self.level * (c["n1"] - c["n0"])))
        self.n_coins = n
        if self.coins is not None and len(self.coins) != n:
            if len(self.coins) > n:
                self.coins = self.coins[self.rng.permutation(len(self.coins))[:n]]
            else:
                self.coins = np.concatenate([self.coins, np.stack([self._spawn() for _ in range(n - len(self.coins))])])

    # ---- state
    def _spawn(self):
        while True:
            c = self.rng.uniform(-self.world, self.world, 2)
            if np.linalg.norm(c - self.body) > 3 * self.coin_radius:
                return c

    @property
    def gaze(self):
        return self.body + self.gaze_off

    def lit(self, p):
        """Lit places: within `light` of the body or of the gaze point (p: (..., 2))."""
        p = np.asarray(p, dtype=np.float64)
        return (np.linalg.norm(p - self.body, axis=-1) <= self.light) | (np.linalg.norm(p - self.gaze, axis=-1) <= self.light)

    def nearest(self):
        d = self.coins - self.body
        return d[int(np.argmin(np.linalg.norm(d, axis=1)))]

    @property
    def offset(self):
        return self.nearest()

    @property
    def ctype(self):
        d = self.nearest()
        return int(np.floor((np.arctan2(d[1], d[0]) + np.pi) / (2 * np.pi) * 8)) % 8

    @property
    def good(self):
        return bool(self.last_cos > np.cos(np.pi / 4))

    def move_info(self):
        dist = float("nan") if self.coins is None else float(np.linalg.norm(self.nearest()))
        seen = 0.0 if self.coins is None else float(self.lit(self.coins).any())
        return {"cos": self.last_cos, "speed": self.last_speed, "dist": dist, "lit": seen,
                "gaze": float(np.linalg.norm(self.gaze_off)), "light": self.light, "level": self.level}

    # ---- view
    def _grid(self):
        """World coordinates of the frame's cell centres, (h, w, 2)."""
        h, w, _ = self.inp_shape
        xs = self.body[0] + ((np.arange(w) + 0.5) / w * 2 - 1) * self.half
        ys = self.body[1] + (1 - (np.arange(h) + 0.5) / h * 2) * self.half
        return np.stack(np.broadcast_arrays(xs[None, :], ys[:, None]), -1)

    def _cell(self, p):
        h, w, _ = self.inp_shape
        u = (np.asarray(p) - self.body) / self.half
        return (int(np.floor((1 - u[1]) / 2 * h)), int(np.floor((u[0] + 1) / 2 * w)))

    def _bump(self, p, amp=1.0):
        h, w, _ = self.inp_shape
        u = (np.asarray(p) - self.body) / self.half
        cx, cy = (u[0] + 1) / 2 * w - 0.5, (1 - u[1]) / 2 * h - 0.5
        return amp * np.exp(-((np.arange(h)[:, None] - cy) ** 2 + (np.arange(w)[None, :] - cx) ** 2)
                            / (2 * self.sigma ** 2))

    def _square(self, img, ch, p):
        h, w, _ = self.inp_shape
        y, x = self._cell(p)
        if 0 <= y < h and 0 <= x < w:
            b = self.blob
            img[max(0, y - b):y + b + 1, max(0, x - b):x + b + 1, ch] = 1.0

    def _render(self):
        h, w, c = self.inp_shape
        img = np.zeros((h, w, c))
        grid = self._grid()
        coins = [p for p in self.coins if self.lit(p)]
        if self.render_mode == "image":
            self._square(img, 0, self.body)
            if c > 1:
                for p in coins:
                    self._square(img, 1, p)
        else:
            img[..., 0] = np.maximum(self._bump(self.body), self._bump(self.gaze, self.gaze_amp))
            if c > 1:
                for p in coins:
                    img[..., 1] = np.maximum(img[..., 1], self._bump(p))
        if c > 2:
            inside = np.all(np.abs(grid) <= self.world, axis=-1)
            img[..., 2] = (self.lit(grid) & inside).astype(float)
        if self.noise > 0:
            img = np.where(self.rng.random(img.shape) < self.noise, 1.0 - img, img)
        self._obs = img

    def reset(self):
        if self.curriculum is not None and self.episode >= 0 and self.curriculum["mode"] == "staircase":
            c, got = self.curriculum, len(self.snake) - 1      # the segment that ended
            if got >= c["up"]:
                self.level = min(1.0, self.level + c["step"])
            elif got <= c["down"]:
                self.level = max(0.0, self.level - c["step"])
            self._apply_level()
        self.snake, self.done, self.steps = [None], False, 0
        self.wait_count = self.wait_inc
        self.episode += 1
        if self.coins is None:                                # the first segment: a new world
            self.body = np.zeros(2); self.gaze_off = np.zeros(2)
            self.coins = np.stack([self._spawn() for _ in range(self.n_coins)])
        self.last_cos, self.last_speed = float("nan"), 0.0
        self._win = (self.body.copy(), self.nearest())
        self._render()
        return self._obs

    def get_local_img_observation(self):
        return self._obs

    def step(self, action):
        if self.done:
            raise Exception("Environment needs reset. Call env.reset().")
        boundary = self.wait_count == 0
        self.wait_count = self.wait_inc if boundary else self.wait_count - 1
        a = np.clip(np.asarray(action, dtype=np.float64).ravel()[:4], -1.0, 1.0)
        self.body = np.clip(self.body + self.body_speed * a[:2], -self.world, self.world)
        self.gaze_off = self.gaze_off + self.gaze_smooth * (self.reach * a[2:] - self.gaze_off)
        self.steps += 1
        self.total_steps += 1
        if self.curriculum is not None and self.curriculum["mode"] == "time" and self.total_steps % 100 == 0:
            lv = min(1.0, self.total_steps / float(self.curriculum["steps"]))
            if lv != self.level:
                self.level = lv; self._apply_level()
        reward = 0.0
        hit = np.linalg.norm(self.coins - self.body, axis=1) < self.coin_radius
        for i in np.flatnonzero(hit):
            reward += self.coin_reward
            self.snake.append(None); self.collected += 1
            self.coins[i] = self._spawn()
        if boundary:
            b0, o0 = self._win
            db = self.body - b0
            nd, no = np.linalg.norm(db), np.linalg.norm(o0)
            self.last_cos = float(db @ o0 / (nd * no)) if nd > 1e-12 and no > 1e-12 else 0.0
            self.last_speed = float(nd)
            self._win = (self.body.copy(), self.nearest())
        if self.steps >= self.segment_steps:
            self.done = True
        self._render()
        return self._obs, reward, self.done

    def world_img(self, size=80):
        """The whole world from above, RGB uint8 (size x size): dark floor, lit discs, coins, body, gaze point."""
        xs = ((np.arange(size) + 0.5) / size * 2 - 1) * self.world
        g = np.stack(np.broadcast_arrays(xs[None, :], xs[::-1][:, None]), -1)
        img = np.full((size, size, 3), 25, np.uint8)
        img[self.lit(g)] = (90, 90, 70)
        def dot(p, col, r):
            m = np.linalg.norm(g - p, axis=-1) <= r
            img[m] = col
        for p in self.coins:
            dot(p, (240, 200, 40) if self.lit(p) else (110, 95, 40), self.coin_radius)
        dot(self.gaze, (200, 200, 255), 0.02)
        dot(self.body, (60, 140, 255), 0.04)
        return img

    def img(self, scale=10):
        view = np.kron((np.clip(self._obs, 0, 1) * 255).astype(np.uint8), np.ones((scale, scale, 1), dtype=np.uint8))
        h = view.shape[0]
        wimg = self.world_img(h)
        return np.concatenate([wimg, np.full((h, 4, 3), 255, np.uint8), view], axis=1)


class ShooterEnv:
    """A simple top-down twin-stick shooter. A point body moves on a bounded square world ([-`world`, `world`]^2,
    clipped at the walls); `n_enemies` enemies spawn at least `spawn_dist` away and walk toward the body at
    `enemy_speed` per step. Every step the action is 4 values in [-1, 1]: the body's velocity x, y (`body_speed` per
    step) and the aim x, y: when the aim is pushed past `fire_threshold` (its length) and the cooldown (`cooldown`
    steps) is over, a projectile leaves the body in the aim's direction at `shot_speed` per step and flies until it
    hits an enemy (within `hit_radius`: `kill_reward`, the enemy respawns) or leaves the world. An enemy that reaches
    the body (within `touch_radius`) costs `hit_penalty` and respawns.
    View (a picture, as in Snake), a frame centred on the body (never rotated) spanning +-`frame` world units:
    channel 0 the body (a square of half-size `blob` cells at the centre) and the projectiles (single cells), channel
    1 the enemies (squares), channel 2 the floor (cells inside the world: the walls are where it ends).
    `curriculum`: None or {"e0": 1, "e1": 4, "v0": 0.005, "v1": 0.015, "step": 0.05, "up": 15, "down": 3}: a level in
    [0, 1] sets the number of enemies (round(e0 + level (e1 - e0))) and their speed (v0 + level (v1 - v0)); after
    each segment it rises by `step` if the segment's kills - hits >= `up` and falls by `step` if <= `down`.
    `done` only every `segment_steps` steps (the world carries on), len(snake) - 1 the kills in the segment, `hits`
    the times the body was touched. Move log: ctype = 8 directions to the nearest enemy; good = the last shot went
    within 22.5 degrees of it; move_info: dist (nearest enemy), kills, hits, shots, level. Exposes the Snake
    interface the training loop uses; img(): the world from above next to the agent's view."""

    n_axes = 4

    def __init__(self, wait_inc, inp_shape, world=1.0, n_enemies=1, body_speed=0.02, enemy_speed=0.005,
                 shot_speed=0.06, cooldown=8, fire_threshold=0.5, hit_radius=0.07, touch_radius=0.06, spawn_dist=0.6,
                 kill_reward=1.0, hit_penalty=1.0, frame=1.0, blob=1, segment_steps=3000, noise=0.0, seed=None,
                 task="shooter", curriculum=None, record_every=0, record_dir=None):   # recording: ignored
        self.wait_inc, self.wait_count = int(wait_inc), int(wait_inc)
        self.inp_shape = tuple(inp_shape)
        self.world, self.n_enemies, self.body_speed = float(world), int(n_enemies), float(body_speed)
        self.enemy_speed, self.shot_speed, self.cooldown = float(enemy_speed), float(shot_speed), int(cooldown)
        self.fire_threshold, self.hit_radius, self.touch_radius = float(fire_threshold), float(hit_radius), float(touch_radius)
        self.spawn_dist, self.kill_reward, self.hit_penalty = float(spawn_dist), float(kill_reward), float(hit_penalty)
        self.half, self.blob, self.segment_steps, self.noise = float(frame), int(blob), int(segment_steps), float(noise)
        if self.n_enemies < 1 or self.spawn_dist >= 2 * self.world or self.cooldown < 1:
            raise ValueError("shooter: n_enemies >= 1, spawn_dist < 2 world, cooldown >= 1")
        self.curriculum = None if curriculum is None else {"e0": 1, "e1": 4, "v0": 0.005, "v1": 0.015, "step": 0.05,
                                                           "up": 15, "down": 3, **dict(curriculum)}
        self.rng = np.random.default_rng(seed)
        self.switches, self.level = 0, 0.0
        self.snake, self.done, self.steps, self.episode = [None], False, 0, -1
        self.body = np.zeros(2)
        self.enemies, self.shots = None, np.zeros((0, 4))       # shots: x, y, vx, vy
        self.cool, self.hits, self.n_shots, self.kills_total = 0, 0, 0, 0
        self.last_aim_cos = float("nan")
        self._obs = None
        if self.curriculum is not None:
            self._apply_level()

    def _apply_level(self):
        c = self.curriculum
        self.enemy_speed = c["v0"] + self.level * (c["v1"] - c["v0"])
        n = int(round(c["e0"] + self.level * (c["e1"] - c["e0"])))
        self.n_enemies = n
        if self.enemies is not None and len(self.enemies) != n:
            if len(self.enemies) > n:
                self.enemies = self.enemies[self.rng.permutation(len(self.enemies))[:n]]
            else:
                self.enemies = np.concatenate([self.enemies, np.stack([self._spawn() for _ in range(n - len(self.enemies))])])

    def _spawn(self):
        for _ in range(10000):
            p = self.rng.uniform(-self.world, self.world, 2)
            if np.linalg.norm(p - self.body) >= self.spawn_dist:
                return p
        return -np.sign(self.body + 1e-9) * self.world * 0.9

    def nearest(self):
        d = self.enemies - self.body
        return d[int(np.argmin(np.linalg.norm(d, axis=1)))]

    @property
    def offset(self):
        return self.nearest()

    @property
    def ctype(self):
        d = self.nearest()
        return int(np.floor((np.arctan2(d[1], d[0]) + np.pi) / (2 * np.pi) * 8)) % 8

    @property
    def good(self):
        return bool(self.last_aim_cos > np.cos(np.pi / 8))

    def move_info(self):
        dist = float("nan") if self.enemies is None else float(np.linalg.norm(self.nearest()))
        return {"dist": dist, "kills": len(self.snake) - 1, "hits": self.hits, "shots": self.n_shots,
                "aim_cos": self.last_aim_cos, "level": self.level}

    # ---- view
    def _cell(self, p):
        h, w, _ = self.inp_shape
        u = (np.asarray(p) - self.body) / self.half
        return (int(np.floor((1 - u[1]) / 2 * h)), int(np.floor((u[0] + 1) / 2 * w)))

    def _square(self, img, ch, p, b):
        h, w, _ = self.inp_shape
        y, x = self._cell(p)
        if -b <= y < h + b and -b <= x < w + b:
            img[max(0, y - b):max(0, y + b + 1), max(0, x - b):max(0, x + b + 1), ch] = 1.0

    def _render(self):
        h, w, c = self.inp_shape
        img = np.zeros((h, w, c))
        self._square(img, 0, self.body, self.blob)
        for sh in self.shots:
            self._square(img, 0, sh[:2], 0)
        if c > 1:
            for e in self.enemies:
                self._square(img, 1, e, self.blob)
        if c > 2:
            xs = self.body[0] + ((np.arange(w) + 0.5) / w * 2 - 1) * self.half
            ys = self.body[1] + (1 - (np.arange(h) + 0.5) / h * 2) * self.half
            img[..., 2] = ((np.abs(ys)[:, None] <= self.world) & (np.abs(xs)[None, :] <= self.world)).astype(float)
        if self.noise > 0:
            img = np.where(self.rng.random(img.shape) < self.noise, 1.0 - img, img)
        self._obs = img

    def reset(self):
        if self.curriculum is not None and self.episode >= 0:
            c, score = self.curriculum, (len(self.snake) - 1) - self.hits      # the segment that ended
            if score >= c["up"]:
                self.level = min(1.0, self.level + c["step"])
            elif score <= c["down"]:
                self.level = max(0.0, self.level - c["step"])
            self._apply_level()
        self.snake, self.done, self.steps, self.hits, self.n_shots = [None], False, 0, 0, 0
        self.wait_count = self.wait_inc
        self.episode += 1
        if self.enemies is None:
            self.body = np.zeros(2)
            self.enemies = np.stack([self._spawn() for _ in range(self.n_enemies)])
        self._render()
        return self._obs

    def get_local_img_observation(self):
        return self._obs

    def step(self, action):
        if self.done:
            raise Exception("Environment needs reset. Call env.reset().")
        boundary = self.wait_count == 0
        self.wait_count = self.wait_inc if boundary else self.wait_count - 1
        a = np.clip(np.asarray(action, dtype=np.float64).ravel()[:4], -1.0, 1.0)
        self.body = np.clip(self.body + self.body_speed * a[:2], -self.world, self.world)
        reward = 0.0
        self.cool = max(0, self.cool - 1)
        aim = a[2:]; na = np.linalg.norm(aim)
        if na >= self.fire_threshold and self.cool == 0:
            d = aim / na
            self.shots = np.concatenate([self.shots, np.r_[self.body, self.shot_speed * d][None]])
            self.cool, self.n_shots = self.cooldown, self.n_shots + 1
            o = self.nearest(); no = np.linalg.norm(o)
            self.last_aim_cos = float(d @ o / no) if no > 1e-12 else 0.0
        if len(self.shots):
            self.shots[:, :2] += self.shots[:, 2:]
        to = self.body - self.enemies
        dist = np.linalg.norm(to, axis=1, keepdims=True)
        self.enemies = self.enemies + self.enemy_speed * to / np.maximum(dist, 1e-9)
        keep = np.ones(len(self.shots), bool)
        for i in range(len(self.enemies)):
            if len(self.shots):
                hit = keep & (np.linalg.norm(self.shots[:, :2] - self.enemies[i], axis=1) < self.hit_radius)
                if hit.any():
                    keep[np.flatnonzero(hit)[0]] = False
                    reward += self.kill_reward; self.snake.append(None); self.kills_total += 1
                    self.enemies[i] = self._spawn()
                    continue
            if np.linalg.norm(self.enemies[i] - self.body) < self.touch_radius:
                reward -= self.hit_penalty; self.hits += 1
                self.enemies[i] = self._spawn()
        if len(self.shots):
            keep &= np.all(np.abs(self.shots[:, :2]) <= self.world, axis=1)
            self.shots = self.shots[keep]
        self.steps += 1
        if self.steps >= self.segment_steps:
            self.done = True
        self._render()
        return self._obs, reward, self.done

    def world_img(self, size=80):
        xs = ((np.arange(size) + 0.5) / size * 2 - 1) * self.world
        g = np.stack(np.broadcast_arrays(xs[None, :], xs[::-1][:, None]), -1)
        img = np.full((size, size, 3), 40, np.uint8)
        def dot(p, col, r):
            img[np.linalg.norm(g - p, axis=-1) <= r] = col
        for e in self.enemies:
            dot(e, (230, 60, 60), 0.05)
        for sh in self.shots:
            dot(sh[:2], (255, 255, 120), 0.02)
        dot(self.body, (60, 140, 255), 0.05)
        return img

    def img(self, scale=10):
        view = np.kron((np.clip(self._obs, 0, 1) * 255).astype(np.uint8), np.ones((scale, scale, 1), dtype=np.uint8))
        h = view.shape[0]
        return np.concatenate([self.world_img(h), np.full((h, 4, 3), 255, np.uint8), view], axis=1)


class ArenaEnv:
    """Coins and enemies in the dark: the coin task's lights joined with the shooter's enemies, seen as an RGB
    picture. A point body moves on a bounded square world ([-`world`, `world`]^2, clipped at the walls), with
    `n_coins` coins (touch: `coin_reward`, respawn) and `n_enemies` enemies that spawn at least `spawn_dist` away and
    walk toward the body at `enemy_speed` per step (touch: -`hit_penalty`, respawn). Lights: the player light (radius
    `light` around the body) and the gaze light (the same radius around the gaze point); the gaze is a joystick
    (offset -> `reach` x stick per axis, smoothed by `gaze_smooth`; released, it returns to the body) and also the aim:
    with the stick pushed past `fire_threshold` and the cooldown over, a projectile leaves the body toward the gaze
    (`shot_speed` per step) and kills the first enemy it touches (`kill_reward`, respawn) or leaves the world.
    Every step the action is 4 values in [-1, 1]: body x, y (`body_speed` per step) and the gaze / aim stick x, y.
    View: a frame centred on the body (never rotated), +-`frame` world units (default reach + the largest light),
    an RGB picture: lit floor grey (`floor` in each channel), unlit floor `dark`, outside the world black; enemies red
    (1, 0, 0), coins yellow (1, 1, 0), the body and its projectiles blue (0, 0, 1); unlit coins and enemies glow at
    `glow` of their colour. `render` "bumps" (default): objects are Gaussian bumps of `sigma` cells at their exact
    positions (sub-pixel: the share of brightness between neighbouring cells encodes where in the cell an object is;
    drawn over the floor with a maximum per channel); "pixels": single cells (`blob` 0; squares of half-size `blob`
    cells otherwise). Projectiles are bumps / single cells too.
    `curriculum`: None or {"n0": 8, "n1": 2, "e0": 1, "e1": 3, "v0": 0.004, "v1": 0.012, "light0": 0.45,
    "light1": 0.2, "step": 0.05, "up": 15, "down": 3}: a level in [0, 1] sets coins, enemies, their speed and both
    lights' radius (each from its first to its second value); after each segment the level rises by `step` if
    coins + kills - hits >= `up`, falls if <= `down`.
    `done` only every `segment_steps` steps (the world carries on); len(snake) - 1 the coins + kills of the segment.
    Move log: ctype = 8 directions to the nearest coin; good = the body moved within 45 degrees of it over the last
    window; move_info: coins, kills, hits, shots, lit (a coin or enemy lit), gaze, light, level.
    img(): the world from above next to the agent's view.
    `n_coin_decoys`, `n_enemy_decoys` (0): confounds that glow exactly like a coin / an enemy in the dark (enemy
    decoys also walk toward the body like enemies) but are nothing: lit, they are not drawn, and touching them does
    nothing, so a glow only says "maybe" and the light is needed to know. With a curriculum, "d0" / "d1" (default 0 /
    0) set the number of decoys of each kind from level 0 to 1 (overriding the two counts).
    `record_every`: every that many segments (0: never) the segment is saved to `record_dir`/seg<index>.npz for
    replays: per step the body, gaze point, coins, enemies (padded with nan to the most), shots (the same), the
    light radius and the reward."""

    n_axes = 4

    def __init__(self, wait_inc, inp_shape, world=1.0, n_coins=8, n_enemies=1, body_speed=0.02, enemy_speed=0.004,
                 reach=0.6, light=0.45, gaze_smooth=0.3, shot_speed=0.06, cooldown=8, fire_threshold=0.5,
                 coin_radius=0.06, hit_radius=0.07, touch_radius=0.06, spawn_dist=0.6, coin_reward=1.0,
                 kill_reward=1.0, hit_penalty=1.0, floor=0.3, dark=0.01, glow=0.05, blob=0, frame=None,
                 segment_steps=3000, noise=0.0, seed=None, task="arena", curriculum=None, render="bumps",
                 sigma=0.8, record_every=0, record_dir="outputs/arena_segments", n_coin_decoys=0, n_enemy_decoys=0):
        self.wait_inc, self.wait_count = int(wait_inc), int(wait_inc)
        self.inp_shape = tuple(inp_shape)
        self.world, self.n_coins, self.n_enemies = float(world), int(n_coins), int(n_enemies)
        self.body_speed, self.enemy_speed, self.reach, self.light = float(body_speed), float(enemy_speed), float(reach), float(light)
        self.gaze_smooth, self.shot_speed, self.cooldown = float(gaze_smooth), float(shot_speed), int(cooldown)
        self.fire_threshold, self.coin_radius, self.hit_radius = float(fire_threshold), float(coin_radius), float(hit_radius)
        self.touch_radius, self.spawn_dist = float(touch_radius), float(spawn_dist)
        self.coin_reward, self.kill_reward, self.hit_penalty = float(coin_reward), float(kill_reward), float(hit_penalty)
        self.floor, self.dark, self.glow, self.blob = float(floor), float(dark), float(glow), int(blob)
        self.segment_steps, self.noise = int(segment_steps), float(noise)
        self.render_mode, self.sigma = render, float(sigma)
        self.record_every, self.record_dir, self._rec = int(record_every), str(record_dir), None
        self.n_coin_decoys, self.n_enemy_decoys = int(n_coin_decoys), int(n_enemy_decoys)
        self.coin_decoys, self.enemy_decoys = None, None
        if render not in ("bumps", "pixels"):
            raise ValueError("arena render: 'bumps' or 'pixels'")
        if self.inp_shape[2] != 3 or self.n_coins < 1 or self.n_enemies < 0 or self.light <= 0 or self.cooldown < 1:
            raise ValueError("arena: 3 input channels (RGB), n_coins >= 1, n_enemies >= 0, light > 0, cooldown >= 1")
        self.curriculum = None if curriculum is None else {"n0": 8, "n1": 2, "e0": 1, "e1": 3, "v0": 0.004,
                                                           "v1": 0.012, "light0": 0.45, "light1": 0.2, "step": 0.05,
                                                           "up": 15, "down": 3, "d0": None, "d1": None,
                                                           **dict(curriculum)}
        c = self.curriculum
        self.half = (self.reach + (self.light if c is None else max(c["light0"], c["light1"]))) if frame is None \
            else float(frame)
        self.rng = np.random.default_rng(seed)
        self.switches, self.level = 0, 0.0
        self.snake, self.done, self.steps, self.episode = [None], False, 0, -1
        self.body, self.gaze_off = np.zeros(2), np.zeros(2)
        self.coins, self.enemies, self.shots = None, None, np.zeros((0, 4))
        self.cool, self.n_got, self.kills, self.hits, self.n_shots = 0, 0, 0, 0, 0
        self.last_cos, self.last_speed = float("nan"), 0.0
        self._win, self._obs = None, None
        if c is not None:
            self._apply_level()

    def _apply_level(self):
        c, lv = self.curriculum, self.level
        self.light = c["light0"] + lv * (c["light1"] - c["light0"])
        self.enemy_speed = c["v0"] + lv * (c["v1"] - c["v0"])
        self.n_coins = int(round(c["n0"] + lv * (c["n1"] - c["n0"])))
        self.n_enemies = int(round(c["e0"] + lv * (c["e1"] - c["e0"])))
        if c.get("d0") is not None or c.get("d1") is not None:
            d0, d1 = c.get("d0") or 0, c.get("d1") or 0
            self.n_coin_decoys = self.n_enemy_decoys = int(round(d0 + lv * (d1 - d0)))
        for name, n, spawn in (("coin_decoys", self.n_coin_decoys, self._spawn_coin),
                               ("enemy_decoys", self.n_enemy_decoys, self._spawn_enemy)):
            arr = getattr(self, name)
            if arr is None or len(arr) == n:
                continue
            arr = arr[:n] if len(arr) > n else np.concatenate([arr.reshape(-1, 2), np.stack([spawn() for _ in range(n - len(arr))])])
            setattr(self, name, arr)
        for name, n, spawn in (("coins", self.n_coins, self._spawn_coin), ("enemies", self.n_enemies, self._spawn_enemy)):
            arr = getattr(self, name)
            if arr is None or len(arr) == n:
                continue
            if len(arr) > n:
                arr = arr[self.rng.permutation(len(arr))[:n]]
            else:
                arr = np.concatenate([arr.reshape(-1, 2), np.stack([spawn() for _ in range(n - len(arr))])])
            setattr(self, name, arr)

    def _spawn_coin(self):
        while True:
            p = self.rng.uniform(-self.world, self.world, 2)
            if np.linalg.norm(p - self.body) > 3 * self.coin_radius:
                return p

    def _spawn_enemy(self):
        for _ in range(10000):
            p = self.rng.uniform(-self.world, self.world, 2)
            if np.linalg.norm(p - self.body) >= self.spawn_dist:
                return p
        return -np.sign(self.body + 1e-9) * self.world * 0.9

    @property
    def gaze(self):
        return self.body + self.gaze_off

    def lit(self, p):
        p = np.asarray(p, dtype=np.float64)
        return (np.linalg.norm(p - self.body, axis=-1) <= self.light) | (np.linalg.norm(p - self.gaze, axis=-1) <= self.light)

    def nearest(self):
        d = self.coins - self.body
        return d[int(np.argmin(np.linalg.norm(d, axis=1)))]

    @property
    def offset(self):
        return self.nearest()

    @property
    def ctype(self):
        d = self.nearest()
        return int(np.floor((np.arctan2(d[1], d[0]) + np.pi) / (2 * np.pi) * 8)) % 8

    @property
    def good(self):
        return bool(self.last_cos > np.cos(np.pi / 4))

    def move_info(self):
        seen = 0.0
        if self.coins is not None:
            seen = float(self.lit(self.coins).any() or (len(self.enemies) > 0 and self.lit(self.enemies).any()))
        return {"cos": self.last_cos, "speed": self.last_speed, "coins": self.n_got, "kills": self.kills,
                "hits": self.hits, "shots": self.n_shots, "lit": seen, "gaze": float(np.linalg.norm(self.gaze_off)),
                "light": self.light, "level": self.level}

    # ---- view
    def _grid(self):
        h, w, _ = self.inp_shape
        xs = self.body[0] + ((np.arange(w) + 0.5) / w * 2 - 1) * self.half
        ys = self.body[1] + (1 - (np.arange(h) + 0.5) / h * 2) * self.half
        return np.stack(np.broadcast_arrays(xs[None, :], ys[:, None]), -1)

    def _cell(self, p):
        h, w, _ = self.inp_shape
        u = (np.asarray(p) - self.body) / self.half
        return (int(np.floor((1 - u[1]) / 2 * h)), int(np.floor((u[0] + 1) / 2 * w)))

    def _paint(self, img, p, colour, b):
        h, w, _ = self.inp_shape
        y, x = self._cell(p)
        if -b <= y < h + b and -b <= x < w + b:
            img[max(0, y - b):max(0, y + b + 1), max(0, x - b):max(0, x + b + 1)] = colour

    def _bump(self, p):
        h, w, _ = self.inp_shape
        u = (np.asarray(p) - self.body) / self.half
        cx, cy = (u[0] + 1) / 2 * w - 0.5, (1 - u[1]) / 2 * h - 0.5
        return np.exp(-((np.arange(h)[:, None] - cy) ** 2 + (np.arange(w)[None, :] - cx) ** 2) / (2 * self.sigma ** 2))

    def _draw(self, img, p, colour, b):
        if self.render_mode == "pixels":
            self._paint(img, p, colour, b)
        else:
            np.maximum(img, self._bump(p)[..., None] * np.asarray(colour)[None, None, :], out=img)

    def _render(self):
        grid = self._grid()
        inside = np.all(np.abs(grid) <= self.world, axis=-1)
        lit = self.lit(grid)
        img = np.where((inside & lit)[..., None], self.floor, np.where(inside[..., None], self.dark, 0.0))
        img = np.broadcast_to(img, self.inp_shape).copy()
        for arr, col in ((self.coins, np.array([1.0, 1.0, 0.0])), (self.enemies, np.array([1.0, 0.0, 0.0]))):
            for p in arr:
                self._draw(img, p, col if self.lit(p) else self.glow * col, self.blob)
        for arr, col in ((self.coin_decoys, np.array([1.0, 1.0, 0.0])), (self.enemy_decoys, np.array([1.0, 0.0, 0.0]))):
            for p in arr:
                if not self.lit(p):                        # a decoy glows like the real thing, and is nothing in the light
                    self._draw(img, p, self.glow * col, self.blob)
        for sh in self.shots:
            self._draw(img, sh[:2], np.array([0.0, 0.0, 1.0]), 0)
        self._draw(img, self.body, np.array([0.0, 0.0, 1.0]), self.blob)
        if self.noise > 0:
            img = np.where(self.rng.random(img.shape) < self.noise, 1.0 - img, img)
        self._obs = img

    def reset(self):
        if self.curriculum is not None and self.episode >= 0:
            c, score = self.curriculum, (len(self.snake) - 1) - self.hits      # the segment that ended
            if score >= c["up"]:
                self.level = min(1.0, self.level + c["step"])
            elif score <= c["down"]:
                self.level = max(0.0, self.level - c["step"])
            self._apply_level()
        self.snake, self.done, self.steps = [None], False, 0
        self.n_got, self.kills, self.hits, self.n_shots = 0, 0, 0, 0
        self.wait_count = self.wait_inc
        self.episode += 1
        if self.coins is None:
            self.body = np.zeros(2); self.gaze_off = np.zeros(2)
            self.coins = np.stack([self._spawn_coin() for _ in range(self.n_coins)])
            self.enemies = np.stack([self._spawn_enemy() for _ in range(self.n_enemies)]) if self.n_enemies \
                else np.zeros((0, 2))
            self.coin_decoys = np.stack([self._spawn_coin() for _ in range(self.n_coin_decoys)]) \
                if self.n_coin_decoys else np.zeros((0, 2))
            self.enemy_decoys = np.stack([self._spawn_enemy() for _ in range(self.n_enemy_decoys)]) \
                if self.n_enemy_decoys else np.zeros((0, 2))
        self.last_cos, self.last_speed = float("nan"), 0.0
        self._win = (self.body.copy(), self.nearest())
        self._rec = ({k: [] for k in ("body", "gaze", "coins", "enemies", "shots", "light", "reward")}
                     if self.record_every > 0 and self.episode % self.record_every == 0 else None)
        self._render()
        return self._obs

    def _record(self, reward):
        r = self._rec
        r["body"].append(self.body.copy()); r["gaze"].append(self.gaze.copy())
        r["coins"].append(self.coins.copy()); r["enemies"].append(self.enemies.copy())
        r["shots"].append(self.shots[:, :2].copy()); r["light"].append(self.light); r["reward"].append(reward)
        if self.done:
            import os
            def pad(seq):
                n = max([len(x) for x in seq] + [1])
                out = np.full((len(seq), n, 2), np.nan)
                for t, x in enumerate(seq):
                    out[t, :len(x)] = x
                return out
            os.makedirs(self.record_dir, exist_ok=True)
            np.savez_compressed(os.path.join(self.record_dir, f"seg{self.episode:06d}.npz"), body=np.array(r["body"]),
                                gaze=np.array(r["gaze"]), coins=pad(r["coins"]), enemies=pad(r["enemies"]),
                                shots=pad(r["shots"]), light=np.array(r["light"]), reward=np.array(r["reward"]),
                                level=self.level, episode=self.episode, world=self.world)
            self._rec = None

    def get_local_img_observation(self):
        return self._obs

    def step(self, action):
        if self.done:
            raise Exception("Environment needs reset. Call env.reset().")
        boundary = self.wait_count == 0
        self.wait_count = self.wait_inc if boundary else self.wait_count - 1
        a = np.clip(np.asarray(action, dtype=np.float64).ravel()[:4], -1.0, 1.0)
        self.body = np.clip(self.body + self.body_speed * a[:2], -self.world, self.world)
        self.gaze_off = self.gaze_off + self.gaze_smooth * (self.reach * a[2:] - self.gaze_off)
        reward = 0.0
        self.cool = max(0, self.cool - 1)
        stick = a[2:]; ns = np.linalg.norm(stick)
        if ns >= self.fire_threshold and self.cool == 0:
            self.shots = np.concatenate([self.shots, np.r_[self.body, self.shot_speed * stick / ns][None]])
            self.cool, self.n_shots = self.cooldown, self.n_shots + 1
        if len(self.shots):
            self.shots[:, :2] += self.shots[:, 2:]
        if len(self.enemies):
            to = self.body - self.enemies
            self.enemies = self.enemies + self.enemy_speed * to / np.maximum(np.linalg.norm(to, axis=1, keepdims=True), 1e-9)
        if self.enemy_decoys is not None and len(self.enemy_decoys):
            to = self.body - self.enemy_decoys
            self.enemy_decoys = self.enemy_decoys + self.enemy_speed * to / np.maximum(
                np.linalg.norm(to, axis=1, keepdims=True), 1e-9)
            for k in np.flatnonzero(np.linalg.norm(self.enemy_decoys - self.body, axis=1) < self.touch_radius):
                self.enemy_decoys[k] = self._spawn_enemy()       # harmless; it reappears elsewhere
        if self.coin_decoys is not None and len(self.coin_decoys):
            for k in np.flatnonzero(np.linalg.norm(self.coin_decoys - self.body, axis=1) < self.coin_radius):
                self.coin_decoys[k] = self._spawn_coin()
        keep = np.ones(len(self.shots), bool)
        for i in range(len(self.enemies)):
            if len(self.shots):
                hit = keep & (np.linalg.norm(self.shots[:, :2] - self.enemies[i], axis=1) < self.hit_radius)
                if hit.any():
                    keep[np.flatnonzero(hit)[0]] = False
                    reward += self.kill_reward; self.kills += 1; self.snake.append(None)
                    self.enemies[i] = self._spawn_enemy()
                    continue
            if np.linalg.norm(self.enemies[i] - self.body) < self.touch_radius:
                reward -= self.hit_penalty; self.hits += 1
                self.enemies[i] = self._spawn_enemy()
        if len(self.shots):
            keep &= np.all(np.abs(self.shots[:, :2]) <= self.world, axis=1)
            self.shots = self.shots[keep]
        for i in np.flatnonzero(np.linalg.norm(self.coins - self.body, axis=1) < self.coin_radius):
            reward += self.coin_reward; self.n_got += 1; self.snake.append(None)
            self.coins[i] = self._spawn_coin()
        if boundary:
            b0, o0 = self._win
            db = self.body - b0
            nd, no = np.linalg.norm(db), np.linalg.norm(o0)
            self.last_cos = float(db @ o0 / (nd * no)) if nd > 1e-12 and no > 1e-12 else 0.0
            self.last_speed = float(nd)
            self._win = (self.body.copy(), self.nearest())
        self.steps += 1
        if self.steps >= self.segment_steps:
            self.done = True
        if self._rec is not None:
            self._record(reward)
        self._render()
        return self._obs, reward, self.done

    def world_img(self, size=80):
        xs = ((np.arange(size) + 0.5) / size * 2 - 1) * self.world
        g = np.stack(np.broadcast_arrays(xs[None, :], xs[::-1][:, None]), -1)
        img = np.full((size, size, 3), 25, np.uint8)
        img[self.lit(g)] = (90, 90, 90)
        def dot(p, col, r):
            img[np.linalg.norm(g - p, axis=-1) <= r] = col
        for p in self.coins:
            dot(p, (240, 220, 40) if self.lit(p) else (100, 90, 30), self.coin_radius)
        for p in self.enemies:
            dot(p, (230, 50, 50) if self.lit(p) else (100, 30, 30), 0.05)
        for p in (self.coin_decoys if self.coin_decoys is not None else []):
            dot(p, (70, 70, 40), self.coin_radius)              # decoys: dull, they glow only to the agent
        for p in (self.enemy_decoys if self.enemy_decoys is not None else []):
            dot(p, (70, 40, 40), 0.05)
        for sh in self.shots:
            dot(sh[:2], (120, 160, 255), 0.02)
        dot(self.gaze, (220, 220, 255), 0.02)
        dot(self.body, (60, 120, 255), 0.05)
        return img

    def img(self, scale=10):
        view = np.kron((np.clip(self._obs, 0, 1) * 255).astype(np.uint8), np.ones((scale, scale, 1), dtype=np.uint8))
        h = view.shape[0]
        return np.concatenate([self.world_img(h), np.full((h, 4, 3), 255, np.uint8), view], axis=1)


def beta_policy(readout, rng, scale=1.0, max_conc=50.0):
    """A Beta policy over bounded actions from antagonist readout pairs: for axis j, alpha = 1 + softplus(scale r_2j)
    and beta = 1 + softplus(scale r_2j+1) (both capped at max_conc, so the policy keeps some exploration), x ~
    Beta(alpha, beta), action = 2 x - 1 in [-1, 1] (no clipping). The mean is alpha / (alpha + beta), the probabilistic
    form of the flexor / extensor ratio decoder; alpha + beta (the pair's total drive) sets the concentration.
    Returns (action, PG, mean action); PG = -d log pi / d readout per unit (the convention of softmax - onehot):
    d log p / d alpha = ln x - psi(alpha) + psi(alpha + beta), d log p / d beta = ln(1 - x) - psi(beta) +
    psi(alpha + beta), times d alpha / d r = scale sigmoid(scale r) (0 where the cap binds)."""
    from scipy.special import digamma
    r = np.asarray(readout, dtype=np.float64).reshape(-1, 2) * float(scale)
    sp = np.logaddexp(0.0, r)                                  # softplus
    capped = 1.0 + sp > max_conc
    ab = np.minimum(1.0 + sp, max_conc)
    a, b = ab[:, 0], ab[:, 1]
    x = np.clip(rng.beta(a, b), 1e-6, 1 - 1e-6)
    common = digamma(a + b)
    dl = np.stack([np.log(x) - digamma(a) + common, np.log1p(-x) - digamma(b) + common], 1)
    dab = float(scale) / (1.0 + np.exp(-r)) * (~capped)
    pg = -(dl * dab).ravel()
    return 2 * x - 1, pg, 2 * a / (a + b) - 1


def joint_beta(traces, scale, rng, max_conc=50.0):
    """The Beta surrogate for the motor pools ("joints" decoder): for joint j, alpha = 1 + scale x the flexor pool's
    trace and beta = 1 + scale x the extensor pool's, x ~ Beta(alpha, beta), action = 2 x - 1 (bounded, no clipping;
    alpha, beta capped at max_conc). The mean alpha / (alpha + beta) is the ratio decoder's command in probabilistic
    form, and alpha + beta (co-activation of the antagonists) sets the precision. The score is exact with respect to
    each pool's trace: PG = -scale (ln x - psi(alpha) + psi(alpha + beta)) for the flexor, the same with ln(1 - x)
    and psi(beta) for the extensor (0 where the cap binds). Returns (action, PG, mean action)."""
    from scipy.special import digamma
    t = np.maximum(np.asarray(traces, dtype=np.float64).reshape(-1, 2), 0.0)
    raw = 1.0 + float(scale) * t
    capped = raw > max_conc
    ab = np.minimum(raw, max_conc)
    a, b = ab[:, 0], ab[:, 1]
    x = np.clip(rng.beta(a, b), 1e-6, 1 - 1e-6)
    common = digamma(a + b)
    dl = np.stack([np.log(x) - digamma(a) + common, np.log1p(-x) - digamma(b) + common], 1)
    pg = -(float(scale) * dl * (~capped)).ravel()
    return 2 * x - 1, pg, 2 * a / (a + b) - 1


class SurrogateGain:
    """A learned correction of the joint surrogate's credit: per joint j, an online estimate of the local Jacobian
    [d omega_j / d flexor_j, d omega_j / d extensor_j] of the decoded command with respect to the pools' traces,
    by normalised LMS on step-to-step changes, J += rate (d omega - J . d s) d s / (|d s|^2 + eps), d s the change of
    the two traces and d omega the change of the command the pools see (the efference copy of the decoded command, or
    proprioception: the executed movement, then with the exploration noise in it). It starts at the fixed
    surrogate's [1, -1] (scale `init`), so the early credit is unchanged, and learns from the body only what the
    surrogate assumes: the decoder's normalisation, asymmetry between the antagonists, saturation (a clipped joint's
    gain decays toward 0). `gains()` is J divided by the running mean of |J| (decay `norm_decay`), so the credit's
    overall scale (and e-prop's effective learning rate) stays that of the fixed surrogate and only the shape is
    learned. Two parameters per joint: the dimensionality is that of the action."""

    def __init__(self, joints, rate=0.05, eps=1e-6, init=1.0, norm_decay=0.999):
        self.J = np.tile(np.array([init, -init], dtype=np.float64), (int(joints), 1))
        self.rate, self.eps, self.nd = float(rate), float(eps), float(norm_decay)
        self.scale = float(init)
        self.prev = None

    def update(self, traces, command):
        s = np.asarray(traces, dtype=np.float64).reshape(-1, 2)
        w = np.asarray(command, dtype=np.float64).ravel()
        if self.prev is not None:
            ds, dw = s - self.prev[0], w - self.prev[1]
            err = dw - (self.J * ds).sum(1)
            self.J += self.rate * err[:, None] * ds / ((ds * ds).sum(1, keepdims=True) + self.eps)
            self.scale = self.nd * self.scale + (1 - self.nd) * float(np.abs(self.J).mean())
        self.prev = (s.copy(), w.copy())
        return self.gains()

    def gains(self):
        return self.J / max(self.scale, 1e-12)


def joint_surrogate(omega, sigma, rng, rho=0.0, state=None, gains=None):
    """Gaussian surrogate policy over the joint velocities: the action is a = omega + sigma * n (executed clipped to
    [-1, 1]); the policy error per pool is the score sent back through the pool membership, to joint j's flexor pool
    and its negative to the extensor pool (PG = -d log pi / d logit, as softmax - onehot is for the discrete
    surrogate; the derivative of the normalisation by the pools' total is left out).
    rho = 0: n is white, n = xi, and the error is (omega - a) / sigma = -xi.
    rho > 0: n is an Ornstein-Uhlenbeck process with per-step correlation rho and unit variance,
    n_t = rho n_{t-1} + sqrt(1 - rho^2) xi_t, so exploration is a committed movement held over ~1 / (1 - rho) steps.
    Given n_{t-1}, a_t is Gaussian with mean omega + sigma rho n_{t-1} and s.d. sigma sqrt(1 - rho^2), so the exact
    score is the innovation's: the error is -xi_t (unit scale, as for white noise).
    gains: None (the fixed surrogate: +1 to the flexor, -1 to the extensor) or (joints, 2) per-pool gains (a learned
    Jacobian, SurrogateGain.gains()): pool k of joint j gets -xi_j * gains[j, k].
    Returns (action, PG, new state); state is n (None: zeros)."""
    omega = np.asarray(omega, dtype=np.float64)
    xi = rng.standard_normal(omega.shape)
    rho = float(rho)
    if rho > 0.0:
        prev = np.zeros(omega.shape) if state is None else np.asarray(state, dtype=np.float64)
        n = rho * prev + np.sqrt(1.0 - rho * rho) * xi
    else:
        n = xi
    raw = omega + float(sigma) * n
    g = -xi                                   # the score of the sample drawn (the environment clips what it executes)
    if gains is not None:
        return np.clip(raw, -1.0, 1.0), (g[:, None] * np.asarray(gains, dtype=np.float64)).ravel(), n
    return np.clip(raw, -1.0, 1.0), np.stack([g, -g], axis=1).ravel(), n
