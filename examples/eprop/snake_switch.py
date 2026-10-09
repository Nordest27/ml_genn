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
                 silent_hz=0.5, dt_ms=1.0, weights=True):
        """populations: hidden ALIF ml_genn Populations; labels: their names in the log.
        weights: also log, per connection group into a hidden population, how far its weights have moved from
        their initial values (||W - W0|| / ||W0||), to see which layers learn when."""
        self.net, self.pops, self.labels = compiled_net, list(populations), list(labels)
        self.report_every, self.psi_eps, self.silent_hz, self.dt_ms = int(report_every), psi_eps, silent_hz, dt_ms
        self.log_path = log_path
        self.psi_sum = [np.zeros(int(np.prod(p.shape))) for p in self.pops]
        self.spikes = [np.zeros(int(np.prod(p.shape))) for p in self.pops]
        self.vloss = [0.0 for _ in self.pops]
        self.prev_spike_count = [None] * len(self.pops)
        self.samples, self.moves = 0, 0
        self.prev_dead = [None] * len(self.pops)
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        name_of = {p: l for p, l in zip(self.pops, self.labels)}
        self.weight_groups = []
        if weights:
            for c, sg in compiled_net.connection_populations.items():
                if c.is_feedback or c.target() not in name_of:
                    continue
                sg.vars["g"].pull_from_device()
                w0 = np.asarray(sg.vars["g"].values, dtype=np.float64).copy()
                src = name_of.get(c.source(), c.source().name)
                self.weight_groups.append((f"{src}->{name_of[c.target()]}", sg, w0, np.linalg.norm(w0) + 1e-12))
        cols = ["timestep", "moves", "switch"]
        for l in self.labels:
            cols += [f"{l}_dead", f"{l}_dead_silent", f"{l}_dead_firing", f"{l}_mean_psi", f"{l}_hz",
                     f"{l}_recovered", f"{l}_vloss"]
        cols += [f"dW|{name}" for name, *_ in self.weight_groups]
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(cols)

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
            row += [dead.mean(), silent.mean(), (dead & ~silent).mean(), psi.mean(), hz.mean(), rec,
                    self.vloss[i] / max(self.samples, 1)]
            self.vloss[i] = 0.0
            self.prev_dead[i] = dead
            self.psi_sum[i][:] = 0
            self.spikes[i][:] = 0
        for name, sg, w0, n0 in self.weight_groups:
            sg.vars["g"].pull_from_device()
            row.append(float(np.linalg.norm(np.asarray(sg.vars["g"].values, dtype=np.float64) - w0) / n0))
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
    last `moves` samples of the run, for linear-probe analysis of what the hidden layer represents."""
    DIRS = {"left": (0, -1), "up": (-1, 0), "right": (0, 1), "down": (1, 0)}

    def __init__(self, compiled_net, populations, labels, moves=5000):
        from collections import deque
        self.net, self.pops, self.labels, self.moves = compiled_net, list(populations), list(labels), int(moves)
        self.first, self.last = [], deque(maxlen=self.moves)

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

    def sample(self, env):
        feats = []
        for p in self.pops:
            npop = self.net.neuron_populations[p]
            for v in ("V", "RefracTime"):
                npop.vars[v].pull_from_device()
            feats.append(np.asarray(npop.vars["V"].view, dtype=np.float32).ravel().copy())
            feats.append((np.asarray(npop.vars["RefracTime"].view).ravel() > 0).astype(np.float32))
        row = (np.concatenate(feats), self.task_variables(env))
        if len(self.first) < self.moves:
            self.first.append(row)
        self.last.append(row)

    def dump(self, path):
        def stack(rows):
            if not rows:
                return np.zeros((0, 0), np.float32), np.zeros((0, 7), np.float32)
            return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])
        fx, fy = stack(self.first)
        lx, ly = stack(list(self.last))
        sizes = [int(np.prod(p.shape)) for p in self.pops]
        np.savez(path, first_x=fx, first_y=fy, last_x=lx, last_y=ly, sizes=np.array(sizes),
                 labels=np.array(self.labels), variables=np.array(["apple_dy", "apple_dx", "danger_left", "danger_up",
                                                                   "danger_right", "danger_down", "heading"]))
