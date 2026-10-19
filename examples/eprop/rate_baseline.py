"""Non-spiking control for Snake: PPO with a policy and value function read directly from the input frame.

    python rate_baseline.py OUT_DIR [--hidden 0] [--moves 1000000] [--gamma 0.5] [--seed 0]

--hidden 0 (default): LINEAR policy and value (logits = W x + b, value = v.x + c on the 20x20x3 frame). This is the
locality control: if a linear readout of the input already plays Snake well, the task's spatial integration can be
done by the output, and a hidden network's job is to build a better representation (memory, non-linear features),
not to carry information across the visual field. --hidden N (or N,M,... for more layers) adds ReLU layers shared by policy
and value, as a non-linear reference. One frame per decision, no memory (the spiking runs see the same frames).

Environment: the SnakeEnv of snake.py with the training defaults (board 5, view 5, scale 4: 20x20x3 frames, the
camera on the head), one call per move (wait_inc 0). Reward +1 food, -1 death or 25 moves without food; gamma per
move as in snake_hparams (gamma_env 0.5). Writes OUT_DIR/episodes.csv (episode, moves, score = total reward,
apples) and prints running means; numpy only (exact gradients, Adam). --continuing: the task the spiking agents learn
(deaths reset the board without ending the value: the objective is the reward rate, not the episode score).
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reservoir_search import snake_env_class  # noqa: E402


class Model:
    """Actor-critic on a flat input with a shared ReLU trunk of any depth (hidden: list of widths; [] = LINEAR);
    exact gradients."""

    def __init__(self, n_in, n_act, hidden, rng):
        if isinstance(hidden, int):
            hidden = [hidden] if hidden else []
        self.hidden = list(hidden)
        p = {}
        width = n_in
        for i, h in enumerate(self.hidden, 1):
            p[f"W{i}"] = rng.normal(0, np.sqrt(2.0 / width), (width, h)); p[f"b{i}"] = np.zeros(h)
            width = h
        p["Wp"] = rng.normal(0, 0.01, (width, n_act)); p["bp"] = np.zeros(n_act)
        p["Wv"] = np.zeros((width, 1)); p["bv"] = np.zeros(1)
        self.p = p
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.t = 0

    def forward(self, x):
        p = self.p
        acts, pres = [x], []
        h = x
        for i in range(1, len(self.hidden) + 1):
            pre = h @ p[f"W{i}"] + p[f"b{i}"]
            h = np.maximum(pre, 0.0)
            pres.append(pre); acts.append(h)
        logits = h @ p["Wp"] + p["bp"]
        value = (h @ p["Wv"] + p["bv"])[:, 0]
        return logits, value, (acts, pres)

    @staticmethod
    def softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def grads(self, cache, dlogits, dvalue):
        acts, pres = cache
        h = acts[-1]
        g = {"Wp": h.T @ dlogits, "bp": dlogits.sum(0), "Wv": h.T @ dvalue[:, None], "bv": np.array([dvalue.sum()])}
        dh = dlogits @ self.p["Wp"].T + dvalue[:, None] @ self.p["Wv"].T
        for i in range(len(self.hidden), 0, -1):
            dpre = dh * (pres[i - 1] > 0)
            g[f"W{i}"] = acts[i - 1].T @ dpre; g[f"b{i}"] = dpre.sum(0)
            dh = dpre @ self.p[f"W{i}"].T
        return g

    def adam(self, g, lr, b1=0.9, b2=0.999, eps=1e-8, clip=0.5, value_lr_scale=1.0):
        norm = np.sqrt(sum(float((v ** 2).sum()) for v in g.values()))
        scale = min(1.0, clip / (norm + 1e-12))                    # global gradient-norm clipping
        self.t += 1
        for k in self.p:
            gk = g[k] * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            step = lr * (value_lr_scale if k in ("Wv", "bv") else 1.0)     # the value head can learn slower
            self.p[k] -= step * mh / (np.sqrt(vh) + eps)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--hidden", default="0", help="hidden ReLU layer widths, comma separated (0: linear; 128,128: "
                                                   "two layers)")
    ap.add_argument("--moves", type=float, default=1e6)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vcoef", type=float, default=0.5)
    ap.add_argument("--trace", type=float, default=None,
                    help="actor trace decay per move, decoupled from gamma as in the spiking rules (default: "
                         "gamma * lam, standard GAE); the critic keeps its standard targets")
    ap.add_argument("--reward-relay", type=float, default=0.0,
                    help="add this fraction of the raw reward to the actor's TD error (the actor signal is then "
                         "delta + beta * r, accumulated with the actor trace); the critic is unaffected")
    ap.add_argument("--relay-baseline", type=float, default=0.0, metavar="TAU",
                    help="relay r - rho instead of the raw reward, rho = running average reward per move (time constant "
                         "TAU moves): a continuous baseline, so the rewards of the next game do not flow back into the "
                         "moves that ended the previous one (average-reward / differential reward)")
    ap.add_argument("--trace-surprise", type=float, default=0.0, metavar="C",
                    help="surprise-gated trace (continuous, sign-blind): the actor trace decays per move at "
                         "trace / (1 + (|delta| / (C * m))^2), m = mean |delta| of the rollout; a large TD error of "
                         "either sign (a death, an unexpected apple) closes the trace, so credit from after the event "
                         "stops flowing back past it. 0: off")
    ap.add_argument("--trace-surprise-asym", action="store_true",
                    help="with --trace-surprise: only NEGATIVE surprises shorten the trace; a large positive TD error "
                         "lengthens it towards 1 (decay = trace + (1 - trace) * (1 - 1 / (1 + (delta / (C * m))^2)))")
    ap.add_argument("--trace-value-gate", type=float, default=0.0, metavar="C",
                    help="state-dependent trace (consistent with TD(lambda) theory: lambda depends on the states, not "
                         "the reward): the actor trace closes when the critic's predicted future collapses, i.e. when "
                         "gamma V(s') - V(s), relative to its usual level, drops far below normal; decay = trace / "
                         "(1 + (drop / (C * m))^2), m = mean |deviation|. Rises leave the trace unchanged. 0: off")
    ap.add_argument("--cut-trace-at-death", action="store_true",
                    help="diagnostic: reset the actor trace when a game ends (uses episode boundaries; the critic stays "
                         "continuing)")
    ap.add_argument("--value-lr-scale", type=float, default=1.0,
                    help="learning-rate factor of the value head (Adam normalises each parameter's step, so --vcoef "
                         "alone does not slow the critic); < 1: a slow critic, whose unpredicted rewards the actor's "
                         "trace then carries back further")
    ap.add_argument("--board", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--continuing", action="store_true",
                    help="continuing task, as the spiking agents learn it: a death still costs -1 and resets the "
                         "board, but the value is bootstrapped across the reset (no terminal cut), so the objective "
                         "is the discounted reward stream (reward rate), not the score of one episode")
    ap.add_argument("--stage-reward", default=None, metavar="N1:R1[,N2:R2,...][,RLAST]",
                    help="reward stages within a game, by the apples already eaten: fewer than N1 -> R1, fewer than N2 "
                         "-> R2, ..., otherwise RLAST (default 1); e.g. 10:0.5 (first 10 apples worth 0.5, the rest 1) "
                         "or 10:0.5,19:1,2 (first 10 worth 0.5, the next 9 worth 1, every later one 2). Makes long "
                         "games worth more than restarting, on the continuing task")
    ap.add_argument("--record", type=int, default=0, help="save the last N decisions (frame, body, apple, heading, "
                                                          "action) to OUT_DIR/states.npz, for decoding analyses")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); rng = np.random.default_rng(args.seed)

    SnakeEnv = snake_env_class()
    env = SnakeEnv(size=args.board, visible_range=5, wait_inc=0, scale=4, inp_shape=(20, 20, 3), bounded_camera=False)
    obs = env.reset().astype(np.float32).ravel()
    model = Model(obs.size, 4, [int(w) for w in str(args.hidden).split(",") if int(w) > 0], rng)
    log = open(out / "episodes.csv", "w", newline="")
    writer = csv.writer(log); writer.writerow(["episode", "moves", "score", "apples"])
    ep, ep_ret, ep_apples, moves, t0 = 0, 0.0, 0, 0, time.time()
    recent = []
    from collections import deque
    rec = deque(maxlen=args.record) if args.record else None
    stage = None
    if args.stage_reward:
        steps, last = [], 1.0                            # [(apples eaten below which, reward), ...], reward after
        for part in args.stage_reward.split(","):
            if ":" in part:
                n, v = part.split(":"); steps.append((int(n), float(v)))
            else:
                last = float(part)
        stage = (steps, last)
    rho = 0.0                                            # running average reward per move (--relay-baseline)
    while moves < args.moves:
        X, A, LP, R, D, V = [], [], [], [], [], []
        RHO = []
        for _ in range(args.rollout):
            logits, value, _ = model.forward(obs[None])
            pi = model.softmax(logits)[0]
            a = int(rng.choice(4, p=pi))
            if rec is not None:
                body = np.full((args.board * args.board, 2), -1, np.int8); body[:len(env.snake)] = env.snake
                rec.append(((obs * 255).astype(np.uint8), body, np.asarray(env.apples[0] if env.apples else (-1, -1),
                            np.int8), ["left", "up", "right", "down"].index(env.direction), a))
            nxt, r, done = env.step(a)
            if stage is not None and r > 0:                    # an apple: its value depends on the game's stage
                r = next((v for n, v in stage[0] if ep_apples < n), stage[1])
            X.append(obs); A.append(a); LP.append(np.log(pi[a] + 1e-12)); R.append(r); D.append(done); V.append(value[0])
            RHO.append(rho)
            if args.relay_baseline > 0:
                rho += (r - rho) / args.relay_baseline
            moves += 1; ep_ret += r; ep_apples += int(r > 0)
            if done:
                ep += 1
                writer.writerow([ep, moves, round(ep_ret, 3), ep_apples])
                recent.append((ep_ret, ep_apples)); recent = recent[-200:]
                ep_ret, ep_apples = 0.0, 0
                nxt = env.reset()
            obs = nxt.astype(np.float32).ravel()
        _, last_v, _ = model.forward(obs[None])
        # generalised advantage estimation
        X = np.array(X); A = np.array(A); LP = np.array(LP); R = np.array(R, dtype=np.float64)
        D = np.array(D, dtype=np.float64); V = np.array(V + [last_v[0]])
        D_game = D.copy()                                    # game ends (for the --cut-trace-at-death diagnostic)
        RHO = np.array(RHO)
        if args.continuing:
            D = np.zeros_like(D)                         # the next state after a death is the fresh board's
        adv = np.zeros(len(R)); gae = 0.0
        for t in reversed(range(len(R))):
            delta = R[t] + args.gamma * V[t + 1] * (1 - D[t]) - V[t]
            gae = delta + args.gamma * args.lam * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V[:-1]                                   # the critic's targets (always the standard estimate)
        if args.trace is not None or args.reward_relay:
            # decoupled actor signal, as in the spiking rules: the trace decays at --trace per move (not gamma*lambda)
            # and integrates the TD error plus a fraction of the raw reward (which reaches back through the trace
            # whatever the critic predicts)
            decay = args.trace if args.trace is not None else args.gamma * args.lam
            td = R + args.gamma * V[1:] * (1 - D) - V[:-1]          # the critic's TD errors (no relay)
            gate = np.ones(len(R))
            if args.trace_surprise > 0:
                m = np.mean(np.abs(td)) + 1e-8
                gate = 1.0 / (1.0 + (np.abs(td) / (args.trace_surprise * m)) ** 2)
                if args.trace_surprise_asym:                    # positive surprises lengthen the trace instead
                    longer = (decay + (1.0 - decay) * (1.0 - gate)) / decay
                    gate = np.where(td > 0, longer, gate)
            if args.trace_value_gate > 0:
                dv = args.gamma * V[1:] * (1 - D) - V[:-1]          # change in predicted value (no reward)
                dev = dv - dv.mean()
                drop = np.maximum(0.0, -dev)
                gate = gate / (1.0 + (drop / (args.trace_value_gate * (np.mean(np.abs(dev)) + 1e-8))) ** 2)
            adv = np.zeros(len(R)); acc = 0.0
            for t in reversed(range(len(R))):
                relayed = R[t] - (RHO[t] if args.relay_baseline > 0 else 0.0)
                delta = R[t] + args.gamma * V[t + 1] * (1 - D[t]) - V[t] + args.reward_relay * relayed
                cut = D_game[t] if args.cut_trace_at_death else D[t]
                acc = delta + decay * gate[t] * (1 - cut) * acc
                adv[t] = acc
        for _ in range(args.epochs):
            order = rng.permutation(len(R))
            for s in range(0, len(R), args.batch):
                i = order[s:s + args.batch]
                logits, value, cache = model.forward(X[i])
                pi = model.softmax(logits)
                logp = np.log(pi[np.arange(len(i)), A[i]] + 1e-12)
                ratio = np.exp(logp - LP[i])
                a_n = (adv[i] - adv[i].mean()) / (adv[i].std() + 1e-8)
                active = ~(((a_n > 0) & (ratio > 1 + args.clip)) | ((a_n < 0) & (ratio < 1 - args.clip)))
                onehot = np.eye(4)[A[i]]
                # d/dlogits of -ratio * a_n (unclipped samples) and of -ent * H
                dlog = -(active * ratio * a_n)[:, None] * (onehot - pi)
                logpi = np.log(pi + 1e-12)
                H = -(pi * logpi).sum(1, keepdims=True)
                dlog += args.ent * pi * (logpi + H)
                dval = args.vcoef * (value - ret[i])
                model.adam(model.grads(cache, dlog / len(i), dval / len(i)), args.lr,
                           value_lr_scale=args.value_lr_scale)
        if recent:
            sc, ap_ = np.mean([r[0] for r in recent]), np.mean([r[1] for r in recent])
            print(f"moves {moves:9d} ep {ep:6d} | last 200 episodes: score {sc:+.2f}, apples {ap_:.2f} | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        log.flush()
    log.close()
    if rec:
        np.savez_compressed(out / "states.npz", frame=np.stack([r[0] for r in rec]), body=np.stack([r[1] for r in rec]),
                            apple=np.stack([r[2] for r in rec]), heading=np.array([r[3] for r in rec]),
                            action=np.array([r[4] for r in rec]), board=args.board)


if __name__ == "__main__":
    main()
