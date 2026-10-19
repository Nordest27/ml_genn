"""Non-spiking control for tic-tac-toe: PPO with a LINEAR policy and value read directly from the board image.

    python rate_baseline_ttt.py OUT_DIR [--opponent minimax|medium|random] [--games 50000] [--seed 0]

Same model and PPO as rate_baseline.py (no hidden layer). The environment is tic_tac_toe_env.TicTacToeEnv as in
tic_tac_toe_train.py (12x12x3 observation, random first player, the opening move random), stepped once per move
(wait_inc 0). Illegal cells are masked out of the policy, as in the spiking training. Reward +1 win, -1 loss, 0 draw.
Unlike Snake, the value of a move depends on combinations of cells (two of mine and an empty third in a line: an
XOR-like interaction), which a linear readout cannot represent; this measures how far a linear policy gets anyway.
Writes OUT_DIR/games.csv and prints win / draw / loss rates over the last 1000 games.
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
from rate_baseline import Model              # noqa: E402
from tic_tac_toe_env import TicTacToeEnv     # noqa: E402


def masked_softmax(logits, mask):
    z = np.where(mask, logits, -1e9)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z) * mask
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--opponent", default="minimax", choices=["minimax", "medium", "random"])
    ap.add_argument("--games", type=int, default=50000)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vcoef", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); rng = np.random.default_rng(args.seed)

    env = TicTacToeEnv(wait_inc=0, opponent=args.opponent, first_player="random")

    def advance(obs, done):
        """Step through the opponent's turns until the agent must move (or the game ends)."""
        reward = 0.0
        while not done and not env.awaiting_agent_action:
            obs, r, done = env.step(0)
            reward += r
        return obs, reward, done

    obs, r0, done = advance(env.reset(), False)
    model = Model(obs.size, 9, 0, rng)
    log = open(out / "games.csv", "w", newline="")
    writer = csv.writer(log); writer.writerow(["game", "decisions", "result"])
    games, decisions, recent, t0 = 0, 0, [], time.time()
    pending = r0                                   # reward from the opponent's moves before the next decision
    while games < args.games:
        X, M, A, LP, R, D, V = [], [], [], [], [], [], []
        while len(R) < args.rollout:
            if done:                               # game ended during the opponent's turn (before any decision)
                obs, pending, done = advance(env.reset(), False)
                continue
            x = obs.ravel().astype(np.float64)
            mask = env.legal_mask()
            logits, value, _ = model.forward(x[None])
            pi = masked_softmax(logits, mask[None])[0]
            a = int(rng.choice(9, p=pi))
            obs, r, done = env.step(a)
            if not done:
                obs, r_opp, done = advance(obs, done)
                r += r_opp
            X.append(x); M.append(mask); A.append(a); LP.append(np.log(pi[a] + 1e-12)); R.append(r)
            D.append(done); V.append(value[0]); decisions += 1
            if done:
                games += 1
                res = {"agent": 1, "opponent": -1}.get(env.winner, 0)
                writer.writerow([games, decisions, res]); recent.append(res); recent = recent[-1000:]
                obs, pending, done = advance(env.reset(), False)
        last_v = model.forward(obs.ravel()[None].astype(np.float64))[1][0] if not done else 0.0
        X = np.array(X); M = np.array(M); A = np.array(A); LP = np.array(LP)
        R = np.array(R, dtype=np.float64); D = np.array(D, dtype=np.float64); V = np.array(V + [last_v])
        adv = np.zeros(len(R)); gae = 0.0
        for t in reversed(range(len(R))):
            delta = R[t] + args.gamma * V[t + 1] * (1 - D[t]) - V[t]
            gae = delta + args.gamma * args.lam * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V[:-1]
        for _ in range(args.epochs):
            order = rng.permutation(len(R))
            for s in range(0, len(R), args.batch):
                i = order[s:s + args.batch]
                logits, value, cache = model.forward(X[i])
                pi = masked_softmax(logits, M[i])
                logp = np.log(pi[np.arange(len(i)), A[i]] + 1e-12)
                ratio = np.exp(logp - LP[i])
                a_n = (adv[i] - adv[i].mean()) / (adv[i].std() + 1e-8)
                active = ~(((a_n > 0) & (ratio > 1 + args.clip)) | ((a_n < 0) & (ratio < 1 - args.clip)))
                dlog = -(active * ratio * a_n)[:, None] * (np.eye(9)[A[i]] - pi)
                logpi = np.log(pi + 1e-12)
                H = -(pi * logpi * M[i]).sum(1, keepdims=True)
                dlog += args.ent * pi * (logpi + H) * M[i]
                dval = args.vcoef * (value - ret[i])
                model.adam(model.grads(cache, dlog / len(i), dval / len(i)), args.lr)
        if recent:
            rc = np.array(recent)
            print(f"games {games:7d} | last {len(rc)}: win {np.mean(rc == 1):.3f} draw {np.mean(rc == 0):.3f} "
                  f"loss {np.mean(rc == -1):.3f} | {(time.time() - t0) / 60:.1f} min", flush=True)
        log.flush()
    log.close()


if __name__ == "__main__":
    main()
