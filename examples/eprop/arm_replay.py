"""Replays of the two-link arm task (snake_switch.ArmEnv with "record_every"): animated GIFs of recorded episodes.

    python arm_replay.py RUN_DIR/outputs/arm_episodes -o replay.gif [--episodes 0 500 1000] [--stride 4] [--fps 30]

Each panel shows the workspace (the reachable disc faint), the two links from the shoulder at the centre, the hand's
trail, and the target (green once reached), with the episode, the step and the cumulative reward. Several episodes
(default: the first, a middle and the last recorded one) play side by side, so early and late behaviour can be
compared; shorter episodes hold their last frame.
"""
import argparse
import glob
import os

import numpy as np


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def pick(files, episodes):
    eps = {int(os.path.basename(f)[2:-4]): f for f in files}
    if episodes:
        return [eps[min(eps, key=lambda e: abs(e - want))] for want in episodes]
    keys = sorted(eps)
    if len(keys) <= 3:
        return [eps[k] for k in keys]
    return [eps[keys[0]], eps[keys[len(keys) // 2]], eps[keys[-1]]]


def render(records, out, stride=4, fps=30, trail=150):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    n = len(records)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.9), squeeze=False)
    axes = axes[0]
    arts = []
    for ax, r in zip(axes, records):
        point = str(r.get("kinematics", "arm")) == "point"
        if point:                                      # an open plane: frame the hand's path and the targets
            pts = np.concatenate([r["hand"], r.get("targets", r["target"][None])])
            lo, hi = pts.min(0) - 0.2, pts.max(0) + 0.2; c, half = (lo + hi) / 2, max(hi - lo) / 2
            ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
        else:
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
            ax.add_patch(plt.Circle((0, 0), float(r["l1"] + r["l2"]), color="0.92", zorder=0))
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        tgt = plt.Circle(tuple(r["target"]), float(r.get("radius_now", r["radius"])), color="tab:red", alpha=0.6, zorder=1)
        ax.add_patch(tgt)
        trail_line, = ax.plot([], [], "-", color="tab:blue", alpha=0.35, lw=1, zorder=2)
        arm_line, = ax.plot([], [], "-o", color="0.15", lw=3, ms=4, zorder=3)
        hand_dot, = ax.plot([], [], "o", color="tab:blue", ms=7, zorder=4)
        title = ax.set_title("", fontsize=9)
        arts.append((r, tgt, trail_line, arm_line, hand_dot, title, point))
    length = max(len(r["theta"]) for r in records)
    frames = list(range(0, length, stride)) + [length - 1]

    def update(t):
        out_arts = []
        for r, tgt, trail_line, arm_line, hand_dot, title, point in arts:
            k = min(t, len(r["theta"]) - 1)
            el, hd = r["elbow"][k], r["hand"][k]
            if "targets" in r:                     # continuous mode: the target and its radius change over time
                tgt.center = tuple(r["targets"][k]); tgt.set_radius(float(r["radii"][k]))
                if k > 0 and not np.allclose(r["targets"][k], r["targets"][k - 1]):
                    tgt.set_color("tab:red")
            if not point:
                arm_line.set_data([0, el[0], hd[0]], [0, el[1], hd[1]])
            lo = max(0, k - trail)
            trail_line.set_data(r["hand"][lo:k + 1, 0], r["hand"][lo:k + 1, 1])
            hand_dot.set_data([hd[0]], [hd[1]])
            done = k == len(r["theta"]) - 1
            if done and bool(r["reached"]) and "targets" not in r:
                tgt.set_color("tab:green")
            if "targets" in r and k + 1 < len(r["targets"]) and not np.allclose(r["targets"][k + 1], r["targets"][k]) \
                    and np.linalg.norm(r["hand"][k + 1] - r["targets"][k]) < r["radii"][k] + 1e-9:
                tgt.set_color("tab:green")          # reached: the next target appears
            lvl = f"  level {float(r['level']):.1f}" if "level" in r else ""
            title.set_text(f"episode {int(r['episode'])}{lvl}  step {k}\nreward {r['reward'][:k + 1].sum():+.2f}"
                           + ("  reached" if done and bool(r["reached"]) else ""))
            out_arts += [tgt, trail_line, arm_line, hand_dot, title]
        return out_arts

    fig.tight_layout(); fig.subplots_adjust(top=0.84)
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes_dir"); ap.add_argument("-o", "--out", default="arm_replay.gif")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=4); ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.episodes_dir, "ep*.npz")))
    if not files:
        raise SystemExit(f"no recorded episodes in {a.episodes_dir} (set arm.record_every)")
    records = [load(f) for f in pick(files, a.episodes)]
    render(records, a.out, a.stride, a.fps)
    print(f"wrote {a.out} ({len(records)} episodes: {[int(r['episode']) for r in records]})")


if __name__ == "__main__":
    main()
