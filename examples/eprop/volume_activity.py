"""Activity of a volume network, sheet by sheet (move_log "activity" / "activity_at" recordings).

    python volume_activity.py RUN_DIR [-o PREFIX] [--fps 10] [--moves 200]

RUN_DIR holds config.json and outputs/*_activity*.npz (one file per recorded window: spike counts per neuron per move).
For a volume layer (network.layers[0].volume), a population of N neurons with depth D is laid out (side, side, D) in C
order (snake_network.volume_positions), so each depth index is one sheet, from the input sheet to the output sheets.
Writes, per window:
  PREFIX_<window>.gif   one frame per move: the excitatory and inhibitory sheets side by side (input left, output
                        right) and the motor (policy) field, spike counts per neuron over the move
  PREFIX_sheets.png     mean spike count per sheet (excitatory and inhibitory) for every window: how activity
                        propagates through the depth, and how learning changes it
"""
import argparse
import glob
import json
import os

import numpy as np


def sheets(counts, depth):
    """(moves, N) -> (moves, side, side, depth), or None if N is not side^2 * depth."""
    n = counts.shape[1]
    side = int(round(np.sqrt(n / depth)))
    if side * side * depth != n:
        return None
    return counts.reshape(len(counts), side, side, depth)


def field_img(counts, channels=3):
    n = counts.shape[1]
    side = int(round(np.sqrt(n / channels)))
    if side * side * channels != n:
        return None
    return counts.reshape(len(counts), side, side, channels).sum(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir"); ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--fps", type=int, default=10); ap.add_argument("--moves", type=int, default=200)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    depth = int(cfg["network"]["layers"][0]["volume"]["depth"])
    prefix = a.out or os.path.join(a.run_dir, "volume")
    files = sorted(glob.glob(os.path.join(a.run_dir, "outputs", "*_activity*.npz")))
    if not files:
        raise SystemExit("no activity recordings (set move_log.activity / activity_at)")
    means = {}
    for f in files:
        z = np.load(f)
        tag = os.path.basename(f).split("_activity")[1].replace(".npz", "") or "_0"
        E = sheets(z["L1_E"].astype(float), depth)
        I = sheets(z["L1_I"].astype(float), depth) if "L1_I" in z.files else None
        F = field_img(z["policy_field"].astype(float)) if "policy_field" in z.files else None
        if E is None:
            raise SystemExit(f"L1_E ({z['L1_E'].shape[1]} neurons) is not a volume of depth {depth}")
        t0 = int(z["timestep"][0]) if "timestep" in z.files and len(z["timestep"]) else 0
        means[t0] = (E.mean((0, 1, 2)), None if I is None else I.mean((0, 1, 2)))
        m = min(a.moves, len(E))
        rows = 2 if I is not None else 1
        cols = depth + (1 if F is not None else 0)
        fig, axes = plt.subplots(rows, cols, figsize=(1.3 * cols, 1.5 * rows + 0.4), squeeze=False)
        vE = max(np.percentile(E[:m], 99), 1e-9); vI = max(np.percentile(I[:m], 99), 1e-9) if I is not None else 1
        vF = max(np.percentile(F[:m], 99), 1e-9) if F is not None else 1
        ims = []
        for d in range(depth):
            ims.append(axes[0][d].imshow(E[0, :, :, d], vmin=0, vmax=vE, cmap="magma"))
            axes[0][d].set_title(f"E z{d}", fontsize=7)
            if I is not None:
                ims.append(axes[1][d].imshow(I[0, :, :, d], vmin=0, vmax=vI, cmap="viridis"))
                axes[1][d].set_title(f"I z{d}", fontsize=7)
        if F is not None:
            ims.append(axes[0][depth].imshow(F[0], vmin=0, vmax=vF, cmap="cividis"))
            axes[0][depth].set_title("motor field", fontsize=7)
            if rows > 1:
                axes[1][depth].axis("off")
        for row in axes:
            for ax in row:
                ax.set_xticks([]); ax.set_yticks([])
        sup = fig.suptitle("", fontsize=8)

        def update(k):
            i = 0
            for d in range(depth):
                ims[i].set_data(E[k, :, :, d]); i += 1
                if I is not None:
                    ims[i].set_data(I[k, :, :, d]); i += 1
            if F is not None:
                ims[i].set_data(F[k])
            ts = int(z["timestep"][k]) if "timestep" in z.files and k < len(z["timestep"]) else k
            sup.set_text(f"window {tag.strip('_')}: move {k}  (timestep {ts})  — spike counts per neuron per move")
            return ims + [sup]

        fig.tight_layout()
        FuncAnimation(fig, update, frames=range(m), blit=False).save(f"{prefix}{tag}.gif",
                                                                    writer=PillowWriter(fps=a.fps))
        plt.close(fig)
        print(f"wrote {prefix}{tag}.gif ({m} moves)")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for t0, (me, mi) in sorted(means.items()):
        ax[0].plot(range(depth), me, "-o", ms=3, label=f"from timestep {t0}")
        if mi is not None:
            ax[1].plot(range(depth), mi, "-o", ms=3, label=f"from timestep {t0}")
    for k, name in enumerate(("excitatory", "inhibitory")):
        ax[k].set_xlabel("sheet (input -> output)"); ax[k].set_ylabel("mean spikes per neuron per move")
        ax[k].set_title(name, fontsize=9); ax[k].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{prefix}_sheets.png", dpi=120); plt.close(fig)
    print(f"wrote {prefix}_sheets.png")


if __name__ == "__main__":
    main()
