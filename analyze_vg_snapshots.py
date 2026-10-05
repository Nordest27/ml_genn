import argparse
import glob
import os

import numpy as np
import pandas as pd


def load_trace_csv(path):
    """Load timestep, value, and reward_trace from a CSV file."""
    df = pd.read_csv(path).sort_values("timestep").reset_index(drop=True)

    required = {"timestep", "value", "reward_trace"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")

    return (
        df["timestep"].to_numpy(dtype=np.int64),
        df["value"].to_numpy(dtype=np.float64),
        df["reward_trace"].to_numpy(dtype=np.float64),
    )


def discounted_return(reward_trace, gamma_step):
    """Compute G_t = r_t + gamma_step * G_{t+1} over the whole trace."""
    g = np.zeros_like(reward_trace)
    running = 0.0

    for t in range(len(reward_trace) - 1, -1, -1):
        running = reward_trace[t] + gamma_step * running
        g[t] = running

    return g


def plot_vg_snapshots(
    file_name, timesteps, v, g, n_snapshots, snap_len, out_dir, at_timesteps=None
):
    """Plot raw V and G over evenly spaced sections of the trace."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    n = len(v)
    if n_snapshots <= 0:
        return
    if n < snap_len:
        raise ValueError(
            f"Trace has only {n} rows, shorter than snapshot length {snap_len}"
        )

    max_start = n - snap_len
    half = snap_len // 2

    if at_timesteps:
        if n_snapshots < 3:
            raise ValueError(
                "--snapshot-at requires --snapshots >= 3 "
                "(first/last stay anchored to the trace ends, "
                "--snapshot-at fills the ones in between)"
            )
        if len(at_timesteps) != n_snapshots - 2:
            raise ValueError(
                f"--snapshot-at needs exactly {n_snapshots - 2} timestep(s) "
                f"to fill the middle of {n_snapshots} snapshots "
                f"(got {len(at_timesteps)})"
            )

        middle_starts = []
        for t in at_timesteps:
            center_idx = int(np.searchsorted(timesteps, t))
            start = int(np.clip(center_idx - half, 0, max_start))
            middle_starts.append(start)

        starts = [0] + middle_starts + [max_start]
    elif n_snapshots == 1:
        starts = [max_start // 2]
    else:
        starts = [
            round(i * max_start / (n_snapshots - 1))
            for i in range(n_snapshots)
        ]

    fig, axes = plt.subplots(
        n_snapshots, 1, figsize=(11, 3 * n_snapshots), squeeze=False
    )
    axes = axes[:, 0]

    for ax, start in zip(axes, starts):
        end = start + snap_len
        ts = timesteps[start:end]

        ax.plot(ts, v[start:end], linewidth=0.8, label="V")
        ax.plot(ts, g[start:end], linewidth=0.8, label="G")
        ax.set_title(f"Timestep {ts[0]:,} - {ts[-1]:,}")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("V vs G snapshots")
    plt.tight_layout()

    output_path = os.path.join(out_dir, f"{file_name}.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved snapshot plot to {output_path}")


def find_csv_files(trace_dir, recursive):
    pattern = "**/*.csv" if recursive else "*.csv"
    paths = glob.glob(os.path.join(trace_dir, pattern), recursive=recursive)
    return sorted(paths)


def process_file(trace_path, args):
    print(f"Loading {trace_path}...")
    timesteps, v, reward_trace = load_trace_csv(trace_path)
    file_name = os.path.basename(trace_path).split(".")[0]

    print("Computing discounted return...")
    g = discounted_return(reward_trace, args.gamma_step)

    plot_vg_snapshots(
        file_name,
        timesteps,
        v,
        g,
        args.snapshots,
        args.snapshot_len,
        args.out_dir,
        at_timesteps=args.snapshot_at,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot raw V vs discounted return G snapshots."
    )
    parser.add_argument(
        "--trace-dir",
        required=True,
        help="Folder to scan for CSV trace files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subfolders too",
    )
    parser.add_argument(
        "--gamma-step",
        type=float,
        required=True,
        help="Per-timestep discount, e.g. 0.5**(1/30)",
    )
    parser.add_argument(
        "--snapshot-at",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit timesteps for the middle snapshots (n_snapshots - 2 "
            "values required). The first and last snapshots always stay "
            "anchored to the start/end of the trace; this only overrides "
            "the ones in between."
        ),
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=3,
        help="Number of evenly spaced snapshots",
    )
    parser.add_argument(
        "--snapshot-len",
        type=int,
        default=10000,
        help="Rows shown in each snapshot",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/analysis_continuous",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csv_files = find_csv_files(args.trace_dir, args.recursive)
    if not csv_files:
        print(f"No CSV files found in {args.trace_dir}")
        return

    print(f"Found {len(csv_files)} CSV file(s) in {args.trace_dir}")

    for trace_path in csv_files:
        try:
            process_file(trace_path, args)
        except Exception as e:
            print(f"Skipping {trace_path}: {e}")


if __name__ == "__main__":
    main()