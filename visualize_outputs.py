"""
plot_training.py
Iterates over all CSVs in outputs/, groups files that share the same name
prefix up to a trailing "(N)" (e.g. "run(1).csv", "run(2).csv" -> group "run"),
renders a smoothed summary plot per group, and saves PNGs to outputs/renders/.

- Single-file groups: median + IQR band with raw data faded in background
  (rolling window smoothing), same as before.
- Multi-file groups (several runs of the same experiment): mean (bold line),
  ±1 std band, and min/max band, all aggregated within fixed-size episode
  windows (pooling data points from every run that falls in that window).
"""

import os
import re
import glob
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import uniform_filter1d

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_GLOB  = "outputs/*.csv"
OUT_DIR     = "outputs/renders"
WINDOW          = 50        # rolling window for median / IQR smoothing (single-run plots)
WINDOW_EP       = 100       # episode-window size for multi-run aggregation
RAW_ALPHA       = 0.15      # opacity of raw data trace
BAND_ALPHA      = 0.25      # opacity of IQR / std shaded region
MINMAX_ALPHA    = 0.12      # opacity of min-max shaded region (multi-run plots)
DPI             = 200
WIN_THRESHOLD   = 24        # score >= this value counts as a win

# Matches a trailing "(<number>)" right before the extension, e.g.
# "pacman_training_weight_dist_prop_lambda_05(3)" -> "pacman_training_weight_dist_prop_lambda_05"
GROUP_SUFFIX_RE = re.compile(r"^(.*)\(\d+\)$")

# Subplot definitions  ─────────────────────────────────────────────────────────
PANELS = [
    {
        "title": "Episode Reward",
        "cols":  ["score"],
        "color": "#4C72B0",
    },
    {
        "title": "Reward Rate",
        "cols":  ["reward_rate", "gamma_disc_reward"],   # first present wins
        "color": "#55A868",
    },
    {
        "title": "TD Error |mean|",
        "cols":  ["avg_abs_td_error"],
        "color": "#C44E52",
    },
    {
        "title": "Voltage",
        "cols":  ["voltage"],
        "color": "#8172B2",
    },
    {
        "title": "Voltage Loss",
        "cols":  ["voltage_loss"],
        "color": "#CCB974",
    },
    {
        "title": "Frequency",
        "cols":  ["frequency"],
        "color": "#64B5CD",
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def rolling_stats(y: np.ndarray, w: int):
    """Return (median, q25, q75) via a simple rolling window."""
    n = len(y)
    med = np.empty(n)
    q25 = np.empty(n)
    q75 = np.empty(n)
    half = w // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = y[lo:hi]
        med[i] = np.median(chunk)
        q25[i] = np.percentile(chunk, 25)
        q75[i] = np.percentile(chunk, 75)
    return med, q25, q75


def per_run_window_means(ep: np.ndarray, val: np.ndarray, window: int) -> pd.Series:
    """Reduce one run's (episode, value) pairs to a per-window mean.
    Returns a Series indexed by window bin index."""
    valid = ~np.isnan(ep) & ~np.isnan(val)
    ep, val = ep[valid], val[valid]
    if len(ep) == 0:
        return pd.Series(dtype=np.float64)
    bins = (ep // window).astype(np.int64)
    return pd.Series(val).groupby(bins).mean()


def aggregate_runs_by_window(per_run_series: list[pd.Series], window: int):
    """Combine each run's per-window means into across-run mean / std /
    min / max curves (the min/max curves trace the worst- and best-run
    window-averages, not raw per-episode outliers).
    Returns (x_centers, mean, std, vmin, vmax) or None if no data."""
    per_run_series = [s for s in per_run_series if not s.empty]
    if not per_run_series:
        return None

    combined = pd.concat(per_run_series, axis=1).sort_index()

    x_centers = combined.index.to_numpy(dtype=np.float64) * window + window / 2.0
    mean = combined.mean(axis=1, skipna=True).to_numpy()
    std  = combined.std(axis=1, skipna=True).to_numpy()
    vmin = combined.min(axis=1, skipna=True).to_numpy()
    vmax = combined.max(axis=1, skipna=True).to_numpy()

    return x_centers, mean, std, vmin, vmax


def find_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def group_files(files: list[str]) -> dict[str, list[str]]:
    """Group CSV paths that share the same stem up to a trailing '(N)'."""
    groups: dict[str, list[str]] = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        m = GROUP_SUFFIX_RE.match(stem)
        key = m.group(1) if m else stem
        groups.setdefault(key, []).append(f)
    # keep deterministic ordering of runs within a group
    for key in groups:
        groups[key] = sorted(groups[key])
    return groups


def style_ax(ax, title: str, color: str, x, raw, med, q25, q75, win_mask=None):
    """Draw one single-run panel: faded raw + IQR band + median line, with
    optional win dots."""
    # raw trace
    ax.plot(x, raw, color=color, alpha=RAW_ALPHA, linewidth=0.6, zorder=1)
    # IQR band
    ax.fill_between(x, q25, q75, color=color, alpha=BAND_ALPHA, zorder=2, linewidth=0)
    # median
    ax.plot(x, med, color=color, linewidth=1.8, zorder=3, label="median")
    # win dots on the raw score trace
    # if win_mask is not None and win_mask.any():
    #     ax.scatter(x[win_mask], raw[win_mask],
    #                color="#FFD700", edgecolors="#B8860B",
    #                s=18, linewidths=0.5, zorder=5,
    #                label=f"win (≥{WIN_THRESHOLD})")
    #     ax.legend(fontsize=7, loc="upper left", framealpha=0.6)
    finish_ax(ax, title)


def style_ax_multi(ax, title: str, color: str, x, mean, std, vmin, vmax, win_points=None):
    """Draw one multi-run panel: min/max band + ±1 std band + bold mean line,
    with optional faded win dots (pooled raw episode/score points)."""
    # min-max band
    ax.fill_between(x, vmin, vmax, color=color, alpha=MINMAX_ALPHA, zorder=1,
                    linewidth=0, label="best/worst run")
    # ±1 std band
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=BAND_ALPHA, zorder=2,
                    linewidth=0, label="±1 std")
    # mean line
    ax.plot(x, mean, color=color, linewidth=2.2, zorder=3, label="mean")
    # win dots (pooled raw points across all runs)
    # if win_points is not None and len(win_points[0]):
    #     wx, wy = win_points
    #     ax.scatter(wx, wy, color="#FFD700", edgecolors="#B8860B",
    #                s=14, linewidths=0.4, alpha=0.6, zorder=5,
    #                label=f"win (≥{WIN_THRESHOLD})")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.6)
    finish_ax(ax, title)


def finish_ax(ax, title: str):
    """Shared aesthetics for both single-run and multi-run panels."""
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Episode", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="grey", alpha=0.2, linewidth=0.5)


# ── Main ──────────────────────────────────────────────────────────────────────

def render_single(group_key: str, csv_path: str):
    """Original single-run rendering: median + IQR band with faded raw trace."""
    df = pd.read_csv(csv_path)
    x  = pd.to_numeric(df["episode"], errors="coerce").to_numpy(dtype=np.float64)

    out_png = os.path.join(OUT_DIR, f"{group_key}.png")

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(group_key.replace("-", " ").replace("_", " "), fontsize=13,
                 fontweight="bold", y=1.01)
    axs = axs.flatten()

    score_col = find_col(df, ["score"])
    win_mask  = (pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=np.float64) >= WIN_THRESHOLD) if score_col else None

    for idx, panel in enumerate(PANELS):
        ax    = axs[idx]
        col   = find_col(df, panel["cols"])
        color = panel["color"]

        if col is None:
            ax.set_visible(False)
            continue

        raw            = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        med, q25, q75  = rolling_stats(raw, WINDOW)
        med  = med.astype(np.float64)
        q25  = q25.astype(np.float64)
        q75  = q75.astype(np.float64)
        mask = win_mask if (idx == 0 and win_mask is not None) else None
        style_ax(ax, panel["title"], color, x, raw, med, q25, q75, win_mask=mask)

    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_png}")


def render_group(group_key: str, csv_paths: list[str]):
    """Multi-run rendering: pool runs and aggregate within WINDOW_EP-sized
    episode windows, showing mean, ±1 std and min/max."""
    dfs = [pd.read_csv(p) for p in csv_paths]

    out_png = os.path.join(OUT_DIR, f"{group_key}.png")

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(
        f"{group_key.replace('-', ' ').replace('_', ' ')}  (n={len(dfs)} runs)",
        fontsize=13, fontweight="bold", y=1.01
    )
    axs = axs.flatten()

    for idx, panel in enumerate(PANELS):
        ax    = axs[idx]
        color = panel["color"]

        # per-run window means for this panel
        per_run_series = []
        all_ep, all_val = [], []  # pooled raw points, used only for win dots
        for df in dfs:
            col = find_col(df, panel["cols"])
            if col is None:
                continue
            ep  = pd.to_numeric(df["episode"], errors="coerce").to_numpy(dtype=np.float64)
            val = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            per_run_series.append(per_run_window_means(ep, val, WINDOW_EP))
            all_ep.append(ep)
            all_val.append(val)

        if not per_run_series:
            ax.set_visible(False)
            continue

        agg = aggregate_runs_by_window(per_run_series, WINDOW_EP)
        if agg is None:
            ax.set_visible(False)
            continue
        x, mean, std, vmin, vmax = agg

        win_points = None
        # if idx == 0:  # win dots only on the Episode Reward panel
        #     all_ep_c  = np.concatenate(all_ep)
        #     all_val_c = np.concatenate(all_val)
        #     wmask = all_val_c >= WIN_THRESHOLD
        #     if wmask.any():
        #         win_points = (all_ep_c[wmask], all_val_c[wmask])

        style_ax_multi(ax, panel["title"], color, x, mean, std, vmin, vmax, win_points)

    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_png}")


def render(task):
    """Dispatch to single- or multi-run rendering depending on group size."""
    group_key, csv_paths = task
    if len(csv_paths) == 1:
        render_single(group_key, csv_paths[0])
    else:
        render_group(group_key, csv_paths)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(INPUT_GLOB))

    if not files:
        print(f"No CSV files found matching {INPUT_GLOB!r}.")
        return

    groups = group_files(files)
    tasks = list(groups.items())

    n_multi = sum(1 for _, paths in tasks if len(paths) > 1)
    print(f"Found {len(files)} CSV file(s) — {len(tasks)} output group(s) "
          f"({n_multi} multi-run) — rendering...")

    workers = min(len(tasks), os.cpu_count() or 1)
    with multiprocessing.Pool(processes=workers) as pool:
        pool.map(render, tasks)

    print(f"\nAll renders saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
