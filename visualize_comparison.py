"""
compare_solutions.py
Compares multiple multi-run "solutions" (groups of CSVs that share the same
name prefix up to a trailing "(N)") that have the same number of runs.
Each solution is assigned its own color and, for every panel, its mean
(bold line), ±1 std band, and best/worst-run band (per-run window means,
see plot_training.py) are overlaid on shared axes so solutions can be
compared directly.

Solutions are bucketed by run count: e.g. all solutions with 5 runs are
compared together in one figure, all solutions with 3 runs in another.
Single-run CSVs and run-count buckets with fewer than 2 solutions are
skipped (nothing to compare).
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_GLOB    = "outputs/*.csv"
OUT_DIR       = "outputs/renders/comparisons"
WINDOW_EP     = 100        # episode-window size for per-run aggregation
BAND_ALPHA    = 0.15       # opacity of ±1 std band
MINMAX_ALPHA  = 0.06       # opacity of best/worst-run band
DPI           = 200

# Matches a trailing "(<number>)" right before the extension, e.g.
# "pacman_training_weight_dist_prop_lambda_05(3)" -> "pacman_training_weight_dist_prop_lambda_05"
GROUP_SUFFIX_RE = re.compile(r"^(.*)\(\d+\)$")

# Distinct colors assigned to solutions within a comparison figure (cycled
# if there are more solutions than colors).
COLOR_PALETTE = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B2",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # grey
    "#CCB974",  # gold
    "#64B5CD",  # cyan
]

# Subplot definitions (must match plot_training.py)  ────────────────────────────
PANELS = [
    {"title": "Episode Reward",  "cols": ["score"]},
    {"title": "Reward Rate",     "cols": ["reward_rate", "gamma_disc_reward"]},
    {"title": "TD Error |mean|", "cols": ["avg_abs_td_error"]},
    {"title": "Voltage",         "cols": ["voltage"]},
    {"title": "Voltage Loss",    "cols": ["voltage_loss"]},
    {"title": "Frequency",       "cols": ["frequency"]},
]

# ── Helpers ───────────────────────────────────────────────────────────────────

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
    for key in groups:
        groups[key] = sorted(groups[key])
    return groups


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


def finish_ax(ax, title: str):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Episode", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="grey", alpha=0.2, linewidth=0.5)


# ── Main ──────────────────────────────────────────────────────────────────────

def render_comparison(n_runs: int, solutions: list[tuple[str, list[str]]]):
    out_png = os.path.join(OUT_DIR, f"comparison_{n_runs}_runs.png")

    # pre-load each solution's dataframes once
    loaded = [(key, [pd.read_csv(p) for p in paths]) for key, paths in solutions]

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"Pac-Man Randomized Layouts Problem", fontsize=13,
                  fontweight="bold", y=1.02)
    axs = axs.flatten()

    legend_handles, legend_labels = [], []
    panel_has_data = [False] * len(PANELS)

    for sol_idx, (group_key, dfs) in enumerate(loaded):
        color = COLOR_PALETTE[sol_idx % len(COLOR_PALETTE)]
        label = group_key.replace("-", " ").replace("_", " ")

        for idx, panel in enumerate(PANELS):
            ax = axs[idx]

            per_run_series = []
            for df in dfs:
                col = find_col(df, panel["cols"])
                if col is None:
                    continue
                ep  = pd.to_numeric(df["episode"], errors="coerce").to_numpy(dtype=np.float64)
                val = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
                per_run_series.append(per_run_window_means(ep, val, WINDOW_EP))

            agg = aggregate_runs_by_window(per_run_series, WINDOW_EP)
            if agg is None:
                continue
            x, mean, std, vmin, vmax = agg
            panel_has_data[idx] = True

            ax.fill_between(x, vmin, vmax, color=color, alpha=MINMAX_ALPHA, linewidth=0, zorder=1)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
            line, = ax.plot(x, mean, color=color, linewidth=2.0, zorder=3)

            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(label)

    for idx, panel in enumerate(PANELS):
        if panel_has_data[idx]:
            finish_ax(axs[idx], panel["title"])
        else:
            axs[idx].set_visible(False)

    fig.legend(legend_handles, legend_labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=min(len(legend_labels), 4),
               fontsize=9, frameon=False)

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_png}")


def main():
    files = sorted(glob.glob(INPUT_GLOB))
    if not files:
        print(f"No CSV files found matching {INPUT_GLOB!r}.")
        return

    groups = group_files(files)
    multi_run = {key: paths for key, paths in groups.items() if len(paths) > 1}

    if not multi_run:
        print("No multi-run solutions found (need at least 2 CSVs sharing a "
              "common '<name>(N).csv' prefix).")
        return

    # bucket solutions by their run count
    buckets: dict[int, list[tuple[str, list[str]]]] = {}
    for key, paths in multi_run.items():
        buckets.setdefault(len(paths), []).append((key, paths))

    os.makedirs(OUT_DIR, exist_ok=True)

    any_rendered = False
    for n_runs, solutions in sorted(buckets.items()):
        if len(solutions) < 2:
            print(f"  -  skipping n={n_runs}-run bucket "
                  f"(only 1 solution: {solutions[0][0]!r}, nothing to compare)")
            continue
        names = ", ".join(s[0] for s in solutions)
        print(f"Comparing {len(solutions)} solutions with {n_runs} runs each: {names}")
        render_comparison(n_runs, solutions)
        any_rendered = True

    if any_rendered:
        print(f"\nAll comparisons saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()