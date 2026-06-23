"""
plot_training.py
Iterates over all CSVs in outputs/ and renders one summary plot per file
(no grouping/aggregation across runs — every CSV gets its own PNG in
outputs/renders/, named after the file's stem).

Each panel shows a windowed rolling mean (bold line) with a ±1 std band,
computed the same way as the per-run window aggregation used in the
comparison scripts (fixed-size episode windows). Line/band color is
looked up per file via the same PATTERN_COLORS family-color system used
in compare_solutions.py / compare_lambda_groups.py / compare_layers_groups.py,
so a given file reads as the same color across every script.
"""

import os
import re
import glob
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_GLOB = "outputs/*.csv"
OUT_DIR    = "outputs/renders"
WINDOW_EP  = 50   # episode-window size for rolling mean / std smoothing
BAND_ALPHA = 0.25  # opacity of ±1 std shaded region
DPI        = 200

# Matches a trailing "(<number>)" right before the extension, e.g.
# "pacman_training_weight_dist_prop_lambda_05(3)" -> "pacman_training_weight_dist_prop_lambda_05"
# Stripped from the stem before color lookup, so e.g. "run(1)" and "run(2)"
# (now rendered as separate charts) still share the same color.
GROUP_SUFFIX_RE = re.compile(r"^(.*)\(\d+\)$")

# Subplot definitions  ─────────────────────────────────────────────────────────
PANELS = [
    {"title": "Episode Reward",  "cols": ["score"]},
    {"title": "Reward Rate",     "cols": ["reward_rate", "gamma_disc_reward"]},
    {"title": "TD Error |mean|", "cols": ["avg_abs_td_error"]},
    {"title": "Voltage",         "cols": ["voltage"]},
    {"title": "Voltage Loss",    "cols": ["voltage_loss"]},
    {"title": "Frequency",       "cols": ["frequency"]},
]

# ── Pattern → color mapping ─────────────────────────────────────────────────
# Same family-color system used across the comparison scripts: assigns a
# fixed base color to any file stem containing a given substring, so the
# same kind of run gets the same color everywhere. Substring-anywhere
# matching, longest pattern wins. Keep this list in sync with the other
# scripts' PATTERN_COLORS so a given solution reads as the same color
# family across all of them.
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

PATTERN_COLORS: list[tuple[str, str]] = [
    # --- "weight_dist" family (blues) ---------------------------------------
    ("weight_dist_prop",    "#1F4E79"),  # darkest blue  (most specific variant)
    ("weight_prop_dist",    "#1F4E79"),  # darkest blue  (most specific variant)
    ("weight_dist_uniform", "#2F5C8A"),  # dark blue
    ("weight_dist",         "#4C72B0"),  # base blue     (family base / fallback)
    # --- "lambda" family (oranges) -------------------------------------------
    ("lambda_05", "#A8431F"),  # darkest orange
    ("lambda_10", "#C2562C"),  # dark orange
    ("lambda_20", "#DD8452"),  # base orange   (family base)
    ("lambda",    "#DD8452"),  # any other lambda_* variant -> base orange
    # --- "node_dist" family (greens) -------------------------------------------
    ("node_dist_prop", "#2E5E37"),  # darkest green
    ("node_dist",      "#55A868"),  # base green   (family base)
    # --- "voltage_ctrl" family (reds) -----------------------------------------
    ("voltage_ctrl_pid",      "#8E2E33"),  # darkest red
    ("voltage_ctrl_adaptive", "#A93D43"),  # dark red
    ("voltage_ctrl",          "#C44E52"),  # base red     (family base)
    ("adaptive",                "#C44E52"),  # base red     (family base)
    # --- "combined" / entropy_reg family (purples) -----------------------------
    ("entropy_reg_high", "#574A7A"),  # darkest purple
    ("entropy_reg_low",  "#6C5C96"),  # dark purple
    ("entropy_reg",      "#8172B2"),  # base purple  (family base)
    ("combined_prop",    "#574A7A"),  # darkest purple (specific variant)
    ("combined",         "#8172B2"),  # base purple  (family base)
    # --- "freq_response" family (cyans) ----------------------------------------
    ("freq_response_fast",   "#3A8295"),  # darkest cyan
    ("freq_response_slow",   "#4F9CB0"),  # dark cyan
    ("freq_response",        "#64B5CD"),  # base cyan    (family base)
    ("weight_dist-ind-noise", "#3A8295"),  # darkest cyan (standalone variant)
    # --- standalone / unrelated patterns (no close siblings yet) ---------------
    ("baseline",    "#8C8C8C"),  # grey  — unmodified baseline runs
    ("ablation",    "#CCB974"),  # gold  — ablation studies
    ("curriculum",  "#937860"),  # brown — curriculum learning variants
    ("multi_agent", "#DA8BC3"),  # pink  — multi-agent setups
]

# Cache of name -> color so the fallback palette assignment for unmatched
# names is stable within a single resolution pass. Colors are resolved
# once per file in main() (single process, before dispatching to workers),
# so this cache only needs to be consistent within that pass — not across
# worker processes.
_color_cache: dict[str, str] = {}
_fallback_counter = 0


def get_color_for_solution(name: str) -> str:
    """Return a color for a file/solution name. Names containing a pattern
    from PATTERN_COLORS get that pattern's fixed color (longest matching
    pattern wins on overlaps). Names matching nothing fall back to
    COLOR_PALETTE, cycled in first-seen order and cached so the same
    unmatched name keeps the same fallback color across this run."""
    if name in _color_cache:
        return _color_cache[name]

    best_pattern, best_color = None, None
    for pattern, color in PATTERN_COLORS:
        if pattern in name:
            if best_pattern is None or len(pattern) > len(best_pattern):
                best_pattern, best_color = pattern, color

    if best_color is not None:
        _color_cache[name] = best_color
        return best_color

    global _fallback_counter
    color = COLOR_PALETTE[_fallback_counter % len(COLOR_PALETTE)]
    _fallback_counter += 1
    _color_cache[name] = color
    return color


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def color_key_for_file(csv_path: str) -> str:
    """File stem with any trailing '(N)' suffix stripped, used for color
    lookup so e.g. 'run(1)' and 'run(2)' share a color even though each
    now renders as its own chart."""
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    m = GROUP_SUFFIX_RE.match(stem)
    return m.group(1) if m else stem


def window_rolling_mean_std(ep: np.ndarray, val: np.ndarray, window: int):
    """Aggregate one run's (episode, value) pairs into a windowed rolling
    mean ± std (fixed-size episode windows, same binning approach as the
    comparison scripts' per-run window means).
    Returns (x_centers, mean, std) or None if no data."""
    valid = ~np.isnan(ep) & ~np.isnan(val)
    ep, val = ep[valid], val[valid]
    if len(ep) == 0:
        return None

    bins = (ep // window).astype(np.int64)
    grouped = pd.Series(val).groupby(bins)

    x_centers = grouped.mean().index.to_numpy(dtype=np.float64) * window + window / 2.0
    mean = grouped.mean().to_numpy()
    std  = grouped.std().to_numpy()
    std  = np.nan_to_num(std, nan=0.0)  # windows with a single point have NaN std

    return x_centers, mean, std


def style_ax(ax, title: str, color: str, x, mean, std):
    """Draw one panel: ±1 std band + bold rolling-mean line."""
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=BAND_ALPHA,
                     zorder=2, linewidth=0, label="±1 std")
    ax.plot(x, mean, color=color, linewidth=2.0, zorder=3, label="mean")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.6)
    finish_ax(ax, title)


def finish_ax(ax, title: str):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Episode", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="grey", alpha=0.2, linewidth=0.5)


# ── Main ──────────────────────────────────────────────────────────────────────

def render(task):
    """Render one chart for one CSV file: windowed rolling mean ± std per
    panel, colored via the shared PATTERN_COLORS family-color system.
    Takes a (csv_path, color) tuple — color is resolved up front in the
    main process (see main()) so fallback-palette assignment for unmatched
    names stays consistent even though rendering is parallelized across
    worker processes."""
    csv_path, color = task
    stem = os.path.splitext(os.path.basename(csv_path))[0]

    df = pd.read_csv(csv_path)

    out_png = os.path.join(OUT_DIR, f"{stem}.png")

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(stem.replace("-", " ").replace("_", " "), fontsize=13,
                 fontweight="bold", y=1.01)
    axs = axs.flatten()

    for idx, panel in enumerate(PANELS):
        ax = axs[idx]
        col = find_col(df, panel["cols"])

        if col is None:
            ax.set_visible(False)
            continue

        ep  = pd.to_numeric(df["episode"], errors="coerce").to_numpy(dtype=np.float64)
        val = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)

        if col == "frequency":
            val = val * 1000

        agg = window_rolling_mean_std(ep, val, WINDOW_EP)
        if agg is None:
            ax.set_visible(False)
            continue
        x, mean, std = agg

        style_ax(ax, panel["title"], color, x, mean, std)

    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_png}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(INPUT_GLOB))

    if not files:
        print(f"No CSV files found matching {INPUT_GLOB!r}.")
        return

    print(f"Found {len(files)} CSV file(s) — rendering one chart each...")

    # Resolve every file's color up front, in this (single) process, so
    # fallback-palette assignment for unmatched names is deterministic and
    # consistent — workers just use the color they're handed, they never
    # compute their own.
    tasks = [(f, get_color_for_solution(color_key_for_file(f))) for f in files]

    workers = min(len(tasks), os.cpu_count() or 1)
    with multiprocessing.Pool(processes=workers) as pool:
        pool.map(render, tasks)

    print(f"\nAll renders saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()