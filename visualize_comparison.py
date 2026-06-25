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

# Matches a "[<chart title>]" tag anywhere in the filename stem, e.g.
# "pacman_training[Lambda Sweep](3)" -> tag "Lambda Sweep". Used to route a
# file into its own comparison chart, independent of the (N) run-count
# suffix. Files with no tag fall into the MISC_CHART_KEY chart.
CHART_TAG_RE = re.compile(r"\[([^\[\]]+)\]")

MISC_CHART_KEY = "Misc"

# Distinct colors assigned to solutions within a comparison figure. Used as
# the fallback palette for solution names that don't match any entry in
# PATTERN_COLORS below (cycled if there are more unmatched names than
# colors).
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

# ── Pattern → color mapping ─────────────────────────────────────────────────
# Assigns a fixed color to any solution name containing a given substring,
# so the same kind of solution gets the same color across every chart it
# appears in (instead of an arbitrary per-figure index). Matching is
# substring-anywhere, and the LONGEST matching pattern wins, so e.g.
# "weight_dist_prop" is matched before the shorter "weight_dist" even
# though "weight_dist" is also a substring of it. List order below doesn't
# matter — longest-match-wins is resolved automatically at lookup time.
#
# Group related patterns with visually-similar hex shades (e.g. all
# "weight_dist*" variants in blues, all "lambda*" variants in oranges) so
# related solutions read as a family at a glance. These are placeholders —
# fill in / extend with your actual naming patterns and chosen hexes.
PATTERN_COLORS: list[tuple[str, str]] = [
    # --- "weight_dist" family (blues) ---------------------------------------
    ("weight_dist_prop",   "#1F4E79"),  # darkest blue  (most specific variant)
    ("weight_prop_dist",   "#1F4E79"),  # darkest blue  (most specific variant)
    ("weight_dist_uniform","#2F5C8A"),  # dark blue
    ("weight_dist",        "#4C72B0"),  # base blue     (family base / fallback within family)
    ("weight_dist_no_heuristic",        "#A6B7D3"),  # base blue     (family base / fallback within family)
    ("weight_dist_update_per_episode",        "#87A6DB"),  # base blue     (family base / fallback within family)
    
    ("weight_dist_no_creg_update_per_episode",        "#506588"),  # base blue     (family base / fallback within family)

    # --- "lambda" family (oranges) -------------------------------------------
    ("asfadsfsadfasdf",          "#A8431F"),  # darkest orange
    ("asfadsfsadfasdf",          "#C2562C"),  # dark orange
    ("asfadsfsadfasdf",          "#DD8452"),  # base orange   (family base)
    ("asfadsfsadfasdf",             "#DD8452"),  # any other lambda_* variant -> base orange

    # --- "reward_shaping" family (greens) -------------------------------------
    ("node_dist_prop",  "#2E5E37"),  # darkest green
    ("asfadsfsadfasdf", "#3F7A4A"),  # dark green
    ("node_dist",        "#55A868"),  # base green   (family base)

    # --- "voltage_ctrl" family (reds) -----------------------------------------
    ("asfadsfsadfasdf",   "#8E2E33"),  # darkest red
    ("asfadsfsadfasdf", "#A93D43"),  # dark red
    ("adaptive",       "#C44E52"),  # base red     (family base)
    ("random",       "#C44E52"),  # base red     (family base)
    ("adaptive_update_per_episode",       "#B8726D"),  # base red     (family base)

    # --- "entropy_reg" family (purples) ---------------------------------------
    ("combined_prop",   "#574A7A"),  # darkest purple
    ("asfadsfsadfasdf",    "#6C5C96"),  # dark purple
    ("combined",        "#8172B2"),  # base purple  (family base)

    # --- "freq_response" family (cyans) ----------------------------------------
    ("weight_dist-ind-noise", "#3A8295"),  # darkest cyan
    ("asfadsfsadfasdf", "#4F9CB0"),  # dark cyan
    ("asfadsfsadfasdf",      "#64B5CD"),  # base cyan    (family base)

    # --- standalone / unrelated patterns (no close siblings yet) ---------------
    ("baseline",                "#8C8C8C"),  # grey  — unmodified baseline runs
    ("symmetric",           "#CCB974"),  # gold  — ablation studies
    ("symmetric_with_fields",         "#BEB392"),  # brown — curriculum learning variants
    ("asfadsfsadfasdf",        "#DA8BC3"),  # pink  — multi-agent setups
]

# Cache of name -> color so the fallback palette assignment for unmatched
# names stays stable across multiple render_comparison() calls (i.e. across
# different charts / run-count buckets), not just within one figure.
_color_cache: dict[str, str] = {}
_fallback_counter = 0


def get_color_for_solution(name: str) -> str:
    """Return a color for a solution name. Names containing a pattern from
    PATTERN_COLORS get that pattern's fixed color (longest matching pattern
    wins on overlaps). Names matching nothing fall back to COLOR_PALETTE,
    cycled in first-seen order and cached so the same unmatched name keeps
    the same fallback color across all charts in this run."""
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

# Subplot definitions (must match plot_training.py)  ────────────────────────────
PANELS = [
    {"title": "Episode Reward",  "cols": ["score"]},
    {"title": "Reward Rate",     "cols": ["reward_rate", "gamma_disc_reward"]},
    {"title": "TD Error |mean|", "cols": ["avg_abs_td_error"]},
    {"title": "Voltage",         "cols": ["voltage"]},
    {"title": "Voltage Loss",    "cols": ["voltage_loss"]},
    {"title": "Frequency (Hz)",       "cols": ["frequency"]},
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def window_rolling_mean_std(ep: np.ndarray, val: np.ndarray, window: int):
    """Aggregate one run's (episode, value) pairs into a windowed rolling
    mean ± std (within each fixed-size episode window).
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
    std  = np.nan_to_num(std, nan=0.0)  # single-point windows → NaN std → 0
    return x_centers, mean, std

def extract_chart_tag(stem: str) -> tuple[str, str]:
    """Pull a '[chart title]' tag out of a filename stem, wherever it
    appears. Returns (chart_key, stem_without_tag). If no tag is present,
    chart_key is MISC_CHART_KEY and the stem is returned unchanged."""
    m = CHART_TAG_RE.search(stem)
    if not m:
        return MISC_CHART_KEY, stem
    chart_key = m.group(1).strip()
    # remove the tag (and any doubled-up whitespace it leaves behind)
    cleaned = (stem[:m.start()] + stem[m.end():])
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return chart_key, cleaned


def group_files(files: list[str]) -> dict[str, dict[str, list[str]]]:
    """Group CSV paths first by chart tag ('[TEXT]', anywhere in the
    filename), then by solution name (the stem up to a trailing '(N)',
    with the chart tag removed). Returns {chart_key: {group_key: [paths]}}."""
    charts: dict[str, dict[str, list[str]]] = {}
    for f in files:
        raw_stem = os.path.splitext(os.path.basename(f))[0]
        chart_key, stem = extract_chart_tag(raw_stem)
        m = GROUP_SUFFIX_RE.match(stem)
        group_key = m.group(1) if m else stem
        charts.setdefault(chart_key, {}).setdefault(group_key, []).append(f)
    for chart_key in charts:
        for group_key in charts[chart_key]:
            charts[chart_key][group_key] = sorted(charts[chart_key][group_key])
    return charts


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
def render_comparison(chart_key: str, n_runs: int, solutions: list[tuple[str, list[str]]]):
    chart_slug = re.sub(r"[^A-Za-z0-9]+", "_", chart_key).strip("_").lower()
    out_png = os.path.join(OUT_DIR, f"comparison_{chart_slug}_{n_runs}_runs.png")

    # pre-load each solution's dataframes once
    loaded = [(key, [pd.read_csv(p) for p in paths]) for key, paths in solutions]

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    base_title = "Snake 5 × 5"
    run_label  = "1 run" if n_runs == 1 else f"{n_runs} runs"
    full_title = (base_title if chart_key == MISC_CHART_KEY else f"{base_title} - {chart_key}")
    full_title += f"  [{run_label} per solution]"
    fig.suptitle(full_title, fontsize=13, fontweight="bold", y=1.02)
    axs = axs.flatten()

    legend_handles, legend_labels = [], []
    panel_has_data = [False] * len(PANELS)

    # ── One pass per solution; branch on run count inside each panel ──────────
    for sol_idx, (group_key, dfs) in enumerate(loaded):
        color = get_color_for_solution(group_key)
        label = group_key.replace("-", " ").replace("_", " ")

        for idx, panel in enumerate(PANELS):
            ax = axs[idx]
            col = find_col(dfs[0], panel["cols"])
            if col is None:
                continue

            if n_runs == 1:
                # ── Single-run branch: within-window mean ± std ──────────────
                df  = dfs[0]
                ep  = pd.to_numeric(df["episode"], errors="coerce").to_numpy(dtype=np.float64)
                val = pd.to_numeric(df[col],       errors="coerce").to_numpy(dtype=np.float64)
                if col == "frequency":
                    val = val * 1000
                agg = window_rolling_mean_std(ep, val, WINDOW_EP)
                if agg is None:
                    continue
                x, mean, std = agg
                panel_has_data[idx] = True
                ax.fill_between(x, mean - std, mean + std,
                                color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
                line, = ax.plot(x, mean, color=color, linewidth=2.0, zorder=3)

            else:
                # ── Multi-run branch: across-run mean ± std + min/max band ───
                per_run_series = []
                for df in dfs:
                    df_col = find_col(df, panel["cols"])
                    if df_col is None:
                        continue
                    ep  = pd.to_numeric(df["episode"], errors="coerce").to_numpy(dtype=np.float64)
                    val = pd.to_numeric(df[df_col],    errors="coerce").to_numpy(dtype=np.float64)
                    if df_col == "frequency":
                        val = val * 1000
                    per_run_series.append(per_run_window_means(ep, val, WINDOW_EP))

                agg = aggregate_runs_by_window(per_run_series, WINDOW_EP)
                if agg is None:
                    continue
                x, mean, std, vmin, vmax = agg
                panel_has_data[idx] = True
                ax.fill_between(x, vmin, vmax,
                                color=color, alpha=MINMAX_ALPHA, linewidth=0, zorder=1)
                ax.fill_between(x, mean - std, mean + std,
                                color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
                line, = ax.plot(x, mean, color=color, linewidth=2.0, zorder=3)

            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(label)

    # ── Finishing loop (THIS is what was missing → restores titles/axes) ─────
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

    charts = group_files(files)

    os.makedirs(OUT_DIR, exist_ok=True)

    any_rendered = False
    for chart_key in sorted(charts.keys()):
        groups = charts[chart_key]

        # Include ALL groups now (single-run and multi-run alike)
        multi_run = {key: paths for key, paths in groups.items()}
        if not multi_run:
            continue

        # bucket this chart's solutions by their run count
        buckets: dict[int, list[tuple[str, list[str]]]] = {}
        for key, paths in multi_run.items():
            buckets.setdefault(len(paths), []).append((key, paths))

        for n_runs, solutions in sorted(buckets.items()):
            if len(solutions) < 2:
                print(f"  -  [{chart_key}] skipping n={n_runs}-run bucket "
                      f"(only 1 solution: {solutions[0][0]!r}, nothing to compare)")
                continue
            names = ", ".join(s[0] for s in solutions)
            print(f"[{chart_key}] Comparing {len(solutions)} solutions with "
                  f"{n_runs} runs each: {names}")
            render_comparison(chart_key, n_runs, solutions)
            any_rendered = True

    if any_rendered:
        print(f"\nAll comparisons saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()