"""
Analyze one or more continuous value-vs-return traces.

Each CSV is one uninterrupted stream with columns:
    timestep, value, reward_trace

For every file, G_t is computed with one backward pass over that file.
Only ONE input CSV is held in memory at a time.

The per-file metrics use the same windowing as the original continuous
analysis:
  - NMSE: median of per-row normalized squared errors in each window.
  - corr(lag0): Pearson r in each window.

Files are grouped by solution name the same way compare_solutions.py
groups multi-run CSVs: the filename stem (optionally minus a trailing
"[chart tag]") up to a trailing "(N)" is the group key. Aggregation is
performed independently PER GROUP -- runs belonging to different
solutions are never pooled together into the same aggregate. Within a
group, per-window metrics are aggregated like the comparison charts:
  - bold mean across the group's files,
  - mean +/- 1 std band across the group's files,
  - min/max (best/worst run) band across the group's files.

The aggregation is streaming: mean/std use a file counter and Welford's
online algorithm, while min/max are updated in-place. Per-file metric
arrays are discarded before the next CSV is loaded, so memory usage is
approximately that of one input file plus the small aggregate arrays for
the group currently being processed.

GREEDY LENGTH ASSUMPTION: rather than doing a full first pass (e.g.
`wc -l`) over every file to find the true minimum row count before
aggregating, files within a group are sorted by byte size ascending
(a cheap os.path.getsize() stat, not a read), and the smallest-by-size
file is assumed to also have the fewest rows/windows -- its window
count becomes the fixed common length for the whole group, with no
first pass and no aggregator resizing. This holds for CSVs with
roughly uniform row width, which is the case here. See the docstring
on aggregate_trace_files() for what happens if that assumption is
violated.
"""

import argparse
import csv
import glob
import os
import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Filename/solution -> color mapping
# ---------------------------------------------------------------------------

# Assign a fixed color to any solution/filename containing a given substring.
# The longest matching pattern wins, so more specific variants override
# shorter family-level patterns.
PATTERN_COLORS: list[tuple[str, str]] = [
    # --- "weight_dist" family (blues) ---------------------------------------
    ("weight_dist_prop",   "#1F4E79"),  # darkest blue  (most specific variant)
    ("weight_prop_dist",   "#1F4E79"),  # darkest blue  (most specific variant)
    ("weight_dist_uniform","#2F5C8A"),  # dark blue
    ("weight_dist_ind_noise","#367BB8"),
    ("weight_dist_ada_belief","#5E07FF"),

    ("weight_dist",        "#5E07FF"),  # base blue     (family base / fallback within family)
    ("weight_dist_no_psi",        "#A6B7D3"),  # base blue     (family base / fallback within family)
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

COLOR_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#64B5CD", "#CCB974", "#8C8C8C",
]

_color_cache: dict[str, str] = {}
_fallback_counter = 0


def get_color_for_solution(name: str) -> str:
    """Return a stable color based on the solution/file name (longest substring match wins)."""
    global _fallback_counter
    if name in _color_cache:
        return _color_cache[name]

    name_lower = name.lower()
    best_pattern, best_color = None, None
    for pattern, color in PATTERN_COLORS:
        pattern_lower = pattern.lower()
        if pattern_lower in name_lower and (best_pattern is None or len(pattern_lower) > len(best_pattern)):
            best_pattern, best_color = pattern_lower, color

    if best_color is not None:
        _color_cache[name] = best_color
        return best_color

    color = COLOR_PALETTE[_fallback_counter % len(COLOR_PALETTE)]
    _fallback_counter += 1
    _color_cache[name] = color
    return color


# ---------------------------------------------------------------------------
# Grouping by solution name (mirrors compare_solutions.py)
# ---------------------------------------------------------------------------

# Matches a trailing "(<number>)" right before the extension, e.g.
# "pacman_training_weight_dist_prop_lambda_05(3)" -> "pacman_training_weight_dist_prop_lambda_05"
GROUP_SUFFIX_RE = re.compile(r"^(.*)\(\d+\)$")

# Matches a "[<chart title>]" tag anywhere in the filename stem, e.g.
# "pacman_training[Lambda Sweep](3)" -> tag "Lambda Sweep". Stripped out of
# the group key so it doesn't split one solution's runs into two groups.
CHART_TAG_RE = re.compile(r"\[([^\[\]]+)\]")


def solution_name_from_path(path: str) -> str:
    """Use the filename (without extension) as the raw solution name."""
    return os.path.splitext(os.path.basename(path))[0]


def group_key_from_path(path: str) -> str:
    """Derive the solution/group key for a trace file path.

    Strips any trailing "(N)" run-count suffix and any "[chart tag]",
    the same way compare_solutions.py's group_files() does, so that
    e.g. "weight_dist_prop(1)" and "weight_dist_prop(2)" collapse to
    the same group "weight_dist_prop" while files for different
    solutions never collapse into each other.
    """
    stem = solution_name_from_path(path)

    m = CHART_TAG_RE.search(stem)
    if m:
        stem = (stem[: m.start()] + stem[m.end():])
        stem = re.sub(r"\s{2,}", " ", stem).strip()

    m = GROUP_SUFFIX_RE.match(stem)
    if m:
        stem = m.group(1)

    return stem


def group_trace_paths(paths: list[str]) -> "dict[str, list[str]]":
    """Group trace file paths by solution name. Order of paths within
    each group, and of groups themselves, follows first-seen order."""
    groups: "dict[str, list[str]]" = {}
    for p in paths:
        key = group_key_from_path(p)
        groups.setdefault(key, []).append(p)
    return groups


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_trace_csv(path):
    """Load exactly one continuous trace CSV."""
    df = pd.read_csv(path)
    required = {"timestep", "value", "reward_trace"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")

    df = df.sort_values("timestep").head(int(30e6)).reset_index(drop=True)
    return (
        df["timestep"].to_numpy(dtype=np.int64),
        df["value"].to_numpy(dtype=np.float64),
        df["reward_trace"].to_numpy(dtype=np.float64),
    )


def discounted_return(reward_trace, gamma_step):
    """Backward cumulative sum over the WHOLE file: G_t = r_t + gamma_step * G_{t+1}."""
    G = np.zeros_like(reward_trace)
    running = 0.0
    for t in range(len(reward_trace) - 1, -1, -1):
        running = reward_trace[t] + gamma_step * running
        G[t] = running
    return G


# ---------------------------------------------------------------------------
# Per-file window metrics
# ---------------------------------------------------------------------------

def nmse_median(v, g, var_g_global):
    """Median per-row normalized squared error for one window: (v-g)^2 / var(g).

    Uses the GLOBAL variance of G over the whole file (passed in), not the
    variance within this window. Per-window variance can be near-zero when
    the return is locally flat, which blows up the ratio into huge spikes
    that don't reflect a real value/return mismatch.
    """
    if var_g_global < 1e-12:
        return np.nan
    return np.median((v - g) ** 2 / var_g_global)


def corr0(v, g):
    """Pearson corr(V, G) for one window."""
    if len(v) < 3 or np.std(v) < 1e-12 or np.std(g) < 1e-12:
        return np.nan
    return np.corrcoef(v, g)[0, 1]


def window_metrics(timesteps, v, g, window_size):
    """Compute the original continuous-analysis metric once per non-overlapping window."""
    n = len(v)
    if n < window_size:
        raise ValueError(f"Trace has only {n} rows, shorter than --window-size={window_size}")

    n_windows = n // window_size
    starts = range(0, n_windows * window_size, window_size)

    # Normalize NMSE by the return's GLOBAL variance (whole file), computed
    # once, rather than a fresh per-window variance. See nmse_median().
    var_g_global = np.var(g)

    x = np.empty(n_windows, dtype=np.float64)
    nmse = np.full(n_windows, np.nan, dtype=np.float64)
    corr = np.full(n_windows, np.nan, dtype=np.float64)

    for i, start in enumerate(starts):
        end = start + window_size
        x[i] = timesteps[(start + end) // 2]
        v_win, g_win = v[start:end], g[start:end]
        nmse[i] = nmse_median(v_win, g_win, var_g_global)
        corr[i] = corr0(v_win, g_win)

    return x, nmse, corr

# ---------------------------------------------------------------------------
# Streaming cross-file aggregation
# ---------------------------------------------------------------------------

class StreamingMetricAggregator:
    """
    Aggregate per-window values from files without retaining all files.

    Per window index: count, online mean/M2 (Welford), and running min/max.
    NaNs are ignored per window.
    """

    def __init__(self, n_windows):
        self.count = np.zeros(n_windows, dtype=np.int64)
        self.mean = np.zeros(n_windows, dtype=np.float64)
        self.m2 = np.zeros(n_windows, dtype=np.float64)
        self.min = np.full(n_windows, np.inf, dtype=np.float64)
        self.max = np.full(n_windows, -np.inf, dtype=np.float64)

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if len(values) != len(self.mean):
            raise ValueError(
                f"Metric has {len(values)} windows but aggregator expects {len(self.mean)}"
            )

        idx = np.flatnonzero(np.isfinite(values))
        if len(idx) == 0:
            return

        x = values[idx]
        new_count = self.count[idx] + 1
        delta = x - self.mean[idx]
        self.mean[idx] += delta / new_count
        delta2 = x - self.mean[idx]
        self.m2[idx] += delta * delta2
        self.count[idx] = new_count

        self.min[idx] = np.minimum(self.min[idx], x)
        self.max[idx] = np.maximum(self.max[idx], x)

    def finalize(self):
        """Return mean, population std (ddof=0), min, max, and valid-file count."""
        mean = np.full_like(self.mean, np.nan)
        std = np.full_like(self.mean, np.nan)
        lo = np.full_like(self.mean, np.nan)
        hi = np.full_like(self.mean, np.nan)

        valid = self.count > 0
        mean[valid] = self.mean[valid]

        more_than_one = self.count > 1
        std[more_than_one] = np.sqrt(self.m2[more_than_one] / self.count[more_than_one])
        std[self.count == 1] = 0.0

        lo[valid] = self.min[valid]
        hi[valid] = self.max[valid]

        return mean, std, lo, hi, self.count.copy()


def expand_trace_paths(trace_paths):
    """Accept paths and/or glob patterns; de-duplicate while preserving order."""
    expanded = []
    for item in trace_paths:
        matches = sorted(glob.glob(item))
        expanded.extend(matches if matches else [item])
    return list(dict.fromkeys(expanded))


def aggregate_trace_files(paths, gamma_step, window_size):
    """
    Process paths belonging to a SINGLE solution/group strictly one at a
    time, one file in memory at any point.

    GREEDY SIZE ASSUMPTION: instead of doing a `wc -l` first pass over
    every file to find the true minimum row count, we sort paths by byte
    size ascending and assume the smallest file (by bytes) also has the
    fewest rows/windows. For CSVs with roughly uniform row width (as here:
    timestep, value, reward_trace are all fixed-ish-width numeric fields)
    this is true in practice, and `os.path.getsize()` is an O(1) stat
    call rather than a full scan of every file, so it avoids the second
    full read entirely. We do NOT verify this assumption at runtime and
    we do NOT dynamically shrink the aggregator if a later file turns out
    shorter than expected -- that truncation logic was tried and made the
    hot path (a same-length elementwise `update()` on every file) pay a
    conditional-resize cost on every iteration, which is worse in
    aggregate than the rare case this guards against. If byte size and
    row count are decorrelated for your data (e.g. wildly different value
    magnitudes changing field width), widen --window-size margins or
    pre-sort your own file list accordingly; a file longer than assumed
    is simply clipped to the first file's window count, and a file
    SHORTER than assumed will raise, since there's nothing to safely
    clip to.
    """
    if not paths:
        raise ValueError("No trace files were found.")

    # O(1) stat per file, no file contents are read here.
    paths_by_size = sorted(paths, key=lambda p: os.path.getsize(p))

    nmse_agg = None
    corr_agg = None
    reference_x = None
    n_files = 0
    common_n_windows = None

    for path in paths_by_size:
        print(f"\n[{n_files + 1}/{len(paths_by_size)}] Loading {path} "
              f"({os.path.getsize(path):,} bytes) ...")
        timesteps, v, reward_trace = load_trace_csv(path)

        if len(timesteps) == 0:
            print("  Skipping empty file.")
            del timesteps, v, reward_trace
            continue

        print(f"  {len(timesteps):,} rows, timestep range [{timesteps[0]:,}, {timesteps[-1]:,}]")

        if common_n_windows is None:
            # First file processed = smallest by byte size = assumed to
            # have the fewest windows. This fixes the aggregator size for
            # the rest of the group; see the greedy-assumption note above.
            common_n_windows = len(timesteps) // window_size
            if common_n_windows == 0:
                raise ValueError(
                    f"The smallest-by-size trace ({path}) has only {len(timesteps)} rows, "
                    f"shorter than --window-size={window_size}."
                )
            print(
                f"  Using this file's {common_n_windows:,} complete windows "
                f"({common_n_windows * window_size:,} rows) as the common length "
                f"for the rest of this group (greedy: smallest-by-size assumed shortest)."
            )
            nmse_agg = StreamingMetricAggregator(common_n_windows)
            corr_agg = StreamingMetricAggregator(common_n_windows)

        n_rows_to_use = common_n_windows * window_size
        if len(timesteps) < n_rows_to_use:
            raise ValueError(
                f"{path} has {len(timesteps)} rows, fewer than the "
                f"{n_rows_to_use} assumed common rows from the smallest-by-size file. "
                f"The greedy size-based ordering assumption doesn't hold for this group -- "
                f"see the note in aggregate_trace_files()."
            )

        timesteps, v, reward_trace = (
            timesteps[:n_rows_to_use],
            v[:n_rows_to_use],
            reward_trace[:n_rows_to_use],
        )

        print("  Computing discounted return G_t ...")
        g = discounted_return(reward_trace, gamma_step)

        print(f"  Computing per-window metrics (window={window_size:,}) ...")
        x, nmse, corr = window_metrics(timesteps, v, g, window_size)

        if reference_x is None:
            reference_x = x.copy()

        nmse_agg.update(nmse)
        corr_agg.update(corr)
        n_files += 1
        print(f"  Aggregated {n_files} file(s).")

        del timesteps, v, reward_trace, g, x, nmse, corr

    if n_files == 0:
        raise ValueError("No non-empty trace files were processed.")

    return reference_x, nmse_agg, corr_agg, n_files


# ---------------------------------------------------------------------------
# Plot smoothing
# ---------------------------------------------------------------------------

def smooth_series(values, window):
    """Centered moving average for plotting only. NaNs are excluded from each window's average."""
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(values) == 0:
        return values.copy()

    window = min(window, len(values))
    kernel = np.ones(window, dtype=np.float64)
    valid = np.isfinite(values)

    summed = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
    counts = np.convolve(valid.astype(np.float64), kernel, mode="same")

    out = np.full_like(values, np.nan)
    np.divide(summed, counts, out=out, where=counts > 0)
    return out


def rolling_extrema(values, window):
    """
    Centered rolling min/max for plotting only (as opposed to a moving
    average). Used for the correlation chart's best/worst-run band, so the
    band reflects the min/max over the smoothing window rather than a
    smoothed average of the per-run extremes.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if window <= 1 or n == 0:
        return values.copy(), values.copy()

    window = min(window, n)
    half = window // 2

    lo_out = np.full(n, np.nan)
    hi_out = np.full(n, np.nan)
    for i in range(n):
        chunk = values[max(0, i - half): min(n, i + (window - half))]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk) > 0:
            lo_out[i] = np.min(chunk)
            hi_out[i] = np.max(chunk)
    return lo_out, hi_out


def smooth_for_plot(x, *series, window):
    """Apply the same centered moving-average smoothing to x and all series."""
    if window <= 1:
        return (x,) + series
    return (smooth_series(x, window),) + tuple(smooth_series(s, window) for s in series)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_summary_csv(path, x, nmse, corr, n_files):
    nmse_mean, nmse_std, nmse_min, nmse_max, nmse_count = nmse.finalize()
    corr_mean, corr_std, corr_min, corr_max, corr_count = corr.finalize()

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window_center_timestep",
            "nmse_mean", "nmse_std", "nmse_min_run", "nmse_max_run", "nmse_n_runs",
            "corr0_mean", "corr0_std", "corr0_min_run", "corr0_max_run", "corr0_n_runs",
        ])
        for i in range(len(x)):
            writer.writerow([
                x[i],
                nmse_mean[i], nmse_std[i], nmse_min[i], nmse_max[i], nmse_count[i],
                corr_mean[i], corr_std[i], corr_min[i], corr_max[i], corr_count[i],
            ])

    return (
        nmse_mean, nmse_std, nmse_min, nmse_max,
        corr_mean, corr_std, corr_min, corr_max,
    )


def plot_comparison_metrics(out_path, group_results, smooth_window):
    """Overlay every solution group's mean / ±1std / min-max bands on the
    SAME pair of axes (NMSE panel + corr panel), like compare_solutions.py
    does for its panels. group_results is a list of dicts, one per group,
    each holding that group's x/metric arrays, n_files, and name."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    BAND_ALPHA, MINMAX_ALPHA = 0.25, 0.10

    def finish_ax(ax, title, ylabel):
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("Timestep", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="grey", alpha=0.2, linewidth=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    legend_handles, legend_labels = [], []

    for g in group_results:
        color = get_color_for_solution(g["name"])
        label = g["name"].replace("-", " ").replace("_", " ")

        # ---- smooth this group's series for plotting ----
        corr_min_raw, corr_max_raw = g["corr_min"], g["corr_max"]
        x, nmse_mean, nmse_std, nmse_min, nmse_max, corr_mean, corr_std = smooth_for_plot(
            g["x"], g["nmse_mean"], g["nmse_std"], g["nmse_min"], g["nmse_max"],
            g["corr_mean"], g["corr_std"],
            window=smooth_window,
        )
        corr_min, _ = rolling_extrema(corr_min_raw, smooth_window)
        _, corr_max = rolling_extrema(corr_max_raw, smooth_window)

        # ---- NMSE panel ----
        valid = np.isfinite(nmse_mean)
        axes[0].fill_between(x, nmse_min, nmse_max, where=valid, color=color,
                              alpha=MINMAX_ALPHA, zorder=1, linewidth=0)
        axes[0].fill_between(x, nmse_mean - nmse_std, nmse_mean + nmse_std, where=valid,
                              color=color, alpha=BAND_ALPHA, zorder=2, linewidth=0)
        line, = axes[0].plot(x, nmse_mean, color=color, linewidth=0.5, zorder=3)

        # ---- corr panel ----
        valid_c = np.isfinite(corr_mean)
        axes[1].fill_between(x, corr_min, corr_max, where=valid_c, color=color,
                              alpha=MINMAX_ALPHA, zorder=1, linewidth=0)
        axes[1].fill_between(x, corr_mean - corr_std, corr_mean + corr_std, where=valid_c,
                              color=color, alpha=BAND_ALPHA, zorder=2, linewidth=0)
        axes[1].plot(x, corr_mean, color=color, linewidth=0.5, zorder=3)

        legend_handles.append(line)
        legend_labels.append(f"{label} ({g['n_files']} runs)")

    finish_ax(
        axes[0],
        f"Normalized MSE(V, G) (smooth={smooth_window})",
        "NMSE (lower = closer)",
    )
    finish_ax(
        axes[1],
        f"Correlation(V, G) at lag 0 (smooth={smooth_window})",
        "Pearson r",
    )
    # Pearson correlation is capped at 1.0, so keep that as a fixed ceiling;
    # the lower bound auto-scales to the data instead of being pinned to -1.0.
    axes[1].set_ylim(top=1.0)

    fig.suptitle("Value vs. Return — solution comparison", fontsize=13, fontweight="bold", y=1.03)
    fig.legend(legend_handles, legend_labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=min(len(legend_labels), 4),
               fontsize=9, frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_group(group_key, group_paths, args):
    """Run the aggregate -> CSV pipeline for ONE solution group and
    return its results (for later combined plotting). Groups are always
    aggregated independently; runs from different groups are never
    pooled into the same aggregation."""
    print(f"\n{'=' * 70}")
    print(f"Solution group: {group_key!r}  ({len(group_paths)} file(s))")
    print(f"{'=' * 70}")

    x, nmse_agg, corr_agg, n_files = aggregate_trace_files(
        group_paths, args.gamma_step, args.window_size
    )

    group_slug = re.sub(r"[^A-Za-z0-9]+", "_", group_key).strip("_").lower() or "group"
    summary_path = os.path.join(args.out_dir, f"value_return_summary_{group_slug}.csv")

    (
        nmse_mean, nmse_std, nmse_min, nmse_max,
        corr_mean, corr_std, corr_min, corr_max,
    ) = write_summary_csv(summary_path, x, nmse_agg, corr_agg, n_files)

    print(f"\n[{group_key}] Aggregated {n_files} file(s), {len(x)} complete windows.")
    print(
        f"\n{'timestep':>14}"
        f"{'nmse_mean':>13}{'nmse_std':>11}{'nmse_min':>11}{'nmse_max':>11}"
        f"{'corr_mean':>11}{'corr_std':>10}{'corr_min':>10}{'corr_max':>10}"
    )
    for i in range(len(x)):
        print(
            f"{x[i]:>14,.0f}"
            f"{nmse_mean[i]:>13.4f}{nmse_std[i]:>11.4f}{nmse_min[i]:>11.4f}{nmse_max[i]:>11.4f}"
            f"{corr_mean[i]:>11.4f}{corr_std[i]:>10.4f}{corr_min[i]:>10.4f}{corr_max[i]:>10.4f}"
        )

    print(f"\n[{group_key}] Saved aggregate summary CSV to {summary_path}")

    return {
        "name": group_key,
        "x": x,
        "n_files": n_files,
        "nmse_mean": nmse_mean, "nmse_std": nmse_std, "nmse_min": nmse_min, "nmse_max": nmse_max,
        "corr_mean": corr_mean, "corr_std": corr_std, "corr_min": corr_min, "corr_max": corr_max,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace-path", nargs="+", required=True,
                         help="one or more continuous trace CSV paths or glob patterns")
    parser.add_argument("--gamma-step", type=float, required=True,
                         help="per-timestep discount, e.g. 0.5**(1/WAIT_INC)")
    parser.add_argument("--window-size", type=int, default=2000,
                         help="rows per non-overlapping metric window")
    parser.add_argument("--smooth-window", type=int, default=1,
                         help="centered moving-average window, in plotted metric points (1 disables smoothing)")
    parser.add_argument("--out-dir", type=str, default="outputs/analysis_continuous")
    args = parser.parse_args()

    if args.window_size <= 0:
        raise ValueError("--window-size must be > 0")
    if args.smooth_window <= 0:
        raise ValueError("--smooth-window must be > 0")
    if not (0.0 < args.gamma_step <= 1.0):
        raise ValueError("--gamma-step must be in (0, 1]")

    os.makedirs(args.out_dir, exist_ok=True)

    paths = expand_trace_paths(args.trace_path)
    if not paths:
        raise ValueError("No trace files matched the supplied --trace-path.")

    print(f"Found {len(paths)} trace file(s).")
    print("Only one trace is loaded into memory at a time.")

    groups = group_trace_paths(paths)
    print(f"Grouped into {len(groups)} solution(s): {list(groups.keys())}")

    group_results = []
    for group_key, group_paths in groups.items():
        group_results.append(process_group(group_key, group_paths, args))

    print(f"\nAll {len(groups)} solution group(s) aggregated. Outputs written to {args.out_dir}")

    try:
        plot_path = os.path.join(args.out_dir, "value_return_comparison.png")
        plot_comparison_metrics(plot_path, group_results, args.smooth_window)
        print(f"Saved comparison plot (all {len(groups)} solution(s) overlaid) to {plot_path}")
    except ImportError:
        print("matplotlib not available -- skipping comparison plot, per-group CSV summaries still saved.")


if __name__ == "__main__":
    main()