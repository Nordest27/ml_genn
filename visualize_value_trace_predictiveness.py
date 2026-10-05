"""
Analyze value-vs-return traces saved every TRACE_SAVE_EVERY episodes.

For each checkpoint CSV (columns: timestep, value, reward_trace):
  1. Compute G_t, the per-timestep discounted return, via the backward
     cumulative sum G_t = reward_trace_t + gamma_step * G_{t+1}.
  2. Compare V_t and G_t two ways:
       - squared difference (MSE, and normalized MSE / NMSE)
       - Pearson correlation, both at lag 0 and swept over a small
         range of lags, so we can see whether V leads G (anticipation).
  3. Plot both metrics across training (one point per checkpoint episode),
     for one or more solutions/rules, so e.g. distributed vs. adaptive
     e-prop can be compared directly.

Folder layout expected under --traces-dir (mirrors compare_solutions.py's
CSV grouping, but one level up -- folders of checkpoints instead of files):

    value_traces/
        distributed(1)/ep0.csv, ep1000.csv, ep2000.csv, ...
        distributed(2)/ep0.csv, ep1000.csv, ep2000.csv, ...
        adaptive(1)/ep0.csv, ...
        adaptive(2)/ep0.csv, ...

Folders are grouped into "solutions" by stripping a trailing "(N)"
run-index suffix -- the exact same GROUP_SUFFIX_RE convention
compare_solutions.py uses for its CSV filenames. Every run-folder
belonging to a solution is analyzed individually (via analyze_run) and
then the per-checkpoint metrics (nmse, corr0, peak_corr) are aggregated
across those runs using the SAME windowed-bucket aggregation
compare_solutions.py uses for its training-curve panels: each run's
(episode, metric) pairs are floor-divided into fixed-size episode windows
(--window-ep, matching compare_solutions.py's WINDOW_EP) via
per_run_window_means, then combined across runs into a per-window mean
+/- std plus a best/worst-run band via aggregate_runs_by_window -- both
copied verbatim from compare_solutions.py so the two scripts aggregate
identically and can be told apart only by which metric is on the y-axis.
Solutions are colored using the same name-pattern -> color scheme
(PATTERN_COLORS / get_color_for_solution) so a given solution keeps a
consistent color across both scripts' figures.

A single-run solution (only one "name(1)"-style folder, or a bare folder
with no "(N)" suffix at all) is still supported -- it just skips the
std/min-max bands and plots a single line, same as compare_solutions.py's
single-run branch (which uses a within-run windowed mean +/- std instead
of a cross-run band).

Usage:
    python visualize_value_trace_predictiveness.py \
        --traces-dir outputs/complete_value_traces \
        --gamma-step 0.9772372209558107 \
        --max-lag 20 \
        --out-dir outputs/analysis

Notes on gamma:
  Training uses an action-level gamma=0.5 (Table 2), converted to a
  per-simulation-timestep decay via gamma_step = gamma ** (1/WAIT_INC).
  With WAIT_INC=30 that's gamma_step = 0.5**(1/30) ~= 0.9772. Pass the
  actual per-step value you trained with via --gamma-step.
"""
import argparse
import glob
import os
import re
import csv
import numpy as np
import pandas as pd


# ── Per-run trace analysis (unchanged from the single-run version) ─────────

def load_trace_csv(path):
    """Load a single (timestep, value, reward_trace) checkpoint CSV."""
    values, rewards = [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row["value"]))
            rewards.append(float(row["reward_trace"]))
    return np.array(values, dtype=np.float64), np.array(rewards, dtype=np.float64)


def discounted_return(reward_trace, gamma_step):
    """
    Backward cumulative sum: G_t = reward_trace_t + gamma_step * G_{t+1}.
    Exact up to the truncation at the end of the logged episode (negligible
    given gamma_step's effective horizon relative to a full episode length).
    """
    G = np.zeros_like(reward_trace)
    running = 0.0
    for t in range(len(reward_trace) - 1, -1, -1):
        running = reward_trace[t] + gamma_step * running
        G[t] = running
    return G


def normalized_mse(v, g):
    """MSE normalized by variance of G, so it's comparable across
    checkpoints/runs whose reward/value scale may drift over training."""
    mse = np.mean((v - g) ** 2)
    var_g = np.var(g)
    if var_g < 1e-12:
        return np.nan
    return mse / var_g


def lag_correlation(v, g, max_lag):
    """
    Cross-correlation of V_t against G_{t+lag} for lag in [-max_lag, max_lag].
    Positive lag means G is shifted forward in time relative to V, i.e. we're
    testing whether V_t correlates with a *future* value of G (V leading G,
    which is NOT what we want -- see convention note below).

    Convention used here:
        corr(lag) = Pearson(V[t], G[t + lag])  for valid overlapping t
    lag > 0  => compares V_t to a LATER G, i.e. V evaluated before G's window
                has fully unfolded -- this is not usually what you want.
    lag < 0  => compares V_t to an EARLIER G, i.e. checks whether V_t echoes
                a G that already happened (lagging behind), not anticipation.

    For "does V anticipate the future reward stream" in the sense of Fig. 5
    (value rises before reward), the relevant comparison is actually V_t vs.
    G_t at lag 0, since G_t already *is* the forward-looking return from t
    onward by construction. The lag sweep here instead answers a *secondary*
    question: at what shift does the overall SHAPE of the two traces align
    best, useful for sanity-checking timing/offset artifacts (e.g. one-step
    bookkeeping offsets between when V and reward_trace are appended in the
    logging loop). Treat lag=0 as the primary number; use the sweep as a
    diagnostic, not the headline metric.
    """
    n = len(v)
    lags = range(-max_lag, max_lag + 1)
    correlations = []
    for lag in lags:
        if lag >= 0:
            v_seg = v[: n - lag] if lag > 0 else v
            g_seg = g[lag:]
        else:
            v_seg = v[-lag:]
            g_seg = g[: n + lag]
        if len(v_seg) < 2 or np.std(v_seg) < 1e-12 or np.std(g_seg) < 1e-12:
            correlations.append(np.nan)
            continue
        corr = np.corrcoef(v_seg, g_seg)[0, 1]
        correlations.append(corr)
    return np.array(list(lags)), np.array(correlations)


def parse_episode_from_filename(path):
    m = re.search(r"ep(\d+)\.csv$", os.path.basename(path))
    if m is None:
        raise ValueError(f"Could not parse episode number from {path}")
    return int(m.group(1))


def analyze_run(trace_dir, gamma_step, max_lag):
    """
    Analyze a single run-folder of ep*.csv checkpoints. Returns a dict:
        episodes:      sorted array of checkpoint episode numbers
        nmse:          normalized MSE(V, G) at lag 0, per checkpoint
        corr0:         Pearson corr(V, G) at lag 0, per checkpoint
        peak_lag:      lag (in timesteps) at which correlation peaks, per checkpoint
        peak_corr:     correlation value at that peak lag, per checkpoint
        lag_curves:    list of (lags, correlations) arrays, per checkpoint
                       (kept for optional detailed plotting)
    """
    paths = sorted(glob.glob(os.path.join(trace_dir, "ep*.csv")),
                   key=parse_episode_from_filename)
    if not paths:
        raise FileNotFoundError(f"No ep*.csv files found under {trace_dir}")

    episodes, nmse_list, corr0_list, peak_lag_list, peak_corr_list = [], [], [], [], []
    lag_curves = []

    for path in paths:
        ep = parse_episode_from_filename(path)
        v, r = load_trace_csv(path)
        if len(v) < 3:
            continue  # too short to be meaningful
        g = discounted_return(r, gamma_step)

        episodes.append(ep)
        nmse_list.append(normalized_mse(v, g))

        lags, corrs = lag_correlation(v, g, max_lag)
        lag_curves.append((lags, corrs))

        # lag=0 is at index max_lag in the `lags`/`corrs` arrays
        zero_idx = max_lag
        corr0_list.append(corrs[zero_idx])

        if np.all(np.isnan(corrs)):
            peak_lag_list.append(np.nan)
            peak_corr_list.append(np.nan)
        else:
            best_idx = np.nanargmax(corrs)
            peak_lag_list.append(lags[best_idx])
            peak_corr_list.append(corrs[best_idx])

    return {
        "episodes": np.array(episodes),
        "nmse": np.array(nmse_list),
        "corr0": np.array(corr0_list),
        "peak_lag": np.array(peak_lag_list),
        "peak_corr": np.array(peak_corr_list),
        "lag_curves": lag_curves,
    }


# ── Folder grouping + coloring, mirroring compare_solutions.py ─────────────

# Matches a trailing "(<number>)" right before the end of the folder name,
# e.g. "distributed(2)" -> "distributed". Same convention as
# compare_solutions.py's GROUP_SUFFIX_RE, just applied to directory names
# instead of CSV filename stems.
GROUP_SUFFIX_RE = re.compile(r"^(.*)\(\d+\)$")

# Distinct colors assigned to solutions within a comparison figure. Used as
# the fallback palette for solution names that don't match any entry in
# PATTERN_COLORS below (cycled if there are more unmatched names than
# colors). Kept identical to compare_solutions.py so a solution's fallback
# color only depends on first-seen order, same as there.
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
# Copied verbatim from compare_solutions.py so that a solution named e.g.
# "weight_dist_prop(1)" gets the exact same color in both scripts' figures.
# If you edit the palette in one script, mirror the edit in the other (or
# better: factor this block out into a shared module both scripts import).
PATTERN_COLORS: list[tuple[str, str]] = [
    # --- "weight_dist" family (blues) ---------------------------------------
    ("weight_dist_prop",   "#1F4E79"),  # darkest blue  (most specific variant)
    ("weight_prop_dist",   "#1F4E79"),  # darkest blue  (most specific variant)
    ("weight_dist_uniform","#2F5C8A"),  # dark blue
    ("weight_dist_cadam","#1F8CFA"),
    ("weight_dist",        "#4C72B0"),  # base blue     (family base / fallback within family)
    ("weight_dist_no_heuristic",        "#A6B7D3"),  # base blue     (family base / fallback within family)
    
    ("no_async_weight_dist",        "#A6B7D3"),  # base blue     (family base / fallback within family)
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

# Cache of name -> color so fallback palette assignment for unmatched names
# stays stable across the whole run of this script, not just within one
# figure (same pattern as compare_solutions.py).
_color_cache: dict[str, str] = {}
_fallback_counter = 0


def get_color_for_solution(name: str) -> str:
    """Return a color for a solution name. Names containing a pattern from
    PATTERN_COLORS get that pattern's fixed color (longest matching pattern
    wins on overlaps). Names matching nothing fall back to COLOR_PALETTE,
    cycled in first-seen order and cached so the same unmatched name keeps
    the same fallback color for the whole run."""
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


def group_run_dirs(base_dir: str) -> dict[str, list[str]]:
    """Scan base_dir for run subfolders and group them into solutions by
    stripping a trailing "(N)" run-index suffix, exactly like
    compare_solutions.py's group_files() does for CSV filenames -- just one
    level up, since here each "file" is a folder of ep*.csv checkpoints.

    Returns {solution_name: [run_dir_path, ...]} with run dirs sorted so
    "(1)" < "(2)" < ... within a solution."""
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"--traces-dir not found: {base_dir}")

    entries = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )
    if not entries:
        raise FileNotFoundError(f"No run subfolders found under {base_dir}")

    groups: dict[str, list[str]] = {}
    for name in entries:
        m = GROUP_SUFFIX_RE.match(name)
        group_key = m.group(1) if m else name
        groups.setdefault(group_key, []).append(os.path.join(base_dir, name))
    for key in groups:
        groups[key] = sorted(groups[key])
    return groups


# ── Cross-run aggregation (identical convention to compare_solutions.py) ───
# These two helpers are copied verbatim (only renamed for clarity of what
# they're bucketing here: checkpoint episode numbers, not per-episode
# training-log rows) from compare_solutions.py's per_run_window_means /
# aggregate_runs_by_window. Aggregation floor-divides each run's real
# checkpoint episode numbers into fixed-size --window-ep-wide bins
# (bins = episode // window), takes the per-bin mean within a run, then
# combines each run's per-bin series across runs into a per-bin mean/std/
# min/max -- exactly the same two-stage windowed-bucket aggregation
# compare_solutions.py's multi-run branch uses for its training-curve
# panels, so both scripts' figures are aggregated identically and remain
# directly comparable.

def per_run_window_means(ep: np.ndarray, val: np.ndarray, window: int) -> pd.Series:
    """Reduce one run's (episode, value) pairs to a per-window mean.
    Returns a Series indexed by window bin index. Copied verbatim from
    compare_solutions.py."""
    valid = ~np.isnan(ep) & ~np.isnan(val)
    ep, val = ep[valid], val[valid]
    if len(ep) == 0:
        return pd.Series(dtype=np.float64)
    bins = (ep // window).astype(np.int64)
    return pd.Series(val).groupby(bins).mean()


def aggregate_runs_by_window(per_run_series: list, window: int):
    """Combine each run's per-window means into across-run mean / std /
    min / max curves (the min/max curves trace the worst- and best-run
    window-averages, not raw per-checkpoint outliers). Copied verbatim from
    compare_solutions.py. Returns (x_centers, mean, std, vmin, vmax) or
    None if no data."""
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


def window_rolling_mean_std(ep: np.ndarray, val: np.ndarray, window: int):
    """Aggregate one run's (episode, value) pairs into a windowed mean +/-
    std (within each fixed-size episode window). Used for the single-run
    branch, exactly like compare_solutions.py's identically-named helper.
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
    std  = np.nan_to_num(std, nan=0.0)  # single-point windows -> NaN std -> 0
    return x_centers, mean, std


def analyze_solution(run_dirs, gamma_step, max_lag):
    """Run analyze_run() on every run-folder belonging to one solution.
    Aggregation across those runs' metrics happens later, at plot time, via
    window_rolling_mean_std (n_runs==1) or per_run_window_means +
    aggregate_runs_by_window (n_runs>1) -- matching compare_solutions.py's
    per-panel branch structure. Returns None if every run-folder failed to
    load."""
    per_run_results = []
    for run_dir in run_dirs:
        try:
            per_run_results.append((run_dir, analyze_run(run_dir, gamma_step, max_lag)))
        except FileNotFoundError as e:
            print(f"  ! skipping {run_dir}: {e}")

    if not per_run_results:
        return None

    return {"n_runs": len(per_run_results), "per_run": per_run_results}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces-dir", type=str, required=True,
                         help="base folder containing run subfolders, e.g. "
                              "'distributed(1)/', 'distributed(2)/', 'adaptive(1)/', "
                              "each holding that run's ep*.csv checkpoints")
    parser.add_argument("--gamma-step", type=float, required=True,
                         help="per-timestep discount, e.g. 0.5**(1/WAIT_INC)")
    parser.add_argument("--max-lag", type=int, default=20,
                         help="max +/- lag (in timesteps) for the correlation sweep")
    parser.add_argument("--window-ep", type=int, default=1000,
                         help="episode-window size used to bucket checkpoints before "
                              "aggregating (both within a single run's own mean/std, "
                              "and -- for multi-run solutions -- across runs). Same "
                              "convention and same default role as compare_solutions.py's "
                              "WINDOW_EP: checkpoint episode numbers are floor-divided by "
                              "this window size, matching how the comparison script "
                              "buckets its own training-log episodes into windows before "
                              "aggregating, so pick a window at least as wide as your "
                              "TRACE_SAVE_EVERY checkpoint spacing.")
    parser.add_argument("--out-dir", type=str, default="outputs/analysis")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    groups = group_run_dirs(args.traces_dir)
    print(f"Found {len(groups)} solution(s) under {args.traces_dir}:")
    for name, dirs in groups.items():
        print(f"  {name}: {len(dirs)} run(s) -> {[os.path.basename(d) for d in dirs]}")

    solutions = {}
    for name, run_dirs in groups.items():
        result = analyze_solution(run_dirs, args.gamma_step, args.max_lag)
        if result is None:
            print(f"  ! no usable checkpoints for solution {name!r}, skipping")
            continue
        solutions[name] = result

    if not solutions:
        print("No solutions produced usable results. Exiting.")
        return

    # ---- print a plain-text per-run summary table (unaggregated) ----
    print(f"\n{'solution':<28}{'run':<14}{'episode':>10}{'nmse':>12}"
          f"{'corr(lag=0)':>14}{'peak_lag':>10}{'peak_corr':>12}")
    for name, sol in solutions.items():
        for run_dir, res in sol["per_run"]:
            run_label = os.path.basename(run_dir)
            for i in range(len(res["episodes"])):
                print(f"{name:<28}{run_label:<14}{res['episodes'][i]:>10}"
                      f"{res['nmse'][i]:>12.4f}{res['corr0'][i]:>14.4f}"
                      f"{res['peak_lag'][i]:>10}{res['peak_corr'][i]:>12.4f}")

    # ---- save per-run CSV summary ----
    per_run_path = os.path.join(args.out_dir, "value_return_summary_per_run.csv")
    with open(per_run_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solution", "run", "episode", "nmse", "corr_lag0", "peak_lag", "peak_corr"])
        for name, sol in solutions.items():
            for run_dir, res in sol["per_run"]:
                run_label = os.path.basename(run_dir)
                for i in range(len(res["episodes"])):
                    writer.writerow([
                        name, run_label, res["episodes"][i],
                        res["nmse"][i], res["corr0"][i],
                        res["peak_lag"][i], res["peak_corr"][i],
                    ])
    print(f"\nSaved per-run summary CSV to {per_run_path}")

    # ---- compute windowed aggregates (mirrors compare_solutions.py) and
    #      save an aggregated CSV summary ----
    # agg_by_solution[name][metric] = (x, mean, std, vmin, vmax) for n_runs>1
    #                                  (x, mean, std)             for n_runs==1
    #                                  None if that metric had no data
    agg_by_solution: dict[str, dict[str, tuple]] = {}
    metrics = ("nmse", "corr0", "peak_corr")
    for name, sol in solutions.items():
        n_runs = sol["n_runs"]
        agg_by_solution[name] = {}
        for metric in metrics:
            if n_runs == 1:
                _, res = sol["per_run"][0]
                agg_by_solution[name][metric] = window_rolling_mean_std(
                    res["episodes"], res[metric], args.window_ep)
            else:
                per_run_series = [
                    per_run_window_means(res["episodes"], res[metric], args.window_ep)
                    for _, res in sol["per_run"]
                ]
                agg_by_solution[name][metric] = aggregate_runs_by_window(
                    per_run_series, args.window_ep)

    agg_path = os.path.join(args.out_dir, "value_return_summary_aggregated.csv")
    with open(agg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solution", "n_runs", "window_episode_center",
                          "nmse_mean", "nmse_std", "nmse_min", "nmse_max",
                          "corr0_mean", "corr0_std", "corr0_min", "corr0_max",
                          "peak_corr_mean", "peak_corr_std", "peak_corr_min", "peak_corr_max"])
        for name, sol in solutions.items():
            n_runs = sol["n_runs"]
            metric_aggs = agg_by_solution[name]
            if all(agg is None for agg in metric_aggs.values()):
                continue

            # Look each metric up by its window-center x value, so a metric
            # missing/NaN at one window (e.g. corr0 undefined when a
            # segment has ~zero variance) doesn't blank out unrelated
            # metrics that DO have a value at that window.
            lookups = {}
            for m, agg in metric_aggs.items():
                if agg is None:
                    lookups[m] = {}
                elif n_runs == 1:
                    x, mean, std = agg
                    lookups[m] = {xi: (mi, si, np.nan, np.nan) for xi, mi, si in zip(x, mean, std)}
                else:
                    x, mean, std, vmin, vmax = agg
                    lookups[m] = {xi: (mi, si, vi, vxi) for xi, mi, si, vi, vxi in
                                  zip(x, mean, std, vmin, vmax)}

            all_x = sorted(set().union(*(lu.keys() for lu in lookups.values())))
            for x in all_x:
                row = [name, n_runs, x]
                for m in metrics:
                    row += list(lookups[m].get(x, (np.nan, np.nan, np.nan, np.nan)))
                writer.writerow(row)
    print(f"Saved aggregated (across-run, windowed) summary CSV to {agg_path}")

    # ---- plots ----
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        # Same band opacities and finishing touches as compare_solutions.py,
        # so the two scripts' figures read as a matched pair.
        BAND_ALPHA = 0.07
        MINMAX_ALPHA = 0.00

        def finish_ax(ax, title, ylabel):
            ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
            ax.set_xlabel("Episode", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="grey", alpha=0.2, linewidth=0.5)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        metric_titles = [
            ("nmse", "Normalized MSE(V, G)", "NMSE (lower = closer)"),
            ("corr0", "Correlation(V, G) at lag 0", "Pearson r"),
            ("peak_corr", "Peak correlation over lag sweep", "Pearson r (best lag)"),
        ]

        legend_handles, legend_labels = [], []
        panel_has_data = [False] * len(metric_titles)

        for name, sol in solutions.items():
            color = get_color_for_solution(name)
            n_runs = sol["n_runs"]
            label = f"{name} (n={n_runs})"

            for ax_idx, (metric, _, _) in enumerate(metric_titles):
                ax = axes[ax_idx]
                agg = agg_by_solution[name][metric]
                if agg is None:
                    continue

                if n_runs == 1:
                    # ── Single-run branch: within-window mean +/- std,
                    #    same as compare_solutions.py's single-run branch. ──
                    x, mean, std = agg
                    panel_has_data[ax_idx] = True
                    ax.fill_between(x, mean - std, mean + std,
                                     color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
                    line, = ax.plot(x, mean, color=color, linewidth=1.0, zorder=3)
                else:
                    # ── Multi-run branch: across-run mean +/- std, plus a
                    #    best/worst-run band, on window-bucketed per-run
                    #    series -- same as compare_solutions.py's multi-run
                    #    branch. ──
                    x, mean, std, vmin, vmax = agg
                    panel_has_data[ax_idx] = True
                    ax.fill_between(x, vmin, vmax,
                                     color=color, alpha=MINMAX_ALPHA, linewidth=0, zorder=1)
                    ax.fill_between(x, mean - std, mean + std,
                                     color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
                    line, = ax.plot(x, mean, color=color, linewidth=1.0, zorder=3)

                if ax_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(label)

        for ax_idx, (_, title, ylabel) in enumerate(metric_titles):
            if panel_has_data[ax_idx]:
                finish_ax(axes[ax_idx], title, ylabel)
            else:
                axes[ax_idx].set_visible(False)

        fig.legend(legend_handles, legend_labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.05), ncol=min(len(legend_labels), 4),
                   fontsize=9, frameon=False)
        plt.tight_layout(rect=(0, 0.06, 1, 1))
        fig_path = os.path.join(args.out_dir, "value_return_metrics.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved summary plot to {fig_path}")

        # Lag-correlation curves, one panel per solution: every checkpoint of
        # every run in that solution is overlaid in the solution's color,
        # with opacity increasing over training (faint = early, opaque =
        # late) so you can see whether the curve's shape/peak stabilizes as
        # training progresses and whether it's consistent run-to-run.
        n_solutions = len(solutions)
        fig3, axes3 = plt.subplots(1, n_solutions, figsize=(6 * n_solutions, 4.5), squeeze=False)
        axes3 = axes3[0]

        for ax, (name, sol) in zip(axes3, solutions.items()):
            color = get_color_for_solution(name)
            # Pool every (episode, lags, corrs) triple across all runs of
            # this solution, then sort by episode so opacity still encodes
            # chronological training progress even when multiple runs are
            # interleaved.
            pooled = []
            for run_dir, res in sol["per_run"]:
                for ep, (lags, corrs) in zip(res["episodes"], res["lag_curves"]):
                    pooled.append((ep, lags, corrs))
            pooled.sort(key=lambda t: t[0])

            n_ckpts = len(pooled)
            if n_ckpts == 0:
                continue
            alphas = np.linspace(0.15, 1.0, n_ckpts)

            for i, (ep, lags, corrs) in enumerate(pooled):
                is_last = (i == n_ckpts - 1)
                ax.plot(
                    lags, corrs,
                    color=color,
                    alpha=alphas[i],
                    linewidth=1.8 if is_last else 1.0,
                    label=f"ep {ep}" if is_last else None,
                    zorder=10 if is_last else 1,
                )

            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Lag (timesteps)")
            ax.set_ylabel("Pearson r")
            ax.set_title(f"{name} (n={sol['n_runs']}): lag-correlation across training\n"
                         f"(faint = early, opaque = late)")
            ax.legend(loc="upper right", fontsize=8)

        fig3.tight_layout()
        fig3_path = os.path.join(args.out_dir, "lag_correlation_all_checkpoints.png")
        fig3.savefig(fig3_path, dpi=150)
        print(f"Saved all-checkpoints lag-correlation plot to {fig3_path}")

    except ImportError:
        print("matplotlib not available -- skipping plots, CSV summaries still saved.")


if __name__ == "__main__":
    main()