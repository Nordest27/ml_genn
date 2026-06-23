"""
compare_lambda_groups.py
Compares CSVs that share a common name *prefix* but differ by a trailing
"_lambda_0XX" suffix, e.g.:
    pacman_training_weight_dist_prop_lambda_05.csv
    pacman_training_weight_dist_prop_lambda_075.csv
    pacman_training_weight_dist_prop_lambda_1.csv
    pacman_training_other_method_lambda_025.csv
    pacman_training_other_method_lambda_05.csv

Layout: one ROW per group, one COLUMN per metric panel.
Within each cell only that group's lambda curves are drawn, so shades
never mix between groups. Lambda values are labeled directly at the
right edge of each line — no legend needed.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_GLOB = "outputs/*.csv"
OUT_DIR = "outputs/renders/comparisons"
OUT_NAME = "comparison_lambda_groups.png"
WINDOW_EP = 250  # episode-window size for smoothing
DPI = 200

# Shade range: lightest (smallest λ) → darkest (largest λ).
# Wider range = easier to tell λ values apart.
SHADE_LIGHTEST = 0.15
SHADE_DARKEST = 1.00

# Line width range: thinnest for smallest λ, thickest for largest λ.
LW_MIN = 0.8
LW_MAX = 2.2

# Base hues, used as the fallback palette for group prefixes that don't
# match any entry in PATTERN_COLORS below (cycled if there are more
# unmatched prefixes than colors).
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
# Assigns a fixed base color to any group prefix containing a given
# substring, so the same kind of solution gets the same hue across every
# chart it appears in (instead of an arbitrary per-figure index). Matching
# is substring-anywhere, and the LONGEST matching pattern wins, so e.g.
# "weight_dist_prop" is matched before the shorter "weight_dist" even
# though "weight_dist" is also a substring of it. List order below doesn't
# matter — longest-match-wins is resolved automatically at lookup time.
#
# This is the same mapping used in compare_solutions.py, kept in sync so a
# given solution reads as the same color family across both scripts. The
# lambda shading (light->dark per λ) is layered on top of whatever base
# color is returned here.
PATTERN_COLORS: list[tuple[str, str]] = [
    # --- "weight_dist" family (blues) ---------------------------------------
    ("weight_dist_prop",   "#1F4E79"),  # darkest blue  (most specific variant)
    ("weight_prop_dist",   "#1F4E79"),  # darkest blue  (most specific variant)
    ("weight_dist_uniform","#2F5C8A"),  # dark blue
    ("weight_dist",        "#4C72B0"),  # base blue     (family base / fallback within family)

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

    # --- "entropy_reg" family (purples) ---------------------------------------
    ("combined_prop",   "#574A7A"),  # darkest purple
    ("asfadsfsadfasdf",    "#6C5C96"),  # dark purple
    ("combined",        "#8172B2"),  # base purple  (family base)

    # --- "freq_response" family (cyans) ----------------------------------------
    ("weight_dist-ind-noise", "#3A8295"),  # darkest cyan
    ("asfadsfsadfasdf", "#4F9CB0"),  # dark cyan
    ("asfadsfsadfasdf",      "#64B5CD"),  # base cyan    (family base)

    # --- standalone / unrelated patterns (no close siblings yet) ---------------
    ("asfadsfsadfasdf",           "#8C8C8C"),  # grey  — unmodified baseline runs
    ("asfadsfsadfasdf",           "#CCB974"),  # gold  — ablation studies
    ("asfadsfsadfasdf",         "#937860"),  # brown — curriculum learning variants
    ("asfadsfsadfasdf",        "#DA8BC3"),  # pink  — multi-agent setups
]


# Cache of prefix -> color so the fallback palette assignment for unmatched
# prefixes stays stable across multiple render_comparison() calls (i.e.
# across different charts), not just within one figure.
_color_cache = {}
_fallback_counter = 0


def get_color_for_solution(name):
    """Return a base color for a group prefix. Names containing a pattern
    from PATTERN_COLORS get that pattern's fixed color (longest matching
    pattern wins on overlaps). Names matching nothing fall back to
    COLOR_PALETTE, cycled in first-seen order and cached so the same
    unmatched name keeps the same fallback color across all charts in this
    run."""
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


LAMBDA_SUFFIX_RE = re.compile(r"^(.*)_lambda_(\d+)$")

# Matches a "[<chart title>]" tag anywhere in the filename stem, e.g.
# "pacman_training_weight_dist_prop[Lambda Sweep]_lambda_05" -> tag
# "Lambda Sweep". Used to route a file into its own comparison chart,
# independent of the "_lambda_0XX" suffix. Files with no tag fall into the
# MISC_CHART_KEY chart.
CHART_TAG_RE = re.compile(r"\[([^\[\]]+)\]")

MISC_CHART_KEY = "Misc"

PANELS = [
    {"title": "Episode Reward", "cols": ["score"]},
    {"title": "Reward Rate", "cols": ["reward_rate", "gamma_disc_reward"]},
    {"title": "TD Error |mean|", "cols": ["avg_abs_td_error"]},
    {"title": "Voltage", "cols": ["voltage"]},
    {"title": "Voltage Loss", "cols": ["voltage_loss"]},
    {"title": "Frequency (Hz)", "cols": ["frequency"]},
]

# How far (in data-x units) to push the right-edge label past the last point.
# Expressed as a fraction of the total x-range.
LABEL_X_OFFSET_FRAC = 0.01

# ── Helpers ───────────────────────────────────────────────────────────────────


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_lambda_str(lambda_str):
    """
    Examples:
        09   -> 0.9
        05   -> 0.5
        075  -> 0.75
        025  -> 0.25
        1    -> 1.0
    """
    if len(lambda_str) == 1:
        return float(lambda_str)

    return int(lambda_str) / (10 ** (len(lambda_str) - 1))


def extract_chart_tag(stem):
    """Pull a '[chart title]' tag out of a filename stem, wherever it
    appears. Returns (chart_key, stem_without_tag). If no tag is present,
    chart_key is MISC_CHART_KEY and the stem is returned unchanged."""
    m = CHART_TAG_RE.search(stem)
    if not m:
        return MISC_CHART_KEY, stem
    chart_key = m.group(1).strip()
    # remove the tag (and any doubled-up underscores/whitespace it leaves)
    cleaned = stem[: m.start()] + stem[m.end() :]
    cleaned = re.sub(r"_{2,}", "_", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip("_ ")
    return chart_key, cleaned


def group_files_by_lambda(files):
    """Group CSV paths first by chart tag ('[TEXT]', anywhere in the
    filename), then by prefix/lambda as before (tag removed before suffix
    matching). Returns {chart_key: {prefix: {lam: path}}}."""
    charts = {}
    for f in files:
        raw_stem = os.path.splitext(os.path.basename(f))[0]
        chart_key, stem = extract_chart_tag(raw_stem)
        m = LAMBDA_SUFFIX_RE.match(stem)
        if not m:
            print(f"  -  skipping {f!r} (doesn't match '<prefix>_lambda_<N>')")
            continue
        prefix, lambda_str = m.group(1), m.group(2)
        lam = parse_lambda_str(lambda_str)
        charts.setdefault(chart_key, {}).setdefault(prefix, {})[lam] = f
    return charts


def window_aggregate(ep, val, window):
    valid = ~np.isnan(ep) & ~np.isnan(val)
    ep, val = ep[valid], val[valid]
    if len(ep) == 0:
        return None
    bins = (ep // window).astype(np.int64)
    s = pd.Series(val).groupby(bins)
    idx = s.mean().index.to_numpy(dtype=np.float64)
    x_centers = idx * window + window / 2.0
    mean = s.mean().to_numpy()
    return x_centers, mean


def shade_color(base_hex, t):
    """Blend from near-white (t=0) to base color (t=1)."""
    t = float(np.clip(t, 0.0, 1.0))
    r, g, b = mcolors.to_rgb(base_hex)
    return (1.0 + (r - 1.0) * t, 1.0 + (g - 1.0) * t, 1.0 + (b - 1.0) * t)


def lambda_styles(lambdas, base_hex):
    """Return {lam: (color, linewidth)} sorted lightest→darkest."""
    lambdas_sorted = sorted(lambdas)
    n = len(lambdas_sorted)
    styles = {}
    for i, lam in enumerate(lambdas_sorted):
        t = (
            SHADE_DARKEST
            if n == 1
            else (SHADE_LIGHTEST + (SHADE_DARKEST - SHADE_LIGHTEST) * (i / (n - 1)))
        )
        lw = LW_MIN if n == 1 else (LW_MIN + (LW_MAX - LW_MIN) * (i / (n - 1)))
        styles[lam] = (shade_color(base_hex, t), lw)
    return styles


def pretty_prefix(prefix):
    return prefix.replace("-", " ").replace("_", " ")


def finish_ax(ax, title, show_xlabel):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
    if show_xlabel:
        ax.set_xlabel("Episode", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="grey", alpha=0.18, linewidth=0.5)


# ── Main ──────────────────────────────────────────────────────────────────────
def render_comparison(chart_key, groups):
    chart_slug = re.sub(r"[^A-Za-z0-9]+", "_", chart_key).strip("_").lower()
    out_name = (
        OUT_NAME
        if chart_key == MISC_CHART_KEY
        else f"comparison_lambda_groups_{chart_slug}.png"
    )
    out_png = os.path.join(OUT_DIR, out_name)

    sorted_groups = sorted(groups.items())
    n_groups = len(sorted_groups)

    # ------------------------------------------------------------------
    # Layout:
    #
    # Episode Reward      Reward Rate
    #   Method A            Method A
    #   Method B            Method B
    #   Method C            Method C
    #
    # TD Error            Voltage
    #   Method A            Method A
    #   Method B            Method B
    #   Method C            Method C
    #
    # Voltage Loss        Frequency
    #   Method A            Method A
    #   Method B            Method B
    #   Method C            Method C
    # ------------------------------------------------------------------

    METRIC_COLS = 2
    METRIC_ROWS = int(np.ceil(len(PANELS) / METRIC_COLS))

    total_rows = METRIC_ROWS * n_groups

    PLOT_WIDTH = 8.0
    PLOT_HEIGHT = 3

    fig_w = PLOT_WIDTH * METRIC_COLS
    fig_h = PLOT_HEIGHT * total_rows

    # preload csvs
    loaded = {}
    for prefix, lam_map in sorted_groups:
        loaded[prefix] = {}
        for lam, path in lam_map.items():
            loaded[prefix][lam] = pd.read_csv(path)

    fig, axs = plt.subplots(
        total_rows, METRIC_COLS, figsize=(fig_w, fig_h), squeeze=False
    )

    fig.suptitle(
        (
            "Snake 5 × 5"
            if chart_key == MISC_CHART_KEY
            else f"Snake 5 × 5 - {chart_key}"
        ),
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    # ==========================================================
    # METRIC-FIRST ORGANIZATION
    # ==========================================================

    for panel_idx, panel in enumerate(PANELS):
        metric_row = panel_idx // METRIC_COLS
        metric_col = panel_idx % METRIC_COLS

        for group_idx, (prefix, lam_dfs) in enumerate(loaded.items()):
            ax_row = metric_row * n_groups + group_idx
            ax = axs[ax_row, metric_col]

            base_color = get_color_for_solution(prefix)

            lam_sorted = sorted(lam_dfs.keys())
            styles = lambda_styles(lam_sorted, base_color)

            # Metric title only on top chart of block
            title = panel["title"] if group_idx == 0 else ""

            # Method label on every row
            ax.set_ylabel(
                pretty_prefix(prefix),
                fontsize=8,
                fontweight="bold",
                rotation=90,
                labelpad=10,
            )

            has_data = False
            all_x_ends = []

            for lam in lam_sorted:
                df = lam_dfs[lam]

                col = find_col(df, panel["cols"])
                if col is None:
                    continue

                ep = pd.to_numeric(df["episode"], errors="coerce").to_numpy(
                    dtype=np.float64
                )

                val = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)

                # Keep your custom conversions
                if col == "frequency":
                    val = val * 1000

                agg = window_aggregate(ep, val, WINDOW_EP)

                if agg is None:
                    continue

                x, mean = agg

                color, lw = styles[lam]

                ax.plot(x, mean, color=color, linewidth=lw, zorder=3)

                all_x_ends.append((x[-1], mean[-1], lam, color))

                has_data = True

            # --------------------------------------------------
            # Lambda labels
            # --------------------------------------------------

            if all_x_ends:
                x_max = max(v[0] for v in all_x_ends)
                x_min = min(v[0] for v in all_x_ends)

                x_offset = (x_max - x_min) * LABEL_X_OFFSET_FRAC

                for x_end, y_end, lam, color in all_x_ends:
                    ax.annotate(
                        f"λ={lam:g}",
                        xy=(x_end, y_end),
                        xytext=(x_end + x_offset, y_end),
                        fontsize=6.5,
                        color=color,
                        va="center",
                        annotation_clip=False,
                    )

            show_xlabel = group_idx == n_groups - 1

            if has_data:
                finish_ax(ax, title, show_xlabel)

                # Leave room for λ labels
                xl = ax.get_xlim()

                ax.set_xlim(xl[0], xl[1] * 1.15)

            else:
                ax.set_visible(False)

    # ==========================================================
    # Metric block separators
    # ==========================================================

    for metric_row in range(1, METRIC_ROWS):
        y = 1 - (metric_row / METRIC_ROWS)

        fig.add_artist(
            plt.Line2D(
                [0.04, 0.98],
                [y, y],
                transform=fig.transFigure,
                color="lightgrey",
                linewidth=1.0,
                alpha=0.6,
            )
        )

    fig.text(
        0.5,
        0.005,
        "Lighter, thinner lines = smaller λ   ·   Darker, thicker lines = larger λ",
        ha="center",
        va="bottom",
        fontsize=8,
        style="italic",
        color="grey",
    )

    plt.tight_layout(rect=(0, 0.02, 1, 0.98))

    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")

    plt.close(fig)

    print(f"  ✓  {out_png}")


def main():
    files = sorted(glob.glob(INPUT_GLOB))
    if not files:
        print(f"No CSV files found matching {INPUT_GLOB!r}.")
        return

    charts = group_files_by_lambda(files)
    if not charts:
        print("No files matched the '<prefix>_lambda_<N>' naming pattern.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    for chart_key in sorted(charts.keys()):
        groups = charts[chart_key]
        print(f"[{chart_key}]")
        for prefix, lam_map in sorted(groups.items()):
            lam_list = ", ".join(f"λ={lam:g}" for lam in sorted(lam_map.keys()))
            print(f"  Group {prefix!r}: {len(lam_map)} lambda value(s) → {lam_list}")

        render_comparison(chart_key, groups)

    print(f"\nComparison(s) saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()