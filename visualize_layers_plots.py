"""
compare_layers_groups.py
Compares CSVs that share a common name *prefix* but differ by a trailing
"_layers_X" suffix, e.g.:
    pacman_training_weight_dist_prop_layers_2.csv
    pacman_training_weight_dist_prop_layers_4.csv
    pacman_training_weight_dist_prop_layers_8.csv
    pacman_training_other_method_layers_2.csv
    pacman_training_other_method_layers_4.csv

Layout: one ROW per group, one COLUMN per metric panel.
Within each cell only that group's layer-count curves are drawn, so shades
never mix between groups. Layer counts are labeled directly at the
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
OUT_NAME = "comparison_layers_groups.png"
WINDOW_EP = 50  # episode-window size for smoothing
DPI = 200

# Line width range: thinnest for smallest layer count, thickest for largest.
LW_MIN = 0.8
LW_MAX = 2.2

# Base hues, used as the fallback palette for group prefixes that don't
# match any entry in PATTERN_COLORS below (cycled if there are more
# unmatched prefixes than colors). Each fallback entry is a single hex —
# its endpoint pair is derived as (hex, hex) so the layer-count gradient
# collapses to a single flat color for unmatched groups.
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
# Assigns a fixed (start_hex, end_hex) color pair to any group prefix
# containing a given substring, so the same kind of solution gets the same
# gradient across every chart it appears in (instead of an arbitrary
# per-figure index). Matching is substring-anywhere, and the LONGEST
# matching pattern wins, so e.g. "weight_dist_prop" is matched before the
# shorter "weight_dist" even though "weight_dist" is also a substring of
# it. List order below doesn't matter — longest-match-wins is resolved
# automatically at lookup time.
#
# Each entry is (pattern, start_hex, end_hex): start_hex is used for the
# smallest layer count in the group, end_hex for the largest, with a
# straight RGB blend for any layer counts in between (instead of the
# opacity-based shading used for lambda, which doesn't read as cleanly
# here). Keep related patterns in the same hue family (similar start/end
# pairs) so they still read as related at a glance.
PATTERN_COLORS: list[tuple[str, str, str]] = [
    # --- "weight_dist" family (blues) ---------------------------------------
    ("weight_dist_prop", "#A9C6E8", "#1F4E79"),  # light->dark blue (specific variant)
    ("weight_prop_dist", "#A9C6E8", "#1F4E79"),  # light->dark blue (specific variant)
    ("weight_dist_uniform", "#9DB8DE", "#2F5C8A"),  # light->dark blue
    ("weight_dist", "#8FAEDB", "#4C72B0"),  # base blue (family base / fallback)
    # --- "node_dist" family (greens) -------------------------------------------
    ("node_dist_prop", "#9FCBA8", "#2E5E37"),  # light->dark green (specific variant)
    ("node_dist", "#A8D4B2", "#55A868"),  # base green (family base)
    # --- "random" family (reds) -----------------------------------------------
    ("adaptive", "#E3A6A9", "#C44E52"),  # light->dark red (family base)
    # --- "combined" family (purples) --------------------------------------------
    ("combined_prop", "#B9AFD6", "#574A7A"),  # light->dark purple (specific variant)
    ("combined", "#C2B8DC", "#8172B2"),  # base purple (family base)
    # --- "weight_dist-ind-noise" (cyan, standalone variant) ----------------------
    ("weight_dist-ind-noise", "#9CD3DF", "#3A8295"),  # light->dark cyan
]


# Cache of prefix -> color pair so the fallback palette assignment for
# unmatched prefixes stays stable across multiple render_comparison() calls
# (i.e. across different charts), not just within one figure.
_color_cache = {}
_fallback_counter = 0


def get_color_pair_for_solution(name):
    """Return a (start_hex, end_hex) color pair for a group prefix. Names
    containing a pattern from PATTERN_COLORS get that pattern's fixed pair
    (longest matching pattern wins on overlaps). Names matching nothing
    fall back to COLOR_PALETTE (used as a flat start==end pair), cycled in
    first-seen order and cached so the same unmatched name keeps the same
    fallback color across all charts in this run."""
    if name in _color_cache:
        return _color_cache[name]

    best_pattern, best_pair = None, None
    for pattern, start_hex, end_hex in PATTERN_COLORS:
        if pattern in name:
            if best_pattern is None or len(pattern) > len(best_pattern):
                best_pattern, best_pair = pattern, (start_hex, end_hex)

    if best_pair is not None:
        _color_cache[name] = best_pair
        return best_pair

    global _fallback_counter
    color = COLOR_PALETTE[_fallback_counter % len(COLOR_PALETTE)]
    _fallback_counter += 1
    pair = (color, color)
    _color_cache[name] = pair
    return pair


LAYERS_SUFFIX_RE = re.compile(r"^(.*)_layers_(\d)$")

# Matches a "[<chart title>]" tag anywhere in the filename stem, e.g.
# "pacman_training_weight_dist_prop[Layer Sweep]_layers_4" -> tag
# "Layer Sweep". Used to route a file into its own comparison chart,
# independent of the "_layers_X" suffix. Files with no tag fall into the
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


def group_files_by_layers(files):
    """Group CSV paths first by chart tag ('[TEXT]', anywhere in the
    filename), then by prefix/layer-count as before (tag removed before
    suffix matching). Returns {chart_key: {prefix: {n_layers: path}}}."""
    charts = {}
    for f in files:
        raw_stem = os.path.splitext(os.path.basename(f))[0]
        chart_key, stem = extract_chart_tag(raw_stem)
        m = LAYERS_SUFFIX_RE.match(stem)
        if not m:
            print(f"  -  skipping {f!r} (doesn't match '<prefix>_layers_<N>')")
            continue
        prefix, layers_str = m.group(1), m.group(2)
        n_layers = int(layers_str)
        charts.setdefault(chart_key, {}).setdefault(prefix, {})[n_layers] = f
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


def blend_color(start_hex, end_hex, t):
    """Linearly blend from start_hex (t=0) to end_hex (t=1) in RGB space."""
    t = float(np.clip(t, 0.0, 1.0))
    r0, g0, b0 = mcolors.to_rgb(start_hex)
    r1, g1, b1 = mcolors.to_rgb(end_hex)
    return (r0 + (r1 - r0) * t, g0 + (g1 - g0) * t, b0 + (b1 - b0) * t)


def layer_styles(layer_counts, start_hex, end_hex):
    """Return {n_layers: (color, linewidth)} sorted fewest->most layers.
    Color is a straight RGB blend between start_hex (fewest layers) and
    end_hex (most layers); linewidth scales the same way."""
    counts_sorted = sorted(layer_counts)
    n = len(counts_sorted)
    styles = {}
    for i, n_layers in enumerate(counts_sorted):
        t = 1.0 if n == 1 else i / (n - 1)
        lw = LW_MIN if n == 1 else (LW_MIN + (LW_MAX - LW_MIN) * (i / (n - 1)))
        styles[n_layers] = (blend_color(start_hex, end_hex, t), lw)
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
        else f"comparison_layers_groups_{chart_slug}.png"
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
    for prefix, layer_map in sorted_groups:
        loaded[prefix] = {}
        for n_layers, path in layer_map.items():
            loaded[prefix][n_layers] = pd.read_csv(path)

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

        for group_idx, (prefix, layer_dfs) in enumerate(loaded.items()):
            ax_row = metric_row * n_groups + group_idx
            ax = axs[ax_row, metric_col]

            start_hex, end_hex = get_color_pair_for_solution(prefix)

            layers_sorted = sorted(layer_dfs.keys())
            styles = layer_styles(layers_sorted, start_hex, end_hex)

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

            for n_layers in layers_sorted:
                df = layer_dfs[n_layers]

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

                color, lw = styles[n_layers]

                ax.plot(x, mean, color=color, linewidth=lw, zorder=3)

                all_x_ends.append((x[-1], mean[-1], n_layers, color))

                has_data = True

            # --------------------------------------------------
            # Layer-count labels
            # --------------------------------------------------

            if all_x_ends:
                x_max = max(v[0] for v in all_x_ends)
                x_min = min(v[0] for v in all_x_ends)

                x_offset = (x_max - x_min) * LABEL_X_OFFSET_FRAC

                for x_end, y_end, n_layers, color in all_x_ends:
                    ax.annotate(
                        f"L={n_layers}",
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

                # Leave room for layer-count labels
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
        "Lighter color = fewer layers   ·   Darker color = more layers",
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

    charts = group_files_by_layers(files)
    if not charts:
        print("No files matched the '<prefix>_layers_<N>' naming pattern.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    for chart_key in sorted(charts.keys()):
        groups = charts[chart_key]
        print(f"[{chart_key}]")
        for prefix, layer_map in sorted(groups.items()):
            layer_list = ", ".join(f"L={n}" for n in sorted(layer_map.keys()))
            print(f"  Group {prefix!r}: {len(layer_map)} layer count(s) → {layer_list}")

        render_comparison(chart_key, groups)

    print(f"\nComparison(s) saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()