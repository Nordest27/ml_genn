"""
plot_training.py
Iterates over all CSVs in outputs/, renders a smoothed summary plot
(median + IQR band with raw data faded in background) and saves PNGs to
outputs/renders/.
"""

import os
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
WINDOW          = 50        # rolling window for median / IQR smoothing
RAW_ALPHA       = 0.15      # opacity of raw data trace
BAND_ALPHA      = 0.25      # opacity of IQR shaded region
DPI             = 200
WIN_THRESHOLD   = 24        # score >= this value counts as a win

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


def find_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def style_ax(ax, title: str, color: str, x, raw, med, q25, q75, win_mask=None):
    """Draw one panel: faded raw + IQR band + median line, with optional win dots."""
    # raw trace
    ax.plot(x, raw, color=color, alpha=RAW_ALPHA, linewidth=0.6, zorder=1)
    # IQR band
    ax.fill_between(x, q25, q75, color=color, alpha=BAND_ALPHA, zorder=2, linewidth=0)
    # median
    ax.plot(x, med, color=color, linewidth=1.8, zorder=3, label="median")
    # win dots on the raw score trace
    if win_mask is not None and win_mask.any():
        ax.scatter(x[win_mask], raw[win_mask],
                   color="#FFD700", edgecolors="#B8860B",
                   s=18, linewidths=0.5, zorder=5,
                   label=f"win (≥{WIN_THRESHOLD})")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.6)
    # aesthetics
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Episode", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="grey", alpha=0.2, linewidth=0.5)


# ── Main ──────────────────────────────────────────────────────────────────────

def render(csv_path: str):
    df = pd.read_csv(csv_path)
    x  = pd.to_numeric(df["episode"], errors="coerce").to_numpy(dtype=np.float64)

    stem    = os.path.splitext(os.path.basename(csv_path))[0]
    out_png = os.path.join(OUT_DIR, f"{stem}.png")

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(stem.replace("-", " ").replace("_", " "), fontsize=13,
                 fontweight="bold", y=1.01)
    axs = axs.flatten()

    # Pre-compute win mask from the score column (used only on the reward panel)
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
        # Only overlay win dots on the Episode Reward panel (index 0)
        mask = win_mask if (idx == 0 and win_mask is not None) else None
        style_ax(ax, panel["title"], color, x, raw, med, q25, q75, win_mask=mask)

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

    workers = min(len(files), os.cpu_count() or 1)
    print(f"Found {len(files)} CSV file(s) — rendering with {workers} worker(s)...")

    with multiprocessing.Pool(processes=workers) as pool:
        pool.map(render, files)

    print(f"\nAll renders saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()