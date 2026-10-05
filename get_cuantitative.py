"""
summarise_results.py
For every CSV in outputs/*.csv, computes:
  - Best 100-episode rolling mean of reward_rate  (normalised the same way
    compare_solutions.py does: divided by 30)
  - Best 100-episode rolling mean of episode reward (score)
  - Total wins  (episodes where score >= 24)

Results are printed as a sorted table (descending by best reward rate) and
also saved to outputs/summary.csv.
"""

import os
import glob
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_GLOB  = "outputs/*.csv"
OUT_CSV     = "outputs/summary.csv"
WINDOW_EP   = 100      # rolling window size (episodes)
WIN_THRESH  = 24       # score >= this counts as a win
REWARD_DIV  = 30       # match compare_solutions.py normalisation for reward_rate

# Column name candidates (first found wins)
SCORE_COLS       = ["score"]
REWARD_RATE_COLS = ["reward_rate", "gamma_disc_reward"]
EPISODE_COL      = "episode"


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def best_rolling_mean(series: pd.Series, window: int) -> float:
    """Peak value of a rolling mean over `window` episodes (min 1 valid point
    required per window so short runs still get a result)."""
    rolled = series.rolling(window, min_periods=1).mean()
    return float(rolled.max()) if not rolled.empty else float("nan")


def process_file(path: str) -> dict:
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"file": name, "error": str(e)}

    # Coerce numeric
    if EPISODE_COL in df.columns:
        df[EPISODE_COL] = pd.to_numeric(df[EPISODE_COL], errors="coerce")
    
    score_col = find_col(df, SCORE_COLS)
    rr_col    = find_col(df, REWARD_RATE_COLS)

    # ── score / wins ─────────────────────────────────────────────────────────
    if score_col:
        score = pd.to_numeric(df[score_col], errors="coerce").dropna()
        best_score_avg = best_rolling_mean(score.reset_index(drop=True), WINDOW_EP)
        total_wins     = int((score >= WIN_THRESH).sum())
        win_pct        = 100.0 * total_wins / len(score) if len(score) else float("nan")
    else:
        best_score_avg = float("nan")
        total_wins     = 0
        win_pct        = float("nan")

    # ── reward rate ──────────────────────────────────────────────────────────
    if rr_col:
        rr = pd.to_numeric(df[rr_col], errors="coerce").dropna()
        rr_norm = rr 
        best_rr_avg = best_rolling_mean(rr_norm.reset_index(drop=True), WINDOW_EP)
    else:
        best_rr_avg = float("nan")

    total_episodes = len(df)

    return {
        "file":                   name,
        "total_episodes":         total_episodes,
        "best_100ep_reward_rate": round(best_rr_avg, 4),
        "best_100ep_score":       round(best_score_avg, 4),
        "total_wins":             total_wins,
        "win_pct":                round(win_pct, 2),
    }


def main():
    files = sorted(glob.glob(INPUT_GLOB))
    if not files:
        print(f"No CSV files found matching {INPUT_GLOB!r}. "
              f"Run this from the project root.")
        return

    rows = [process_file(f) for f in files]

    # Split out any errored files
    errors = [r for r in rows if "error" in r]
    rows   = [r for r in rows if "error" not in r]

    if errors:
        print("\n── Errors ──────────────────────────────────────────────────")
        for e in errors:
            print(f"  {e['file']}: {e['error']}")

    if not rows:
        print("No files processed successfully.")
        return

    df_out = pd.DataFrame(rows).sort_values(
        "best_100ep_reward_rate", ascending=False
    ).reset_index(drop=True)

    df_out.index += 1  # 1-based rank

    # ── Pretty-print ─────────────────────────────────────────────────────────
    col_widths = {
        "file":                   max(30, df_out["file"].str.len().max() + 2),
        "total_episodes":         10,
        "best_100ep_reward_rate": 22,
        "best_100ep_score":       18,
        "total_wins":             11,
        "win_pct":                9,
    }
    headers = {
        "file":                   "File",
        "total_episodes":         "Episodes",
        "best_100ep_reward_rate": "Best100 RewardRate",
        "best_100ep_score":       "Best100 Score",
        "total_wins":             "Wins",
        "win_pct":                "Win %",
    }

    def fmt_row(rank, row):
        parts = [f"{rank:<4}"]
        for col, w in col_widths.items():
            val = row[col]
            if isinstance(val, float):
                parts.append(f"{val:{w}.4f}" if not np.isnan(val) else f"{'—':>{w}}")
            else:
                parts.append(f"{str(val):<{w}}")
        return "  ".join(parts)

    header_line = "Rank  " + "  ".join(
        f"{headers[c]:<{w}}" for c, w in col_widths.items()
    )
    sep = "─" * len(header_line)

    print(f"\n{sep}")
    print(f"  Results sorted by Best 100-episode Reward Rate  "
          f"(win threshold: score ≥ {WIN_THRESH})")
    print(sep)
    print(header_line)
    print(sep)
    for rank, row in df_out.iterrows():
        print(fmt_row(rank, row))
    print(sep)
    print(f"\nTotal files: {len(rows)}\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df_out.to_csv(OUT_CSV, index_label="rank")
    print(f"Summary saved → {OUT_CSV}")


if __name__ == "__main__":
    main()