"""
_lambda_timeslice_scatter_csv.py
--------------------------------
CSV-driven version of _lambda_timeslice_scatter.py.

Reads a date/value price series from a CSV, splits it into non-overlapping
chunks of CHUNK_SIZE trading days, and for each chunk produces the same 3-panel
scatter (peak λ vs payoff-to-premium ratio, by time slice).

Per-chunk parameters derived automatically:
  S0    = first closing price in the chunk
  sigma = annualised realised volatility of log-returns in the chunk
  K_lo  = S0 * K_LO_FRAC
  K_hi  = S0 * K_HI_FRAC
  T     = CHUNK_SIZE / TRADING_DAYS_PER_YEAR  (fraction of a year)

All other parameters (r, q, cap, n_paths …) stay fixed from the CONFIG block.
Chunks shorter than MIN_CHUNK_ROWS are silently skipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from ramp_hedging import RampStripPayoff, DeltaHedgingSimulation

# ═══════════════════════════════════════════════════════════════════════════
# ── CONFIG ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

CSV_PATH   = "test_prices.csv"   # path to the input CSV (relative or absolute)
DATE_COL   = "date"              # name of the date column
VALUE_COL  = "value"             # name of the price column

CHUNK_SIZE = 250                 # number of trading days per chunk  [configurable]
MIN_CHUNK_ROWS = 50              # skip tail chunks shorter than this

TRADING_DAYS_PER_YEAR = 252      # used to convert chunk length → T

# Strike levels expressed as a fraction of S0 at the start of each chunk
K_LO_FRAC  = 0.97
K_HI_FRAC  = 1.05

# Fixed financial / simulation parameters
R          = 0.035               # continuously compounded risk-free rate
Q          = 0.035               # dividend yield / cost-of-carry
CAP        = 2.0                 # lambda cap
N_PATHS    = 50_000              # Monte-Carlo paths per chunk
HEDGE_FREQ = 1                   # hedge every N steps
RETENTION  = 0.0
SEED       = 421                 # base seed; each chunk gets SEED + chunk_idx
N_SCATTER  = 5_000               # points shown per panel in the scatter
N_BUCKETS  = 40                  # equal-count buckets for bucket-mean line

# Output
OUT_DIR    = "."                 # directory for saved PNGs
OUT_PREFIX = "lambda_timeslice_csv"   # filename prefix; chunk label appended

# ═══════════════════════════════════════════════════════════════════════════
# ── STYLE ──────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

BG       = "#0e1117"
AX       = "#13161f"
GRID     = "#2a2a3e"
WHT      = "white"
DIM      = "#cccccc"
TEAL     = "#2a9d8f"
ORG      = "#e09430"
RED_LINE = "#d62728"
CDF_COL  = "#a0a8ff"
PANEL_COLS = [TEAL, ORG, RED_LINE]


# ═══════════════════════════════════════════════════════════════════════════
# ── HELPERS ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def realised_vol(prices: np.ndarray, ann_factor: float = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised realised vol from a daily closing-price array."""
    log_rets = np.diff(np.log(prices))
    return float(np.std(log_rets, ddof=1) * np.sqrt(ann_factor))


def load_chunks(csv_path: str) -> list[pd.DataFrame]:
    """
    Read CSV, sort by date, split into non-overlapping chunks of CHUNK_SIZE rows.
    Returns list of DataFrames; each has columns [DATE_COL, VALUE_COL].
    """
    df = pd.read_csv(csv_path, parse_dates=[DATE_COL])
    df = df[[DATE_COL, VALUE_COL]].sort_values(DATE_COL).reset_index(drop=True)

    chunks = []
    n = len(df)
    for start in range(0, n, CHUNK_SIZE):
        end = start + CHUNK_SIZE
        chunk = df.iloc[start:end].copy().reset_index(drop=True)
        if len(chunk) >= MIN_CHUNK_ROWS:
            chunks.append(chunk)
    return chunks


def chunk_label(chunk_df: pd.DataFrame, idx: int) -> str:
    """Human-readable chunk label: chunk index + date range."""
    d0 = chunk_df[DATE_COL].iloc[0].strftime("%Y-%m-%d")
    d1 = chunk_df[DATE_COL].iloc[-1].strftime("%Y-%m-%d")
    return f"chunk{idx+1:02d}_{d0}_to_{d1}"


# ═══════════════════════════════════════════════════════════════════════════
# ── PER-CHUNK ANALYSIS ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def run_chunk(chunk_df: pd.DataFrame, chunk_idx: int) -> dict:
    """
    Derive parameters from `chunk_df`, run the delta-hedging simulation,
    return a dict with everything needed for plotting.
    """
    prices   = chunk_df[VALUE_COL].to_numpy(dtype=float)
    n_rows   = len(prices)
    S0       = float(prices[0])
    sigma    = realised_vol(prices)
    T        = n_rows / TRADING_DAYS_PER_YEAR
    N        = n_rows                        # one step per trading day
    K_lo     = S0 * K_LO_FRAC
    K_hi     = S0 * K_HI_FRAC

    print(
        f"  chunk {chunk_idx+1}: rows={n_rows}  S0={S0:.2f}  "
        f"σ={sigma:.4f}  K=[{K_lo:.2f}, {K_hi:.2f}]  T={T:.4f}"
    )

    payoff = RampStripPayoff(
        S0=S0, T=T, N=N,
        K_lo=K_lo, K_hi=K_hi,
        r=R, q=Q, sigma=sigma,
    )

    sim = DeltaHedgingSimulation(
        payoff,
        n_paths=N_PATHS,
        hedge_freq=HEDGE_FREQ,
        seed=SEED + chunk_idx,
        lambda_cap=CAP,
        retention=RETENTION,
    )
    _, vl = sim.run()

    grow       = np.exp(R * T)
    premium    = float(vl.future_pvs[0, 0])
    payoff_arr = vl.realised_pvs[:, -1] * grow
    ratio      = payoff_arr / premium

    lam = vl.lambda_trace          # (n_paths, N+1)
    b1  = N // 3
    b2  = 2 * N // 3
    b3  = N

    slices = [
        dict(
            max_lam     = lam[:, 1       : b1 + 1].max(axis=1),
            cdf_ratio   = vl.option_values[:, b1] / premium,
            label       = f"T/3  (steps 1–{b1})",
        ),
        dict(
            max_lam     = lam[:, b1 + 1  : b2 + 1].max(axis=1),
            cdf_ratio   = vl.option_values[:, b2] / premium,
            label       = f"2T/3  (steps {b1+1}–{b2})",
        ),
        dict(
            max_lam     = lam[:, b2 + 1  : b3 + 1].max(axis=1),
            cdf_ratio   = vl.option_values[:, b3] / premium,
            label       = f"T  (steps {b2+1}–{b3})",
        ),
    ]

    return dict(
        payoff=payoff, premium=premium, ratio=ratio,
        slices=slices, sigma=sigma, S0=S0, T=T, N=N,
        chunk_df=chunk_df, chunk_idx=chunk_idx,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ── PLOTTING ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def plot_chunk(result: dict, label: str, out_dir: str) -> str:
    """Draw the 3-panel scatter for one chunk; return the saved filename."""
    payoff   = result["payoff"]
    premium  = result["premium"]
    slices   = result["slices"]
    sigma    = result["sigma"]
    S0       = result["S0"]
    T        = result["T"]
    chunk_df = result["chunk_df"]

    # Date range for the title
    d0 = chunk_df[DATE_COL].iloc[0].strftime("%Y-%m-%d")
    d1 = chunk_df[DATE_COL].iloc[-1].strftime("%Y-%m-%d")

    rng   = np.random.default_rng(0)
    idx_s = rng.choice(N_PATHS, size=min(N_PATHS, N_SCATTER), replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor=BG)
    fig.suptitle(
        f"Peak λ (max notional multiplier in slice) vs payoff-to-premium ratio\n"
        f"Period: {d0} → {d1}   |   S0={S0:.2f}   "
        f"K=[{payoff.ramps[0].K_lo:.2f}, {payoff.ramps[0].K_hi:.2f}]   "
        f"σ={sigma:.4f} (realised)   T={T:.3f}yr   cap={CAP}   "
        f"{N_PATHS//1_000}k paths   |   premium={premium:.4f}",
        color=WHT, fontsize=11, fontweight="bold",
    )

    for ax, sl, col in zip(axes, slices, PANEL_COLS):
        max_lam   = sl["max_lam"]
        cdf_ratio = sl["cdf_ratio"]
        snap_lbl  = sl["label"]

        ax.set_facecolor(AX)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors="#888888")
        ax.xaxis.label.set_color(DIM)
        ax.yaxis.label.set_color(DIM)
        ax.grid(True, color=GRID, linewidth=0.7)

        # ── scatter ──────────────────────────────────────────────────────────
        ax.scatter(
            cdf_ratio[idx_s], max_lam[idx_s],
            s=4, alpha=0.20, color=col, linewidths=0, rasterized=True,
        )

        # ── bucket mean ± IQR ────────────────────────────────────────────────
        order    = np.argsort(cdf_ratio)
        buckets  = np.array_split(order, N_BUCKETS)
        bx       = np.array([cdf_ratio[b].mean()             for b in buckets])
        by_mean  = np.array([max_lam[b].mean()               for b in buckets])
        by_p25   = np.array([np.percentile(max_lam[b], 25)   for b in buckets])
        by_p75   = np.array([np.percentile(max_lam[b], 75)   for b in buckets])

        ax.fill_between(bx, by_p25, by_p75, alpha=0.25, color=col)
        ax.plot(bx, by_mean, lw=2.4, color=col, zorder=4, label="bucket mean λ")

        # reference lines
        ax.axhline(CAP, color=RED_LINE, lw=1.5, ls="--", alpha=0.9,
                   label=f"cap = {CAP}")
        ax.axhline(1.0, color=WHT,     lw=0.9, ls=":",  alpha=0.4,
                   label="λ = 1  (no adj.)")
        ax.axvline(1.0, color=WHT,     lw=0.9, ls=":",  alpha=0.4,
                   label="payoff = premium")

        # ── CDF on secondary y ───────────────────────────────────────────────
        ax2 = ax.twinx()
        ax2.set_facecolor("none")
        sorted_cdf = np.sort(cdf_ratio)
        cdf_y      = np.linspace(0, 1, len(sorted_cdf))
        ax2.plot(sorted_cdf, cdf_y, lw=1.0, color=CDF_COL, alpha=0.7,
                 zorder=3, label="CDF (total MTM / premium)")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Cumulative probability", fontsize=8, color=CDF_COL)
        ax2.tick_params(axis="y", colors=CDF_COL, labelsize=7)
        ax2.spines["right"].set_color(CDF_COL)
        for sp in ["top", "left", "bottom"]:
            ax2.spines[sp].set_visible(False)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, facecolor=BG,
                  edgecolor=GRID, labelcolor=DIM, markerscale=3,
                  loc="center right")

        corr       = float(np.corrcoef(cdf_ratio, max_lam)[0, 1])
        pct_capped = float((max_lam >= CAP * 0.9999).mean() * 100)
        x_std = cdf_ratio.std()
        x_iqr = float(np.percentile(cdf_ratio, 75) - np.percentile(cdf_ratio, 25))

        ax.set_title(
            f"Slice up to  {snap_lbl}\n"
            f"corr = {corr:.3f}   |   {pct_capped:.1f}% paths at cap",
            color=WHT, fontweight="bold", fontsize=10,
        )
        ax.set_xlabel(
            "Total MTM / premium at slice boundary  "
            "(0 = fully OTM,  1 = breakeven)",
            fontsize=9,
        )
        ax.set_ylabel("Max λ in slice", fontsize=9)
        ax.set_ylim(0, CAP * 1.1)
        ax.legend(loc="lower right", fontsize=7, facecolor=BG,
                  edgecolor=GRID, labelcolor=DIM)

        # std / IQR annotation on x-axis spread
        ax.text(0.98, 0.52, f"x std = {x_std:.3f}\nx IQR = {x_iqr:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color=CDF_COL,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=GRID, alpha=0.8))

        # zone labels
        yhi = ax.get_ylim()[1]
        ax.text(0.12, yhi * 0.97, "OTM", color=TEAL, fontsize=9,
                ha="center", va="top", alpha=0.9, fontweight="bold")
        ax.text(2.8,  yhi * 0.97, "ITM / paying", color=ORG, fontsize=9,
                ha="center", va="top", alpha=0.9)

        print(
            f"    [{snap_lbl[:14]:14s}]  corr={corr:+.4f}  "
            f"mean_max_lam={max_lam.mean():.4f}  pct_at_cap={pct_capped:.1f}%  "
            f"x_range=[{cdf_ratio.min():.3f}, {cdf_ratio.max():.3f}]"
        )

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = str(Path(out_dir) / f"{OUT_PREFIX}_{label}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → saved {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# ── MAIN ───────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"Loading '{CSV_PATH}' …")
    chunks = load_chunks(CSV_PATH)
    print(f"  {len(chunks)} chunk(s) of ≤{CHUNK_SIZE} rows found.\n")

    summary_rows = []

    for idx, chunk_df in enumerate(chunks):
        label = chunk_label(chunk_df, idx)
        print(f"── Chunk {idx+1}/{len(chunks)}: {label}")
        result = run_chunk(chunk_df, idx)
        plot_chunk(result, label, OUT_DIR)

        summary_rows.append(
            dict(
                chunk       = idx + 1,
                start       = chunk_df[DATE_COL].iloc[0].strftime("%Y-%m-%d"),
                end         = chunk_df[DATE_COL].iloc[-1].strftime("%Y-%m-%d"),
                rows        = len(chunk_df),
                S0          = round(result["S0"], 2),
                sigma       = round(result["sigma"], 4),
                T_yr        = round(result["T"], 4),
                K_lo        = round(result["payoff"].ramps[0].K_lo, 2),
                K_hi        = round(result["payoff"].ramps[0].K_hi, 2),
                premium     = round(result["premium"], 5),
            )
        )
        print()

    print("═" * 90)
    print("SUMMARY")
    print("═" * 90)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
