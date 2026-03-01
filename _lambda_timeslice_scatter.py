"""
_lambda_timeslice_scatter.py
----------------------------
3×1 scatter: x = payoff / initial-premium,  y = max λ within each time slice.

Panel 1: max λ over steps in  [0,     T/3]
Panel 2: max λ over steps in  (T/3,  2T/3]
Panel 3: max λ over steps in  (2T/3,   T]

The mean λ is a poor signal because on OTM paths λ spikes to the cap then
collapses back as the strip's fair value approaches zero — mean washes out.
Max λ within a slice preserves the spike and clearly separates OTM from ITM.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ramp_hedging import RampStripPayoff, DeltaHedgingSimulation

# ── setup ────────────────────────────────────────────────────────────────────
payoff  = RampStripPayoff(S0=100, T=1, N=250, K_lo=97, K_hi=105,
                          r=0.035, q=0.035, sigma=0.04)
grow    = np.exp(payoff.r * payoff.T)
N_PATHS = 50_000
CAP     = 2.0

sim = DeltaHedgingSimulation(payoff, n_paths=N_PATHS, hedge_freq=1,
                              seed=421, lambda_cap=CAP, retention=0.0)
_, vl = sim.run()

N    = payoff.N              # 250
lam  = vl.lambda_trace       # (n_paths, N+1);  col 0 = t=0 (always 1.0)

# initial premium
premium = vl.future_pvs[0, 0]           # same for all paths at t=0

# payoff / premium ratio per path
payoff_arr = vl.realised_pvs[:, -1] * grow   # undiscounted, grown to T
ratio      = payoff_arr / premium             # x-axis

# ── time-slice boundaries (step indices into lambda_trace cols 1..N) ─────────
b1 = N // 3          # ~83
b2 = 2 * N // 3      # ~167
b3 = N               # 250

# max λ within each slice — preserves OTM spike before it collapses
max_lam_s1 = lam[:, 1       : b1 + 1].max(axis=1)
max_lam_s2 = lam[:, b1 + 1  : b2 + 1].max(axis=1)
max_lam_s3 = lam[:, b2 + 1  : b3 + 1].max(axis=1)

# CDF x-data: total MTM at each slice boundary / premium
# option_values[:, k] = realised_pv(k) + exp(-r*t_k)*future_pv(k)  (t=0 NPV)
cdf_ratio_s1 = vl.option_values[:, b1] / premium
cdf_ratio_s2 = vl.option_values[:, b2] / premium
cdf_ratio_s3 = vl.option_values[:, b3] / premium

slices = [
    (max_lam_s1, f"T/3  (steps 1–{b1})",      cdf_ratio_s1),
    (max_lam_s2, f"2T/3  (steps {b1+1}–{b2})", cdf_ratio_s2),
    (max_lam_s3, f"T  (steps {b2+1}–{b3})",    cdf_ratio_s3),
]

# ── per-slice MTM decomposition ───────────────────────────────────────────────
# option_values = realised_pv + disc*future_pv.  We want to know how much of
# the x-axis spread comes from each component vs the actual spot sigma.
dt       = payoff.T / N
d0       = payoff.initial_delta()   # dV/dS at t=0, S=S0
slice_stats = []
for k in [b1, b2, b3]:
    t_k      = k * dt
    disc_k   = np.exp(-payoff.r * t_k)
    rpv_k    = vl.realised_pvs[:, k]
    fpv_k    = disc_k * vl.future_pvs[:, k]
    tot_k    = vl.option_values[:, k]
    sig_spot = payoff.sigma * np.sqrt(t_k)          # cumulative spot sigma at t_k
    # 1st-order estimate: how much does a 1-sigma spot move shift V / premium?
    dv_per_sig = d0 * payoff.S0 * sig_spot / premium * 100
    slice_stats.append(dict(
        sig_spot_pct = sig_spot * 100,
        std_rpv_prem = rpv_k.std() / premium,
        pct_future   = fpv_k.std() / (tot_k.std() + 1e-12) * 100,
        dv_per_sig   = dv_per_sig,
    ))

# ── subsample for scatter readability (5k points per panel) ──────────────────
rng   = np.random.default_rng(0)
idx_s = rng.choice(N_PATHS, size=min(N_PATHS, 5_000), replace=False)

# ── style ────────────────────────────────────────────────────────────────────
BG   = "#0e1117"
AX   = "#13161f"
GRID = "#2a2a3e"
WHT  = "white"
DIM  = "#cccccc"
TEAL = "#2a9d8f"
ORG  = "#e09430"
RED  = "#d62728"
PANEL_COLS = [TEAL, ORG, RED]

fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor=BG)
fig.suptitle(
    f"Peak λ (max notional multiplier in slice) vs payoff-to-premium ratio\n"
    f"S0=100  K=[{payoff.ramps[0].K_lo}, {payoff.ramps[0].K_hi}]  "
    f"σ={payoff.sigma}  T={payoff.T}  cap={CAP}  {N_PATHS//1_000}k paths  |  "
    f"premium = {premium:.4f}",
    color=WHT, fontsize=12, fontweight="bold",
)

CDF_COL = "#a0a8ff"   # pale blue — distinct from all panel colours

for ax, (lam_slice, snap_label, cdf_ratio), col, ss in zip(axes, slices, PANEL_COLS, slice_stats):
    ax.set_facecolor(AX)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors="#888888")
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)
    ax.grid(True, color=GRID, linewidth=0.7)

    # ── scatter of subsampled paths ──────────────────────────────────────────
    ax.scatter(
        cdf_ratio[idx_s], lam_slice[idx_s],
        s=4, alpha=0.20, color=col, linewidths=0, rasterized=True,
    )

    # ── overlay: bucket mean ± IQR in 40 equal-count payoff-ratio buckets ────
    order   = np.argsort(cdf_ratio)
    buckets = np.array_split(order, 40)
    bx      = np.array([cdf_ratio[b].mean()            for b in buckets])
    by_mean = np.array([lam_slice[b].mean()             for b in buckets])
    by_p25  = np.array([np.percentile(lam_slice[b], 25) for b in buckets])
    by_p75  = np.array([np.percentile(lam_slice[b], 75) for b in buckets])

    ax.fill_between(bx, by_p25, by_p75, alpha=0.25, color=col)
    ax.plot(bx, by_mean, lw=2.4, color=col, zorder=4, label="bucket mean λ")

    # reference lines
    ax.axhline(CAP, color=RED,  lw=1.5, ls="--", alpha=0.9, label=f"cap = {CAP}")
    ax.axhline(1.0, color=WHT,  lw=0.9, ls=":",  alpha=0.4, label="λ = 1  (no adj.)")
    ax.axvline(1.0, color=WHT,  lw=0.9, ls=":",  alpha=0.4, label="payoff = premium")

    # ── CDF of payoff/premium on secondary y-axis ────────────────────────────
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
    # merge legends from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, facecolor=BG, edgecolor=GRID,
              labelcolor=DIM, markerscale=3, loc="center right")

    corr = np.corrcoef(cdf_ratio, lam_slice)[0, 1]
    pct_capped = (lam_slice >= CAP * 0.9999).mean() * 100
    x_std = cdf_ratio.std()
    x_iqr = np.percentile(cdf_ratio, 75) - np.percentile(cdf_ratio, 25)
    ax.set_title(
        f"Slice up to  {snap_label}\n"
        f"corr = {corr:.3f}   |   {pct_capped:.1f}% paths at cap",
        color=WHT, fontweight="bold", fontsize=10,
    )
    ax.set_xlabel("Total MTM / premium at slice boundary  (0 = fully OTM,  1 = breakeven)", fontsize=9)
    ax.set_ylabel("Max λ in slice", fontsize=9)
    ax.set_ylim(0, CAP * 1.1)
    ax.legend(loc="lower right", fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=DIM)

    # std / IQR annotation on x-axis spread + MTM decomposition
    decomp_line = (
        f"x std = {x_std:.3f}   IQR = {x_iqr:.3f}\n"
        f"spot 1-sig = {ss['sig_spot_pct']:.2f}%   "
        f"dV/prem per sig = {ss['dv_per_sig']:.1f}%\n"
        f"future_pv drives {ss['pct_future']:.0f}% of x-spread\n"
        f"realised_pv std/prem = {ss['std_rpv_prem']:.4f}"
    )
    ax.text(0.98, 0.54, decomp_line,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color=CDF_COL,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=GRID, alpha=0.85))

    # OTM / ITM zone labels
    yhi = ax.get_ylim()[1]
    ax.text(0.12, yhi * 0.97, "OTM", color=TEAL, fontsize=9,
            ha="center", va="top", alpha=0.9, fontweight="bold")
    ax.text(2.8,  yhi * 0.97, "ITM / paying", color=ORG, fontsize=9,
            ha="center", va="top", alpha=0.9)

    print(f"Slice {snap_label[:12]:12s}  corr={corr:+.4f}  "
          f"max_lam_mean={lam_slice.mean():.4f}  pct_at_cap={pct_capped:.1f}%  "
          f"x_range=[{cdf_ratio.min():.3f}, {cdf_ratio.max():.3f}]")

plt.tight_layout(rect=(0, 0, 1, 0.93))
out = "lambda_timeslice_scatter_safe.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {out}")
