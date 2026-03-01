"""
_lambda_payoff_scatter.py
-------------------------
Scatterplot: per-path mean lambda adjustment vs client payoff at maturity.

Hypothesis: large upward lambda adjustments (hedge account > strip fair value)
occur predominantly when the ramp strip is deep OTM — i.e. on paths that will
ultimately yield little or no payoff to the client.

Layout: two panels side by side
  Left  – hexbin density  (all 50k paths)
  Right – mean lambda by payoff percentile bucket (confirms the pattern cleanly)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ramp_hedging import RampStripPayoff, DeltaHedgingSimulation

# ── parameters (match existing scripts) ─────────────────────────────────────
payoff  = RampStripPayoff(S0=100, T=1, N=250, K_lo=102, K_hi=105,
                          r=0.035, q=0.035, sigma=0.04)
grow    = np.exp(payoff.r * payoff.T)
N_PATHS = 50_000
CAP     = 2.0          # a representative cap; shows both capped and uncapped paths

sim = DeltaHedgingSimulation(payoff, n_paths=N_PATHS, hedge_freq=1,
                              seed=421, lambda_cap=CAP, retention=0.0)
_, vl = sim.run()

# ── per-path summary stats ───────────────────────────────────────────────────
# lambda_trace[:, 0] = 1.0 always (before any adjustment)
# adjustments are made at steps 1 .. N-1  (last step k=N has no ramps left)
lam_trace   = vl.lambda_trace          # (n_paths, N+1)

mean_lam    = lam_trace[:, 1:].mean(axis=1)       # avg λ over steps 1..N  per path
max_lam     = lam_trace[:, 1:].max(axis=1)        # peak λ per path
final_lam   = lam_trace[:, -1]                    # λ at final step

client_payoff = vl.realised_pvs[:, -1] * grow     # undiscounted payoff grown to maturity

# fraction of steps where lambda was at the cap (binding)
frac_capped  = (lam_trace[:, 1:] >= CAP * 0.9999).mean(axis=1)

# terminal spot relative to strikes
S_T      = vl.spot_paths[:, -1]
K_lo     = payoff.ramps[0].K_lo   # 102
K_hi     = payoff.ramps[0].K_hi   # 105

print(f"N_PATHS={N_PATHS}  cap={CAP}")
print(f"client payoff: mean={client_payoff.mean():.4f}  "
      f"p10={np.percentile(client_payoff,10):.4f}  "
      f"p90={np.percentile(client_payoff,90):.4f}")
print(f"mean_lam: mean={mean_lam.mean():.4f}  max={mean_lam.max():.4f}")
print(f"fraction of paths hitting cap at ≥1 step: "
      f"{(frac_capped > 0).mean()*100:.1f}%")

# correlation
corr_mean = np.corrcoef(client_payoff, mean_lam)[0, 1]
corr_max  = np.corrcoef(client_payoff, max_lam)[0, 1]
print(f"corr(payoff, mean_lam)={corr_mean:.4f}  "
      f"corr(payoff, max_lam)={corr_max:.4f}")

# ── bucket analysis: 20 equal-count payoff percentile buckets ────────────────
N_BUCKETS   = 20
bucket_idx  = np.argsort(client_payoff)
bucket_ids  = np.array_split(bucket_idx, N_BUCKETS)

bucket_payoff_mid  = np.array([client_payoff[b].mean() for b in bucket_ids])
bucket_mean_lam    = np.array([mean_lam[b].mean()       for b in bucket_ids])
bucket_max_lam     = np.array([max_lam[b].mean()        for b in bucket_ids])
bucket_frac_capped = np.array([frac_capped[b].mean()    for b in bucket_ids])
bucket_n_capped    = np.array([(frac_capped[b] > 0).mean() * 100 for b in bucket_ids])

# ── figure ───────────────────────────────────────────────────────────────────
BG   = "#0e1117"
AX   = "#13161f"
GRID = "#2a2a3e"
WHT  = "white"
DIM  = "#cccccc"
TEAL = "#2a9d8f"
ORG  = "#e09430"
RED  = "#d62728"

fig = plt.figure(figsize=(20, 9), facecolor=BG)
fig.suptitle(
    f"Are large λ adjustments wasted on OTM paths?\n"
    f"S0=100  K=[{K_lo}, {K_hi}]  σ={payoff.sigma}  T={payoff.T}  "
    f"cap={CAP}  N={N_PATHS//1_000}k paths  |  "
    f"corr(payoff, mean-λ) = {corr_mean:.3f}",
    color=WHT, fontsize=12, fontweight="bold",
)

gs = gridspec.GridSpec(1, 3, figure=fig,
                       left=0.06, right=0.97,
                       top=0.88, bottom=0.10,
                       wspace=0.35)

def style(ax):
    ax.set_facecolor(AX)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors="#888888", which="both")
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)
    ax.grid(True, color=GRID, linewidth=0.7)

# ── Panel A: hexbin density scatter ─────────────────────────────────────────
axA = fig.add_subplot(gs[0])
style(axA)

hb = axA.hexbin(
    client_payoff, mean_lam,
    gridsize=60,
    cmap="YlOrRd",
    mincnt=1,
    bins="log",
)
cb = fig.colorbar(hb, ax=axA, pad=0.02)
cb.set_label("log₁₀(path count)", color=DIM, fontsize=9)
cb.ax.yaxis.set_tick_params(color="#888888")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="#888888")

axA.axhline(CAP,     color=RED,  lw=1.5, ls="--", label=f"cap = {CAP}")
axA.axhline(1.0,     color=WHT,  lw=0.8, ls=":",  alpha=0.5, label="λ = 1 (no adj.)")
axA.axvline(0.0,     color=TEAL, lw=0.8, ls=":",  alpha=0.5, label="zero payoff")

axA.set_xlabel("Client payoff at maturity  (undiscounted, grown to T)", fontsize=10)
axA.set_ylabel("Mean λ across simulation  (per path)", fontsize=10)
axA.set_title("A  |  Path density: mean λ vs client payoff", color=WHT,
              fontweight="bold", fontsize=11)
axA.legend(fontsize=9, facecolor=BG, edgecolor=GRID,
           labelcolor=DIM, loc="upper right")

# ── Panel B: bucket means – mean λ by payoff percentile ─────────────────────
axB = fig.add_subplot(gs[1])
style(axB)

bars = axB.bar(
    range(N_BUCKETS), bucket_mean_lam,
    color=[TEAL if v <= 1.0 else ORG if v < CAP else RED
           for v in bucket_mean_lam],
    edgecolor=BG, linewidth=0.5,
)
axB.plot(range(N_BUCKETS), bucket_max_lam,
         "o--", color=RED, lw=1.4, ms=5, zorder=3, label="mean of per-path MAX λ")

axB.axhline(CAP, color=RED,  lw=1.4, ls="--", alpha=0.7, label=f"cap = {CAP}")
axB.axhline(1.0, color=WHT,  lw=0.8, ls=":",  alpha=0.5, label="λ = 1")

axB.set_xticks(range(0, N_BUCKETS, 4))
axB.set_xticklabels(
    [f"p{int((i+0.5)/N_BUCKETS*100)}" for i in range(0, N_BUCKETS, 4)],
    fontsize=8, color="#888888",
)
axB.set_xlabel("Client payoff percentile bucket  (left = low payoff / OTM)", fontsize=10)
axB.set_ylabel("Mean λ within bucket", fontsize=10)
axB.set_title("B  |  Mean λ by payoff percentile", color=WHT,
              fontweight="bold", fontsize=11)
axB.legend(fontsize=9, facecolor=BG, edgecolor=GRID, labelcolor=DIM)

# annotate the OTM vs ITM boundary in Panel B
zero_payoff_frac = (client_payoff == 0).mean()
otm_bucket = int(zero_payoff_frac * N_BUCKETS)
if 0 < otm_bucket < N_BUCKETS:
    axB.axvline(otm_bucket - 0.5, color=TEAL, lw=1.2, ls=":",
                label="~zero-payoff boundary")
    axB.text(otm_bucket * 0.4, axB.get_ylim()[1] * 0.95,
             "← OTM / zero payoff", color=TEAL, fontsize=8, ha="center",
             va="top")
    axB.text(otm_bucket + (N_BUCKETS - otm_bucket) * 0.5,
             axB.get_ylim()[1] * 0.95,
             "ITM / paying →", color=ORG, fontsize=8, ha="center", va="top")

# ── Panel C: % paths hitting cap, binned by payoff percentile ───────────────
axC = fig.add_subplot(gs[2])
style(axC)

axC.bar(range(N_BUCKETS), bucket_n_capped,
        color=[RED if v > 50 else ORG if v > 20 else TEAL
               for v in bucket_n_capped],
        edgecolor=BG, linewidth=0.5)

axC.set_xticks(range(0, N_BUCKETS, 4))
axC.set_xticklabels(
    [f"p{int((i+0.5)/N_BUCKETS*100)}" for i in range(0, N_BUCKETS, 4)],
    fontsize=8, color="#888888",
)
axC.set_xlabel("Client payoff percentile bucket  (left = low payoff / OTM)", fontsize=10)
axC.set_ylabel("% of paths that hit the cap (≥1 step)", fontsize=10)
axC.set_title("C  |  Cap-binding rate by payoff percentile", color=WHT,
              fontweight="bold", fontsize=11)
axC.set_ylim(0, 105)

if 0 < otm_bucket < N_BUCKETS:
    axC.axvline(otm_bucket - 0.5, color=TEAL, lw=1.2, ls=":")
    axC.text(otm_bucket * 0.4, 98,
             "← OTM", color=TEAL, fontsize=8, ha="center")
    axC.text(otm_bucket + (N_BUCKETS - otm_bucket) * 0.5, 98,
             "ITM →", color=ORG, fontsize=8, ha="center")

# print summary stats per panel C
for i, (pm, pn) in enumerate(zip(bucket_payoff_mid, bucket_n_capped)):
    if i % 4 == 0:
        axC.text(i, pn + 2, f"{pn:.0f}%", fontsize=6,
                 ha="center", va="bottom", color=WHT, alpha=0.7)

out = "lambda_payoff_scatter.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {out}")
