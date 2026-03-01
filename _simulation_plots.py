"""
_simulation_plots.py
--------------------
Generates the two built-in diagnostic figures from ramp_hedging.py:

  simulation_diagnostic.png  –  four-panel single-run diagnostic
                                 (spot paths, option MTM, hedge vs disc.future,
                                  final P&L histogram)

  simulation_comparison.png  –  four-panel plain vs vol-lock comparison
                                 (P&L overlaid histograms, tracking-error fan,
                                  lambda trace with % capped, TE std over time)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ramp_hedging import RampStripPayoff, DeltaHedgingSimulation, plot_simulation, plot_comparison

# ── CONFIG ───────────────────────────────────────────────────────────────────
S0      = 100
K_LO    = 102
K_HI    = 105
T       = 1
N       = 250
R       = 0.035
Q       = 0.035
SIGMA   = 0.04
CAP     = 2.0
N_PATHS = 50_000
SEED    = 421
# ─────────────────────────────────────────────────────────────────────────────

payoff = RampStripPayoff(S0=S0, T=T, N=N, K_lo=K_LO, K_hi=K_HI,
                         r=R, q=Q, sigma=SIGMA)

sim = DeltaHedgingSimulation(payoff, n_paths=N_PATHS, hedge_freq=1,
                              seed=SEED, lambda_cap=CAP, retention=0.0)

print("Running simulation …")
res_plain, res_vl = sim.run()
print("Done.\n")

# ── Figure 1a: single-run diagnostic (plain delta-hedge, no vol-lock) ────────
fig1a = plot_simulation(res_plain, n_show=40)
fig1a.texts[0].set_text(fig1a.texts[0].get_text() + "  [NO VOL-LOCK]")
out1a = "simulation_diagnostic_plain.png"
fig1a.savefig(out1a, dpi=150, bbox_inches="tight")
plt.close(fig1a)
print(f"Saved → {out1a}")

# ── Figure 1b: single-run diagnostic (vol-lock result) ───────────────────────
fig1 = plot_simulation(res_vl, n_show=40)
out1 = "simulation_diagnostic.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"Saved → {out1}")

# ── Figure 2: plain vs vol-lock comparison ───────────────────────────────────
fig2 = plot_comparison(res_plain, res_vl)
out2 = "simulation_comparison.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved → {out2}")
