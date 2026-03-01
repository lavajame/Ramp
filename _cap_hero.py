import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from ramp_hedging import RampStripPayoff, DeltaHedgingSimulation

# ── setup ──────────────────────────────────────────────────────────────────
payoff  = RampStripPayoff(S0=100, T=1, N=250, K_lo=102, K_hi=105,
                           r=0.035, q=0.035, sigma=0.04)
grow    = np.exp(payoff.r * payoff.T)
N_PATHS = 50_000

caps   = [1.05, 1.10, 1.20, 1.30, 1.50, 2.00, 3.00, np.inf]
clbls  = ['1.05', '1.10', '1.20', '1.30', '1.50', '2.00', '3.00', '∞']

# run sims
rows = []
for lc, lbl in zip(caps, clbls):
    sim = DeltaHedgingSimulation(payoff, n_paths=N_PATHS, hedge_freq=1,
                                  seed=421, lambda_cap=lc, retention=0.0)
    _, vl = sim.run()
    pnl = vl.hedge_port_value[:, -1]
    ctv = vl.realised_pvs[:, -1] * grow
    lam = vl.lambda_trace
    var5  = np.percentile(pnl, 5)
    cvar5 = pnl[pnl < var5].mean()
    rows.append(dict(cap=lc, lbl=lbl, pnl=pnl, ctv=ctv, lam=lam,
                     std=pnl.std(), var5=var5, cvar5=cvar5,
                     clt_mean=ctv.mean(), clt_p10=np.percentile(ctv, 10),
                     clt_p90=np.percentile(ctv, 90), times=vl.times))
    print(f"cap={lbl:>5}  std={pnl.std():.5f}  clt_mean={ctv.mean():.4f}")

ref = rows[-1]  # cap=∞

# ── derived metrics ────────────────────────────────────────────────────────
for r in rows:
    r['std_red_pct']  = (ref['std']      - r['std'])      / ref['std']      * 100
    # CVaR is negative; improvement = less negative = higher value
    r['cvar_imp_pct'] = (r['cvar5'] - ref['cvar5']) / abs(ref['cvar5']) * 100
    r['clt_loss_pct'] = (ref['clt_mean'] - r['clt_mean']) / ref['clt_mean'] * 100
    r['efficiency']   = r['cvar_imp_pct'] / r['clt_loss_pct'] if r['clt_loss_pct'] > 0.001 else np.inf

# ── colour palette: warm to cool by cap (low cap = danger red, high cap = safe teal) ──
PALETTE = ['#d62728','#d6722a','#e09430','#e8c03a',
           '#82c45e','#2a9d8f','#1a6e8a','#0d2f4d']
for r, col in zip(rows, PALETTE):
    r['col'] = col

# ══════════════════ FIGURE ═══════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 11), facecolor='#0e1117')
fig.patch.set_facecolor('#0e1117')

TITLE_KW  = dict(color='white', fontweight='bold')
LABEL_KW  = dict(color='#cccccc', fontsize=10)
TICK_KW   = dict(colors='#888888')
GRID_KW   = dict(color='#2a2a3e', linewidth=0.7)
SPINE_COL = '#2a2a3e'

def style_ax(ax):
    ax.set_facecolor('#13161f')
    for sp in ax.spines.values(): sp.set_color(SPINE_COL)
    ax.tick_params(colors='#888888', which='both')
    ax.xaxis.label.set_color('#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.grid(True, **GRID_KW)

# ── layout: 1 wide hero + 2 supporting on right, 1 bottom bar ──────────────
gs = fig.add_gridspec(2, 3, figure=fig,
                      left=0.06, right=0.97,
                      top=0.88, bottom=0.09,
                      hspace=0.45, wspace=0.35)

# ── PANEL A (tall left): Pareto scatter ────────────────────────────────────
axA = fig.add_subplot(gs[:, 0])
style_ax(axA)

# gradient background zones by client cost
axA.axvspan(-0.05, 0.20, alpha=0.07, color='#2a9d8f')   # sweet-spot: big tail relief, tiny client cost
axA.axvspan(0.20,  1.0,  alpha=0.04, color='#e8c03a')
axA.axvspan(1.0,   4.0,  alpha=0.04, color='#d62728')

for r in rows:
    axA.scatter(r['clt_loss_pct'], r['cvar_imp_pct'],
                s=220, color=r['col'], zorder=5,
                edgecolors='white', linewidths=0.6)
    axA.annotate(f"cap = {r['lbl']}",
                 xy=(r['clt_loss_pct'], r['cvar_imp_pct']),
                 xytext=(6, 4), textcoords='offset points',
                 fontsize=9, color=r['col'], fontweight='bold',
                 path_effects=[pe.withStroke(linewidth=2, foreground='#0e1117')])

# connect curve
xs = [r['clt_loss_pct']  for r in rows]
ys = [r['cvar_imp_pct']  for r in rows]
axA.plot(xs, ys, '-', color='#555577', lw=1.2, zorder=2)

# zero axes
axA.axhline(0, color='#555577', lw=1.0, ls='--')
axA.axvline(0, color='#555577', lw=1.0, ls='--')

# annotate zones
axA.text(0.06, 10, 'SWEET SPOT', color='#2a9d8f',
         fontsize=9, fontweight='bold', alpha=0.9,
         path_effects=[pe.withStroke(linewidth=2, foreground='#13161f')])
axA.text(2.2, 10, 'Client pays\ntoo much', color='#d62728',
         fontsize=8, alpha=0.8, ha='center')

axA.set_xlabel('Client mean payoff reduction  (% vs no cap)', **LABEL_KW)
axA.set_ylabel('Dealer worst-5% tail improvement  (% of baseline CVaR)', **LABEL_KW)
axA.set_title('A  |  Tail Risk Efficiency Frontier', **TITLE_KW, fontsize=12, pad=10)

# callout arrow for cap=3.0
r3 = next(r for r in rows if r['lbl'] == '3.00')
axA.annotate('',
             xy=(r3['clt_loss_pct'], r3['cvar_imp_pct']),
             xytext=(r3['clt_loss_pct'] + 0.45, r3['cvar_imp_pct'] - 8),
             arrowprops=dict(arrowstyle='->', color='#2a9d8f', lw=1.5))
axA.text(r3['clt_loss_pct'] + 0.47, r3['cvar_imp_pct'] - 10,
         '+33% tail improvement\n−0.05% client payoff',
         color='#2a9d8f', fontsize=8,
         path_effects=[pe.withStroke(linewidth=2, foreground='#13161f')])

# ── PANEL B (top middle): P&L distribution overlay ─────────────────────────
axB = fig.add_subplot(gs[0, 1])
style_ax(axB)

show = ['∞', '3.00', '2.00', '1.50']
all_pnl = np.concatenate([r['pnl'] for r in rows if r['lbl'] in show])
bins = np.linspace(np.percentile(all_pnl, 0.2), np.percentile(all_pnl, 99.8), 90)
for r in rows:
    if r['lbl'] not in show: continue
    axB.hist(r['pnl'], bins=bins, alpha=0.55, color=r['col'], density=True,
             label=f"cap {r['lbl']}  σ={r['std']:.4f}", linewidth=0)

axB.axvline(0, color='white', lw=1.0, ls='--', alpha=0.4)
axB.set_xlabel('Dealer P&L at maturity', **LABEL_KW)
axB.set_ylabel('Density', **LABEL_KW)
axB.set_title('B  |  P&L Distribution', **TITLE_KW, fontsize=11, pad=8)
leg = axB.legend(fontsize=8, facecolor='#13161f', edgecolor='#2a2a3e', labelcolor='#cccccc')

# ── PANEL C (top right): Client CDF ────────────────────────────────────────
axC = fig.add_subplot(gs[0, 2])
style_ax(axC)

for r in rows:
    if r['lbl'] not in show: continue
    xs = np.sort(r['ctv'])
    ys = np.linspace(1 / len(xs), 1, len(xs))
    axC.plot(xs, ys, color=r['col'], lw=2.0, label=f"cap {r['lbl']}")

axC.axvline(ref['clt_mean'], color='#888888', lw=0.9, ls=':', alpha=0.7)
axC.set_xlabel('Client payoff at maturity', **LABEL_KW)
axC.set_ylabel('Cumulative probability', **LABEL_KW)
axC.set_title('C  |  Client Payoff CDF', **TITLE_KW, fontsize=11, pad=8)
axC.legend(fontsize=8, facecolor='#13161f', edgecolor='#2a2a3e', labelcolor='#cccccc')

# zoom in on upper tail to show cap effect
xlim = axC.get_xlim()
axC.set_xlim(0, xlim[1])

# ── PANEL D (bottom middle+right): Efficiency bars ─────────────────────────
axD = fig.add_subplot(gs[1, 1:])
style_ax(axD)

finite = [r for r in rows if not np.isinf(r['cap'])]
x      = np.arange(len(finite))
width  = 0.35

# Scale client loss for visibility: find a round multiplier so max client bar is ~30% of dealer bar
max_cvar  = max(r['cvar_imp_pct'] for r in finite)
max_clt   = max(r['clt_loss_pct'] for r in finite)
scale_clt = round(max_cvar / max_clt / 5) * 5   # round to nearest 5
scale_clt = max(scale_clt, 5)

bars1 = axD.bar(x - width/2, [r['cvar_imp_pct']  for r in finite],
                width, color=[r['col'] for r in finite], alpha=0.85,
                label='Dealer worst-5% tail improvement %', edgecolor='none')
bars2 = axD.bar(x + width/2, [r['clt_loss_pct'] * scale_clt for r in finite],
                width, color=[r['col'] for r in finite], alpha=0.40,
                hatch='//', edgecolor=[r['col'] for r in finite], linewidth=0.5,
                label=f'Client payoff loss ×{scale_clt} (scaled for visibility)')

# value labels on CVaR bar
for bar, r in zip(bars1, finite):
    h = bar.get_height()
    axD.text(bar.get_x() + bar.get_width()/2, h + 0.5,
             f"+{h:.0f}%", ha='center', va='bottom', fontsize=8,
             color='white', fontweight='bold',
             path_effects=[pe.withStroke(linewidth=1.5, foreground='#13161f')])

# client loss labels (actual %)
for bar, r in zip(bars2, finite):
    axD.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f"−{r['clt_loss_pct']:.2f}%", ha='center', va='bottom',
             fontsize=7, color='#aaaaaa',
             path_effects=[pe.withStroke(linewidth=1, foreground='#13161f')])

axD.axhline(0, color='#555577', lw=0.9)
axD.set_xticks(x)
axD.set_xticklabels([f"cap={r['lbl']}" for r in finite], color='#cccccc', fontsize=9)
axD.set_ylabel('% improvement vs no-cap baseline', **LABEL_KW)
axD.set_title('D  |  Dealer Worst-5% Tail Improvement vs Client Payoff Cost  (retention = 0)',
              **TITLE_KW, fontsize=11, pad=8)
leg = axD.legend(fontsize=8, facecolor='#13161f', edgecolor='#2a2a3e', labelcolor='#cccccc',
                 loc='upper left')

# highlight sweet-spot zone (cap=2–3: large CVaR gain, tiny client cost)
axD.axvspan(4.5, 6.5, color='#2a9d8f', alpha=0.07, zorder=0)
ylim_top = axD.get_ylim()[1] if axD.get_ylim()[1] > 0 else 80
axD.text(5.5, ylim_top * 0.88, 'sweet\nspot', color='#2a9d8f',
         ha='center', fontsize=9, fontweight='bold',
         path_effects=[pe.withStroke(linewidth=2, foreground='#13161f')])

# ── master title + subtitle ─────────────────────────────────────────────────
fig.text(0.5, 0.955,
         'Lambda Cap: Controlling Hedger Tail Risk at Minimal Cost to the Client',
         ha='center', va='top', fontsize=15, fontweight='bold', color='white')
fig.text(0.5, 0.933,
         r'S₀=100  K=[102, 105]  σ=0.04  r=3.5%  T=1yr  N=250 daily hedges  50,000 paths  |  retention = 0  |  tail metric = CVaR 5% (mean of worst-5% losses)',
         ha='center', va='top', fontsize=9.5, color='#888888')

fig.savefig('cap_sweep_hero.png', dpi=160, bbox_inches='tight',
            facecolor='#0e1117')
print('Saved -> cap_sweep_hero.png')
