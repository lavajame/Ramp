import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ramp_hedging import RampStripPayoff, DeltaHedgingSimulation

payoff = RampStripPayoff(S0=100, T=1, N=250, K_lo=102, K_hi=105, r=0.035, q=0.035, sigma=0.04)
grow   = np.exp(payoff.r * payoff.T)
N_PATHS = 50_000

# ── grid ──────────────────────────────────────────────────────────────────
caps       = [1.1, 1.2, 1.5, 2.0, 3.0, np.inf]
retentions = [0.0, 0.10, 0.20, 0.40]
cap_labels = ['cap=1.1', 'cap=1.2', 'cap=1.5', 'cap=2.0', 'cap=3.0', 'cap=inf']
cap_colors = ['#1a1a2e', '#16213e', '#0f3460', '#e94560', '#f5a623', '#6a0dad']

print(f"{'cap':>8}  {'ret':>6}  {'dlr_std':>9}  {'twoway_std':>11}  {'std_red%':>9}  "
      f"{'clt_mean':>10}  {'clt_p10':>9}  {'VaR5%':>8}  {'CVaR5%':>8}")

rows = []
for lc, clbl, ccol in zip(caps, cap_labels, cap_colors):
    for ret in retentions:
        sim = DeltaHedgingSimulation(payoff, n_paths=N_PATHS, hedge_freq=1, seed=421,
                                      lambda_cap=lc, retention=ret)
        _, vl = sim.run()

        pnl  = vl.hedge_port_value[:, -1]
        buf  = vl.buffer_trace[:, -1]
        ctv  = vl.realised_pvs[:, -1] * grow

        # two-way: draw buffer on losses
        draw   = np.minimum(np.maximum(-pnl, 0.0), buf)
        tw_pnl = pnl + draw

        std_red = (pnl.std() - tw_pnl.std()) / pnl.std() * 100 if pnl.std() > 0 else 0.0
        var5    = np.percentile(tw_pnl, 5)
        cvar5   = tw_pnl[tw_pnl < var5].mean() if (tw_pnl < var5).any() else var5

        row = dict(cap=lc, cap_lbl=clbl, cap_col=ccol, ret=ret,
                   dlr_std=pnl.std(), tw_std=tw_pnl.std(), std_red=std_red,
                   dlr_mean=tw_pnl.mean(), var5=var5, cvar5=cvar5,
                   clt_mean=ctv.mean(), clt_p10=np.percentile(ctv, 10),
                   clt_p90=np.percentile(ctv, 90))
        rows.append(row)
        print(f"{clbl:>8}  {ret:>6.2f}  {pnl.std():>9.5f}  {tw_pnl.std():>11.5f}  "
              f"{std_red:>+8.2f}%  {ctv.mean():>10.4f}  {np.percentile(ctv,10):>9.4f}  "
              f"{var5:>8.4f}  {cvar5:>8.4f}")

# ── reference: cap=inf, ret=0 ──────────────────────────────────────────────
ref = next(r for r in rows if r['cap'] == np.inf and r['ret'] == 0.0)

# ── figures ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle(
    'Hedger Resilience vs Investor Performance Cost  |  Two-Way Reserve Buffer\n'
    'S0=100  K=[102,105]  sigma=0.04  50k paths  '
    '|  two-way = buffer drawn on dealer losses, funded by client surplus retention',
    fontweight='bold', fontsize=11)

# ── Panel 1: Pareto frontier  dealer two-way std vs client mean payoff ─────
ax = axes[0, 0]
for ccol, clbl in zip(cap_colors, cap_labels):
    pts = [r for r in rows if r['cap_lbl'] == clbl]
    xs  = [r['clt_mean'] for r in pts]
    ys  = [r['tw_std']   for r in pts]
    ax.plot(xs, ys, 'o-', color=ccol, lw=1.8, ms=7, label=clbl)
    for r in pts:
        lbl = f"r={r['ret']}"
        ax.annotate(lbl, (r['clt_mean'], r['tw_std']),
                    textcoords='offset points', xytext=(3, 3), fontsize=6, color=ccol, alpha=0.8)
ax.scatter([ref['clt_mean']], [ref['tw_std']], s=180, color='black', zorder=5, marker='*',
           label=f"Baseline std={ref['tw_std']:.4f}")
ax.set_xlabel('Client mean payoff (current money)', fontsize=9)
ax.set_ylabel('Dealer two-way P&L std  (gap risk)', fontsize=9)
ax.set_title('Pareto Frontier\n(lower-right = better: less dealer risk + more client payoff)', fontweight='bold')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# ── Panel 2: CVaR-5% frontier ─────────────────────────────────────────────
ax = axes[0, 1]
for ccol, clbl in zip(cap_colors, cap_labels):
    pts = [r for r in rows if r['cap_lbl'] == clbl]
    xs  = [r['clt_mean'] for r in pts]
    ys  = [r['cvar5']    for r in pts]
    ax.plot(xs, ys, 'o-', color=ccol, lw=1.8, ms=7, label=clbl)
ax.scatter([ref['clt_mean']], [ref['cvar5']], s=180, color='black', zorder=5, marker='*',
           label=f"Baseline CVaR={ref['cvar5']:.4f}")
ax.axhline(0, color='grey', lw=0.7, ls=':')
ax.set_xlabel('Client mean payoff', fontsize=9)
ax.set_ylabel('Dealer CVaR 5%  (higher = less severe tail loss)', fontsize=9)
ax.set_title('Left-Tail Risk vs Client Payoff\n(CVaR 5% — worst-5% average loss)', fontweight='bold')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# ── Panel 3: client p10 vs dealer std (tail-vs-tail) ──────────────────────
ax = axes[0, 2]
for ccol, clbl in zip(cap_colors, cap_labels):
    pts = [r for r in rows if r['cap_lbl'] == clbl]
    xs  = [r['clt_p10'] for r in pts]
    ys  = [r['tw_std']  for r in pts]
    ax.plot(xs, ys, 'o-', color=ccol, lw=1.8, ms=7, label=clbl)
ax.set_xlabel('Client p10 payoff  (unlucky scenario)', fontsize=9)
ax.set_ylabel('Dealer two-way std', fontsize=9)
ax.set_title('Tail-vs-Tail\n(client downside vs dealer gap risk)', fontweight='bold')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# ── Panel 4: heatmap dealer std (two-way) ─────────────────────────────────
ax = axes[1, 0]
cap_finite     = [c for c in caps if not np.isinf(c)]
cap_fin_lbls   = [l for c, l in zip(caps, cap_labels) if not np.isinf(c)]
heat_std = np.array([[next(r['tw_std'] for r in rows if r['cap'] == c and r['ret'] == ret)
                      for ret in retentions] for c in cap_finite])
im = ax.imshow(heat_std, aspect='auto', cmap='RdYlGn_r')
ax.set_xticks(range(len(retentions))); ax.set_xticklabels([f'ret={r}' for r in retentions])
ax.set_yticks(range(len(cap_finite))); ax.set_yticklabels(cap_fin_lbls)
ax.set_xlabel('Retention (two-way buffer)'); ax.set_ylabel('Lambda cap')
ax.set_title('Heatmap: Dealer Two-Way Std\n(green = less gap risk)', fontweight='bold')
for i in range(len(cap_finite)):
    for j in range(len(retentions)):
        ax.text(j, i, f'{heat_std[i, j]:.4f}', ha='center', va='center', fontsize=9, color='white')
fig.colorbar(im, ax=ax, label='Dealer P&L std')

# ── Panel 5: heatmap client mean payoff ───────────────────────────────────
ax = axes[1, 1]
heat_clt = np.array([[next(r['clt_mean'] for r in rows if r['cap'] == c and r['ret'] == ret)
                      for ret in retentions] for c in cap_finite])
im2 = ax.imshow(heat_clt, aspect='auto', cmap='RdYlGn')
ax.set_xticks(range(len(retentions))); ax.set_xticklabels([f'ret={r}' for r in retentions])
ax.set_yticks(range(len(cap_finite))); ax.set_yticklabels(cap_fin_lbls)
ax.set_xlabel('Retention'); ax.set_ylabel('Lambda cap')
ax.set_title('Heatmap: Client Mean Payoff\n(green = better for client)', fontweight='bold')
for i in range(len(cap_finite)):
    for j in range(len(retentions)):
        ax.text(j, i, f'{heat_clt[i, j]:.3f}', ha='center', va='center', fontsize=9, color='black')
fig.colorbar(im2, ax=ax, label='Client mean payoff')

# ── Panel 6: efficiency scatter  std reduction% vs client cost% ───────────
ax = axes[1, 2]
for ccol, clbl in zip(cap_colors[:-1], cap_labels[:-1]):   # skip cap=inf
    pts = [r for r in rows if r['cap_lbl'] == clbl]
    for r in pts:
        std_gain = (ref['tw_std'] - r['tw_std']) / ref['tw_std'] * 100
        clt_loss = (ref['clt_mean'] - r['clt_mean']) / ref['clt_mean'] * 100
        ax.scatter(clt_loss, std_gain, s=90, color=ccol, alpha=0.85, zorder=3)
        lbl = f"{clbl}\nr={r['ret']}"
        ax.annotate(lbl, (clt_loss, std_gain),
                    textcoords='offset points', xytext=(4, 2), fontsize=6, color=ccol)

xlim = max(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1, 1)
ylim = max(ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1, 1)
top  = max(xlim, ylim) * 1.2
ax.plot([0, top], [0, top],     'k--', lw=0.9, alpha=0.4, label='1:1')
ax.plot([0, top], [0, 2 * top], 'grey', lw=0.9, ls=':', alpha=0.4, label='2:1 (dealer 2x better)')
ax.set_xlabel('Client mean payoff loss  (% of baseline)', fontsize=9)
ax.set_ylabel('Dealer std reduction  (% of baseline)', fontsize=9)
ax.set_title('Efficiency: How Much Dealer Risk Saved\nper 1% Sacrificed by Client', fontweight='bold')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0); ax.set_ylim(bottom=0)

fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig('resilience_frontier.png', dpi=150, bbox_inches='tight')
print('\nSaved -> resilience_frontier.png')

# ── print best-efficiency configs ──────────────────────────────────────────
print('\n--- Best configs by efficiency (dealer std reduction / client payoff loss) ---')
efficiencies = []
for r in rows:
    if r['cap'] == np.inf and r['ret'] == 0.0:
        continue
    std_gain = (ref['tw_std'] - r['tw_std']) / ref['tw_std'] * 100
    clt_loss = (ref['clt_mean'] - r['clt_mean']) / ref['clt_mean'] * 100
    eff = std_gain / clt_loss if clt_loss > 0.01 else np.inf
    efficiencies.append((eff, std_gain, clt_loss, r))

efficiencies.sort(reverse=True)
print(f"{'config':>20}  {'std_red%':>9}  {'clt_loss%':>10}  {'efficiency':>11}  {'abs_tw_std':>11}  {'clt_mean':>10}")
for eff, sg, cl, r in efficiencies[:12]:
    cfg = f"{r['cap_lbl']}  r={r['ret']}"
    eff_str = f'{eff:.2f}x' if not np.isinf(eff) else 'inf'
    print(f"{cfg:>20}  {sg:>+8.2f}%  {cl:>9.3f}%  {eff_str:>11}  {r['tw_std']:>11.5f}  {r['clt_mean']:>10.4f}")
