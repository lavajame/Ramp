import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ramp_hedging import RampStripPayoff, DeltaHedgingSimulation

payoff  = RampStripPayoff(S0=100, T=1, N=250, K_lo=102, K_hi=105,
                           r=0.035, q=0.035, sigma=0.04)
grow    = np.exp(payoff.r * payoff.T)
N_PATHS = 50_000

caps   = [1.05, 1.10, 1.20, 1.30, 1.50, 2.00, 3.00, np.inf]
clbls  = ['1.05','1.10','1.20','1.30','1.50','2.00','3.00','∞']
colors = ['#0d0221','#1a1a6e','#0f3460','#155f82','#1d8a99','#57b894','#aed987','#e9f5a1']

print(f"{'cap':>6}  {'dlr_std':>9}  {'dlr_VaR5':>9}  {'dlr_CVaR5':>10}  "
      f"{'% paths cap-bound':>18}  {'clt_mean':>10}  {'clt_p50':>9}  "
      f"{'clt_p10':>9}  {'clt_p90':>9}  {'clt_loss%':>10}")

results = []
for lc, lbl in zip(caps, clbls):
    sim = DeltaHedgingSimulation(payoff, n_paths=N_PATHS, hedge_freq=1,
                                  seed=421, lambda_cap=lc, retention=0.0)
    _, vl = sim.run()
    pnl  = vl.hedge_port_value[:, -1]
    ctv  = vl.realised_pvs[:, -1] * grow
    lam  = vl.lambda_trace

    var5  = np.percentile(pnl, 5)
    cvar5 = pnl[pnl < var5].mean()
    # fraction of path-steps where lambda was at the cap (binding)
    if not np.isinf(lc):
        bound_frac = (lam >= lc * 0.9999).mean() * 100
    else:
        bound_frac = 0.0

    results.append(dict(cap=lc, lbl=lbl, col=colors[len(results)],
                        pnl=pnl, ctv=ctv, lam=lam,
                        std=pnl.std(), var5=var5, cvar5=cvar5,
                        bound_frac=bound_frac,
                        clt_mean=ctv.mean(),
                        clt_p50=np.percentile(ctv, 50),
                        clt_p10=np.percentile(ctv, 10),
                        clt_p90=np.percentile(ctv, 90),
                        times=vl.times))

ref = results[-1]   # cap=inf baseline
for r in results:
    clt_loss_pct = (ref['clt_mean'] - r['clt_mean']) / ref['clt_mean'] * 100
    print(f"{r['lbl']:>6}  {r['std']:>9.5f}  {r['var5']:>9.5f}  {r['cvar5']:>10.5f}  "
          f"{r['bound_frac']:>17.2f}%  {r['clt_mean']:>10.4f}  {r['clt_p50']:>9.4f}  "
          f"{r['clt_p10']:>9.4f}  {r['clt_p90']:>9.4f}  {clt_loss_pct:>+9.3f}%")

# ── figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 15))
fig.suptitle(
    f'Lambda Cap Sweep  |  retention=0  S0=100  K=[102,105]  σ=0.04  T=1  {N_PATHS//1000}k paths\n'
    f'How cap level controls dealer gap risk at minimal cost to client',
    fontweight='bold', fontsize=12)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.33)

# ── 1: Dealer P&L distribution overlaid ────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
all_pnl = np.concatenate([r['pnl'] for r in results])
bins = np.linspace(np.percentile(all_pnl, 0.5), np.percentile(all_pnl, 99.5), 120)
for r in results:
    ax.hist(r['pnl'], bins=bins, alpha=0.45, color=r['col'], density=True,
            label=f"cap={r['lbl']}  σ={r['std']:.4f}")
ax.axvline(0, color='black', lw=1.2)
ax.set_xlabel('Dealer P&L at maturity')
ax.set_ylabel('Density')
ax.set_title('Dealer P&L Distribution\n(all cap levels)', fontweight='bold')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# ── 2: Dealer std vs cap ────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
cap_finite = [r for r in results if not np.isinf(r['cap'])]
cap_inf    = results[-1]
xs_num  = [r['cap']  for r in cap_finite]
xs_lbl  = [r['lbl']  for r in cap_finite]
stds    = [r['std']   for r in cap_finite]
var5s   = [r['var5']  for r in cap_finite]
cvar5s  = [r['cvar5'] for r in cap_finite]

ax.plot(xs_num, stds,   'o-', color='#e94560', lw=2.2, ms=8, label='P&L std (gap risk)')
ax.plot(xs_num, var5s,  's--', color='#f5a623', lw=1.8, ms=7, label='VaR 5%')
ax.plot(xs_num, cvar5s, '^:', color='#0f3460', lw=1.8, ms=7, label='CVaR 5%')
ax.axhline(cap_inf['std'],   color='#e94560', lw=1, ls=':', alpha=0.5, label=f'Baseline std={cap_inf["std"]:.4f}')
ax.axhline(cap_inf['cvar5'], color='#0f3460', lw=1, ls=':', alpha=0.5)
ax.set_xlabel('Lambda cap value')
ax.set_ylabel('Dealer P&L metric')
ax.set_title('Dealer Risk vs Cap Level\n(std, VaR 5%, CVaR 5%)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── 3: Client mean / p10 / p90 vs cap ──────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
clt_means = [r['clt_mean']  for r in cap_finite]
clt_p10s  = [r['clt_p10']   for r in cap_finite]
clt_p90s  = [r['clt_p90']   for r in cap_finite]
ax.plot(xs_num, clt_means, 'o-',  color='#2a9d8f', lw=2.2, ms=8, label='Client mean payoff')
ax.fill_between(xs_num, clt_p10s, clt_p90s, color='#2a9d8f', alpha=0.15, label='Client p10–p90 band')
ax.axhline(ref['clt_mean'], color='#2a9d8f', lw=1, ls=':', alpha=0.6, label=f'Baseline={ref["clt_mean"]:.4f}')
ax.axhline(ref['clt_p10'],  color='#2a9d8f', lw=1, ls=':', alpha=0.3)
ax.axhline(ref['clt_p90'],  color='#2a9d8f', lw=1, ls=':', alpha=0.3)
ax.set_xlabel('Lambda cap value')
ax.set_ylabel('Client payoff (current money)')
ax.set_title('Client Payoff vs Cap Level\n(mean + p10/p90 band)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── 4: Pareto scatter  dealer std vs client mean ────────────────────────────
ax = fig.add_subplot(gs[1, 0])
for r in results:
    ax.scatter(r['clt_mean'], r['std'], s=120, color=r['col'], zorder=4)
    ax.annotate(f"cap={r['lbl']}", (r['clt_mean'], r['std']),
                textcoords='offset points', xytext=(5, 3), fontsize=8, color=r['col'])
# connect the dots
ax.plot([r['clt_mean'] for r in results], [r['std'] for r in results],
        '-', color='grey', lw=1.2, alpha=0.5, zorder=2)
ax.set_xlabel('Client mean payoff', fontsize=9)
ax.set_ylabel('Dealer P&L std', fontsize=9)
ax.set_title('Pareto Frontier\n(down-right = Pareto-dominant)', fontweight='bold')
ax.grid(True, alpha=0.3)

# ── 5: Std reduction % vs client payoff loss % ──────────────────────────────
ax = fig.add_subplot(gs[1, 1])
std_reds  = [(ref['std'] - r['std']) / ref['std'] * 100 for r in cap_finite]
clt_losses = [(ref['clt_mean'] - r['clt_mean']) / ref['clt_mean'] * 100 for r in cap_finite]
for r, sr, cl in zip(cap_finite, std_reds, clt_losses):
    ax.scatter(cl, sr, s=120, color=r['col'], zorder=4)
    ax.annotate(f"cap={r['lbl']}", (cl, sr),
                textcoords='offset points', xytext=(4, 3), fontsize=8, color=r['col'])
ax.plot(clt_losses, std_reds, '-', color='grey', lw=1.2, alpha=0.5, zorder=2)
top = max(max(std_reds), max(clt_losses)) * 1.25 + 1
ax.plot([0, top], [0, top],     'k--', lw=0.9, alpha=0.3, label='1:1 equal')
ax.plot([0, top], [0, 5 * top], 'k:',  lw=0.9, alpha=0.3, label='5:1 (dealer 5× better)')
ax.set_xlabel('Client mean payoff reduction  (% vs no-cap)', fontsize=9)
ax.set_ylabel('Dealer std reduction  (% vs no-cap)', fontsize=9)
ax.set_title('Efficiency: Dealer Risk Saved\nper % Client Payoff Sacrificed', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=-0.1); ax.set_ylim(bottom=-2)

# ── 6: Cap-binding fraction over time ──────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
times = results[0]['times']
for r in cap_finite:
    if np.isinf(r['cap']): continue
    lam = r['lam']
    bound_ts = (lam >= r['cap'] * 0.9999).mean(axis=0) * 100
    ax.plot(times, bound_ts, color=r['col'], lw=2.0, label=f"cap={r['lbl']}")
ax.set_xlabel('Time (yrs)')
ax.set_ylabel('% of paths at cap (lambda binding)')
ax.set_title('Cap-Binding Frequency Over Time\n(when does the cap actively engage?)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── 7: Lambda fan: cap=inf vs cap=1.5 vs cap=1.2 ──────────────────────────
ax = fig.add_subplot(gs[2, 0])
show_caps = ['∞', '1.50', '1.20', '1.10']
for r in results:
    if r['lbl'] not in show_caps: continue
    lam  = r['lam']
    mean = lam.mean(axis=0)
    p16  = np.percentile(lam, 16, axis=0)
    p84  = np.percentile(lam, 84, axis=0)
    ax.plot(times, mean, color=r['col'], lw=2.2, label=f"cap={r['lbl']}  mean_T={mean[-1]:.3f}")
    ax.fill_between(times, p16, p84, color=r['col'], alpha=0.15)
ax.axhline(1.0, color='grey', lw=0.8, ls=':')
ax.set_xlabel('Time (yrs)'); ax.set_ylabel('Lambda (notional multiplier)')
ax.set_title('Lambda Fan  (mean ± 1σ)\ncap=∞ vs selected caps', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── 8: Tracking error std over time ────────────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
for r in results:
    if r['lbl'] not in show_caps: continue
    te = r['pnl']   # scalar at T — use full time series instead
for r in results:
    if r['lbl'] not in show_caps: continue
    sim2 = DeltaHedgingSimulation(payoff, n_paths=N_PATHS, hedge_freq=1,
                                   seed=421, lambda_cap=r['cap'], retention=0.0)
    _, vl2 = sim2.run()
    te = vl2.hedge_port_value - vl2.future_pvs
    ax.plot(times, te.std(axis=0), color=r['col'], lw=2.0, label=f"cap={r['lbl']}")
ax.set_xlabel('Time (yrs)'); ax.set_ylabel('Std(hedge − future PV)')
ax.set_title('Tracking Error Std Over Time\n(how tightly hedge follows obligation)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── 9: Client CDF (cap=inf vs cap=1.5 vs cap=1.2 vs cap=1.1) ──────────────
ax = fig.add_subplot(gs[2, 2])
for r in results:
    if r['lbl'] not in show_caps: continue
    xs = np.sort(r['ctv'])
    ys = np.linspace(1 / len(xs), 1, len(xs))
    ax.plot(xs, ys, color=r['col'], lw=2.0,
            label=f"cap={r['lbl']}  mean={r['clt_mean']:.3f}")
ax.axvline(ref['clt_mean'], color='grey', lw=0.8, ls=':', label='Baseline mean')
ax.set_xlabel('Client payoff at maturity (current money)')
ax.set_ylabel('Cumulative probability')
ax.set_title('Client Payoff CDF\n(cap=∞ vs selected caps)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig('cap_sweep.png', dpi=150, bbox_inches='tight')
print('\nSaved -> cap_sweep.png')

# ── summary table ─────────────────────────────────────────────────────────
print('\n--- Efficiency table (retention=0) ---')
print(f"{'cap':>6}  {'std_red%':>9}  {'CVaR_impr%':>11}  {'clt_loss%':>10}  "
      f"{'efficiency':>12}  {'abs_std':>9}  {'clt_mean':>10}")
for r in cap_finite:
    sr   = (ref['std']   - r['std'])   / ref['std']   * 100
    cvr  = (r['cvar5']   - ref['cvar5']) / abs(ref['cvar5']) * 100
    cl   = (ref['clt_mean'] - r['clt_mean']) / ref['clt_mean'] * 100
    eff  = sr / cl if cl > 0.001 else float('inf')
    eff_s = f'{eff:.0f}x' if not np.isinf(eff) else '∞'
    print(f"{r['lbl']:>6}  {sr:>+8.2f}%  {cvr:>+10.2f}%  {cl:>+9.3f}%  "
          f"{eff_s:>12}  {r['std']:>9.5f}  {r['clt_mean']:>10.4f}")
