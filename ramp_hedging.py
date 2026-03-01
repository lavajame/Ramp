"""
ramp_hedging.py
---------------
Builds on ramp_strip.py to add:

  RampStripPayoff       – ramp strip priced at 1/dK per ramp, with methods for
                          realised payoff collection and forward PV / delta of
                          the remaining (unrealised) ramps.

  DeltaHedgingSimulation – Monte Carlo engine that simulates GBM spot paths,
                           evolves the option mark-to-market (realised + future),
                           runs a delta-hedging account alongside it, and stores
                           the full per-path, per-step history.

  plot_simulation         – Produces a four-panel diagnostic figure.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")   # headless – works without a display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Sequence

from ramp_strip import (
    Ramp,
    bs_call_price,
    bs_call_delta,
    bs_call_strike_delta,
)


# ---------------------------------------------------------------------------
# RampStripPayoff
# ---------------------------------------------------------------------------

class RampStripPayoff:
    """
    A strip of N mini call-spreads, each with notional = 1 / dK_i,
    expiring at times t_i = i * dt,  i = 1 … N.

    The 1/dK weighting makes each ramp approximate a digital (indicator)
    as dK → 0: the strip converges to a density-weighted corridor.

    Parameters
    ----------
    S0, T, N, K_lo, K_hi : same semantics as RampStrip.
        K_lo / K_hi can be scalar (uniform) or length-N arrays (per-slice).
    r, q, sigma           : Black-Scholes parameters.

    Key methods
    -----------
    ramp_payoff(i, S_i)           – intrinsic payoff of ramp i at its expiry
    realised_pv(spots_so_far, k)  – PV to time 0 of ramps 0..k-1 that have paid
    future_pv(k, S, t_now)        – BS value at time t_now of ramps k..N-1
    total_value(spots_so_far, k, S, t_now) – realised_pv + exp(-r*t_now)*future_pv
                                              expressed as a t=0 PV
    future_delta(k, S, t_now)     – spot delta of the remaining ramps
    future_strike_delta(k, S, t_now)
    initial_value()               – future_pv(0, S0, 0)
    initial_delta()               – future_delta(0, S0, 0)
    """

    def __init__(
        self,
        S0: float,
        T: float,
        N: int,
        K_lo: float | Sequence[float],
        K_hi: float | Sequence[float],
        r: float,
        q: float,
        sigma: float,
    ) -> None:
        if N < 1:
            raise ValueError("N must be >= 1")
        if T <= 0:
            raise ValueError("T must be positive")

        self.S0 = S0
        self.T = T
        self.N = N
        self.r = r
        self.q = q
        self.sigma = sigma
        self.dt = T / N
        self.times = np.arange(1, N + 1) * self.dt   # t_1 … t_N

        K_los = np.full(N, K_lo) if np.isscalar(K_lo) else np.asarray(K_lo, dtype=float)
        K_his = np.full(N, K_hi) if np.isscalar(K_hi) else np.asarray(K_hi, dtype=float)
        if K_los.shape != (N,) or K_his.shape != (N,):
            raise ValueError("K_lo / K_hi must be scalar or length-N arrays")

        self.dKs = K_his - K_los          # per-ramp width (array)
        self.ramps: list[Ramp] = [
            Ramp(t=self.times[i], K_lo=K_los[i], K_hi=K_his[i],
                 notional=1.0 / self.dKs[i])
            for i in range(N)
        ]

    # ------------------------------------------------------------------
    # Realised payoff
    # ------------------------------------------------------------------

    def ramp_payoff(self, ramp_idx: int, S_at_expiry: float) -> float:
        """
        Intrinsic payoff of ramp `ramp_idx` when spot at its expiry is S.
        = (1/dK) * (max(S-K_lo, 0) - max(S-K_hi, 0))
        """
        r = self.ramps[ramp_idx]
        raw = max(S_at_expiry - r.K_lo, 0.0) - max(S_at_expiry - r.K_hi, 0.0)
        return r.notional * raw

    def realised_pv(self, realised_spots: np.ndarray, k: int) -> float:
        """
        NPV (discounted to t=0) of ramps 0 … k-1 that have already paid.

        realised_spots[i]  = spot at time t_{i+1}  (index 0 = first expiry)
        k                  = number of ramps that have expired so far
        """
        pv = 0.0
        for i in range(k):
            payoff = self.ramp_payoff(i, realised_spots[i])
            pv += payoff * np.exp(-self.r * self.times[i])
        return pv

    # ------------------------------------------------------------------
    # Future (unrealised) ramps
    # ------------------------------------------------------------------

    def future_pv(self, k: int, S: float, t_now: float = 0.0) -> float:
        """
        BS value at time t_now of ramps k … N-1.

        k       : first unrealised ramp index (0-based)
        S       : current spot at t_now
        t_now   : current calendar time (default 0 = pricing from inception)
        """
        total = 0.0
        for i in range(k, self.N):
            tau = self.times[i] - t_now        # time-to-expiry for ramp i
            if tau <= 0.0:
                # already expired; intrinsic only (should not normally happen)
                continue
            ramp = self.ramps[i]
            c_lo = bs_call_price(S, ramp.K_lo, self.r, self.q, self.sigma, tau)
            c_hi = bs_call_price(S, ramp.K_hi, self.r, self.q, self.sigma, tau)
            total += ramp.notional * (c_lo - c_hi)
        return total

    def future_delta(self, k: int, S: float, t_now: float = 0.0) -> float:
        """Spot delta of the unrealised ramps k … N-1 at time t_now."""
        total = 0.0
        for i in range(k, self.N):
            tau = self.times[i] - t_now
            if tau <= 0.0:
                continue
            ramp = self.ramps[i]
            d_lo = bs_call_delta(S, ramp.K_lo, self.r, self.q, self.sigma, tau)
            d_hi = bs_call_delta(S, ramp.K_hi, self.r, self.q, self.sigma, tau)
            total += ramp.notional * (d_lo - d_hi)
        return total

    def future_strike_delta(self, k: int, S: float, t_now: float = 0.0) -> float:
        """Parallel strike-shift sensitivity of unrealised ramps k … N-1."""
        total = 0.0
        for i in range(k, self.N):
            tau = self.times[i] - t_now
            if tau <= 0.0:
                continue
            ramp = self.ramps[i]
            sk_lo = bs_call_strike_delta(S, ramp.K_lo, self.r, self.q, self.sigma, tau)
            sk_hi = bs_call_strike_delta(S, ramp.K_hi, self.r, self.q, self.sigma, tau)
            total += ramp.notional * (sk_lo - sk_hi)
        return total

    # ------------------------------------------------------------------
    # Combined mark-to-market  (always expressed as a t=0 NPV)
    # ------------------------------------------------------------------

    def total_value(
        self,
        realised_spots: np.ndarray,
        k: int,
        S_now: float,
        t_now: float,
    ) -> float:
        """
        Total t=0 NPV = realised_pv(k) + exp(-r*t_now) * future_pv(k, S_now, t_now).

        realised_spots : length-N array; indices 0..k-1 are used
        k              : number of ramps that have expired
        S_now          : current spot at t_now
        t_now          : current calendar time
        """
        r_pv = self.realised_pv(realised_spots, k)
        f_pv = self.future_pv(k, S_now, t_now)
        return r_pv + np.exp(-self.r * t_now) * f_pv

    # ------------------------------------------------------------------
    # Convenience: values at inception
    # ------------------------------------------------------------------

    def initial_value(self) -> float:
        return self.future_pv(0, self.S0, 0.0)

    def initial_delta(self) -> float:
        return self.future_delta(0, self.S0, 0.0)

    def __repr__(self) -> str:
        return (
            f"RampStripPayoff(S0={self.S0}, T={self.T}, N={self.N}, dt={self.dt:.4f}, "
            f"K_lo={self.ramps[0].K_lo}, K_hi={self.ramps[0].K_hi}, "
            f"r={self.r}, q={self.q}, sigma={self.sigma})"
        )


# ---------------------------------------------------------------------------
# Fast vectorised normal CDF (A&S 7.1.26, max abs error < 1.5e-7)
# Avoids scipy overhead for large arrays in tight loops.
# ---------------------------------------------------------------------------

def _ndtr_fast(x: np.ndarray) -> np.ndarray:
    p  = 0.2316419
    b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    ax = np.abs(x)
    t  = 1.0 / (1.0 + p * ax)
    poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
    pdf  = 0.3989422804014327 * np.exp(-0.5 * x * x)
    cdf  = 1.0 - pdf * poly
    return np.where(x >= 0.0, cdf, 1.0 - cdf)


# ---------------------------------------------------------------------------
# Strip value interpolator  (precompute once; evaluate via np.interp)
# ---------------------------------------------------------------------------

class StripValueInterpolator:
    """
    Precomputes future-PV and delta tables on a 1-D grid of

        x  =  S / K_lo_ref        (spot rescaled by the lower strike reference)

    for every remaining-step index k = 0 … N.  During simulation, any number
    of spot paths can be looked up simultaneously with a single np.interp call.

    This completely decouples the (expensive) Black-Scholes work from the
    (cheap but potentially path-dependent) cash-accounting loop.

    Parameters
    ----------
    payoff   : RampStripPayoff
    n_grid   : number of grid points in x  (default 1 000)
    n_sigma  : half-width of the grid in multiples of sigma*sqrt(T)
               (default 6 – covers all realistic spot moves)

    Attributes
    ----------
    K_lo_ref    : reference lower strike (= ramps[0].K_lo)
    log_x_grid  : (n_grid,) uniformly spaced log-moneyness grid
    x_grid      : (n_grid,) corresponding x values
    S_grid      : (n_grid,) absolute spot values
    pv_table    : (N+1, n_grid) future PV at each (step, grid point)
    delta_table : (N+1, n_grid) future delta    build_time  : seconds taken to build tables
    """

    def __init__(
        self,
        payoff: "RampStripPayoff",
        n_grid: int = 1_000,
        n_sigma: float = 6.0,
    ) -> None:
        import time as _time
        _t0 = _time.perf_counter()

        p = self.payoff = payoff
        N = p.N

        # --- grid setup in log(x) space, x = S / K_lo_ref -------------------
        self.K_lo_ref   = float(p.ramps[0].K_lo)
        half_range      = n_sigma * p.sigma * np.sqrt(p.T)
        log_x_min       = -half_range
        log_x_max       = +half_range
        self.log_x_grid = np.linspace(log_x_min, log_x_max, n_grid)  # (n_grid,)
        self.x_grid     = np.exp(self.log_x_grid)                      # (n_grid,)
        self.S_grid     = self.x_grid * self.K_lo_ref                  # (n_grid,)

        # --- ramp arrays ----------------------------------------------------
        _K_lo = np.array([r.K_lo     for r in p.ramps])   # (N,)
        _K_hi = np.array([r.K_hi     for r in p.ramps])   # (N,)
        _nots = np.array([r.notional for r in p.ramps])   # (N,)

        uniform = np.allclose(_K_lo, _K_lo[0]) and np.allclose(_K_hi, _K_hi[0])

        pv_table    = np.zeros((N + 1, n_grid))
        delta_table = np.zeros((N + 1, n_grid))
        S_g = self.S_grid    # alias

        if uniform:
            # ----------------------------------------------------------------
            # O(N) precomputation via lag decomposition.
            #
            # For uniform strikes all ramps share K_lo, K_hi, notional.
            # The BS call-spread value/delta with lag tau = m*dt is independent
            # of which step k we are at.  So we compute it once per lag m and
            # scatter-add into pv_table[0 : N-m+1, :].
            # ----------------------------------------------------------------
            Klo = _K_lo[0]; Khi = _K_hi[0]; not_ = _nots[0]
            log_S = np.log(S_g)              # (n_grid,) — constant across lags

            for m in range(1, N + 1):
                tau     = m * p.dt
                sqT     = np.sqrt(tau)
                sig_sqT = p.sigma * sqT
                drift   = (p.r - p.q + 0.5 * p.sigma ** 2) * tau
                dq      = np.exp(-p.q * tau)
                dr      = np.exp(-p.r * tau)

                d1_lo = (log_S - np.log(Klo) + drift) / sig_sqT
                d1_hi = (log_S - np.log(Khi) + drift) / sig_sqT
                d2_lo = d1_lo - sig_sqT
                d2_hi = d1_hi - sig_sqT

                N1lo = _ndtr_fast(d1_lo); N2lo = _ndtr_fast(d2_lo)
                N1hi = _ndtr_fast(d1_hi); N2hi = _ndtr_fast(d2_hi)

                v_m = not_ * (S_g * dq * N1lo - Klo * dr * N2lo
                              - S_g * dq * N1hi + Khi * dr * N2hi)   # (n_grid,)
                d_m = not_ * dq * (N1lo - N1hi)                       # (n_grid,)

                # Ramp with lag m contributes to steps k = 0 … N-m
                pv_table[: N - m + 1 :, :] += v_m
                delta_table[: N - m + 1, :] += d_m

        else:
            # ----------------------------------------------------------------
            # General per-ramp loop.  For each ramp i (0-based) and its lag
            # m = i-k+1 from each step k=0..i, we need a distinct BS
            # evaluation (different K_lo[i]).  We vectorise over (lags × grid)
            # inside the per-ramp iteration, giving O(N^2/2) lag vectors each
            # of length n_grid — still evaluated in numpy without Python loops
            # over individual evaluations.
            # ----------------------------------------------------------------
            for i in range(N):
                Klo_i = _K_lo[i]; Khi_i = _K_hi[i]; not_i = _nots[i]
                n_alive = i + 1                         # steps k=0..i see ramp i
                m_arr   = np.arange(i + 1, 0, -1)      # lags i+1..1  (n_alive,)
                tau_arr = m_arr * p.dt                  # (n_alive,)

                sqT_arr = np.sqrt(tau_arr)              # (n_alive,)
                sig_arr = p.sigma * sqT_arr
                dft_arr = (p.r - p.q + 0.5*p.sigma**2) * tau_arr
                dq_arr  = np.exp(-p.q * tau_arr)
                dr_arr  = np.exp(-p.r * tau_arr)

                # Broadcast: (n_grid, 1) vs (1, n_alive)
                logS2   = np.log(S_g)[:, np.newaxis]   # (n_grid, 1)

                d1_lo = (logS2 - np.log(Klo_i) + dft_arr) / sig_arr  # (n_grid, n_alive)
                d1_hi = (logS2 - np.log(Khi_i) + dft_arr) / sig_arr
                d2_lo = d1_lo - sig_arr
                d2_hi = d1_hi - sig_arr

                N1lo = _ndtr_fast(d1_lo); N2lo = _ndtr_fast(d2_lo)
                N1hi = _ndtr_fast(d1_hi); N2hi = _ndtr_fast(d2_hi)

                S2  = S_g[:, np.newaxis]                               # (n_grid, 1)
                c_lo = S2 * dq_arr * N1lo - Klo_i * dr_arr * N2lo    # (n_grid, n_alive)
                c_hi = S2 * dq_arr * N1hi - Khi_i * dr_arr * N2hi

                # pv_table[k, :] for k=0..i  ← transpose to (n_alive, n_grid)
                pv_table[:n_alive, :]    += not_i * (c_lo - c_hi).T
                delta_table[:n_alive, :] += not_i * (dq_arr * (N1lo - N1hi)).T

        self.pv_table    = pv_table       # (N+1, n_grid)
        self.delta_table = delta_table    # (N+1, n_grid)
        # pv_table[k, :] itself is the forward map  V = f(x) = V_unit(x)
        # used for the barycentric inverse in lookup_alpha_inv.
        self.build_time  = _time.perf_counter() - _t0

    # ------------------------------------------------------------------
    # Fast path-vector lookup
    # ------------------------------------------------------------------

    def lookup(self, k: int, S_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate future PV and delta for n_paths spots at step k.

        Parameters
        ----------
        k     : remaining-step index (0 = all ramps live, N = none left)
        S_vec : (n_paths,) current spot prices

        Returns
        -------
        pv    : (n_paths,) future PV
        delta : (n_paths,) future delta
        """
        log_x = np.log(S_vec / self.K_lo_ref)
        pv    = np.interp(log_x, self.log_x_grid, self.pv_table[k])
        delta = np.interp(log_x, self.log_x_grid, self.delta_table[k])
        return pv, delta

    def lookup_alpha_inv(
        self,
        k: int,
        S_vec: np.ndarray,
        T_vec: np.ndarray,
        alpha_prev: np.ndarray | None = None,
        n_iter: int = 6,
    ) -> np.ndarray:
        """
        Find α per-path such that  α · V_unit(S / (α K_lo_ref)) = T_vec.

        The homogeneity identity  α · V_unit(S/α) = T  with  x = S/(α K_ref)  gives:

            pv_table(x)  =  (T · K_ref / S) · x   ≡  C · x

        where C = T · K_ref / S is a per-path constant.  This is solved via
        Newton iteration on  f(x) = pv_table(x) − C · x,  using
        f′(x) = K_ref · delta_table(x) − C.

        Warm start: x₀ = S / (α_prev · K_ref)  if α_prev is supplied, else
        x₀ = S / K_ref  (corresponds to α = 1, the no-adjustment point).
        The Newton step is clipped to the grid and typically converges in 4–6
        iterations for α near 1.

        Parameters
        ----------
        k          : remaining-step index
        S_vec      : (n_paths,) current spot
        T_vec      : (n_paths,) undiscounted future-PV target
        alpha_prev : (n_paths,) previous α for warm-start (optional)
        n_iter     : Newton iterations (default 6)
        """
        Kref   = self.K_lo_ref
        C      = T_vec * Kref / S_vec          # per-path slope of target line
        xp_pv  = self.pv_table[k]              # (n_grid,) forward map
        xp_d   = self.delta_table[k]           # (n_grid,) dV/dS

        # warm start
        if alpha_prev is not None:
            x = S_vec / np.where(alpha_prev > 0, alpha_prev * Kref, Kref)
        else:
            x = S_vec / Kref                   # α = 1 initial guess

        x = np.clip(x, self.x_grid[0], self.x_grid[-1])
        log_xgrid = self.log_x_grid

        for _ in range(n_iter):
            log_x = np.log(np.maximum(x, 1e-12))
            pv    = np.interp(log_x, log_xgrid, xp_pv)   # V_unit(x·K_ref)
            dv_dS = np.interp(log_x, log_xgrid, xp_d)    # dV/dS at x·K_ref
            dv_dx = dv_dS * Kref                          # dV/dx = dV/dS · K_ref
            f  = pv - C * x
            fp = dv_dx - C
            # guard denominator; clip step to ±50% of current x
            step = f / np.where(np.abs(fp) > 1e-30, fp, 1e-30)
            step = np.clip(step, -0.5 * x, 0.5 * x)
            x   = x - step
            x   = np.clip(x, self.x_grid[0], self.x_grid[-1])

        return S_vec / (x * Kref)   # α = S / (x · K_ref)

    def __repr__(self) -> str:
        n_grid = len(self.x_grid)
        return (
            f"StripValueInterpolator(N={self.payoff.N}, n_grid={n_grid}, "
            f"K_lo_ref={self.K_lo_ref}, "
            f"x_range=[{self.x_grid[0]:.3f}, {self.x_grid[-1]:.3f}], "
            f"build_time={self.build_time:.3f}s)"
        )



@dataclass
class SimulationResults:
    """
    Holds the full history for every simulated path.

    Arrays have shape (n_paths, N+1) where column k corresponds to time k*dt.

    Attributes
    ----------
    spot_paths       : simulated spot at each time node
    option_values    : total option MTM (realised_pv + disc*future_pv) at each node
    future_pvs       : future-only PV component
    realised_pvs     : realised-only PV component
    hedge_port_value : value of the hedging portfolio (cash + stock)
    hedge_pnl        : hedge P&L relative to initial premium
                       = hedge_port_value - initial option value
    delta_trace      : delta position held (shares) at each rebalance
    option_params    : dict of S0, T, N, r, q, sigma, K_lo, K_hi
    times            : time axis (length N+1, starts at 0)
    """
    spot_paths:        np.ndarray
    option_values:     np.ndarray   # realised_pv + disc_future_pv  (full MTM)
    future_pvs:        np.ndarray   # undiscounted future-only value at t_now
    disc_future_pvs:   np.ndarray   # exp(-r*t) * future_pvs  (t=0 NPV of remaining)
    realised_pvs:      np.ndarray
    hedge_port_value:  np.ndarray   # cash + shares*S  — tracks disc_future_pvs
    hedge_pnl:         np.ndarray
    delta_trace:       np.ndarray
    lambda_trace:      np.ndarray   # per-path notional multiplier λ (1 unless mode='notional')
    alpha_trace:       np.ndarray   # per-path strike multiplier α  (1 unless mode='strike')
    buffer_trace:      np.ndarray   # per-path cumulative dealer buffer (notional mode only)
    vol_lock:          bool         # whether any vol-lock adjustment was active
    vol_lock_mode:     str          # 'none' | 'notional' | 'strike'
    option_params:     dict
    times:             np.ndarray


# ---------------------------------------------------------------------------
# DeltaHedgingSimulation
# ---------------------------------------------------------------------------

class DeltaHedgingSimulation:
    """
    Monte Carlo delta-hedging engine for a RampStripPayoff.

    The trader is SHORT the ramp strip:
      * At t=0  : receives premium V0, buys delta0 shares, invests rest at r.
      * At each t_k (k*dt) : cash accrues at r, ramp k-1 pays to client,
                              rebalance delta to new_delta of the remaining strip.
      * At t=T  : liquidate remaining shares; final cash = residual P&L.

    If BS hedging is perfect, residual P&L ≈ 0 across all paths.

    Parameters
    ----------
    payoff     : RampStripPayoff instance
    n_paths    : number of Monte Carlo paths
    hedge_freq : rebalances per ramp interval dt (default 1 = at each expiry)
    seed       : random seed for reproducibility
    """

    def __init__(
        self,
        payoff: RampStripPayoff,
        n_paths: int = 2_000,
        hedge_freq: int = 1,
        seed: int = 42,
        lambda_cap: float = 1.5,
        retention: float = 0.0,
        ema_alpha: float = 1.0,
    ) -> None:
        self.payoff      = payoff
        self.n_paths     = n_paths
        self.hedge_freq  = hedge_freq
        self.seed        = seed
        self.lambda_cap  = lambda_cap   # max notional multiplier for vol-lock notional mode
        self.retention   = retention    # fraction of surplus retained by dealer (0=pass all to client, 1=keep all)
        self.ema_alpha   = ema_alpha     # EMA weight on α update (1.0 = full update each step)
        p = payoff
        self.mu        = p.r - p.q      # risk-neutral drift
        self.sigma_sim = p.sigma

    # ------------------------------------------------------------------
    # Path simulation
    # ------------------------------------------------------------------

    def _simulate_paths(self) -> np.ndarray:
        """
        Return array of shape (n_paths, n_steps+1) with GBM spot paths.
        n_steps = N * hedge_freq; nodes at multiples of dt/hedge_freq.
        """
        p = self.payoff
        n_steps = p.N * self.hedge_freq
        sub_dt = p.dt / self.hedge_freq

        rng = np.random.default_rng(self.seed)
        Z = rng.standard_normal((self.n_paths, n_steps))

        # log-normal increments
        drift = (self.mu - 0.5 * self.sigma_sim ** 2) * sub_dt
        diffusion = self.sigma_sim * np.sqrt(sub_dt)

        log_returns = drift + diffusion * Z           # (n_paths, n_steps)
        log_paths = np.cumsum(log_returns, axis=1)    # (n_paths, n_steps)

        S = np.empty((self.n_paths, n_steps + 1))
        S[:, 0] = p.S0
        S[:, 1:] = p.S0 * np.exp(log_paths)
        return S                                       # (n_paths, n_steps+1)

    # ------------------------------------------------------------------
    # Core accounting loop
    # ------------------------------------------------------------------

    def _run_accounting(
        self,
        all_S: np.ndarray,
        interp: "StripValueInterpolator",
        vol_lock_mode: str = "none",
        lambda_cap: float = np.inf,
        retention: float = 0.0,
        ema_alpha: float = 1.0,
    ) -> SimulationResults:
        """
        Cash-accounting pass over pre-generated paths.

        Parameters
        ----------
        all_S         : (n_paths, N*hedge_freq+1) spot paths
        interp        : pre-built StripValueInterpolator
        vol_lock_mode : 'none'     – plain delta hedge (λ ≡ 1, α ≡ 1)
                        'notional' – scale future notional by λ so that the
                                     discounted future strip = retention×hedge_pre
                        'strike'   – shift strikes multiplicatively by α so that
                                     the discounted future strip = retention×hedge_pre
                                     (α found via vectorised Newton; see below)
        lambda_cap    : upper cap on λ (notional mode only; default: no cap)
        retention     : fraction of surplus retained by dealer as a buffer (0–1).
                        0 = pass all surplus to client (tightest hedge track);
                        1 = dealer keeps all surplus, using buffer to absorb deficits.

        Notional mode mechanics
        -----------------------
        At each ramp expiry k, after paying ramp k-1 at current λ, set:

            λ_new = clip( retention × hedge_pre / (e^{-r t_k} × base_pv), 0, λ_cap )

        Strike mode mechanics
        ---------------------
        BS call-spread homogeneity: V(S, αK_lo, αK_hi, 1/dK) = α · V_table(S/α).
        So the precomputed table is reused exactly — just look up at S/α and
        scale PV by α; delta is delta_table(S/α) with NO α factor (it cancels).

        We find α per-path via vectorised Newton on:

            f(α) = α · V_table(S/α) − PV_target,   PV_target = retention × hedge_pre × e^{r t_k}

        with f'(α) = V_table(S/α) − (S/α) · Δ_table(S/α)  (product + chain rule).

        Payoff at expiry of ramp with scale α_prev:
            (1/dK) × max(0, min(S − α·K_lo, α·dK))
        α·dK preserves the ramp width proportionally; notional 1/dK is unchanged.
        """
        p      = self.payoff
        N      = p.N
        hf     = self.hedge_freq
        n_sub  = N * hf
        sub_dt = p.dt / hf
        n_paths = self.n_paths
        coarse_S = all_S[:, ::hf]           # (n_paths, N+1)

        option_values    = np.zeros((n_paths, N + 1))
        future_pvs       = np.zeros((n_paths, N + 1))
        realised_pvs     = np.zeros((n_paths, N + 1))
        hedge_port_value = np.zeros((n_paths, N + 1))
        delta_trace      = np.zeros((n_paths, N + 1))
        lambda_trace     = np.ones((n_paths,  N + 1))   # notional mode scale
        alpha_trace      = np.ones((n_paths,  N + 1))   # strike mode scale
        buffer_trace     = np.zeros((n_paths, N + 1))   # cumulative dealer buffer

        disc_r_ramp  = np.exp(-p.r * p.times)   # (N,)
        cash_accrual = np.exp(p.r * sub_dt)      # scalar

        # --- t=0 initialise -------------------------------------------------
        V0_vec, d0_vec = interp.lookup(0, coarse_S[:, 0])
        lambda_vec    = np.ones(n_paths)    # notional multiplier per path
        alpha_vec     = np.ones(n_paths)    # strike multiplier per path
        dealer_buffer = np.zeros(n_paths)   # cumulative retained surplus (notional mode)

        future_pvs[:, 0]    = V0_vec
        option_values[:, 0] = V0_vec
        delta_trace[:, 0]   = d0_vec

        cash   = V0_vec.copy() - d0_vec * coarse_S[:, 0]
        shares = d0_vec.copy()
        hedge_port_value[:, 0] = cash + shares * coarse_S[:, 0]

        running_realised_pv = np.zeros(n_paths)

        # --- sequential cash-accounting loop --------------------------------
        for step in range(1, n_sub + 1):
            S_prev = all_S[:, step - 1]
            S_curr = all_S[:, step]
            t_curr = step * sub_dt

            # Cash accrues at r; earn continuous dividend on held shares
            cash *= cash_accrual
            cash += p.q * shares * S_prev * sub_dt

            is_coarse = (step % hf == 0)
            k = step // hf          # coarse index 1..N

            if is_coarse:
                # ------ Ramp k-1 expires ------------------------------------
                ramp = p.ramps[k - 1]
                if vol_lock_mode == "strike":
                    # shifted strikes: K_lo' = α*K_lo, width = α*dK
                    aKlo = alpha_vec * ramp.K_lo
                    aKhi = alpha_vec * ramp.K_hi
                    base_pay = ramp.notional * np.maximum(
                        0.0, np.minimum(S_curr - aKlo, aKhi - aKlo)
                    )
                else:
                    # plain or notional mode: original strikes, scaled by λ
                    base_pay = ramp.notional * np.maximum(
                        0.0, np.minimum(S_curr - ramp.K_lo, ramp.K_hi - ramp.K_lo)
                    )
                    base_pay = lambda_vec * base_pay

                cash -= base_pay
                running_realised_pv += base_pay * disc_r_ramp[k - 1]

            # ------ Look up PV/delta for remaining N-k ramps ----------------
            if vol_lock_mode == "strike":
                # V(S, αK_lo, αK_hi) = α · V_table(S/α)  ← homogeneity
                S_eff      = S_curr / alpha_vec
                base_pv_u, base_delta = interp.lookup(k, S_eff)   # unit-notional
                base_pv    = alpha_vec * base_pv_u                 # scaled PV
                # delta = d/dS[α · V(S/α)] = (α/α) · Δ(S/α) = base_delta (no α)
            else:
                base_pv, base_delta = interp.lookup(k, S_curr)

            # ------ Vol-Lock adjustment at each ramp expiry -----------------
            if is_coarse and vol_lock_mode != "none" and k < N:
                # H_min guard: blown hedge account -> near-zero target -> alpha->OTM
                # -> delta=0 -> position frozen. Guard at 1e-10.
                hedge_pre         = cash + shares * S_curr
                hedge_pre_guarded = np.maximum(hedge_pre, 1e-10)

                if vol_lock_mode == "notional":
                    # Dealer buffer mechanics:
                    #   surplus  = hedge_pre − base_pv  (signed, current money)
                    #
                    #   retention=0: all surplus passed to client via λ (no buffer)
                    #   retention=1: dealer keeps all surplus; buffer grows monotonically
                    #
                    #   SURPLUS  (hedge_pre > base_pv): dealer pockets retention×surplus
                    #            permanently.  λ only rises by (1−retention) fraction.
                    #
                    #   DEFICIT  (hedge_pre < base_pv): buffer is NOT drawn.  λ falls
                    #            exactly as it would with no buffer.  The buffer is
                    #            dealer capital — it does not protect the client.
                    #
                    #   At maturity the accumulated buffer appears as positive dealer P&L
                    #   on top of the hedge account residual.
                    valid   = base_pv > 1e-12
                    surplus = hedge_pre_guarded - np.where(valid, base_pv, hedge_pre_guarded)
                    # buffer grows on surplus only; never drawn
                    add_to_buf    = retention * np.maximum(surplus, 0.0)
                    dealer_buffer = dealer_buffer + add_to_buf
                    # effective amount the hedge account forwards to set λ
                    # surplus: hedge_pre - add = base_pv + (1-retention)*surplus  → λ > 1 but damped
                    # deficit: hedge_pre unchanged                                  → λ < 1 as normal
                    effective = hedge_pre_guarded - add_to_buf
                    raw_lam = np.where(
                        valid,
                        effective / np.where(valid, base_pv, 1.0),
                        lambda_vec,
                    )
                    lambda_vec = np.clip(raw_lam, 0.0, lambda_cap)

                elif vol_lock_mode == "strike":
                    # tau_min guard: when tau_remaining < 0.5/365 the pv_table
                    # approaches intrinsic and may not be strictly monotone.
                    # Use k_safe = k-1 (previous step's smooth curve) so the
                    # barycentric inverse is always well-behaved; the adjustment
                    # is still applied — we just invert against a non-degenerate
                    # table.
                    tau_remaining = (N - k) * p.dt
                    k_safe = (k - 1) if tau_remaining < 0.5 / 365.0 else k
                    k_safe = max(k_safe, 0)

                    disc_factor = np.exp(-p.r * t_curr)
                    pv_target   = retention * hedge_pre_guarded / np.where(
                        disc_factor > 1e-12, disc_factor, 1.0
                    )
                    alpha_raw = interp.lookup_alpha_inv(
                        k_safe, S_curr, pv_target, alpha_prev=alpha_vec
                    )
                    alpha_raw = np.maximum(alpha_raw, 0.0)
                    # EMA dampener: blend target α with previous α to reduce
                    # step-to-step oscillation caused by discrete-hedge noise.
                    # ema_alpha=1.0 → full update each step (no smoothing);
                    # ema_alpha<1 → geometric blend toward previous value.
                    alpha_vec = ema_alpha * alpha_raw + (1.0 - ema_alpha) * alpha_vec
                    S_eff         = S_curr / np.where(alpha_vec > 0, alpha_vec, 1.0)
                    base_pv_u, base_delta = interp.lookup(k, S_eff)
                    base_pv       = alpha_vec * base_pv_u

            # ------ Rebalance to new delta ----------------------------------
            if vol_lock_mode == "notional":
                new_shares = lambda_vec * base_delta
            else:
                new_shares = base_delta   # strike mode: no λ factor

            cash -= (new_shares - shares) * S_curr
            shares = new_shares

            if is_coarse:
                # notional mode: hedge holds λ*delta shares, so tracked future
                # exposure is λ * base_pv, not base_pv.
                # strike mode: base_pv = α * V_unit(S/α) — already scaled.
                f_pv = lambda_vec * base_pv if vol_lock_mode == "notional" else base_pv
                realised_pvs[:, k]     = running_realised_pv
                future_pvs[:, k]       = f_pv
                option_values[:, k]    = running_realised_pv + np.exp(-p.r * t_curr) * f_pv
                delta_trace[:, k]      = new_shares
                hedge_port_value[:, k] = cash + shares * S_curr
                lambda_trace[:, k]     = lambda_vec.copy()
                alpha_trace[:, k]      = alpha_vec.copy()
                buffer_trace[:, k]     = dealer_buffer.copy()

        hedge_port_value[:, N] = cash + shares * coarse_S[:, N]
        lambda_trace[:, N]     = lambda_vec.copy()
        alpha_trace[:, N]      = alpha_vec.copy()
        buffer_trace[:, N]     = dealer_buffer.copy()

        _times      = np.arange(N + 1) * p.dt
        disc_future = future_pvs * np.exp(-p.r * _times)[np.newaxis, :]

        return SimulationResults(
            spot_paths       = coarse_S,
            option_values    = option_values,
            future_pvs       = future_pvs,
            disc_future_pvs  = disc_future,
            realised_pvs     = realised_pvs,
            hedge_port_value = hedge_port_value,
            hedge_pnl        = hedge_port_value,
            delta_trace      = delta_trace,
            lambda_trace     = lambda_trace,
            alpha_trace      = alpha_trace,
            buffer_trace     = buffer_trace,
            vol_lock         = vol_lock_mode != "none",
            vol_lock_mode    = vol_lock_mode,
            option_params    = dict(
                S0=p.S0, T=p.T, N=p.N, r=p.r, q=p.q, sigma=p.sigma,
                K_lo=p.ramps[0].K_lo, K_hi=p.ramps[0].K_hi,
            ),
            times = _times,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> tuple[SimulationResults, SimulationResults]:
        """
        Build the interpolator, generate one set of GBM paths, then run
        two accounting passes on the same paths:

        Returns
        -------
        res_plain    : plain delta hedge  (λ ≡ 1)
        res_notional : vol-lock notional  (λ adjusted each ramp expiry)
        """
        interp = StripValueInterpolator(self.payoff)
        self.interpolator = interp

        all_S = self._simulate_paths()

        res_plain    = self._run_accounting(all_S, interp, vol_lock_mode="none")
        res_notional = self._run_accounting(
            all_S, interp,
            vol_lock_mode="notional",
            lambda_cap=self.lambda_cap,
            retention=self.retention,
        )
        return res_plain, res_notional


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    res_plain:    SimulationResults,
    res_notional: SimulationResults,
    figsize=(18, 14),
):
    """
    Four-panel comparison: plain vs vol-lock notional.

    Panel 1 : Overlaid final P&L histograms
    Panel 2 : Mean hedge portfolio value ± 1-std fan
    Panel 3 : λ-trace fan (notional mode)
    Panel 4 : Tracking-error std over time
    """
    times = res_plain.times
    p     = res_plain.option_params

    runs = [
        ("Plain",           res_plain,    "mediumpurple"),
        ("VL-Notional (λ)",  res_notional, "teal"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f"Delta-Hedge Comparison  |  S0={p['S0']}  K=[{p['K_lo']}, {p['K_hi']}]  "
        f"T={p['T']}  N={p['N']}  r={p['r']}  σ={p['sigma']}",
        fontsize=11, fontweight="bold",
    )

    # Panel 1: P&L histograms ------------------------------------------------
    ax = axes[0, 0]
    pnl_all = np.concatenate([r.hedge_port_value[:, -1] for _, r, _ in runs])
    bins = np.linspace(pnl_all.min(), pnl_all.max(), 70)
    for lbl, r, col in runs:  # plot vol-lock second so it overlays plain
        pnl = r.hedge_port_value[:, -1]
        ax.hist(pnl, bins=bins, alpha=0.25, color=col,
                label=f"{lbl}  μ={pnl.mean():+.4f}  σ={pnl.std():.4f}")
    ax.axvline(0, color="black", lw=1.2)
    ax.set_title("Final P&L Distribution", fontweight="bold")
    ax.set_xlabel("P&L at maturity")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # Panel 2: tracking error fan (hedge − future PV) ± 1-std ----------------
    # Plot the ERROR relative to target so both series are centred near 0 and
    # fan widths are directly comparable (plain ≈ ±3, VL ≈ ±0.18).
    ax = axes[0, 1]
    for lbl, r, col in reversed(runs):
        err      = r.hedge_port_value - r.future_pvs   # (n_paths, N+1) current-money error
        err_mean = err.mean(axis=0)
        err_std  = err.std(axis=0)
        ax.plot(times, err_mean, color=col, lw=2.0, label=f"{lbl}  σ={err_std[-1]:.3f}")
        ax.fill_between(times, err_mean - err_std, err_mean + err_std,
                        color=col, alpha=0.25, label=f"{lbl} ±1σ")
    ax.axhline(0, color="black", lw=0.9, ls="--")
    ax.set_title("Tracking Error Fan  (hedge − future PV,  mean ± 1σ)", fontweight="bold")
    ax.set_xlabel("Time (yrs)")
    ax.set_ylabel("Error (current money)")
    ax.legend(fontsize=8)

    # Panel 3: λ trace (notional mode) ----------------------------------------
    ax    = axes[1, 0]
    width = times[1] - times[0]
    lam      = res_notional.lambda_trace
    lam_mean = lam.mean(axis=0)
    lam_lo   = np.percentile(lam, 16, axis=0)
    lam_hi   = np.percentile(lam, 84, axis=0)
    lam_p5   = np.percentile(lam,  5, axis=0)
    lam_p95  = np.percentile(lam, 95, axis=0)
    ax.plot(times, lam_mean, color="teal", lw=2.0, label="λ mean")
    ax.fill_between(times, lam_lo, lam_hi, color="teal", alpha=0.25,
                    label="16–84th pct")
    ax.fill_between(times, lam_p5, lam_p95, color="teal", alpha=0.12,
                    label="5–95th pct")
    ax.axhline(1.0, color="grey", ls=":", lw=0.9, label="λ = 1")
    # Right axis: % paths capped
    lambda_cap_val = lam.max()   # infer cap from data
    ax2 = ax.twinx()
    pct_capped = (lam >= lambda_cap_val - 1e-6).mean(axis=0) * 100
    ax2.bar(times, pct_capped, color="firebrick", alpha=0.25, width=width,
            label="% capped (right)")
    ax2.set_ylabel("% paths at cap", fontsize=8, color="firebrick")
    ax2.tick_params(axis="y", labelsize=7, labelcolor="firebrick")
    ax.set_title("λ Trace (Notional Vol-Lock)", fontweight="bold")
    ax.set_xlabel("Time (yrs)")
    ax.set_ylabel("λ")
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7, ncol=2)

    # Panel 4: tracking error std --------------------------------------------
    ax = axes[1, 1]
    for lbl, r, col in runs:
        # TE = hedge_port − future_pvs: both in current money at t_k
        te_std = (r.hedge_port_value - r.future_pvs).std(axis=0)
        ax.plot(times, te_std, color=col, lw=2.0, label=lbl)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Tracking Error Std Over Time", fontweight="bold")
    ax.set_xlabel("Time (yrs)")
    ax.set_ylabel("Std(hedge − future PV)")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_simulation(res: SimulationResults, n_show: int = 30, figsize=(16, 12)):
    """
    Four-panel diagnostic plot.

      Panel 1  (top-left)   : sample spot paths
      Panel 2  (top-right)  : option MTM evolution – mean ± 1 std across all paths,
                               with individual sample paths overlaid
      Panel 3  (bottom-left): hedge portfolio value vs option value (mean ± std)
      Panel 4  (bottom-right): final hedge P&L distribution (histogram)
    """
    times  = res.times
    rng    = np.random.default_rng(99)
    n_all  = res.spot_paths.shape[0]
    idx    = rng.choice(n_all, size=min(n_show, n_all), replace=False)

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.33)

    # ---- Panel 1: spot paths -----------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times, res.spot_paths[idx].T, color="steelblue", alpha=0.35, lw=0.8)
    ax1.axhline(res.option_params["K_lo"], color="red",   ls="--", lw=1.0, label="K_lo")
    ax1.axhline(res.option_params["K_hi"], color="green", ls="--", lw=1.0, label="K_hi")
    ax1.set_title("Simulated Spot Paths", fontweight="bold")
    ax1.set_xlabel("Time (yrs)")
    ax1.set_ylabel("Spot")
    ax1.legend(fontsize=8)

    # ---- Panel 2: option MTM evolution -------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ov_mean = res.option_values.mean(axis=0)
    ov_std  = res.option_values.std(axis=0)
    ax2.fill_between(times, ov_mean - ov_std, ov_mean + ov_std,
                     alpha=0.20, color="darkorange", label="mean ± 1 std")
    ax2.plot(times, ov_mean, color="darkorange", lw=2.0, label="mean")
    ax2.plot(times, res.option_values[idx].T, color="darkorange", alpha=0.25, lw=0.7)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_title("Option MTM (Realised + Future PV)", fontweight="bold")
    ax2.set_xlabel("Time (yrs)")
    ax2.set_ylabel("NPV (t=0)")
    ax2.legend(fontsize=8)

    # ---- Panel 3: hedge portfolio vs discounted future PV ------------------
    ax3 = fig.add_subplot(gs[1, 0])
    dfv_mean = res.disc_future_pvs.mean(axis=0)
    hv_mean  = res.hedge_port_value.mean(axis=0)
    hv_std   = res.hedge_port_value.std(axis=0)
    ax3.fill_between(times, hv_mean - hv_std, hv_mean + hv_std,
                     alpha=0.20, color="teal", label="hedge mean ± 1 std")
    ax3.plot(times, hv_mean,  color="teal",       lw=2.0,       label="hedge portfolio mean")
    ax3.plot(times, dfv_mean, color="darkorange", lw=2.0, ls="--", label="disc. future PV mean")
    ax3.set_title("Hedge Portfolio vs Disc. Future PV", fontweight="bold")
    ax3.set_xlabel("Time (yrs)")
    ax3.set_ylabel("Value (t=0 NPV)")
    ax3.legend(fontsize=8)

    # ---- Panel 4: final P&L distribution -----------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    final_pnl = res.hedge_port_value[:, -1]
    ax4.hist(final_pnl, bins=60, color="purple", alpha=0.65, edgecolor="white", lw=0.4)
    ax4.axvline(final_pnl.mean(),  color="red",   lw=1.5, ls="-",  label=f"mean  {final_pnl.mean():.4f}")
    ax4.axvline(np.percentile(final_pnl,  5), color="orange", lw=1.2, ls="--",
                label=f"5th pct {np.percentile(final_pnl, 5):.4f}")
    ax4.axvline(np.percentile(final_pnl, 95), color="orange", lw=1.2, ls="--",
                label=f"95th pct {np.percentile(final_pnl, 95):.4f}")
    ax4.axvline(0, color="black", lw=1.0)
    ax4.set_title("Final Hedge P&L Distribution", fontweight="bold")
    ax4.set_xlabel("P&L")
    ax4.set_ylabel("Count")
    ax4.legend(fontsize=8)

    p = res.option_params
    fig.suptitle(
        f"Ramp Strip Delta Hedge  |  S0={p['S0']}  K=[{p['K_lo']}, {p['K_hi']}]  "
        f"T={p['T']}  N={p['N']}  r={p['r']}  q={p['q']}  σ={p['sigma']}",
        fontsize=11, fontweight="bold",
    )
    return fig


# ---------------------------------------------------------------------------
# Single-path diagnostic plot
# ---------------------------------------------------------------------------

def plot_single_path(
    res: SimulationResults,
    path_idx: int | None = None,
    figsize=(16, 14),
):
    """
    Five-panel diagnostic for one simulated path.

    Layout
    ------
    Row 0 (full width) : spot path with strike bands and ramp-expiry markers
    Row 1, left        : PV decomposition  – realised PV, future PV, total MTM
    Row 1, right       : hedge portfolio value vs option MTM
    Row 2, left        : delta position held at each rebalance node
    Row 2, right       : tracking error  (hedge_port − option_MTM)

    Parameters
    ----------
    res      : SimulationResults from DeltaHedgingSimulation.run()
    path_idx : which path to display; if None picks the path whose
               final spot is closest to the median final spot
    """
    if path_idx is None:
        final_spots = res.spot_paths[:, -1]
        path_idx = int(np.argmin(np.abs(final_spots - np.median(final_spots))))

    times = res.times
    p     = res.option_params
    N     = len(times) - 1

    spot     = res.spot_paths[path_idx]         # (N+1,)
    r_pv     = res.realised_pvs[path_idx]       # (N+1,)
    f_pv     = res.future_pvs[path_idx]         # (N+1,)  – undiscounted at t_now
    dfv      = res.disc_future_pvs[path_idx]    # (N+1,)  – disc. to t=0; t=0 NPV of remaining
    fpv      = res.future_pvs[path_idx]         # (N+1,)  – current-money future PV (hedge tracks this)
    ov       = res.option_values[path_idx]      # (N+1,)  – realised_pv + dfv
    hv       = res.hedge_port_value[path_idx]   # (N+1,)
    delta    = res.delta_trace[path_idx]        # (N+1,)
    tracking = hv - fpv                         # hedge minus current future PV (correct comparison)

    ramp_times = times[1:]   # the N expiry dates (exclude t=0)

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(
        3, 2,
        height_ratios=[1.1, 1.2, 1.0],
        hspace=0.48, wspace=0.33,
    )

    # ---- Panel 1: spot path (full width) -----------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, spot, color="steelblue", lw=1.6, label="Spot")
    ax1.axhline(p["K_lo"], color="firebrick",     ls="--", lw=1.1, label=f"K_lo = {p['K_lo']}")
    ax1.axhline(p["K_hi"], color="forestgreen",   ls="--", lw=1.1, label=f"K_hi = {p['K_hi']}")
    ax1b = ax1.twinx()
    ax1b.set_ylabel("Delta held", color="grey", fontsize=8)
    ax1b.plot(times, delta, color="grey", lw=0.9, ls=":", alpha=0.7)
    ax1b.tick_params(axis="y", labelcolor="grey", labelsize=7)
    # mark each ramp expiry with a vertical tick
    for t_k in ramp_times:
        ax1.axvline(t_k, color="black", lw=0.5, alpha=0.3)
    ax1.set_title(f"Spot Path  (path #{path_idx})", fontweight="bold")
    ax1.set_xlabel("Time (yrs)")
    ax1.set_ylabel("Spot")
    ax1.legend(fontsize=8, loc="upper left")

    # ---- Panel 2: PV decomposition -----------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.stackplot(
        times,
        r_pv,
        np.clip(dfv, 0, None),          # disc. future PV (always ≥ 0)
        labels=["Realised PV (disc. to t=0)", "Disc. Future PV"],
        colors=["#e07b54", "#5b9bd5"],
        alpha=0.75,
    )
    ax2.plot(times, ov,  color="black",  lw=1.8, ls="-",  label="Total MTM (realised + future)")
    ax2.plot(times, dfv, color="#5b9bd5", lw=1.5, ls="-",  label="Disc. future PV")
    ax2.plot(times, hv,  color="purple",  lw=1.5, ls="--", label="Hedge portfolio")
    ax2.axhline(0, color="black", lw=0.5)
    # mark ramp expiry steps with a thin step line along the realised PV border
    ax2.step(times, r_pv, where="post", color="#b85c30", lw=0.8, zorder=5)
    ax2.set_title("PV Decomposition: Realised vs Future", fontweight="bold")
    ax2.set_xlabel("Time (yrs)")
    ax2.set_ylabel("NPV (t=0)")
    ax2.legend(fontsize=7, loc="upper left")

    # ---- Panel 3: hedge portfolio vs current future PV --------------------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(times, fpv, color="darkorange", lw=2.0,       label="Future PV (what hedge tracks)")
    ax3.plot(times, hv,  color="teal",       lw=2.0, ls="--", label="Hedge portfolio")
    ax3.fill_between(times, fpv, hv, alpha=0.15, color="red",
                     label="Tracking gap")
    ax3.axhline(0, color="black", lw=0.5)
    for t_k in ramp_times:
        ax3.axvline(t_k, color="black", lw=0.5, alpha=0.3)
    ax3.set_title("Hedge Portfolio vs Future PV", fontweight="bold")
    ax3.set_xlabel("Time (yrs)")
    ax3.set_ylabel("Value (current money)")
    ax3.legend(fontsize=8)

    # ---- Panel 4: delta position -------------------------------------------
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.step(times, delta, where="post", color="navy", lw=1.5, label="Delta (shares held)")
    ax4.axhline(0, color="black", lw=0.5)
    for t_k in ramp_times:
        ax4.axvline(t_k, color="black", lw=0.5, alpha=0.3)
    ax4.set_title("Delta Position (Shares Held Short)", fontweight="bold")
    ax4.set_xlabel("Time (yrs)")
    ax4.set_ylabel("Shares")
    ax4.legend(fontsize=8)

    # ---- Panel 5: tracking error (+ optional scale overlay) ---------------
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.fill_between(times, tracking, alpha=0.35,
                     color="green" if tracking[-1] >= 0 else "red")
    ax5.plot(times, tracking, color="darkgreen", lw=1.5, label="Hedge P&L  (hedge − option)")
    ax5.axhline(0, color="black", lw=1.0)
    for t_k in ramp_times:
        ax5.axvline(t_k, color="black", lw=0.5, alpha=0.3)
    mode = res.vol_lock_mode
    title_extra = f" + λ" if mode == "notional" else (f" + α" if mode == "strike" else "")
    ax5.set_title(f"Cumulative Tracking Error{title_extra}", fontweight="bold")
    ax5.set_xlabel("Time (yrs)")
    ax5.set_ylabel("P&L")
    final_te = tracking[-1]
    ax5.annotate(
        f"Final: {final_te:+.4f}",
        xy=(times[-1], final_te),
        xytext=(-55, 10),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8),
        color="darkgreen",
    )
    ax5.legend(fontsize=8)
    # Overlay scale trace on a twin axis for vol-lock modes
    scale     = None
    scale_lbl = ""
    scale_col = "grey"
    if mode == "notional":
        scale = res.lambda_trace[path_idx]
        scale_lbl = "λ (notional multiplier)"
        scale_col = "teal"
    elif mode == "strike":
        scale = res.alpha_trace[path_idx]
        scale_lbl = "α (strike multiplier)"
        scale_col = "darkorange"
    if scale is not None:
        ax5b = ax5.twinx()
        ax5b.step(times, scale, where="post", color=scale_col, lw=1.3,
                  ls="--", alpha=0.85, label=f"{scale_lbl} (right)")
        ax5b.axhline(1.0, color=scale_col, lw=0.6, ls=":")
        ax5b.set_ylabel(scale_lbl, color=scale_col, fontsize=8)
        ax5b.tick_params(axis="y", labelcolor=scale_col, labelsize=7)

    fig.suptitle(
        f"Single-Path Detail  |  S0={p['S0']}  K=[{p['K_lo']}, {p['K_hi']}]  "
        f"T={p['T']}  N={p['N']}  r={p['r']}  q={p['q']}  σ={p['sigma']}  "
        f"(path #{path_idx})",
        fontsize=10, fontweight="bold",
    )
    return fig


# ---------------------------------------------------------------------------
# Client vs Hedger scatter
# ---------------------------------------------------------------------------

def plot_client_vs_hedger(
    res_plain:    SimulationResults,
    res_notional: SimulationResults,
    figsize=(14, 6),
):
    """
    Two-panel scatter: each dot is one path.
      x-axis : client terminal value  (sum of ramp payoffs grown to T)
      y-axis : dealer final P&L / hedge slippage  (hedge_port_value[:, -1])

    Reveals the client-gain / dealer-risk trade-off and how notional
    vol-lock reshapes that relationship relative to plain delta hedging.
    """
    p    = res_plain.option_params
    grow = np.exp(p["r"] * p["T"])

    runs = [
        ("Plain",          res_plain,    "mediumpurple"),
        ("VL-Notional (λ)", res_notional, "teal"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    fig.suptitle(
        f"Client Terminal Value vs Dealer P&L  |  "
        f"S0={p['S0']}  K=[{p['K_lo']},{p['K_hi']}]  "
        f"T={p['T']}  σ={p['sigma']}",
        fontweight="bold", fontsize=11,
    )

    for ax, (lbl, res, col) in zip(axes, runs):
        ctv = res.realised_pvs[:, -1] * grow      # client terminal value
        pnl = res.hedge_port_value[:, -1]          # dealer P&L

        ax.scatter(ctv, pnl, c=col, alpha=0.12, s=6, linewidths=0)

        # binned-mean overlay
        edges = np.percentile(ctv, np.linspace(0, 100, 31))
        edges = np.unique(edges)
        mids, means = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (ctv >= lo) & (ctv < hi)
            if mask.sum() > 5:
                mids.append(0.5 * (lo + hi))
                means.append(pnl[mask].mean())
        ax.plot(mids, means, color="black", lw=2, label="bin mean")

        ax.axhline(0, color="black", lw=0.8, ls="--")
        corr = np.corrcoef(ctv, pnl)[0, 1]
        ax.set_title(
            f"{lbl}  |  corr(client, dealer) = {corr:+.3f}",
            fontweight="bold",
        )
        ax.set_xlabel("Client terminal value (ramp payoffs × e^{rT})")
        ax.set_ylabel("Dealer final P&L")
        ax.legend(fontsize=8, markerscale=3)

        # summary text box
        textstr = (f"μ_client={ctv.mean():.1f}\n"
                   f"μ_dealer={pnl.mean():+.3f}\n"
                   f"σ_dealer={pnl.std():.3f}")
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
                fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ---------------------------------------------------------------------------
# Summary statistics helper
# ---------------------------------------------------------------------------

def hedge_summary(res: SimulationResults) -> pd.DataFrame:
    """
    Print a tidy summary of the hedging performance.
    """
    final_pnl = res.hedge_port_value[:, -1]
    V0 = res.option_values[:, 0].mean()
    rows = {
        "initial_option_value_mean":  [V0],
        "final_pnl_mean":             [final_pnl.mean()],
        "final_pnl_std":              [final_pnl.std()],
        "final_pnl_pct5":             [np.percentile(final_pnl, 5)],
        "final_pnl_pct95":            [np.percentile(final_pnl, 95)],
        "pnl_as_pct_of_V0":          [final_pnl.std() / V0 * 100],
    }
    return pd.DataFrame(rows).T.rename(columns={0: "value"})


# ---------------------------------------------------------------------------
# Quick smoke test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    # --- set up payoff -------------------------------------------------------
    payoff = RampStripPayoff(
        S0=100.0, T=1.0, N=250,   # daily strip: 250 ramps, dt = 1/250
        K_lo=102.0, K_hi=105.0,
        r=0.035, q=0.035, sigma=0.04,
    )
    print(payoff)
    print(f"  Initial value  : {payoff.initial_value():.6f}")
    print(f"  Initial delta  : {payoff.initial_delta():.6f}")
    print(f"  Ramp notional  : {payoff.ramps[0].notional:.4f}  (= 1/dK = 1/{payoff.dKs[0]})")
    print()

    # --- run simulation (3 passes on same paths) ----------------------------
    print("Running 5 000-path MC with daily rebalancing (N=250, hedge_freq=1)…")
    t0 = time.perf_counter()
    sim = DeltaHedgingSimulation(
        payoff=payoff,
        n_paths=50_000,
        hedge_freq=1,
        seed=421,
        lambda_cap=2.5,
        retention=0.0,
    )
    res_plain, res_notional = sim.run()
    print(f"  Interp build : {sim.interpolator.build_time:.3f}s")
    print(f"  {sim.interpolator}")
    print(f"  Total elapsed: {time.perf_counter()-t0:.1f}s")
    print()

    # --- summary -------------------------------------------------------------
    pd.set_option("display.float_format", "{:.6f}".format)
    for lbl, res in [("Plain", res_plain), ("VL-Notional", res_notional)]:
        print(f"=== {lbl} ===")
        print(hedge_summary(res).to_string())
        print()

    # --- 2-way comparison plot -----------------------------------------------
    fig_cmp = plot_comparison(res_plain, res_notional)
    fig_cmp.savefig("ramp_hedge_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cmp)
    print("Plot saved \u2192 ramp_hedge_comparison.png")

    # --- client vs hedger scatter --------------------------------------------
    fig_sc = plot_client_vs_hedger(res_plain, res_notional)
    fig_sc.savefig("ramp_client_vs_hedger.png", dpi=150, bbox_inches="tight")
    plt.close(fig_sc)
    print("Plot saved \u2192 ramp_client_vs_hedger.png")

    # --- single-path detail plots (median / high / low) ---------------------
    final_spots = res_plain.spot_paths[:, -1]
    median_idx  = int(np.argmin(np.abs(final_spots - np.median(final_spots))))
    high_idx    = int(np.argmax(final_spots))
    low_idx     = int(np.argmin(final_spots))

    for run_label, res in [("plain", res_plain), ("vl_notional", res_notional)]:
        for path_label, idx in [("median", median_idx), ("high", high_idx), ("low", low_idx)]:
            fig2 = plot_single_path(res, path_idx=idx)
            fname = f"ramp_single_path_{run_label}_{path_label}.png"
            fig2.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            print(f"Plot saved \u2192 {fname}")
