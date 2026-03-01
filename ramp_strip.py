"""
ramp_strip.py
-------------
Prices a strip of mini call-spreads ("ramps") under Black-Scholes.

Each ramp is a bull call spread  [long call @ K_lo, short call @ K_hi]
with expiry t_i = i * dt,  i = 1 … N,  so the last ramp expires at T.

Greeks exposed
--------------
  price        : PV of the call spread
  delta        : dV / dS  (spot delta)
  strike_delta : dV / d(parallel strike shift)
                 i.e. shift both K_lo and K_hi by the same epsilon
                 = dC/dK|_{K_lo} - dC/dK|_{K_hi}
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Sequence
import pandas as pd


# ---------------------------------------------------------------------------
# Black-Scholes building blocks
# ---------------------------------------------------------------------------

def _d1(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    return _d1(S, K, r, q, sigma, T) - sigma * np.sqrt(T)


def bs_call_price(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0.0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, r, q, sigma, T)
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """dC/dS under Black-Scholes."""
    if T <= 0.0:
        return 1.0 if S > K else 0.0
    d1 = _d1(S, K, r, q, sigma, T)
    return np.exp(-q * T) * norm.cdf(d1)


def bs_call_strike_delta(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """dC/dK under Black-Scholes  (= -e^{-rT} * N(d2), always <= 0)."""
    if T <= 0.0:
        return -1.0 if S > K else 0.0
    d2 = _d2(S, K, r, q, sigma, T)
    return -np.exp(-r * T) * norm.cdf(d2)


# ---------------------------------------------------------------------------
# Single ramp (mini call spread)
# ---------------------------------------------------------------------------

@dataclass
class Ramp:
    """
    A single mini call spread: long 1 call @ K_lo, short 1 call @ K_hi.

    Attributes
    ----------
    t        : expiry in years
    K_lo     : lower (long) strike
    K_hi     : upper (short) strike  (must be > K_lo)
    notional : scales all outputs linearly
    """
    t: float
    K_lo: float
    K_hi: float
    notional: float = 1.0

    def __post_init__(self) -> None:
        if self.K_hi <= self.K_lo:
            raise ValueError(f"K_hi ({self.K_hi}) must be strictly greater than K_lo ({self.K_lo})")
        if self.t < 0:
            raise ValueError("Expiry t must be non-negative")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def dK(self) -> float:
        return self.K_hi - self.K_lo

    @property
    def K_mid(self) -> float:
        return 0.5 * (self.K_lo + self.K_hi)

    # ------------------------------------------------------------------
    # Pricing and greeks
    # ------------------------------------------------------------------

    def price(self, S: float, r: float, q: float, sigma: float) -> float:
        """PV of the call spread = C(K_lo) - C(K_hi)."""
        c_lo = bs_call_price(S, self.K_lo, r, q, sigma, self.t)
        c_hi = bs_call_price(S, self.K_hi, r, q, sigma, self.t)
        return self.notional * (c_lo - c_hi)

    def delta(self, S: float, r: float, q: float, sigma: float) -> float:
        """Spot delta dV/dS = delta(K_lo) - delta(K_hi)."""
        d_lo = bs_call_delta(S, self.K_lo, r, q, sigma, self.t)
        d_hi = bs_call_delta(S, self.K_hi, r, q, sigma, self.t)
        return self.notional * (d_lo - d_hi)

    def strike_delta(self, S: float, r: float, q: float, sigma: float) -> float:
        """
        Parallel strike-shift sensitivity: dV/d(eps) where
        K_lo -> K_lo + eps, K_hi -> K_hi + eps simultaneously.

        = dC/dK|_{K_lo} - dC/dK|_{K_hi}
        """
        sk_lo = bs_call_strike_delta(S, self.K_lo, r, q, sigma, self.t)
        sk_hi = bs_call_strike_delta(S, self.K_hi, r, q, sigma, self.t)
        return self.notional * (sk_lo - sk_hi)

    def strike_delta_lo(self, S: float, r: float, q: float, sigma: float) -> float:
        """Sensitivity to the lower strike only: dV/dK_lo = dC/dK|_{K_lo}."""
        return self.notional * bs_call_strike_delta(S, self.K_lo, r, q, sigma, self.t)

    def strike_delta_hi(self, S: float, r: float, q: float, sigma: float) -> float:
        """Sensitivity to the upper strike only: dV/dK_hi = -dC/dK|_{K_hi}."""
        return -self.notional * bs_call_strike_delta(S, self.K_hi, r, q, sigma, self.t)

    def __repr__(self) -> str:
        return (
            f"Ramp(t={self.t:.4f}, K_lo={self.K_lo}, K_hi={self.K_hi}, "
            f"dK={self.dK}, notional={self.notional})"
        )


# ---------------------------------------------------------------------------
# Strip of ramps
# ---------------------------------------------------------------------------

class RampStrip:
    """
    A strip of mini call spreads equally spaced in time.

    Observation times:  t_i = i * dt,  i = 1 … N  (last slice expires at T).

    Parameters
    ----------
    S0       : float              – current spot
    T        : float              – total time horizon (years)
    N        : int                – number of ramps in the strip
    K_lo     : float | array-like – lower strike(s).  Scalar = same for all slices.
    K_hi     : float | array-like – upper strike(s).  Scalar = same for all slices.
    r        : float              – continuously-compounded risk-free rate
    q        : float              – continuous dividend / repo yield
    sigma    : float              – Black-Scholes implied vol
    notional : float              – total notional (split evenly across slices)

    Notes
    -----
    * Each individual Ramp receives notional / N so that the strip's aggregate
      notional equals the supplied value.
    * All pricing methods accept an optional `S` argument; if omitted, S0 is used.
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
        notional: float = 1.0,
    ) -> None:
        if N < 1:
            raise ValueError("N must be at least 1")
        if T <= 0:
            raise ValueError("T must be positive")

        self.S0 = S0
        self.T = T
        self.N = N
        self.r = r
        self.q = q
        self.sigma = sigma
        self.notional = notional

        self.dt = T / N
        self.times: np.ndarray = np.arange(1, N + 1) * self.dt

        # Broadcast scalar or per-slice strikes
        K_los = np.full(N, K_lo) if np.isscalar(K_lo) else np.asarray(K_lo, dtype=float)
        K_his = np.full(N, K_hi) if np.isscalar(K_hi) else np.asarray(K_hi, dtype=float)

        if K_los.shape != (N,) or K_his.shape != (N,):
            raise ValueError("K_lo and K_hi must be scalar or length-N arrays")

        slice_notional = notional / N
        self.ramps: list[Ramp] = [
            Ramp(t=self.times[i], K_lo=K_los[i], K_hi=K_his[i], notional=slice_notional)
            for i in range(N)
        ]

    # ------------------------------------------------------------------
    # Aggregate greeks over the whole strip
    # ------------------------------------------------------------------

    def price(self, S: float | None = None) -> float:
        """Total PV of the ramp strip."""
        S = self._resolve_S(S)
        return sum(ramp.price(S, self.r, self.q, self.sigma) for ramp in self.ramps)

    def delta(self, S: float | None = None) -> float:
        """Total spot delta of the strip."""
        S = self._resolve_S(S)
        return sum(ramp.delta(S, self.r, self.q, self.sigma) for ramp in self.ramps)

    def strike_delta(self, S: float | None = None) -> float:
        """
        Total parallel strike-shift sensitivity.
        Applies the same epsilon shift to every K_lo and K_hi simultaneously.
        """
        S = self._resolve_S(S)
        return sum(ramp.strike_delta(S, self.r, self.q, self.sigma) for ramp in self.ramps)

    # ------------------------------------------------------------------
    # Per-slice arrays
    # ------------------------------------------------------------------

    def slice_prices(self, S: float | None = None) -> np.ndarray:
        S = self._resolve_S(S)
        return np.array([ramp.price(S, self.r, self.q, self.sigma) for ramp in self.ramps])

    def slice_deltas(self, S: float | None = None) -> np.ndarray:
        S = self._resolve_S(S)
        return np.array([ramp.delta(S, self.r, self.q, self.sigma) for ramp in self.ramps])

    def slice_strike_deltas(self, S: float | None = None) -> np.ndarray:
        S = self._resolve_S(S)
        return np.array([ramp.strike_delta(S, self.r, self.q, self.sigma) for ramp in self.ramps])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, S: float | None = None) -> pd.DataFrame:
        """
        Return a DataFrame with per-slice and aggregate greeks.

        Columns: time, K_lo, K_hi, dK, price, delta, strike_delta
        The final row ('TOTAL') holds strip-level aggregates.
        """
        S = self._resolve_S(S)
        rows = {
            "time":         self.times,
            "K_lo":         [r.K_lo for r in self.ramps],
            "K_hi":         [r.K_hi for r in self.ramps],
            "dK":           [r.dK   for r in self.ramps],
            "price":        self.slice_prices(S),
            "delta":        self.slice_deltas(S),
            "strike_delta": self.slice_strike_deltas(S),
        }
        df = pd.DataFrame(rows)

        totals = pd.DataFrame(
            {
                "time":         [np.nan],
                "K_lo":         [np.nan],
                "K_hi":         [np.nan],
                "dK":           [np.nan],
                "price":        [df["price"].sum()],
                "delta":        [df["delta"].sum()],
                "strike_delta": [df["strike_delta"].sum()],
            },
            index=["TOTAL"],
        )
        return pd.concat([df, totals])

    # ------------------------------------------------------------------
    # Bump-and-reprice finite-difference greeks (validation helpers)
    # ------------------------------------------------------------------

    def fd_delta(self, S: float | None = None, bump: float = 0.01) -> float:
        """Finite-difference spot delta (central difference)."""
        S = self._resolve_S(S)
        return (self.price(S + bump) - self.price(S - bump)) / (2.0 * bump)

    def fd_strike_delta(self, S: float | None = None, bump: float = 0.01) -> float:
        """
        Finite-difference parallel strike-shift delta (central difference).
        Bumps every K_lo and K_hi by +/- bump simultaneously.
        """
        S = self._resolve_S(S)

        def _price_with_shift(eps: float) -> float:
            total = 0.0
            for ramp in self.ramps:
                shifted = Ramp(
                    t=ramp.t,
                    K_lo=ramp.K_lo + eps,
                    K_hi=ramp.K_hi + eps,
                    notional=ramp.notional,
                )
                total += shifted.price(S, self.r, self.q, self.sigma)
            return total

        return (_price_with_shift(bump) - _price_with_shift(-bump)) / (2.0 * bump)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def _resolve_S(self, S: float | None) -> float:
        return self.S0 if S is None else S

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Ramp:
        return self.ramps[idx]

    def __repr__(self) -> str:
        return (
            f"RampStrip(S0={self.S0}, T={self.T}, N={self.N}, dt={self.dt:.4f}, "
            f"r={self.r}, q={self.q}, sigma={self.sigma}, notional={self.notional})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    strip = RampStrip(
        S0=100.0,
        T=1.0,
        N=12,           # monthly slices
        K_lo=95.0,
        K_hi=105.0,
        r=0.05,
        q=0.02,
        sigma=0.20,
        notional=1.0,
    )

    print(strip)
    print()

    df = strip.summary()
    pd.set_option("display.float_format", "{:.6f}".format)
    print(df.to_string())
    print()

    S_test = 100.0
    print(f"Strip price        : {strip.price(S_test):.6f}")
    print(f"Analytic delta     : {strip.delta(S_test):.6f}")
    print(f"FD delta           : {strip.fd_delta(S_test):.6f}")
    print(f"Analytic str-delta : {strip.strike_delta(S_test):.6f}")
    print(f"FD str-delta       : {strip.fd_strike_delta(S_test):.6f}")
