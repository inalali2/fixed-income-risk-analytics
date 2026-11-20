"""Fixed rate bond pricing and risk.

Prices are per 100 face. Time is measured ACT/365 from the valuation date.
Yields are in percent, compounded at the coupon frequency (Canadian and US
government bonds pay semiannually).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .curves import KEY_TENORS, ZeroCurve

BP = 1e-4


@dataclass
class Bond:
    id: str
    coupon: float                 # annual coupon, percent
    maturity: pd.Timestamp
    issuer: str = ""
    sector: str = "Government"    # Government | Provincial | Corporate
    rating: str = "AAA"
    spread_bp: float = 0.0        # credit spread over the government zero curve
    freq: int = 2
    face: float = 100.0
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        self.maturity = pd.Timestamp(self.maturity)

    # ----- cash flows ------------------------------------------------------- #
    def schedule(self, val_date) -> tuple[np.ndarray, np.ndarray, float]:
        """Remaining cash flow times (years), amounts, and accrued interest."""
        val = pd.Timestamp(val_date)
        key = val.value
        if key in self._cache:
            return self._cache[key]
        if self.maturity <= val:
            raise ValueError(f"{self.id} has matured as of {val.date()}")
        step = pd.DateOffset(months=12 // self.freq)
        dates = [self.maturity]
        while dates[-1] - step > val:
            dates.append(dates[-1] - step)
        prev_coupon = dates[-1] - step
        dates = dates[::-1]
        t = np.array([(d - val).days / 365.0 for d in dates])
        cpn = self.coupon / self.freq * self.face / 100
        cf = np.full(len(t), cpn)
        cf[-1] += self.face
        period = (dates[0] - prev_coupon).days
        accrued = cpn * (val - prev_coupon).days / period
        self._cache[key] = (t, cf, accrued)
        return t, cf, accrued

    def years_to_maturity(self, val_date) -> float:
        return (self.maturity - pd.Timestamp(val_date)).days / 365.0

    # ----- yield based ------------------------------------------------------ #
    def dirty_price_from_yield(self, ytm_pct: float, val_date) -> float:
        t, cf, _ = self.schedule(val_date)
        v = (1 + ytm_pct / 100 / self.freq) ** (-self.freq * t)
        return float(np.sum(cf * v))

    def clean_price_from_yield(self, ytm_pct: float, val_date) -> float:
        return self.dirty_price_from_yield(ytm_pct, val_date) - self.schedule(val_date)[2]

    def ytm(self, dirty_price: float, val_date) -> float:
        f = lambda y: self.dirty_price_from_yield(y, val_date) - dirty_price  # noqa: E731
        return float(brentq(f, -5.0, 50.0, xtol=1e-10))

    def yield_risk(self, ytm_pct: float, val_date) -> dict:
        """Analytic Macaulay and modified duration, convexity and DV01 (per 100 face)."""
        t, cf, _ = self.schedule(val_date)
        y = ytm_pct / 100
        k = self.freq
        v = (1 + y / k) ** (-k * t)
        pv = cf * v
        price = pv.sum()
        mac = float(np.sum(t * pv) / price)
        mod = mac / (1 + y / k)
        conv = float(np.sum(pv * t * (t + 1 / k)) / (price * (1 + y / k) ** 2))
        return {
            "dirty_price": float(price),
            "macaulay_duration": mac,
            "modified_duration": mod,
            "convexity": conv,
            "dv01": mod * price * BP,
        }

    # ----- curve based ------------------------------------------------------ #
    def dirty_price_from_curve(self, curve: ZeroCurve, val_date, spread_bp: float | None = None) -> float:
        t, cf, _ = self.schedule(val_date)
        s = (self.spread_bp if spread_bp is None else spread_bp) * BP
        return float(np.sum(cf * np.exp(-(curve.zero(t) + s) * t)))

    def z_spread(self, curve: ZeroCurve, dirty_price: float, val_date) -> float:
        """Constant spread (bp) over the zero curve that reprices the bond."""
        f = lambda s: self.dirty_price_from_curve(curve, val_date, s) - dirty_price  # noqa: E731
        return float(brentq(f, -500, 5000, xtol=1e-8))

    def curve_risk(self, curve: ZeroCurve, val_date, bump_bp: float = 1.0, keys=KEY_TENORS) -> dict:
        """Effective duration, convexity, spread duration and key rate durations
        by full revaluation under curve bumps."""
        p0 = self.dirty_price_from_curve(curve, val_date)
        up = self.dirty_price_from_curve(curve.shifted(bump_bp), val_date)
        dn = self.dirty_price_from_curve(curve.shifted(-bump_bp), val_date)
        h = bump_bp * BP
        eff_dur = (dn - up) / (2 * p0 * h)
        eff_conv = (up + dn - 2 * p0) / (p0 * h**2)
        s_up = self.dirty_price_from_curve(curve, val_date, self.spread_bp + bump_bp)
        s_dn = self.dirty_price_from_curve(curve, val_date, self.spread_bp - bump_bp)
        spread_dur = (s_dn - s_up) / (2 * p0 * h)
        krd = {}
        for k in keys:
            ku = self.dirty_price_from_curve(curve.key_rate_bump(k, bump_bp, keys), val_date)
            kd = self.dirty_price_from_curve(curve.key_rate_bump(k, -bump_bp, keys), val_date)
            krd[float(k)] = (kd - ku) / (2 * p0 * h)
        return {
            "dirty_price": p0,
            "effective_duration": eff_dur,
            "effective_convexity": eff_conv,
            "spread_duration": spread_dur,
            "krd": krd,
        }


def bond_from_row(row: pd.Series | dict) -> Bond:
    r = dict(row)
    return Bond(
        id=str(r["id"]),
        coupon=float(r["coupon"]),
        maturity=pd.Timestamp(r["maturity"]),
        issuer=str(r.get("issuer", "")),
        sector=str(r.get("sector", "Government")),
        rating=str(r.get("rating", "AAA")),
        spread_bp=float(r.get("spread_bp", 0.0) or 0.0),
        freq=int(r.get("freq", 2) or 2),
    )
