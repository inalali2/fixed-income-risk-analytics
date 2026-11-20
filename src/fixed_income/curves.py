"""Yield curve construction and models.

* ZeroCurve: continuously compounded zero curve bootstrapped from treasury bill
  and par bond yields, with discount factors, forwards, par yields and bumps
  for key rate risk.
* Nelson-Siegel and Svensson parametric models, plus Diebold-Li dynamic
  factors (level, slope, curvature) through time.

Conventions: public functions take and return yields in PERCENT; ZeroCurve
stores decimal continuously compounded zero rates internally.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

KEY_TENORS = np.array([0.25, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
BILL_CUTOFF = 1.0  # tenors at or below this are quoted as money market (simple) yields


# --------------------------------------------------------------------------- #
# Parametric models
# --------------------------------------------------------------------------- #
def _ns_loadings(t: np.ndarray, tau: float) -> np.ndarray:
    t = np.maximum(np.asarray(t, dtype=float), 1e-6)
    x = t / tau
    slope = (1 - np.exp(-x)) / x
    curv = slope - np.exp(-x)
    return np.column_stack([np.ones_like(t), slope, curv])


def nelson_siegel(t, b0, b1, b2, tau):
    return _ns_loadings(t, tau) @ np.array([b0, b1, b2])


def svensson(t, b0, b1, b2, b3, tau1, tau2):
    t = np.maximum(np.asarray(t, dtype=float), 1e-6)
    x2 = t / tau2
    extra = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    return nelson_siegel(t, b0, b1, b2, tau1) + b3 * extra


@dataclass
class CurveFit:
    model: str
    params: dict
    rmse_bp: float
    max_abs_err_bp: float

    def predict(self, t) -> np.ndarray:
        p = self.params
        if self.model == "nelson_siegel":
            return nelson_siegel(t, p["b0"], p["b1"], p["b2"], p["tau"])
        return svensson(t, p["b0"], p["b1"], p["b2"], p["b3"], p["tau1"], p["tau2"])


def _fit_stats(model, params, t, y):
    fit = CurveFit(model, params, 0.0, 0.0)
    err = (fit.predict(t) - y) * 100
    fit.rmse_bp = float(np.sqrt(np.mean(err**2)))
    fit.max_abs_err_bp = float(np.max(np.abs(err)))
    return fit


def fit_nelson_siegel(t, y, tau_grid=None) -> CurveFit:
    """Fit Nelson-Siegel. Betas are linear given tau, so grid search tau with OLS
    for the betas, then polish all four parameters with nonlinear least squares."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    tau_grid = np.linspace(0.3, 8.0, 78) if tau_grid is None else tau_grid
    best = None
    for tau in tau_grid:
        X = _ns_loadings(t, tau)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(np.sum((X @ beta - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, beta, tau)
    _, beta, tau = best
    res = least_squares(
        lambda p: nelson_siegel(t, *p) - y,
        x0=[*beta, tau],
        bounds=([-5, -15, -30, 0.1], [20, 15, 30, 15]),
    )
    b0, b1, b2, tau = res.x
    return _fit_stats("nelson_siegel", {"b0": b0, "b1": b1, "b2": b2, "tau": tau}, t, y)


def fit_svensson(t, y) -> CurveFit:
    """Fit Svensson with several starting points (the objective is not convex)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    ns = fit_nelson_siegel(t, y)
    p = ns.params
    best = None
    for tau1 in (0.5, 1.0, 2.0, p["tau"]):
        for tau2 in (4.0, 8.0, 15.0):
            if tau2 <= tau1:
                continue
            res = least_squares(
                lambda q: svensson(t, *q) - y,
                x0=[p["b0"], p["b1"], p["b2"], 0.0, tau1, tau2],
                bounds=([-5, -15, -30, -30, 0.1, 0.5], [20, 15, 30, 30, 10, 30]),
            )
            if best is None or res.cost < best.cost:
                best = res
    b0, b1, b2, b3, tau1, tau2 = best.x
    params = {"b0": b0, "b1": b1, "b2": b2, "b3": b3, "tau1": tau1, "tau2": tau2}
    return _fit_stats("svensson", params, t, y)


def dynamic_ns_factors(history: pd.DataFrame, tenors: np.ndarray, tau: float | None = None) -> pd.DataFrame:
    """Diebold-Li: fix tau, then each day's level, slope and curvature are an OLS fit.

    Returns a DataFrame with columns level, slope, curvature (percent) plus rmse_bp.
    Slope is reported as short minus long convention flipped to the familiar
    'long minus short' sign (positive = upward sloping curve).
    """
    Y = history.values
    if tau is None:
        tau = float(np.median([fit_nelson_siegel(tenors, row).params["tau"] for row in Y[:: max(1, len(Y) // 40)]]))
    X = _ns_loadings(tenors, tau)
    betas, *_ = np.linalg.lstsq(X, Y.T, rcond=None)
    fitted = (X @ betas).T
    rmse = np.sqrt(np.mean((fitted - Y) ** 2, axis=1)) * 100
    out = pd.DataFrame(
        {"level": betas[0], "slope": -betas[1], "curvature": betas[2], "rmse_bp": rmse},
        index=history.index,
    )
    out.attrs["tau"] = tau
    return out


# --------------------------------------------------------------------------- #
# Zero curve
# --------------------------------------------------------------------------- #
class ZeroCurve:
    """Continuously compounded zero curve: monotone cubic (PCHIP) interpolation of zero
    rates and flat extrapolation at both ends."""

    def __init__(self, times, zero_rates, label: str = "", _base=None, _overlays=()):
        order = np.argsort(times)
        self.times = np.asarray(times, float)[order]
        self.rates = np.asarray(zero_rates, float)[order]  # decimal, continuous
        self.label = label
        # Monotone cubic on zero rates gives smooth forwards without overshoot
        self._interp = _base if _base is not None else PchipInterpolator(self.times, self.rates, extrapolate=False)
        # Shifts are stored as additive functions of t (in bp) so bumps are exact and cheap
        self._overlays = tuple(_overlays)

    def _with_overlay(self, fn) -> ZeroCurve:
        return ZeroCurve(self.times, self.rates, self.label, self._interp, self._overlays + (fn,))

    # ----- core ----------------------------------------------------------- #
    def zero(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        z = self._interp(np.clip(t, self.times[0], self.times[-1]))
        for fn in self._overlays:
            z = z + np.asarray(fn(t), float) / 1e4
        return z

    def df(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        return np.exp(-self.zero(t) * t)

    def forward(self, t1, t2) -> np.ndarray:
        """Continuously compounded forward rate between t1 and t2 (decimal)."""
        t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
        return (self.zero(t2) * t2 - self.zero(t1) * t1) / (t2 - t1)

    def par_yield(self, maturity: float, freq: int = 2) -> float:
        """Par coupon (percent, compounded at freq) for a bond maturing at `maturity`."""
        n = int(round(maturity * freq))
        times = np.arange(1, n + 1) / freq
        d = self.df(times)
        return float(freq * (1 - d[-1]) / d.sum() * 100)

    # ----- transformations -------------------------------------------------- #
    def shifted(self, shift_bp) -> ZeroCurve:
        """Return a new curve shifted by a constant (bp) or a function of time returning bp."""
        if callable(shift_bp):
            return self._with_overlay(shift_bp)
        c = float(shift_bp)
        return self._with_overlay(lambda t: np.full(np.shape(t), c))

    def key_rate_bump(self, key: float, bp: float = 1.0, keys: np.ndarray = KEY_TENORS) -> ZeroCurve:
        """Triangular bump centred on one key tenor, flat beyond the first and last keys.
        Bumps across all keys sum exactly to a parallel shift."""
        keys = np.asarray(keys, float)
        i = int(np.where(np.isclose(keys, key))[0][0])
        w = np.zeros(len(keys))
        w[i] = 1.0
        return self._with_overlay(lambda t: np.interp(t, keys, w) * bp)

    # ----- construction ----------------------------------------------------- #
    @classmethod
    def from_market(
        cls,
        tenors: np.ndarray,
        yields_pct: np.ndarray,
        method: str = "pchip",
        freq: int = 2,
        max_maturity: float = 30.0,
        label: str = "",
    ) -> ZeroCurve:
        """Bootstrap a zero curve from bill yields (simple, ACT/365) and par bond yields.

        method:
            'pchip'         monotone cubic interpolation of par yields (exact through quotes)
            'nelson_siegel' smooth fitted par curve
            'svensson'      smooth fitted par curve with a second hump
        """
        tenors, y = np.asarray(tenors, float), np.asarray(yields_pct, float)
        mask = ~np.isnan(y)
        tenors, y = tenors[mask], y[mask]

        if method == "pchip":
            par_fn = PchipInterpolator(tenors, y, extrapolate=True)
        elif method == "nelson_siegel":
            par_fn = fit_nelson_siegel(tenors, y).predict
        elif method == "svensson":
            par_fn = fit_svensson(tenors, y).predict
        else:
            raise ValueError(method)

        def clamp(t):
            return np.asarray(par_fn(np.clip(t, tenors.min(), tenors.max())), float)

        # Money market segment: DF = 1 / (1 + r t)
        bill_t = np.array(sorted(set(tenors[tenors <= BILL_CUTOFF]) | {0.5, 1.0}))
        bill_r = clamp(bill_t) / 100
        bill_df = 1 / (1 + bill_r * bill_t)
        df_map = dict(zip(np.round(bill_t, 10), bill_df))

        # Bond segment: bootstrap par yields on the coupon grid
        step = 1 / freq
        grid = np.arange(step, max_maturity + 1e-9, step)
        for T in grid:
            key = round(T, 10)
            if T <= BILL_CUTOFF + 1e-9:
                if key not in df_map:
                    r = clamp(np.array([T]))[0] / 100
                    df_map[key] = 1 / (1 + r * T)
                continue
            c = clamp(np.array([T]))[0] / 100 / freq
            prior = sum(df_map[round(s, 10)] for s in grid if s < T - 1e-9)
            df_map[key] = (1 - c * prior) / (1 + c)

        times = np.array(sorted(df_map))
        dfs = np.array([df_map[k] for k in times])
        zeros = -np.log(dfs) / times
        return cls(times, zeros, label)

    def to_frame(self, grid=None) -> pd.DataFrame:
        grid = np.linspace(0.25, 30, 120) if grid is None else grid
        return pd.DataFrame(
            {
                "t": grid,
                "zero_pct": self.zero(grid) * 100,
                "fwd_1y_pct": self.forward(grid, grid + 1.0) * 100,
                "df": self.df(grid),
            }
        )
