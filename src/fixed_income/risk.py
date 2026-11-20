"""Risk engine: stress scenarios, historical and parametric VaR, PCA of curve moves."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .curves import ZeroCurve
from .data import tenor_years
from .portfolio import Portfolio

# Spread shock mapping used by historical VaR: each bond's spread change is
# beta * (daily change in a credit index). Provincials move less than corporates.
SPREAD_FACTOR_MAP = {
    ("Government", None): (None, 0.0),
    ("Provincial", None): ("IG_OAS", 0.35),
    ("Corporate", "A"): ("IG_OAS", 1.0),
    ("Corporate", "BBB"): ("BBB_OAS", 1.0),
}


def _rating_bucket(rating: str) -> str:
    r = rating.upper().replace("+", "").replace("-", "")
    if r.startswith("AAA") or r.startswith("AA") or r == "A":
        return "A"  # A or better
    return "BBB"


def spread_mapping(port: Portfolio) -> list[tuple[str | None, float]]:
    out = []
    for b in port.bonds:
        if b.sector == "Corporate":
            out.append(SPREAD_FACTOR_MAP[("Corporate", _rating_bucket(b.rating))])
        else:
            out.append(SPREAD_FACTOR_MAP.get((b.sector, None), (None, 0.0)))
    return out


# --------------------------------------------------------------------------- #
# Stress scenarios
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    name: str
    curve_points: dict[float, float]                 # tenor (yrs) -> shift (bp), linearly interpolated
    spread_bp: dict[str, float] = field(default_factory=dict)  # sector or rating bucket -> spread shock (bp)
    description: str = ""

    def curve_shift(self):
        ks = np.array(sorted(self.curve_points))
        vs = np.array([self.curve_points[k] for k in ks])
        return lambda t: np.interp(t, ks, vs)

    def spread_shocks(self, port: Portfolio) -> np.ndarray:
        out = []
        for b in port.bonds:
            if b.sector == "Corporate":
                key = f"Corporate {_rating_bucket(b.rating)}"
            else:
                key = b.sector
            out.append(self.spread_bp.get(key, 0.0))
        return np.array(out)


STANDARD_SCENARIOS = [
    Scenario("Parallel +100bp", {0: 100, 30: 100}, description="All rates up 1%"),
    Scenario("Parallel -100bp", {0: -100, 30: -100}, description="All rates down 1%"),
    Scenario("Parallel +25bp", {0: 25, 30: 25}),
    Scenario("Parallel -25bp", {0: -25, 30: -25}),
    Scenario("Bear steepener", {0.25: 0, 2: 10, 10: 50, 30: 60}, description="Long end sells off (term premium)"),
    Scenario("Bull flattener", {0.25: 0, 2: -10, 10: -45, 30: -60}, description="Long end rallies"),
    Scenario("Bear flattener", {0.25: 75, 2: 60, 10: 20, 30: 10}, description="Central bank hikes"),
    Scenario("Bull steepener", {0.25: -75, 2: -60, 10: -20, 30: -10}, description="Central bank cuts"),
    Scenario("Belly up butterfly", {0.25: 0, 2: 0, 5: 25, 7: 25, 10: 0, 30: 0}, description="5 to 7 year sector cheapens"),
    Scenario(
        "Credit risk off",
        {0: -25, 30: -25},
        {"Provincial": 20, "Corporate A": 50, "Corporate BBB": 100},
        description="Flight to quality: govt yields fall, spreads widen",
    ),
    Scenario(
        "Stagflation",
        {0.25: 50, 2: 60, 10: 75, 30: 80},
        {"Provincial": 15, "Corporate A": 40, "Corporate BBB": 80},
        description="Rates and spreads up together",
    ),
]


def worst_historical_window(
    history: pd.DataFrame, credit: pd.DataFrame | None = None, window: int = 63, anchor: str = "5Y"
) -> Scenario:
    """Turn the largest historical sell off over `window` business days into a scenario,
    including the credit spread move over the same dates when credit history is given."""
    chg = history.diff(window)
    end = chg[anchor].idxmax()
    start = history.index[history.index.get_loc(end) - window]
    move = (history.loc[end] - history.loc[start]) * 100
    pts = dict(zip(tenor_years(history.columns), move.values))
    spreads = {}
    if credit is not None:
        c = credit.reindex(history.index).ffill()
        d = (c.loc[end] - c.loc[start]) * 100
        spreads = {
            "Provincial": 0.35 * d.get("IG_OAS", 0.0),
            "Corporate A": d.get("IG_OAS", 0.0),
            "Corporate BBB": d.get("BBB_OAS", 0.0),
        }
    return Scenario(
        f"Historical replay {start:%b %Y} to {end:%b %Y}",
        pts,
        spreads,
        description=f"Worst {window} day rise in the {anchor} yield in the data set",
    )


def run_scenarios(port: Portfolio, curve: ZeroCurve, val_date, scenarios: list[Scenario]) -> pd.DataFrame:
    base = port.market_value(curve, val_date)
    rows = []
    for sc in scenarios:
        shocked = curve.shifted(sc.curve_shift())
        mv = port.market_value(shocked, val_date, sc.spread_shocks(port))
        rate_only = port.market_value(shocked, val_date)
        rows.append(
            {
                "Scenario": sc.name,
                "Description": sc.description,
                "P&L ($)": mv - base,
                "P&L (%)": (mv - base) / base * 100,
                "Rates P&L ($)": rate_only - base,
                "Spread P&L ($)": mv - rate_only,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Value at Risk
# --------------------------------------------------------------------------- #
@dataclass
class VaRResult:
    pnl: pd.Series
    var: dict
    es: dict
    method: str


def historical_var(
    port: Portfolio,
    history: pd.DataFrame,
    val_date,
    credit: pd.DataFrame | None = None,
    lookback: int = 500,
    levels=(0.95, 0.99),
    curve_method: str = "pchip",
) -> VaRResult:
    """1 day historical simulation VaR with full revaluation.

    Each historical daily change in the yield curve (and credit index, if given)
    is applied to today's curve, the zero curve is rebuilt and every bond is
    repriced. No duration approximation, so convexity and curve shape are captured.
    """
    tenors = tenor_years(history.columns)
    today = history.loc[:val_date].iloc[-1].values
    base_curve = ZeroCurve.from_market(tenors, today, curve_method)
    base = port.market_value(base_curve, val_date)

    dY = history.loc[:val_date].diff().dropna().iloc[-lookback:]
    mapping = spread_mapping(port)
    dS = None
    if credit is not None:
        dS = credit.diff().reindex(dY.index).fillna(0.0) * 100  # bp

    pnl = []
    for dt, row in dY.iterrows():
        shocked = ZeroCurve.from_market(tenors, today + row.values, curve_method)
        shocks = np.zeros(len(port.bonds))
        if dS is not None:
            shocks = np.array([beta * dS.at[dt, f] if f else 0.0 for f, beta in mapping])
        pnl.append(port.market_value(shocked, val_date, shocks) - base)
    pnl = pd.Series(pnl, index=dY.index, name="P&L")
    var = {lv: float(-np.quantile(pnl, 1 - lv)) for lv in levels}
    es = {lv: float(-pnl[pnl <= -var[lv]].mean()) for lv in levels}
    return VaRResult(pnl, var, es, "Historical simulation, full revaluation")


def par_tenor_exposures(port: Portfolio, today_yields: pd.Series, val_date, curve_method: str = "pchip") -> pd.Series:
    """Dollar P&L for a +1bp move in each quoted market tenor (rebuilding the curve each time)."""
    t = tenor_years(today_yields.index)
    y0 = today_yields.values.astype(float)
    base = port.market_value(ZeroCurve.from_market(t, y0, curve_method), val_date)
    out = {}
    for i, lab in enumerate(today_yields.index):
        y = y0.copy()
        y[i] += 0.01
        out[lab] = port.market_value(ZeroCurve.from_market(t, y, curve_method), val_date) - base
    return pd.Series(out, name="P&L per +1bp ($)")


def parametric_var(exposures: pd.Series, history: pd.DataFrame, lookback: int = 500, levels=(0.95, 0.99)) -> dict:
    """Delta normal VaR: $ per bp exposures at each quoted tenor and the covariance of daily changes (bp)."""
    from scipy.stats import norm

    cols = [c for c in exposures.index if c in history.columns]
    dY = history[cols].diff().dropna().iloc[-lookback:] * 100
    cov = np.cov(dY.values.T)
    x = exposures[cols].values
    sigma = float(np.sqrt(x @ cov @ x))
    return {lv: float(norm.ppf(lv) * sigma) for lv in levels}


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #
@dataclass
class PCAResult:
    explained: np.ndarray
    loadings: pd.DataFrame       # tenors x components, bp move per 1 sd of the component
    scores: pd.DataFrame


def curve_pca(history: pd.DataFrame, lookback: int = 750, n_components: int = 3) -> PCAResult:
    """PCA on daily yield changes (bp). Loadings are scaled to a 1 standard deviation daily move
    and signed so level is positive, slope is long minus short positive."""
    dY = history.diff().dropna().iloc[-lookback:] * 100
    pca = PCA(n_components=n_components).fit(dY.values)
    names = ["Level", "Slope", "Curvature"][:n_components]
    load = pca.components_.T * np.sqrt(pca.explained_variance_)
    if load[:, 0].mean() < 0:
        load[:, 0] *= -1
    if n_components > 1 and load[-1, 1] - load[0, 1] < 0:
        load[:, 1] *= -1
    if n_components > 2:
        mid = len(load) // 2
        if load[mid, 2] < 0:
            load[:, 2] *= -1
    loadings = pd.DataFrame(load, index=history.columns, columns=names)
    scores = pd.DataFrame(pca.transform(dY.values), index=dY.index, columns=names)
    return PCAResult(pca.explained_variance_ratio_, loadings, scores)


def pca_pnl(port: Portfolio, today_yields: pd.Series, val_date, pca: PCAResult, curve_method: str = "pchip") -> pd.Series:
    """Portfolio P&L for a +1 standard deviation daily move in each principal component (full revaluation)."""
    t = tenor_years(today_yields.index)
    base_curve = ZeroCurve.from_market(t, today_yields.values, curve_method)
    base = port.market_value(base_curve, val_date)
    out = {}
    for comp in pca.loadings.columns:
        shocked = ZeroCurve.from_market(t, today_yields.values + pca.loadings[comp].values / 100, curve_method)
        out[comp] = port.market_value(shocked, val_date) - base
    return pd.Series(out, name="P&L per 1 sd daily move ($)")
