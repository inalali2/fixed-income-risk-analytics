"""End to end analysis: load data, clean, build curves, run portfolio risk, optimize."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import optimize, risk
from .curves import ZeroCurve, dynamic_ns_factors, fit_nelson_siegel, fit_svensson
from .data import (
    PROJECT_ROOT,
    QualityReport,
    YieldStore,
    clean_yields,
    load_sample,
    order_tenors,
    tenor_years,
    update_store,
)
from .portfolio import Portfolio

log = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    source: str
    val_date: pd.Timestamp
    history: pd.DataFrame
    credit: pd.DataFrame | None
    quality: QualityReport
    curve: ZeroCurve
    ns_fit: object
    sv_fit: object
    ns_factors: pd.DataFrame
    analytics: pd.DataFrame
    summary: dict
    krd: pd.Series
    sector: pd.DataFrame
    rating: pd.DataFrame
    scenarios: pd.DataFrame
    var_hist: risk.VaRResult
    var_hist_rates: risk.VaRResult
    var_param: dict
    pca: risk.PCAResult
    pca_pnl: pd.Series
    optim: optimize.OptimResult | None
    optim_current: dict = field(default_factory=dict)
    universe_analytics: pd.DataFrame | None = None


def load_market(source: str, refresh: bool = True, db_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """Return (curve history, credit history, description of source)."""
    if source == "sample":
        curve, credit = load_sample()
        return curve, credit, "Synthetic sample data (offline demo, not real market data)"

    store = YieldStore(db_path or PROJECT_ROOT / "data" / "market.db")
    if refresh:
        for src in ("boc" if source == "boc" else "ust", "credit"):
            try:
                log.info("Refreshing %s: %s", src, update_store(store, src))
            except Exception as exc:
                log.warning("Refresh of %s failed (%s); using stored data", src, exc)
    curve = store.load("boc" if source == "boc" else "ust")
    if curve.empty:
        raise RuntimeError("No stored market data. Check your internet connection or use --source sample.")
    credit = store.load("credit")
    desc = {
        "boc": "Bank of Canada Valet API (Government of Canada bills and benchmark bonds)",
        "ust": "FRED (US Treasury constant maturity yields)",
    }[source]
    desc += "; credit spreads: ICE BofA US corporate OAS via FRED (proxy)"
    return curve, (credit if not credit.empty else None), desc


def run_analysis(
    source: str = "sample",
    portfolio_path: str | Path = PROJECT_ROOT / "config" / "portfolio.csv",
    universe_path: str | Path | None = PROJECT_ROOT / "config" / "universe.csv",
    curve_method: str = "pchip",
    var_lookback: int = 500,
    refresh: bool = True,
    val_date: str | None = None,
) -> AnalysisResult:
    raw, credit, desc = load_market(source, refresh)
    history, qr = clean_yields(order_tenors(raw))
    if val_date:
        history = history.loc[:val_date]
    val = history.index[-1]
    if credit is not None:
        credit = credit.ffill().reindex(history.index).ffill()

    tenors = tenor_years(history.columns)
    today = history.iloc[-1]
    curve = ZeroCurve.from_market(tenors, today.values, curve_method, label=f"{val:%Y-%m-%d}")
    ns, sv = fit_nelson_siegel(tenors, today.values), fit_svensson(tenors, today.values)
    factors = dynamic_ns_factors(history, tenors)

    port = Portfolio.from_csv(portfolio_path).live(val)
    an = port.analytics(curve, val)
    summ = Portfolio.summary(an)
    summ["DTS (yrs x bp)"] = optimize.portfolio_dts(an)

    scen = risk.STANDARD_SCENARIOS + [risk.worst_historical_window(history, credit)]
    scen_df = risk.run_scenarios(port, curve, val, scen)

    vh = risk.historical_var(port, history, val, credit, lookback=var_lookback, curve_method=curve_method)
    vr = risk.historical_var(port, history, val, None, lookback=var_lookback, curve_method=curve_method)
    exposures = risk.par_tenor_exposures(port, today, val, curve_method)
    vp = risk.parametric_var(exposures, history, lookback=var_lookback)

    pca = risk.curve_pca(history)
    ppnl = risk.pca_pnl(port, today, val, pca, curve_method)

    optim, uan, cur = None, None, {}
    if universe_path and Path(universe_path).exists():
        uan = Portfolio.from_csv(universe_path).live(val).analytics(curve, val)
        buckets = optimize.bucket_durations(an, an["weight"].values)
        cons = optimize.Constraints(
            target_duration=summ["Effective duration"], bucket_targets=buckets, max_dts=summ["DTS (yrs x bp)"]
        )
        try:
            optim = optimize.optimize_yield(uan, cons)
            gov = (an["sector"] == "Government")
            cur = {
                "Yield (%)": summ["Yield to maturity (%)"],
                "Effective duration": summ["Effective duration"],
                "Government weight": float(an.loc[gov, "weight"].sum()),
                "Corporate weight": float(an.loc[an["sector"] == "Corporate", "weight"].sum()),
                "BBB weight": float(an.loc[an["rating"].str.upper().str.startswith("BBB"), "weight"].sum()),
                "DTS (yrs x bp)": summ["DTS (yrs x bp)"],
                "Holdings": len(an),
            }
        except RuntimeError as exc:
            log.warning("Optimizer: %s", exc)

    return AnalysisResult(
        source=desc,
        val_date=val,
        history=history,
        credit=credit,
        quality=qr,
        curve=curve,
        ns_fit=ns,
        sv_fit=sv,
        ns_factors=factors,
        analytics=an,
        summary=summ,
        krd=Portfolio.krd_profile(an),
        sector=Portfolio.breakdown(an, "sector"),
        rating=Portfolio.breakdown(an, "rating"),
        scenarios=scen_df,
        var_hist=vh,
        var_hist_rates=vr,
        var_param=vp,
        pca=pca,
        pca_pnl=ppnl,
        optim=optim,
        optim_current=cur,
        universe_analytics=uan,
    )
