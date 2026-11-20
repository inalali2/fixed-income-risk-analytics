"""Portfolio construction: maximize yield subject to risk and allocation limits.

A linear program over market value weights. Portfolio yield, duration and key
rate exposures are all (approximately) linear in weights, so the problem stays
convex and solves instantly. The default use is a security selection exercise:
keep the current portfolio's duration and curve exposure, and look for extra
yield inside the investable universe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

BUCKETS = {"Short (0 to 3Y)": ["3M", "1Y", "2Y", "3Y"], "Belly (5 to 7Y)": ["5Y", "7Y"], "Long (10Y+)": ["10Y", "20Y", "30Y"]}


@dataclass
class Constraints:
    target_duration: float
    duration_tol: float = 0.10
    max_weight: float = 0.10
    min_government: float = 0.20
    max_corporate: float = 0.35
    max_bbb: float = 0.15
    bucket_targets: dict | None = None   # bucket name -> duration contribution (years)
    bucket_tol: float = 0.30
    max_dts: float | None = None         # cap on duration times spread (years x bp)


@dataclass
class OptimResult:
    weights: pd.Series
    summary: pd.DataFrame
    bucket_check: pd.DataFrame
    status: str


def _krd_years(an: pd.DataFrame) -> pd.DataFrame:
    """Key rate durations in years (not dollars) for each bond."""
    cols = [c for c in an.columns if c.startswith("krd_")]
    return an[cols].div(an["market_value"] * 1e-4, axis=0).rename(columns=lambda c: c.replace("krd_", ""))


def bucket_durations(an: pd.DataFrame, weights: np.ndarray) -> dict:
    k = _krd_years(an)
    return {b: float(weights @ k[cols].sum(axis=1).values) for b, cols in BUCKETS.items()}


def optimize_yield(universe_an: pd.DataFrame, cons: Constraints) -> OptimResult:
    """universe_an: output of Portfolio.analytics() on the universe (any par amounts)."""
    n = len(universe_an)
    y = universe_an["ytm"].values
    dur = universe_an["eff_duration"].values
    gov = (universe_an["sector"] == "Government").astype(float).values
    corp = (universe_an["sector"] == "Corporate").astype(float).values
    bbb = universe_an["rating"].str.upper().str.startswith("BBB").astype(float).values
    krd = _krd_years(universe_an)
    dts = (universe_an["spread_duration"] * universe_an["spread_bp"]).values

    A_ub, b_ub = [], []
    A_ub += [dur, -dur]
    b_ub += [cons.target_duration + cons.duration_tol, -(cons.target_duration - cons.duration_tol)]
    A_ub += [-gov, corp, bbb]
    b_ub += [-cons.min_government, cons.max_corporate, cons.max_bbb]
    if cons.max_dts is not None:
        A_ub.append(dts)
        b_ub.append(cons.max_dts)
    if cons.bucket_targets:
        for bucket, cols in BUCKETS.items():
            row = krd[cols].sum(axis=1).values
            tgt = cons.bucket_targets[bucket]
            A_ub += [row, -row]
            b_ub += [tgt + cons.bucket_tol, -(tgt - cons.bucket_tol)]

    res = linprog(
        c=-y,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.ones((1, n)),
        b_eq=[1.0],
        bounds=[(0, cons.max_weight)] * n,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    w = pd.Series(np.where(res.x > 1e-6, res.x, 0.0), index=universe_an["id"].values, name="weight")
    wv = w.values
    summary = pd.DataFrame(
        {
            "Yield (%)": [wv @ y],
            "Effective duration": [wv @ dur],
            "Government weight": [wv @ gov],
            "Corporate weight": [wv @ corp],
            "BBB weight": [wv @ bbb],
            "DTS (yrs x bp)": [wv @ dts],
            "Holdings": [int((wv > 0).sum())],
        }
    )
    bc = pd.DataFrame({"Optimized": bucket_durations(universe_an, wv)})
    if cons.bucket_targets:
        bc["Target"] = pd.Series(cons.bucket_targets)
    return OptimResult(w, summary, bc, res.message)


def portfolio_dts(an: pd.DataFrame) -> float:
    """Duration times spread: spread duration x spread, market value weighted. A standard credit risk measure."""
    return float((an["weight"] * an["spread_duration"] * an["spread_bp"]).sum())


def current_weights_in_universe(current_an: pd.DataFrame, universe_an: pd.DataFrame) -> np.ndarray:
    w = current_an.set_index("id")["weight"]
    return universe_an["id"].map(w).fillna(0.0).values
