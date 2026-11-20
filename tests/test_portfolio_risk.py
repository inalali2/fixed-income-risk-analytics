from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fixed_income import optimize, risk
from fixed_income.curves import ZeroCurve
from fixed_income.data import YieldStore, clean_yields, load_sample, tenor_years
from fixed_income.portfolio import Portfolio

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup():
    hist, credit = load_sample()
    val = hist.index[-1]
    curve = ZeroCurve.from_market(tenor_years(hist.columns), hist.iloc[-1].values)
    port = Portfolio.from_csv(ROOT / "config" / "portfolio.csv").live(val)
    an = port.analytics(curve, val)
    return hist, credit, val, curve, port, an


def test_weights_sum_to_one(setup):
    *_, an = setup
    assert an["weight"].sum() == pytest.approx(1.0)


def test_krd_profile_sums_to_dv01(setup):
    *_, an = setup
    assert Portfolio.krd_profile(an).sum() == pytest.approx(an["dv01"].sum(), rel=1e-6)


def test_dv01_matches_small_parallel_shift(setup):
    _, _, val, curve, port, an = setup
    pnl = port.market_value(curve.shifted(1.0), val) - port.market_value(curve, val)
    assert -pnl == pytest.approx(an["dv01"].sum(), rel=0.01)


def test_convexity_makes_rally_beat_selloff(setup):
    _, _, val, curve, port, _ = setup
    sc = risk.run_scenarios(port, curve, val, risk.STANDARD_SCENARIOS[:2]).set_index("Scenario")["P&L ($)"]
    assert sc["Parallel -100bp"] > -sc["Parallel +100bp"] > 0


def test_spread_scenario_only_hits_credit(setup):
    _, _, val, curve, port, _ = setup
    s = risk.Scenario("spreads", {0: 0, 30: 0}, {"Corporate A": 50, "Corporate BBB": 50, "Provincial": 50})
    out = risk.run_scenarios(port, curve, val, [s]).iloc[0]
    assert out["Rates P&L ($)"] == pytest.approx(0, abs=1e-6)
    assert out["Spread P&L ($)"] < 0


def test_var_is_ordered_and_close_to_parametric(setup):
    hist, credit, val, _, port, _ = setup
    v = risk.historical_var(port, hist, val, credit, lookback=250)
    assert v.var[0.99] > v.var[0.95] > 0
    assert v.es[0.99] >= v.var[0.99]
    exp = risk.par_tenor_exposures(port, hist.iloc[-1], val)
    p = risk.parametric_var(exp, hist, lookback=250)
    assert 0.5 < v.var[0.95] / p[0.95] < 1.5


def test_pca_level_dominates(setup):
    hist = setup[0]
    pca = risk.curve_pca(hist)
    assert pca.explained[0] > 0.7
    assert (pca.loadings["Level"] > 0).all()


def test_optimizer_respects_constraints(setup):
    hist, _, val, curve, _, an = setup
    uan = Portfolio.from_csv(ROOT / "config" / "universe.csv").live(val).analytics(curve, val)
    target = float((an["weight"] * an["eff_duration"]).sum())
    dts = optimize.portfolio_dts(an)
    res = optimize.optimize_yield(uan, optimize.Constraints(target_duration=target, max_dts=dts))
    w = res.weights.values
    assert w.sum() == pytest.approx(1.0)
    assert w.max() <= 0.10 + 1e-9
    assert abs(w @ uan["eff_duration"].values - target) <= 0.10 + 1e-9
    assert res.summary["DTS (yrs x bp)"].iloc[0] <= dts + 1e-6
    current = optimize.current_weights_in_universe(an, uan) @ uan["ytm"].values
    assert res.summary["Yield (%)"].iloc[0] >= current - 1e-9


def test_cleaning_handles_messy_input():
    idx = pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-03", "2026-01-05", "2026-01-06", "2026-01-07"])
    df = pd.DataFrame({"2Y": [3.0, 3.01, 3.02, np.nan, 3.05, 3.80], "10Y": [3.5, 3.5, 3.5, 3.52, np.nan, 3.55]}, index=idx)
    out, qr = clean_yields(df)
    assert qr.duplicate_dates == 1 and qr.weekend_rows == 1
    assert out.isna().sum().sum() == 0
    assert any(c == "2Y" for _, c, _ in qr.large_moves)


def test_sqlite_store_round_trip(tmp_path):
    store = YieldStore(tmp_path / "t.db")
    df = pd.DataFrame({"2Y": [3.0, 3.1], "10Y": [3.5, 3.6]}, index=pd.to_datetime(["2026-01-05", "2026-01-06"]))
    store.upsert("boc", df)
    store.upsert("boc", df.assign(**{"2Y": [3.0, 3.2]}))  # upsert overwrites
    out = store.load("boc")
    assert out.loc["2026-01-06", "2Y"] == pytest.approx(3.2)
    assert store.last_date("boc") == "2026-01-06"
