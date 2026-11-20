import numpy as np
import pandas as pd
import pytest

from fixed_income.bonds import Bond
from fixed_income.curves import ZeroCurve

VAL = pd.Timestamp("2026-06-01")


def flat_curve(rate_cont: float) -> ZeroCurve:
    return ZeroCurve(np.array([0.01, 50.0]), np.array([rate_cont, rate_cont]))


def test_par_bond_prices_at_100_on_coupon_date():
    b = Bond("par", coupon=4.0, maturity="2036-06-01")
    assert b.schedule(VAL)[2] == pytest.approx(0.0)
    # Within ACT/365 day count noise of par
    assert b.clean_price_from_yield(4.0, VAL) == pytest.approx(100.0, abs=0.05)


def test_ytm_round_trip():
    b = Bond("x", coupon=3.25, maturity="2033-12-01")
    px = b.dirty_price_from_yield(3.9, VAL)
    assert b.ytm(px, VAL) == pytest.approx(3.9, abs=1e-8)


def test_zero_coupon_macaulay_equals_maturity():
    b = Bond("zero", coupon=0.0, maturity="2031-06-01")
    r = b.yield_risk(3.0, VAL)
    assert r["macaulay_duration"] == pytest.approx(b.years_to_maturity(VAL), rel=1e-12)


def test_duration_convexity_predict_price_change():
    b = Bond("x", coupon=3.0, maturity="2045-06-01")
    y, dy = 3.5, 0.50
    r = b.yield_risk(y, VAL)
    p0, p1 = r["dirty_price"], b.dirty_price_from_yield(y + dy, VAL)
    approx = p0 * (-r["modified_duration"] * dy / 100 + 0.5 * r["convexity"] * (dy / 100) ** 2)
    assert (p1 - p0) == pytest.approx(approx, rel=0.01)


def test_accrued_interest_half_period():
    b = Bond("x", coupon=4.0, maturity="2030-12-01")
    accrued = b.schedule(pd.Timestamp("2026-09-01"))[2]
    # 92 of 183 days into the period on a 2.00 semiannual coupon
    assert accrued == pytest.approx(2.0 * 92 / 183, rel=1e-9)


def test_krds_sum_to_effective_duration():
    b = Bond("x", coupon=3.5, maturity="2041-03-01", spread_bp=80)
    r = b.curve_risk(flat_curve(0.03), VAL)
    assert sum(r["krd"].values()) == pytest.approx(r["effective_duration"], rel=1e-6)
    assert r["spread_duration"] == pytest.approx(r["effective_duration"], rel=1e-6)


def test_z_spread_recovers_input_spread():
    b = Bond("x", coupon=4.75, maturity="2034-06-01", spread_bp=135)
    curve = flat_curve(0.032)
    px = b.dirty_price_from_curve(curve, VAL)
    assert b.z_spread(curve, px, VAL) == pytest.approx(135, abs=1e-5)


def test_matured_bond_raises():
    with pytest.raises(ValueError):
        Bond("old", coupon=2.0, maturity="2025-01-01").schedule(VAL)
