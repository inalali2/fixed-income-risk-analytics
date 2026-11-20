import numpy as np
import pytest

from fixed_income.curves import (
    ZeroCurve,
    dynamic_ns_factors,
    fit_nelson_siegel,
    fit_svensson,
    nelson_siegel,
    svensson,
)
from fixed_income.data import load_sample, tenor_years

T = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 30])
Y = np.array([2.30, 2.34, 2.40, 2.52, 2.74, 2.85, 3.01, 3.11, 3.25, 3.63])


def test_nelson_siegel_recovers_known_parameters():
    true = dict(b0=4.0, b1=-1.5, b2=1.0, tau=2.0)
    y = nelson_siegel(T, **true)
    fit = fit_nelson_siegel(T, y)
    assert fit.rmse_bp < 0.01
    for k, v in true.items():
        assert fit.params[k] == pytest.approx(v, abs=1e-3)


def test_svensson_fits_at_least_as_well_as_ns():
    ns, sv = fit_nelson_siegel(T, Y), fit_svensson(T, Y)
    assert sv.rmse_bp <= ns.rmse_bp + 1e-9
    assert sv.rmse_bp < 2.0


def test_svensson_reduces_to_ns_when_b3_zero():
    a = svensson(T, 4, -1, 0.5, 0.0, 1.5, 9.0)
    b = nelson_siegel(T, 4, -1, 0.5, 1.5)
    np.testing.assert_allclose(a, b)


def test_bootstrap_reprices_par_quotes():
    zc = ZeroCurve.from_market(T, Y, "pchip")
    for t, y in zip(T, Y):
        if t >= 2:
            assert zc.par_yield(t) == pytest.approx(y, abs=1e-6)


def test_bills_are_simple_yields():
    zc = ZeroCurve.from_market(T, Y)
    assert float(zc.df(0.5)) == pytest.approx(1 / (1 + 0.024 * 0.5), rel=1e-10)


def test_flat_par_curve_gives_nearly_flat_zero_curve():
    flat = np.full_like(T, 3.0)
    zc = ZeroCurve.from_market(T, flat)
    long = zc.zero(np.array([2, 5, 10, 30])) * 100
    # 3% semiannual par = 2*ln(1.015) continuous. Bills are simple yields, so the
    # 1 year point differs slightly and the bootstrap inherits a tiny wedge.
    assert np.allclose(long, 2 * np.log(1.015) * 100, atol=0.01)


def test_key_rate_bumps_sum_to_parallel():
    zc = ZeroCurve.from_market(T, Y)
    grid = np.linspace(0.1, 35, 300)
    from fixed_income.curves import KEY_TENORS

    total = sum(zc.key_rate_bump(k, 1.0).zero(grid) - zc.zero(grid) for k in KEY_TENORS)
    np.testing.assert_allclose(total * 1e4, 1.0, atol=1e-6)


def test_forward_consistency():
    zc = ZeroCurve.from_market(T, Y)
    f = zc.forward(2.0, 5.0)
    assert float(zc.df(5.0)) == pytest.approx(float(zc.df(2.0)) * np.exp(-f * 3.0), rel=1e-12)


def test_dynamic_factors_fit_sample_history():
    hist, _ = load_sample()
    f = dynamic_ns_factors(hist.iloc[-250:], tenor_years(hist.columns))
    assert f["rmse_bp"].median() < 10
    assert set(f.columns) >= {"level", "slope", "curvature"}
