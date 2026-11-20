"""Command line entry point.

    fi-risk update                      # pull new Bank of Canada + FRED data into SQLite
    fi-risk report --source boc         # full risk report on live Government of Canada data
    fi-risk report --source sample      # offline demo on synthetic data
    fi-risk curve --source boc          # print today's curve, zero rates and model fits
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .data import PROJECT_ROOT


def _cmd_update(args):
    from .data import YieldStore, update_store

    store = YieldStore(args.db)
    for src in args.sources:
        try:
            print(update_store(store, src, args.start))
        except Exception as exc:
            print(f"{src}: failed ({exc})", file=sys.stderr)


def _cmd_report(args):
    from .pipeline import run_analysis
    from .report import render

    res = run_analysis(
        source=args.source,
        portfolio_path=args.portfolio,
        universe_path=args.universe,
        curve_method=args.curve_method,
        var_lookback=args.lookback,
        refresh=not args.no_refresh,
        val_date=args.date,
    )
    path = render(res, args.out, args.curve_method, args.lookback)
    s = res.summary
    print(f"Valuation date   {res.val_date:%Y-%m-%d}")
    print(f"Market value     ${s['Market value']:,.0f}")
    print(f"Yield            {s['Yield to maturity (%)']:.2f}%")
    print(f"Duration         {s['Effective duration']:.2f}")
    print(f"DV01             ${s['DV01 ($)']:,.0f}")
    print(f"99% 1 day VaR    ${res.var_hist.var[0.99]:,.0f}")
    print(f"Report written to {path}")


def _cmd_curve(args):
    from .curves import ZeroCurve, fit_nelson_siegel, fit_svensson
    from .data import clean_yields, order_tenors, tenor_years
    from .pipeline import load_market

    raw, _, desc = load_market(args.source, refresh=not args.no_refresh)
    hist, _ = clean_yields(order_tenors(raw))
    today = hist.iloc[-1]
    t = tenor_years(hist.columns)
    zc = ZeroCurve.from_market(t, today.values, args.curve_method)
    print(f"{desc}\nCurve date {hist.index[-1]:%Y-%m-%d}\n")
    print(f"{'Tenor':>6} {'Quote %':>8} {'Zero %':>8} {'1y fwd %':>9}")
    for lab, ti, y in zip(hist.columns, t, today.values):
        print(f"{lab:>6} {y:8.3f} {float(zc.zero(ti))*100:8.3f} {float(zc.forward(ti, ti + 1))*100:9.3f}")
    for name, fit in (("Nelson Siegel", fit_nelson_siegel(t, today.values)), ("Svensson", fit_svensson(t, today.values))):
        p = ", ".join(f"{k}={v:.3f}" for k, v in fit.params.items())
        print(f"\n{name}: RMSE {fit.rmse_bp:.2f}bp  max {fit.max_abs_err_bp:.2f}bp\n  {p}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="fi-risk", description="Fixed income curve and portfolio risk analytics")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("update", help="Incrementally download market data into SQLite")
    u.add_argument("--sources", nargs="+", default=["boc", "credit"], choices=["boc", "ust", "credit"])
    u.add_argument("--start", default="2018-01-01")
    u.add_argument("--db", default=PROJECT_ROOT / "data" / "market.db", type=Path)
    u.set_defaults(func=_cmd_update)

    for name, fn, helptext in (("report", _cmd_report, "Build the HTML risk report"), ("curve", _cmd_curve, "Print the curve")):
        r = sub.add_parser(name, help=helptext)
        r.add_argument("--source", default="boc", choices=["boc", "ust", "sample"])
        r.add_argument("--curve-method", default="pchip", choices=["pchip", "nelson_siegel", "svensson"])
        r.add_argument("--no-refresh", action="store_true", help="Use stored data without downloading")
        r.set_defaults(func=fn)
        if name == "report":
            r.add_argument("--portfolio", default=PROJECT_ROOT / "config" / "portfolio.csv", type=Path)
            r.add_argument("--universe", default=PROJECT_ROOT / "config" / "universe.csv", type=Path)
            r.add_argument("--out", default=PROJECT_ROOT / "reports" / "latest", type=Path)
            r.add_argument("--lookback", default=500, type=int, help="Days of history for VaR")
            r.add_argument("--date", default=None, help="Valuation date (YYYY-MM-DD); defaults to latest")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        args.func(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
