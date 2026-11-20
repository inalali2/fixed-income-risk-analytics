"""Render an AnalysisResult into a self contained HTML risk report, PNG charts and CSV extracts."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Template  # noqa: E402

from .data import tenor_years  # noqa: E402
from .pipeline import AnalysisResult  # noqa: E402

# Validated categorical order (blue, orange, aqua) and a blue/red diverging pair
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
GAIN, LOSS = "#2a78d6", "#e34948"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.size": 10,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "axes.axisbelow": True,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def chart_curve(res: AnalysisResult, path: Path):
    t = tenor_years(res.history.columns)
    today = res.history.iloc[-1]
    grid = np.linspace(0.08, 30, 300)
    fgrid = grid[grid <= 29]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(grid, res.curve.zero(grid) * 100, color=C1, label="Zero curve (bootstrapped)")
    ax.plot(fgrid, res.curve.forward(fgrid, fgrid + 1) * 100, color=C2, label="1 year forward rate")
    ax.plot(grid, res.sv_fit.predict(grid), color=C3, linestyle=(0, (4, 2)), label=f"Svensson fit (RMSE {res.sv_fit.rmse_bp:.1f}bp)")
    ax.scatter(t, today.values, s=36, color=INK, zorder=5, edgecolor=SURFACE, linewidth=1.5, label="Market quotes")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Yield (%)")
    ax.set_title(f"Government curve on {res.val_date:%d %b %Y}")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def chart_curve_shift(res: AnalysisResult, path: Path):
    h = res.history
    t = tenor_years(h.columns)
    val = res.val_date
    pts = [("Today", val, C1)]
    for label, off, col in (("1 month ago", 21, C2), ("1 year ago", 252, C3)):
        if len(h) > off:
            pts.append((label, h.index[-1 - off], col))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for label, d, col in pts:
        ax.plot(t, h.loc[d].values, color=col, marker="o", markersize=5, label=f"{label} ({d:%d %b %Y})")
    ax.set_xscale("log")
    ax.set_xticks(t)
    ax.set_xticklabels(h.columns)
    ax.minorticks_off()
    ax.set_ylabel("Yield (%)")
    ax.set_title("How the curve has moved")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def chart_factors(res: AnalysisResult, path: Path):
    f = res.ns_factors
    fig, axes = plt.subplots(3, 1, figsize=(8, 5.4), sharex=True)
    for ax, col, title in zip(axes, ["level", "slope", "curvature"], ["Level", "Slope (long minus short)", "Curvature"]):
        ax.plot(f.index, f[col], color=C1, linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("%")
        if col != "level":
            ax.axhline(0, color=INK2, linewidth=0.8)
    fig.suptitle(f"Dynamic Nelson Siegel factors (tau = {f.attrs.get('tau', 0):.2f} yrs)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_krd(res: AnalysisResult, path: Path):
    k = res.krd
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar(k.index, k.values, color=C1, width=0.6)
    for b, v in zip(bars, k.values):
        ax.annotate(f"${v/1000:,.1f}k", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8, color=INK2, xytext=(0, 2), textcoords="offset points")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:,.0f}k"))
    ax.set_ylabel("Loss per +1bp")
    ax.set_title(f"Key rate DV01 (total {_money(res.summary['DV01 ($)'])} per bp)")
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def chart_scenarios(res: AnalysisResult, path: Path):
    s = res.scenarios.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [GAIN if v >= 0 else LOSS for v in s["P&L ($)"]]
    ax.barh(s["Scenario"], s["P&L ($)"] / 1e6, color=colors, height=0.6)
    for y, v in enumerate(s["P&L (%)"]):
        x = s["P&L ($)"].iloc[y] / 1e6
        ax.annotate(f"{v:+.1f}%", (x, y), xytext=(4 if x >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if x >= 0 else "right", va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("P&L ($ millions)")
    ax.set_title("Stress test P&L (full revaluation)")
    ax.grid(axis="y", visible=False)
    lo, hi = ax.get_xlim()
    pad = (hi - lo) * 0.12
    ax.set_xlim(lo - pad, hi + pad)
    fig.savefig(path)
    plt.close(fig)


def chart_var(res: AnalysisResult, path: Path):
    pnl = res.var_hist.pnl / 1000
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(pnl, bins=50, color=C1, edgecolor=SURFACE, linewidth=0.8)
    for lv, ls in ((0.95, "--"), (0.99, ":")):
        v = -res.var_hist.var[lv] / 1000
        ax.axvline(v, color=LOSS, linestyle=ls, linewidth=1.5)
        ax.annotate(f"{int(lv*100)}% VaR\n{_money(-v*1000)}", (v, ax.get_ylim()[1] * 0.85), xytext=(-4, 0),
                    textcoords="offset points", ha="right", fontsize=8, color=INK)
    ax.set_xlabel("1 day P&L ($ thousands)")
    ax.set_ylabel("Days")
    ax.set_title(f"Historical simulation: {len(pnl)} daily scenarios")
    fig.savefig(path)
    plt.close(fig)


def chart_pca(res: AnalysisResult, path: Path):
    L = res.pca.loadings
    t = tenor_years(L.index)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for col, c in zip(L.columns, (C1, C2, C3)):
        i = list(L.columns).index(col)
        ax.plot(t, L[col], color=c, marker="o", markersize=5, label=f"{col} ({res.pca.explained[i]*100:.1f}% of variance)")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks(t)
    ax.set_xticklabels(L.index)
    ax.minorticks_off()
    ax.set_ylabel("bp per 1 sd daily move")
    ax.set_title("Principal components of daily curve changes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def chart_optim(res: AnalysisResult, path: Path):
    if res.optim is None:
        return False
    u = res.universe_analytics.set_index("id")
    cur = res.analytics.groupby("sector")["weight"].sum()
    w = res.optim.weights
    opt = w.groupby(u.loc[w.index, "sector"].values).sum()
    sectors = ["Government", "Provincial", "Corporate"]
    x = np.arange(len(sectors))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - 0.17, [cur.get(s, 0) * 100 for s in sectors], 0.32, color=C1, label="Current")
    ax.bar(x + 0.17, [opt.get(s, 0) * 100 for s in sectors], 0.32, color=C2, label="Optimized")
    ax.set_xticks(x)
    ax.set_xticklabels(sectors)
    ax.set_ylabel("Weight (%)")
    ax.set_title("Sector mix: current vs optimized")
    ax.legend()
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fixed Income Risk Report {{ date }}</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:#f4f4f2;color:#0b0b0b;margin:0}
main{max-width:1040px;margin:0 auto;padding:32px 24px 64px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:36px 0 10px;border-bottom:1px solid #e4e3df;padding-bottom:6px}
.sub{color:#52514e;font-size:14px;margin-bottom:20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.tile{background:#fcfcfb;border:1px solid #e4e3df;border-radius:10px;padding:14px 16px}
.tile .k{font-size:12px;color:#52514e} .tile .v{font-size:22px;font-weight:650;margin-top:4px}
.card{background:#fcfcfb;border:1px solid #e4e3df;border-radius:10px;padding:16px;margin:12px 0}
.card img{width:100%;height:auto}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px} @media(max-width:800px){.grid2{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid #eceae6;text-align:right} th{color:#52514e;font-weight:600}
td:first-child,th:first-child{text-align:left} table.scen td:nth-child(2),table.scen th:nth-child(2){text-align:left}
.scroll{overflow-x:auto}
ul.take li{margin:6px 0;line-height:1.45}
.note{font-size:12px;color:#52514e;line-height:1.5}
.badge{display:inline-block;background:#fff3d6;border:1px solid #f0d58a;border-radius:6px;padding:2px 8px;font-size:12px}
</style></head><body><main>
<h1>Fixed Income Portfolio Risk Report</h1>
<div class="sub">Valuation date {{ date }} &middot; Data: {{ source }}{% if synthetic %} <span class="badge">synthetic data</span>{% endif %}</div>

<div class="tiles">
{% for k, v in tiles %}<div class="tile"><div class="k">{{ k }}</div><div class="v">{{ v }}</div></div>{% endfor %}
</div>

<h2>Key takeaways</h2>
<div class="card"><ul class="take">{% for t in takeaways %}<li>{{ t }}</li>{% endfor %}</ul></div>

<h2>1. Yield curve</h2>
<div class="card"><img src="data:image/png;base64,{{ img.curve }}" alt="Yield curve"></div>
<div class="grid2">
<div class="card"><img src="data:image/png;base64,{{ img.curve_shift }}" alt="Curve history"></div>
<div class="card"><div class="scroll">{{ fit_table }}</div>
<p class="note">Nelson Siegel and Svensson are fitted to today's quotes. The zero curve used for pricing is bootstrapped
from bills (simple yields) and par bonds with {{ curve_method }} interpolation, so it reprices every quote exactly.</p></div>
</div>
<div class="card"><img src="data:image/png;base64,{{ img.factors }}" alt="Nelson Siegel factors"></div>

<h2>2. Portfolio positioning</h2>
<div class="card scroll">{{ holdings_table }}</div>
<div class="card scroll"><b>By sector</b>{{ sector_table }}</div>
<div class="card scroll"><b>By rating</b>{{ rating_table }}</div>
<div class="card"><img src="data:image/png;base64,{{ img.krd }}" alt="Key rate DV01"></div>

<h2>3. Stress tests</h2>
<div class="card"><img src="data:image/png;base64,{{ img.scenarios }}" alt="Scenarios"></div>
<div class="card scroll">{{ scenario_table }}</div>

<h2>4. Value at Risk</h2>
<div class="grid2">
<div class="card"><img src="data:image/png;base64,{{ img.var }}" alt="VaR"></div>
<div class="card scroll">{{ var_table }}
<p class="note">Historical simulation applies each of the last {{ lookback }} daily curve changes to today's curve,
rebuilds the zero curve and reprices every bond (no duration shortcut). Parametric VaR uses tenor level
exposures and the covariance of daily yield changes. Credit spread moves use US corporate OAS indices as a proxy,
scaled by sector (provincials 0.35x).</p></div>
</div>

<h2>5. What drives the curve: PCA</h2>
<div class="grid2">
<div class="card"><img src="data:image/png;base64,{{ img.pca }}" alt="PCA"></div>
<div class="card scroll">{{ pca_table }}
<p class="note">Three factors explain {{ pca_total }} of daily curve variance. The table shows portfolio P&L for a
one standard deviation daily move in each factor, computed by full revaluation.</p></div>
</div>

{% if optim_table %}
<h2>6. Portfolio construction: yield optimizer</h2>
<div class="card"><p class="note" style="margin-top:0">Linear program over the investable universe: maximize yield while holding
effective duration within &plusmn;0.10 years of the current portfolio, each curve bucket within &plusmn;0.30 years, credit risk (DTS)
no higher than today, at least 20% government, at most 35% corporate and 15% BBB, and no bond above 10%.</p>
<div class="grid2"><div class="scroll">{{ optim_table }}<br>{{ bucket_table }}</div>
<div><img src="data:image/png;base64,{{ img.optim }}" alt="Optimizer"></div></div>
<div class="scroll"><b>Optimized holdings</b>{{ optim_weights }}</div></div>
{% endif %}

<h2>Data quality</h2>
<div class="card scroll">{{ dq_table }}{% if large_moves %}<p class="note">Large daily moves flagged for review (kept in data):
{{ large_moves }}</p>{% endif %}</div>

<p class="note">Generated automatically by fixed-income-risk-analytics. Corporate issuers in the sample portfolio are illustrative.
For research and education only, not investment advice.</p>
</main></body></html>"""
)


def _fmt_table(df: pd.DataFrame, fmt: dict | None = None, index: bool = True, classes: str = "") -> str:
    fmt = fmt or {}
    out = df.copy()
    out.index.name = None
    for col, f in fmt.items():
        if col in out.columns:
            fn = f if callable(f) else (lambda v, f=f: f.format(v))
            out[col] = out[col].map(lambda v, fn=fn: fn(v) if pd.notna(v) else "")
    return out.to_html(index=index, border=0, escape=True, classes=classes)


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


def takeaways(res: AnalysisResult) -> list[str]:
    s = res.summary
    sc = res.scenarios.set_index("Scenario")
    worst = sc["P&L ($)"].idxmin()
    krd = res.krd
    top = krd.idxmax()
    long_share = krd[[c for c in krd.index if c in ("10Y", "20Y", "30Y")]].sum() / krd.sum()
    out = [
        f"A 1bp parallel rise in government yields costs about {_money(s['DV01 ($)'])} "
        f"(effective duration {s['Effective duration']:.2f} years on {_money(s['Market value'])}).",
        f"Rate risk is concentrated at the long end: {long_share:.0%} of DV01 sits at 10 years and beyond, "
        f"and the single largest key rate exposure is the {top} point.",
        f"The worst stress scenario is '{worst}', with a loss of {_money(abs(sc.loc[worst, 'P&L ($)']))} "
        f"({sc.loc[worst, 'P&L (%)']:.1f}%).",
        f"One day 99% VaR is {_money(res.var_hist.var[0.99])} by historical simulation versus "
        f"{_money(res.var_param[0.99])} parametric; expected shortfall at 99% is {_money(res.var_hist.es[0.99])}.",
    ]
    asym = sc.loc["Parallel -100bp", "P&L ($)"] + sc.loc["Parallel +100bp", "P&L ($)"]
    out.append(
        f"Positive convexity ({s['Convexity']:.0f}) means a 100bp rally gains {_money(asym)} more than a 100bp sell off loses."
    )
    if res.optim is not None:
        pick = (res.optim.summary["Yield (%)"].iloc[0] - res.optim_current["Yield (%)"]) * 100
        out.append(
            f"The optimizer finds {pick:.0f}bp of extra yield in the universe at the same duration, curve profile "
            f"and credit risk (DTS), using {int(res.optim.summary['Holdings'].iloc[0])} bonds."
        )
    return out


def render(res: AnalysisResult, out_dir: str | Path, curve_method: str = "pchip", var_lookback: int = 500) -> Path:
    out = Path(out_dir)
    charts = out / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    chart_curve(res, charts / "curve.png")
    chart_curve_shift(res, charts / "curve_shift.png")
    chart_factors(res, charts / "factors.png")
    chart_krd(res, charts / "krd.png")
    chart_scenarios(res, charts / "scenarios.png")
    chart_var(res, charts / "var.png")
    chart_pca(res, charts / "pca.png")
    has_optim = chart_optim(res, charts / "optim.png")
    img = {p.stem: _b64(p) for p in charts.glob("*.png")}

    s = res.summary
    tiles = [
        ("Market value", f"${s['Market value']/1e6:,.1f}M"),
        ("Yield to maturity", f"{s['Yield to maturity (%)']:.2f}%"),
        ("Effective duration", f"{s['Effective duration']:.2f} yrs"),
        ("DV01", _money(s["DV01 ($)"])),
        ("CS01", _money(s["CS01 ($)"])),
        ("1 day 99% VaR", _money(res.var_hist.var[0.99])),
    ]

    fit = pd.DataFrame(
        {
            "Model": ["Nelson Siegel", "Svensson"],
            "RMSE (bp)": [res.ns_fit.rmse_bp, res.sv_fit.rmse_bp],
            "Max error (bp)": [res.ns_fit.max_abs_err_bp, res.sv_fit.max_abs_err_bp],
            "Long run level b0 (%)": [res.ns_fit.params["b0"], res.sv_fit.params["b0"]],
        }
    )

    an = res.analytics
    hold = an[["id", "rating", "coupon", "maturity", "market_value", "weight", "clean_price", "ytm", "spread_bp",
               "eff_duration", "convexity", "dv01"]].rename(
        columns={"id": "Bond", "rating": "Rating", "coupon": "Coupon", "maturity": "Maturity", "market_value": "Market value",
                 "weight": "Weight", "clean_price": "Clean price", "ytm": "YTM", "spread_bp": "Spread (bp)",
                 "eff_duration": "Duration", "convexity": "Convexity", "dv01": "DV01 ($)"})
    hold_fmt = {"Coupon": "{:.2f}", "Market value": "${:,.0f}", "Weight": "{:.1%}", "Clean price": "{:.3f}", "YTM": "{:.2f}%",
                "Spread (bp)": "{:.0f}", "Duration": "{:.2f}", "Convexity": "{:.1f}", "DV01 ($)": "${:,.0f}"}
    bfmt = {"Weight": "{:.1%}", "Market value": _money, "DV01 ($)": _money, "CS01 ($)": _money,
            "Duration contribution": "{:.2f}", "Share of DV01": "{:.1%}"}

    scen = res.scenarios.copy()
    scen_fmt = {"P&L ($)": _money, "P&L (%)": "{:+.2f}%", "Rates P&L ($)": _money, "Spread P&L ($)": _money}

    var_tbl = pd.DataFrame(
        {
            "Method": ["Historical, rates + credit", "Historical, rates only", "Parametric (delta normal), rates"],
            "95% VaR": [res.var_hist.var[0.95], res.var_hist_rates.var[0.95], res.var_param[0.95]],
            "99% VaR": [res.var_hist.var[0.99], res.var_hist_rates.var[0.99], res.var_param[0.99]],
            "99% ES": [res.var_hist.es[0.99], res.var_hist_rates.es[0.99], np.nan],
        }
    )
    pca_tbl = pd.DataFrame(
        {"Variance explained": res.pca.explained, "Portfolio P&L, +1 sd ($)": res.pca_pnl.values}, index=res.pca.loadings.columns
    )

    ctx = dict(
        date=f"{res.val_date:%d %B %Y}",
        source=res.source,
        synthetic="Synthetic" in res.source,
        tiles=tiles,
        takeaways=takeaways(res),
        img=img,
        curve_method=curve_method.upper() if curve_method == "pchip" else curve_method.replace("_", " ").title(),
        fit_table=_fmt_table(fit, {"RMSE (bp)": "{:.2f}", "Max error (bp)": "{:.2f}", "Long run level b0 (%)": "{:.2f}"}, False),
        holdings_table=_fmt_table(hold, hold_fmt, False),
        sector_table=_fmt_table(res.sector, bfmt),
        rating_table=_fmt_table(res.rating, bfmt),
        scenario_table=_fmt_table(scen, scen_fmt, False, "scen"),
        var_table=_fmt_table(var_tbl, {c: "${:,.0f}" for c in ["95% VaR", "99% VaR", "99% ES"]}, False),
        lookback=len(res.var_hist.pnl),
        pca_table=_fmt_table(pca_tbl, {"Variance explained": "{:.1%}", "Portfolio P&L, +1 sd ($)": _money}),
        pca_total=f"{res.pca.explained.sum():.1%}",
        dq_table=_fmt_table(pd.DataFrame(res.quality.summary().items(), columns=["Check", "Result"]), index=False),
        large_moves=", ".join(f"{d} {c} {m}bp" for d, c, m in res.quality.large_moves[:15]),
        optim_table="",
    )
    if has_optim and res.optim is not None:
        comp = pd.DataFrame({"Current": pd.Series(res.optim_current), "Optimized": res.optim.summary.iloc[0]})
        fmts = {"Yield (%)": "{:.2f}%", "Effective duration": "{:.2f}", "Government weight": "{:.1%}",
                "Corporate weight": "{:.1%}", "BBB weight": "{:.1%}", "DTS (yrs x bp)": "{:.0f}", "Holdings": "{:.0f}"}
        comp_fmt = pd.DataFrame({c: [fmts[i].format(comp.at[i, c]) for i in comp.index] for c in comp.columns}, index=comp.index)
        ctx["optim_table"] = comp_fmt.to_html(border=0)
        ctx["bucket_table"] = _fmt_table(res.optim.bucket_check.rename_axis("Duration by bucket"), {"Optimized": "{:.2f}", "Target": "{:.2f}"})
        u = res.universe_analytics.set_index("id")
        w = res.optim.weights[res.optim.weights > 0].sort_values(ascending=False)
        ow = pd.DataFrame({"Weight": w, "Sector": u.loc[w.index, "sector"], "Rating": u.loc[w.index, "rating"],
                           "YTM": u.loc[w.index, "ytm"], "Duration": u.loc[w.index, "eff_duration"]})
        ctx["optim_weights"] = _fmt_table(ow, {"Weight": "{:.1%}", "YTM": "{:.2f}%", "Duration": "{:.2f}"})

    html_path = out / "risk_report.html"
    html_path.write_text(TEMPLATE.render(**ctx), encoding="utf-8")

    # Machine readable extracts for downstream dashboards
    an.to_csv(out / "holdings_analytics.csv", index=False)
    res.scenarios.to_csv(out / "scenarios.csv", index=False)
    res.var_hist.pnl.to_csv(out / "var_pnl_history.csv")
    summary = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in res.summary.items()}
    summary.update({"valuation_date": f"{res.val_date:%Y-%m-%d}", "source": res.source,
                    "var_99_hist": res.var_hist.var[0.99], "var_99_param": res.var_param[0.99]})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return html_path
