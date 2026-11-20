"""Interactive risk dashboard.

    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fixed_income import risk  # noqa: E402
from fixed_income.curves import ZeroCurve  # noqa: E402
from fixed_income.data import clean_yields, order_tenors, tenor_years  # noqa: E402
from fixed_income.pipeline import load_market  # noqa: E402
from fixed_income.portfolio import Portfolio  # noqa: E402

st.set_page_config(page_title="Fixed Income Risk", layout="wide")
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"


@st.cache_data(show_spinner="Loading market data")
def market(source: str):
    raw, credit, desc = load_market(source, refresh=source != "sample")
    hist, qr = clean_yields(order_tenors(raw))
    if credit is not None:
        credit = credit.ffill().reindex(hist.index).ffill()
    return hist, credit, desc


@st.cache_data(show_spinner="Pricing portfolio")
def analytics(source: str, val: pd.Timestamp, holdings: pd.DataFrame):
    hist, _, _ = market(source)
    curve = ZeroCurve.from_market(tenor_years(hist.columns), hist.loc[val].values)
    port = Portfolio(holdings).live(val)
    return port.analytics(curve, val)


@st.cache_data(show_spinner="Running historical VaR")
def var(source: str, val: pd.Timestamp, holdings: pd.DataFrame, lookback: int):
    hist, credit, _ = market(source)
    port = Portfolio(holdings).live(val)
    return risk.historical_var(port, hist, val, credit, lookback=lookback)


# --------------------------------------------------------------------------- #
st.sidebar.title("Settings")
source = st.sidebar.selectbox(
    "Data source", ["sample", "boc", "ust"],
    format_func={"sample": "Synthetic sample (offline)", "boc": "Bank of Canada (live)", "ust": "US Treasury via FRED (live)"}.get,
)
upload = st.sidebar.file_uploader("Holdings CSV (optional)", type="csv")
holdings = pd.read_csv(upload) if upload else pd.read_csv(ROOT / "config" / "portfolio.csv")

try:
    hist, credit, desc = market(source)
except Exception as exc:  # live source unreachable
    st.error(f"Could not load {source}: {exc}. Try the synthetic sample.")
    st.stop()

dates = hist.index
val = pd.Timestamp(st.sidebar.select_slider("Valuation date", options=list(dates.date), value=dates[-1].date()))
val = dates[dates.get_indexer([val], method="pad")[0]]
st.title("Fixed Income Portfolio Risk")
st.caption(f"{desc} · valuation date {val:%d %b %Y}")

an = analytics(source, val, holdings)
s = Portfolio.summary(an)
cols = st.columns(6)
for c, (k, v) in zip(cols, [
    ("Market value", f"${s['Market value']/1e6:,.1f}M"),
    ("Yield", f"{s['Yield to maturity (%)']:.2f}%"),
    ("Duration", f"{s['Effective duration']:.2f}"),
    ("Convexity", f"{s['Convexity']:.0f}"),
    ("DV01", f"${s['DV01 ($)']:,.0f}"),
    ("CS01", f"${s['CS01 ($)']:,.0f}"),
]):
    c.metric(k, v)

tab_curve, tab_port, tab_whatif, tab_var = st.tabs(["Curve", "Positions", "What if", "VaR"])

with tab_curve:
    t = tenor_years(hist.columns)
    lag = st.radio("Compare with", ["1 week", "1 month", "3 months", "1 year"], horizontal=True, index=1)
    off = {"1 week": 5, "1 month": 21, "3 months": 63, "1 year": 252}[lag]
    i = dates.get_loc(val)
    prev = dates[max(0, i - off)]
    df = pd.concat([
        pd.DataFrame({"Tenor": hist.columns, "Years": t, "Yield": hist.loc[val].values, "Curve": f"{val:%d %b %Y}"}),
        pd.DataFrame({"Tenor": hist.columns, "Years": t, "Yield": hist.loc[prev].values, "Curve": f"{prev:%d %b %Y}"}),
    ])
    chart = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2).encode(
        x=alt.X("Years:Q", scale=alt.Scale(type="log"), title="Maturity (years, log scale)"),
        y=alt.Y("Yield:Q", scale=alt.Scale(zero=False), title="Yield (%)"),
        color=alt.Color("Curve:N", scale=alt.Scale(range=[BLUE, ORANGE]), legend=alt.Legend(orient="top")),
        tooltip=["Curve", "Tenor", alt.Tooltip("Yield:Q", format=".3f")],
    ).properties(height=380)
    st.altair_chart(chart, width="stretch")
    chg = ((hist.loc[val] - hist.loc[prev]) * 100).rename("Change (bp)").to_frame().T
    st.dataframe(chg.style.format("{:+.1f}"), width="stretch")

with tab_port:
    show = an[["id", "sector", "rating", "coupon", "maturity", "market_value", "weight", "ytm", "spread_bp",
               "eff_duration", "dv01"]]
    st.dataframe(show.style.format({"coupon": "{:.2f}", "market_value": "${:,.0f}", "weight": "{:.1%}", "ytm": "{:.2f}%",
                                    "spread_bp": "{:.0f}", "eff_duration": "{:.2f}", "dv01": "${:,.0f}"}),
                 width="stretch", hide_index=True)
    krd = Portfolio.krd_profile(an).rename("DV01").reset_index().rename(columns={"index": "Key rate"})
    st.altair_chart(
        alt.Chart(krd).mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Key rate:N", sort=None), y=alt.Y("DV01:Q", title="$ loss per +1bp"),
            tooltip=["Key rate", alt.Tooltip("DV01:Q", format="$,.0f")],
        ).properties(height=300, title="Key rate DV01"),
        width="stretch",
    )

with tab_whatif:
    st.write("Shift the curve at each key tenor (bp, linearly interpolated) and shock spreads by sector.")
    c1, c2 = st.columns(2)
    with c1:
        pts = {k: st.slider(f"{lab} shift (bp)", -200, 200, 0, 5) for k, lab in [(0.25, "3M"), (2, "2Y"), (5, "5Y"), (10, "10Y"), (30, "30Y")]}
    with c2:
        sp = {k: st.slider(f"{k} spread (bp)", -100, 300, 0, 5) for k in ["Provincial", "Corporate A", "Corporate BBB"]}
    sc = risk.Scenario("Custom", pts, sp)
    curve = ZeroCurve.from_market(t, hist.loc[val].values)
    port = Portfolio(holdings).live(val)
    out = risk.run_scenarios(port, curve, val, [sc] + risk.STANDARD_SCENARIOS)
    r = out.iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Total P&L", f"${r['P&L ($)']:,.0f}", f"{r['P&L (%)']:+.2f}%")
    m2.metric("From rates", f"${r['Rates P&L ($)']:,.0f}")
    m3.metric("From spreads", f"${r['Spread P&L ($)']:,.0f}")
    out["Sign"] = np.where(out["P&L ($)"] >= 0, "Gain", "Loss")
    st.altair_chart(
        alt.Chart(out).mark_bar().encode(
            y=alt.Y("Scenario:N", sort=None, title=None), x=alt.X("P&L ($):Q", title="P&L ($)"),
            color=alt.Color("Sign:N", scale=alt.Scale(domain=["Gain", "Loss"], range=[BLUE, RED]), legend=None),
            tooltip=["Scenario", alt.Tooltip("P&L ($):Q", format="$,.0f"), alt.Tooltip("P&L (%):Q", format="+.2f")],
        ).properties(height=420),
        width="stretch",
    )

with tab_var:
    lookback = st.slider("Lookback (days)", 100, min(1000, len(hist) - 2), 500, 50)
    v = var(source, val, holdings, lookback)
    a, b, c = st.columns(3)
    a.metric("95% 1 day VaR", f"${v.var[0.95]:,.0f}")
    b.metric("99% 1 day VaR", f"${v.var[0.99]:,.0f}")
    c.metric("99% expected shortfall", f"${v.es[0.99]:,.0f}")
    pnl = v.pnl.rename("P&L").reset_index()
    st.altair_chart(
        alt.Chart(pnl).mark_bar(color=BLUE).encode(
            x=alt.X("P&L:Q", bin=alt.Bin(maxbins=50), title="1 day P&L ($)"), y=alt.Y("count()", title="Days"),
            tooltip=[alt.Tooltip("count()", title="Days")],
        ).properties(height=320),
        width="stretch",
    )
    st.caption("Historical simulation with full revaluation: each past daily curve and spread move is applied to "
               "today's curve and every bond is repriced.")
