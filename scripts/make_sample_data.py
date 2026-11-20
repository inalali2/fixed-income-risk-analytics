"""Generate the synthetic offline dataset in data/sample/.

The synthetic curve follows the broad shape of the Canadian rate cycle from
2021 to 2026 (near zero policy rate, the 2022 hiking cycle, the 2024 to 2025
cutting cycle) with realistic daily volatility driven by level, slope and
curvature shocks. It is NOT real market data. It exists so the project runs
offline and in CI. Use `fi-risk update` to pull real Bank of Canada data.

    python scripts/make_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

OUT = Path(__file__).resolve().parents[1] / "data" / "sample"
rng = np.random.default_rng(42)

TENORS = ["1M", "2M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "30Y"]
T = np.array([1 / 12, 2 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 30])

# Monthly anchors (percent) for 3M, 2Y, 10Y, 30Y: a stylised path of the cycle
anchors = pd.DataFrame(
    [
        ("2021-01-01", 0.10, 0.18, 0.75, 1.30),
        ("2021-10-01", 0.15, 0.80, 1.55, 1.95),
        ("2022-03-01", 0.70, 1.90, 2.20, 2.40),
        ("2022-07-01", 2.60, 3.30, 3.10, 3.10),
        ("2022-12-01", 4.10, 3.95, 3.25, 3.25),
        ("2023-06-01", 4.75, 4.50, 3.40, 3.30),
        ("2023-10-01", 5.05, 4.85, 4.15, 3.95),
        ("2024-01-01", 5.00, 4.10, 3.25, 3.25),
        ("2024-07-01", 4.60, 3.80, 3.35, 3.35),
        ("2025-01-01", 3.10, 2.90, 3.25, 3.40),
        ("2025-07-01", 2.65, 2.70, 3.35, 3.65),
        ("2026-01-01", 2.20, 2.55, 3.25, 3.70),
        ("2026-07-01", 2.45, 2.85, 3.40, 3.75),
        ("2026-11-13", 2.35, 2.70, 3.30, 3.70),
    ],
    columns=["date", "3M", "2Y", "10Y", "30Y"],
).set_index("date")
anchors.index = pd.to_datetime(anchors.index)

dates = pd.bdate_range("2021-01-04", "2026-11-13")
anchor_daily = anchors.reindex(anchors.index.union(dates)).interpolate("time").reindex(dates)

# Interpolate anchors across the full tenor grid
anchor_t = np.array([0.25, 2, 10, 30])
base = np.vstack([PchipInterpolator(anchor_t, row, extrapolate=True)(T) for row in anchor_daily.values])
base[:, :3] = np.maximum(base[:, :3], anchor_daily[["3M"]].values - np.array([0.03, 0.02, 0.0]))

# Factor loadings: level, slope (short end moves more), curvature (belly)
level = np.ones_like(T)
slope = np.exp(-T / 2.0)
curv = (T / 2.5) * np.exp(1 - T / 2.5)
short_damp = np.clip(T / 0.5, 0.35, 1.0)  # bills move less day to day

n = len(dates)
dev = np.zeros((n, 3))
phi = np.array([0.985, 0.98, 0.97])
sig = np.array([0.045, 0.030, 0.018])  # percent per day
for i in range(1, n):
    dev[i] = phi * dev[i - 1] + rng.standard_t(5, 3) * sig * np.sqrt(3 / 5)
noise = rng.normal(0, 0.006, size=(n, len(T)))

curve = base + (np.outer(dev[:, 0], level) + np.outer(dev[:, 1], slope) + np.outer(dev[:, 2], curv)) * short_damp + noise
curve = np.maximum(curve, 0.05)
curve_df = pd.DataFrame(np.round(curve, 3), index=dates, columns=TENORS)
curve_df.index.name = "date"

# Credit spreads (percent): mean reverting around regime anchors, widen in 2022
sp_anchor = pd.DataFrame(
    [("2021-01-01", 0.95, 1.25, 3.80), ("2022-06-15", 1.55, 1.95, 5.60), ("2022-10-15", 1.60, 2.00, 5.20),
     ("2023-03-15", 1.45, 1.80, 5.00), ("2024-06-01", 0.95, 1.20, 3.20), ("2025-04-08", 1.15, 1.45, 4.40),
     ("2025-09-01", 0.85, 1.10, 2.90), ("2026-11-13", 0.90, 1.15, 3.05)],
    columns=["date", "IG_OAS", "BBB_OAS", "HY_OAS"],
).set_index("date")
sp_anchor.index = pd.to_datetime(sp_anchor.index)
sp_base = sp_anchor.reindex(sp_anchor.index.union(dates)).interpolate("time").reindex(dates).values
sp_dev = np.zeros((n, 1))
for i in range(1, n):
    sp_dev[i] = 0.97 * sp_dev[i - 1] + rng.normal(0, 0.012)
beta = np.array([1.0, 1.25, 4.0])
credit = sp_base + sp_dev * beta + rng.normal(0, [0.004, 0.005, 0.02], size=(n, 3))
credit_df = pd.DataFrame(np.round(np.maximum(credit, 0.3), 3), index=dates, columns=["IG_OAS", "BBB_OAS", "HY_OAS"])
credit_df.index.name = "date"

OUT.mkdir(parents=True, exist_ok=True)
curve_df.to_csv(OUT / "goc_yields_synthetic.csv")
credit_df.to_csv(OUT / "credit_spreads_synthetic.csv")
print(f"Wrote {len(curve_df)} curve rows and {len(credit_df)} credit rows to {OUT}")
print("Daily change std (bp):", (curve_df.diff().std() * 100).round(1).to_dict())
print(curve_df.tail(3))
