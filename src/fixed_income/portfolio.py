"""Portfolio level analytics: holdings, market values, duration, DV01, key rate profile."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .bonds import BP, Bond, bond_from_row
from .curves import KEY_TENORS, ZeroCurve


def key_label(k: float) -> str:
    return f"{int(round(k * 12))}M" if k < 1 else f"{int(k)}Y"


class Portfolio:
    def __init__(self, holdings: pd.DataFrame, name: str = "Portfolio"):
        required = {"id", "coupon", "maturity", "par_amount"}
        missing = required - set(holdings.columns)
        if missing:
            raise ValueError(f"Holdings missing columns: {missing}")
        self.holdings = holdings.reset_index(drop=True).copy()
        self.holdings["maturity"] = pd.to_datetime(self.holdings["maturity"])
        self.bonds: list[Bond] = [bond_from_row(r) for _, r in self.holdings.iterrows()]
        self.par = self.holdings["par_amount"].astype(float).values
        self.name = name

    @classmethod
    def from_csv(cls, path: str | Path, name: str | None = None) -> Portfolio:
        return cls(pd.read_csv(path), name or Path(path).stem.replace("_", " ").title())

    def live(self, val_date) -> Portfolio:
        """Drop bonds that have matured on or before the valuation date."""
        keep = self.holdings["maturity"] > pd.Timestamp(val_date)
        return Portfolio(self.holdings[keep], self.name)

    # ----------------------------------------------------------------------- #
    def market_value(self, curve: ZeroCurve, val_date, spread_shock_bp: np.ndarray | None = None) -> float:
        shock = np.zeros(len(self.bonds)) if spread_shock_bp is None else spread_shock_bp
        px = [b.dirty_price_from_curve(curve, val_date, b.spread_bp + s) for b, s in zip(self.bonds, shock)]
        return float(np.dot(px, self.par) / 100)

    def analytics(self, curve: ZeroCurve, val_date, keys=KEY_TENORS) -> pd.DataFrame:
        rows = []
        for b, par, (_, h) in zip(self.bonds, self.par, self.holdings.iterrows()):
            _, _, accrued = b.schedule(val_date)
            cr = b.curve_risk(curve, val_date, keys=keys)
            dirty = cr["dirty_price"]
            ytm = b.ytm(dirty, val_date)
            yr = b.yield_risk(ytm, val_date)
            mv = par * dirty / 100
            # Spread risk is measured against the government curve, so government bonds carry none
            sdur = 0.0 if b.sector == "Government" else cr["spread_duration"]
            row = {
                "id": b.id,
                "issuer": b.issuer,
                "sector": b.sector,
                "sub_sector": h.get("sub_sector", b.sector),
                "rating": b.rating,
                "coupon": b.coupon,
                "maturity": b.maturity.date(),
                "years": b.years_to_maturity(val_date),
                "par_amount": par,
                "clean_price": dirty - accrued,
                "accrued": accrued,
                "dirty_price": dirty,
                "market_value": mv,
                "ytm": ytm,
                "spread_bp": b.spread_bp,
                "mod_duration": yr["modified_duration"],
                "eff_duration": cr["effective_duration"],
                "convexity": cr["effective_convexity"],
                "spread_duration": sdur,
                "dv01": cr["effective_duration"] * mv * BP,
                "cs01": sdur * mv * BP,
            }
            for k, d in cr["krd"].items():
                row[f"krd_{key_label(k)}"] = d * mv * BP  # dollar KR01
            rows.append(row)
        df = pd.DataFrame(rows)
        df["weight"] = df["market_value"] / df["market_value"].sum()
        return df

    @staticmethod
    def summary(an: pd.DataFrame) -> dict:
        w = an["weight"]
        return {
            "Holdings": int(len(an)),
            "Market value": float(an["market_value"].sum()),
            "Par amount": float(an["par_amount"].sum()),
            "Yield to maturity (%)": float((w * an["ytm"]).sum()),
            "Average coupon (%)": float((w * an["coupon"]).sum()),
            "Average spread (bp)": float((w * an["spread_bp"]).sum()),
            "Average maturity (yrs)": float((w * an["years"]).sum()),
            "Effective duration": float((w * an["eff_duration"]).sum()),
            "Modified duration": float((w * an["mod_duration"]).sum()),
            "Convexity": float((w * an["convexity"]).sum()),
            "Spread duration": float((w * an["spread_duration"]).sum()),
            "DV01 ($)": float(an["dv01"].sum()),
            "CS01 ($)": float(an["cs01"].sum()),
        }

    @staticmethod
    def krd_profile(an: pd.DataFrame) -> pd.Series:
        cols = [c for c in an.columns if c.startswith("krd_")]
        s = an[cols].sum()
        s.index = [c.replace("krd_", "") for c in cols]
        return s

    @staticmethod
    def breakdown(an: pd.DataFrame, by: str) -> pd.DataFrame:
        g = an.groupby(by)
        out = pd.DataFrame(
            {
                "Weight": g["weight"].sum(),
                "Market value": g["market_value"].sum(),
                "DV01 ($)": g["dv01"].sum(),
                "CS01 ($)": g["cs01"].sum(),
            }
        )
        out["Duration contribution"] = g.apply(lambda x: (x["weight"] * x["eff_duration"]).sum(), include_groups=False)
        out["Share of DV01"] = out["DV01 ($)"] / out["DV01 ($)"].sum()
        if by == "rating":
            scale = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B"]
            rank = {r: i for i, r in enumerate(scale)}
            return out.loc[sorted(out.index, key=lambda r: rank.get(str(r).upper(), len(scale)))]
        return out.sort_values("Weight", ascending=False)
