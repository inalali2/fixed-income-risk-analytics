"""Data collection, cleaning and storage.

Sources
-------
* Bank of Canada Valet API: Government of Canada treasury bill and benchmark
  bond yields (free, no key needed).
* FRED (St. Louis Fed): US Treasury curve and ICE BofA credit spread indices,
  used as a credit spread proxy because Canadian corporate spread indices are
  not freely published.
* Bundled synthetic sample: lets the whole project run offline and in CI.

All yields are stored in percent, as published.
"""
from __future__ import annotations

import io
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

# Tenor label -> (years, Bank of Canada Valet series id)
BOC_SERIES: dict[str, tuple[float, str]] = {
    "1M": (1 / 12, "TB.CDN.30D.MID"),
    "2M": (2 / 12, "TB.CDN.60D.MID"),
    "3M": (3 / 12, "TB.CDN.90D.MID"),
    "6M": (6 / 12, "TB.CDN.180D.MID"),
    "1Y": (1.0, "TB.CDN.1Y.MID"),
    "2Y": (2.0, "BD.CDN.2YR.DQ.YLD"),
    "3Y": (3.0, "BD.CDN.3YR.DQ.YLD"),
    "5Y": (5.0, "BD.CDN.5YR.DQ.YLD"),
    "7Y": (7.0, "BD.CDN.7YR.DQ.YLD"),
    "10Y": (10.0, "BD.CDN.10YR.DQ.YLD"),
    "30Y": (30.0, "BD.CDN.LONG.DQ.YLD"),
}

FRED_UST_SERIES: dict[str, tuple[float, str]] = {
    "1M": (1 / 12, "DGS1MO"),
    "3M": (0.25, "DGS3MO"),
    "6M": (0.5, "DGS6MO"),
    "1Y": (1.0, "DGS1"),
    "2Y": (2.0, "DGS2"),
    "3Y": (3.0, "DGS3"),
    "5Y": (5.0, "DGS5"),
    "7Y": (7.0, "DGS7"),
    "10Y": (10.0, "DGS10"),
    "20Y": (20.0, "DGS20"),
    "30Y": (30.0, "DGS30"),
}

# Option adjusted spreads, percent
FRED_CREDIT_SERIES: dict[str, str] = {
    "IG_OAS": "BAMLC0A0CM",     # ICE BofA US Corporate Index OAS
    "BBB_OAS": "BAMLC0A4CBBB",  # ICE BofA BBB US Corporate Index OAS
    "HY_OAS": "BAMLH0A0HYM2",   # ICE BofA US High Yield Index OAS
}

BOC_URL = "https://www.bankofcanada.ca/valet/observations/{series}/json"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def tenor_years(labels: list[str] | pd.Index) -> np.ndarray:
    """Convert tenor labels such as '3M' or '10Y' to year fractions."""
    out = []
    for lab in labels:
        lab = str(lab).upper()
        if lab.endswith("M"):
            out.append(float(lab[:-1]) / 12)
        elif lab.endswith("Y"):
            out.append(float(lab[:-1]))
        else:
            raise ValueError(f"Unrecognised tenor label: {lab}")
    return np.array(out)


# --------------------------------------------------------------------------- #
# Fetchers
# --------------------------------------------------------------------------- #
def _session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "fixed-income-risk-analytics/1.0"
    return s


def fetch_boc_yields(start: str = "2018-01-01", end: str | None = None) -> pd.DataFrame:
    """Download Government of Canada yields from the Bank of Canada Valet API.

    Returns a wide DataFrame indexed by date with one column per tenor label.
    Series are requested one at a time so a single missing series does not
    break the whole download.
    """
    sess = _session()
    frames = []
    for tenor, (_, series) in BOC_SERIES.items():
        params = {"start_date": start}
        if end:
            params["end_date"] = end
        try:
            r = sess.get(BOC_URL.format(series=series), params=params, timeout=30)
            r.raise_for_status()
            obs = r.json().get("observations", [])
        except Exception as exc:  # network or schema issue: skip, log, continue
            log.warning("Bank of Canada series %s (%s) failed: %s", series, tenor, exc)
            continue
        rows = [
            (o["d"], o[series]["v"])
            for o in obs
            if series in o and o[series].get("v") not in (None, "")
        ]
        if not rows:
            log.warning("No observations for %s", series)
            continue
        s = pd.Series(
            pd.to_numeric([v for _, v in rows], errors="coerce"),
            index=pd.to_datetime([d for d, _ in rows]),
            name=tenor,
        )
        frames.append(s)
    if not frames:
        raise RuntimeError("Could not download any Bank of Canada series.")
    df = pd.concat(frames, axis=1).sort_index()
    df.index.name = "date"
    return df


def fetch_fred(series_ids: dict[str, str], start: str = "2018-01-01") -> pd.DataFrame:
    """Download FRED series as CSV (no API key needed). Columns renamed to the dict keys."""
    sess = _session()
    frames = []
    for name, sid in series_ids.items():
        try:
            r = sess.get(FRED_URL, params={"id": sid, "cosd": start}, timeout=30)
            r.raise_for_status()
            raw = pd.read_csv(io.StringIO(r.text))
        except Exception as exc:
            log.warning("FRED series %s failed: %s", sid, exc)
            continue
        date_col = raw.columns[0]  # 'observation_date' (current) or 'DATE' (older)
        s = pd.Series(
            pd.to_numeric(raw[sid], errors="coerce").values,
            index=pd.to_datetime(raw[date_col]),
            name=name,
        )
        frames.append(s)
    if not frames:
        raise RuntimeError("Could not download any FRED series.")
    df = pd.concat(frames, axis=1).sort_index()
    df.index.name = "date"
    return df


def fetch_ust_yields(start: str = "2018-01-01") -> pd.DataFrame:
    return fetch_fred({k: v for k, (_, v) in FRED_UST_SERIES.items()}, start)


def fetch_credit_spreads(start: str = "2018-01-01") -> pd.DataFrame:
    return fetch_fred(FRED_CREDIT_SERIES, start)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
@dataclass
class QualityReport:
    rows_in: int = 0
    rows_out: int = 0
    duplicate_dates: int = 0
    weekend_rows: int = 0
    missing_filled: int = 0
    rows_dropped_incomplete: int = 0
    large_moves: list[tuple[str, str, float]] = field(default_factory=list)
    inverted_points: int = 0

    def summary(self) -> dict:
        return {
            "Rows received": self.rows_in,
            "Rows after cleaning": self.rows_out,
            "Duplicate dates removed": self.duplicate_dates,
            "Weekend rows removed": self.weekend_rows,
            "Gaps forward filled (max 3 days)": self.missing_filled,
            "Incomplete rows dropped": self.rows_dropped_incomplete,
            "Daily moves over threshold flagged": len(self.large_moves),
        }


def clean_yields(
    df: pd.DataFrame,
    max_ffill: int = 3,
    move_threshold_bp: float = 40.0,
    min_coverage: float = 0.8,
) -> tuple[pd.DataFrame, QualityReport]:
    """Clean a wide yield panel and return the cleaned panel plus a quality report.

    Steps: sort, de-duplicate, drop weekends, forward fill short gaps (holidays
    that differ between series), drop rows still missing too many tenors, and
    flag unusually large daily moves for review instead of silently deleting
    them.
    """
    qr = QualityReport(rows_in=len(df))
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    dup = out.index.duplicated(keep="last")
    qr.duplicate_dates = int(dup.sum())
    out = out[~dup]

    wk = out.index.dayofweek >= 5
    qr.weekend_rows = int(wk.sum())
    out = out[~wk]

    out = out.apply(pd.to_numeric, errors="coerce")
    before = int(out.isna().sum().sum())
    out = out.ffill(limit=max_ffill)
    qr.missing_filled = before - int(out.isna().sum().sum())

    keep = out.notna().mean(axis=1) >= min_coverage
    qr.rows_dropped_incomplete = int((~keep).sum())
    out = out[keep]

    moves = out.diff().abs() * 100
    for col in out.columns:
        for dt, mv in moves[col][moves[col] > move_threshold_bp].items():
            qr.large_moves.append((dt.strftime("%Y-%m-%d"), col, round(float(mv), 1)))

    qr.rows_out = len(out)
    return out, qr


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
class YieldStore:
    """Small SQLite store in long format so updates are incremental and queryable with SQL."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS observations (
        source  TEXT NOT NULL,
        date    TEXT NOT NULL,
        field   TEXT NOT NULL,
        value   REAL,
        PRIMARY KEY (source, date, field)
    );
    CREATE INDEX IF NOT EXISTS ix_obs_source_date ON observations (source, date);
    """

    def __init__(self, path: str | Path = PROJECT_ROOT / "data" / "market.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(self.SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path)

    def upsert(self, source: str, wide: pd.DataFrame) -> int:
        long = (
            wide.rename_axis("date")
            .reset_index()
            .melt(id_vars="date", var_name="field", value_name="value")
            .dropna(subset=["value"])
        )
        long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
        rows = [(source, d, f, float(v)) for d, f, v in long[["date", "field", "value"]].itertuples(index=False)]
        with self._conn() as c:
            c.executemany(
                "INSERT INTO observations (source, date, field, value) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(source, date, field) DO UPDATE SET value = excluded.value",
                rows,
            )
        return len(rows)

    def last_date(self, source: str) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT MAX(date) FROM observations WHERE source = ?", (source,)).fetchone()
        return row[0] if row and row[0] else None

    def load(self, source: str, start: str | None = None) -> pd.DataFrame:
        sql = "SELECT date, field, value FROM observations WHERE source = ?"
        params: list = [source]
        if start:
            sql += " AND date >= ?"
            params.append(start)
        with self._conn() as c:
            long = pd.read_sql_query(sql, c, params=params)
        if long.empty:
            return pd.DataFrame()
        wide = long.pivot(index="date", columns="field", values="value")
        wide.index = pd.to_datetime(wide.index)
        return wide.sort_index()


def update_store(store: YieldStore, source: str = "boc", start: str = "2018-01-01") -> dict:
    """Incremental refresh: only download dates after the last stored date."""
    last = store.last_date(source)
    fetch_start = (pd.Timestamp(last) - pd.Timedelta(days=7)).strftime("%Y-%m-%d") if last else start
    if source == "boc":
        raw = fetch_boc_yields(fetch_start)
    elif source == "ust":
        raw = fetch_ust_yields(fetch_start)
    elif source == "credit":
        raw = fetch_credit_spreads(fetch_start)
    else:
        raise ValueError(source)
    n = store.upsert(source, raw)
    return {"source": source, "fetched_from": fetch_start, "rows_upserted": n, "last_date": store.last_date(source)}


# --------------------------------------------------------------------------- #
# Offline sample
# --------------------------------------------------------------------------- #
def load_sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the bundled synthetic curve and credit spread history."""
    curve = pd.read_csv(SAMPLE_DIR / "goc_yields_synthetic.csv", index_col="date", parse_dates=True)
    credit = pd.read_csv(SAMPLE_DIR / "credit_spreads_synthetic.csv", index_col="date", parse_dates=True)
    return curve, credit


def order_tenors(df: pd.DataFrame) -> pd.DataFrame:
    cols = sorted(df.columns, key=lambda c: tenor_years([c])[0])
    return df[cols]
