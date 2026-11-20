"""Network free tests of the API parsers, using responses shaped like the real ones."""
import pandas as pd

from fixed_income import data


class FakeResponse:
    def __init__(self, payload=None, text=""):
        self._payload, self.text = payload, text

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, handler):
        self.handler = handler

    def get(self, url, params=None, timeout=None):
        return self.handler(url, params or {})


def test_boc_parser(monkeypatch):
    def handler(url, params):
        series = url.split("/observations/")[1].split("/")[0]
        if series == "BD.CDN.7YR.DQ.YLD":
            raise ConnectionError("simulated outage")  # one series failing must not break the rest
        obs = [{"d": "2026-07-23", series: {"v": "2.91"}}, {"d": "2026-07-24", series: {"v": "2.90"}},
               {"d": "2026-07-25", series: {"v": ""}}]
        return FakeResponse({"observations": obs})

    monkeypatch.setattr(data, "_session", lambda: FakeSession(handler))
    df = data.fetch_boc_yields("2026-07-01")
    assert "7Y" not in df.columns and "10Y" in df.columns
    assert df.loc["2026-07-24", "2Y"] == 2.90
    assert len(df) == 2  # blank value dropped


def test_fred_parser_handles_missing_dots(monkeypatch):
    csv = "observation_date,BAMLC0A0CM\n2026-07-23,0.88\n2026-07-24,.\n2026-07-27,0.90\n"
    monkeypatch.setattr(data, "_session", lambda: FakeSession(lambda u, p: FakeResponse(text=csv)))
    df = data.fetch_fred({"IG_OAS": "BAMLC0A0CM"})
    assert pd.isna(df.loc["2026-07-24", "IG_OAS"])
    assert df.loc["2026-07-27", "IG_OAS"] == 0.90


def test_tenor_years():
    assert list(data.tenor_years(["3M", "1Y", "30Y"])) == [0.25, 1.0, 30.0]
