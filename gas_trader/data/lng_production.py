"""LNG export and natural gas production data feed."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import requests
from loguru import logger

from gas_trader.config import Settings


class LNGProductionFeed:
    """Fetches LNG feed gas and dry gas production data from EIA."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.raw["data"]["eia"]["api_key"] or ""

    def fetch_lng_exports(self, months: int = 24) -> pd.DataFrame:
        """Fetch monthly LNG export volumes."""
        if not self.api_key:
            return self._synthetic_lng(months)

        try:
            url = self.settings.raw["data"]["lng"]["feed_gas_url"]
            params = {
                "api_key": self.api_key,
                "frequency": "monthly",
                "data[0]": "value",
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "length": months,
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("response", {}).get("data", [])
            if not rows:
                return self._synthetic_lng(months)

            df = pd.DataFrame(rows)
            df["period"] = pd.to_datetime(df["period"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.rename(columns={"value": "lng_bcf"})
            df = df.set_index("period").sort_index()
            return df[["lng_bcf"]]
        except Exception as e:
            logger.warning(f"LNG data fetch error: {e}")
            return self._synthetic_lng(months)

    def fetch_production(self, months: int = 24) -> pd.DataFrame:
        """Fetch monthly dry gas production."""
        if not self.api_key:
            return self._synthetic_production(months)

        try:
            url = self.settings.raw["data"]["lng"]["production_url"]
            params = {
                "api_key": self.api_key,
                "frequency": "monthly",
                "data[0]": "value",
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "length": months,
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("response", {}).get("data", [])
            if not rows:
                return self._synthetic_production(months)

            df = pd.DataFrame(rows)
            df["period"] = pd.to_datetime(df["period"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.rename(columns={"value": "production_bcf"})
            df = df.set_index("period").sort_index()
            return df[["production_bcf"]]
        except Exception as e:
            logger.warning(f"Production data fetch error: {e}")
            return self._synthetic_production(months)

    def get_supply_demand_balance(self) -> dict:
        """Compute simple supply/demand balance signal."""
        lng = self.fetch_lng_exports(months=6)
        prod = self.fetch_production(months=6)

        if lng.empty or prod.empty:
            return {"signal": 0.0, "lng_trend": 0.0, "prod_trend": 0.0}

        # LNG exports rising = bullish (more demand for US gas)
        lng_trend = lng["lng_bcf"].pct_change().mean() if len(lng) > 1 else 0.0
        # Production falling = bullish
        prod_trend = prod["production_bcf"].pct_change().mean() if len(prod) > 1 else 0.0

        # Signal: positive = bullish
        signal = lng_trend - prod_trend
        return {
            "signal": float(np.clip(signal * 10, -1, 1)),
            "lng_trend": float(lng_trend),
            "prod_trend": float(prod_trend),
        }

    @staticmethod
    def _synthetic_lng(months: int) -> pd.DataFrame:
        dates = pd.date_range(end=dt.datetime.now(), periods=months, freq="MS")
        # LNG exports trending up over time
        base = np.linspace(300, 400, months) + np.random.randn(months) * 20
        return pd.DataFrame({"lng_bcf": base}, index=dates)

    @staticmethod
    def _synthetic_production(months: int) -> pd.DataFrame:
        dates = pd.date_range(end=dt.datetime.now(), periods=months, freq="MS")
        base = np.linspace(3000, 3100, months) + np.random.randn(months) * 50
        return pd.DataFrame({"production_bcf": base}, index=dates)
