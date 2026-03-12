"""EIA Weekly Natural Gas Storage Report feed."""

from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from gas_trader.config import Settings


class EIAStorageFeed:
    """Fetches and parses EIA weekly natural gas storage data."""

    BASE_URL = "https://api.eia.gov/v2/natural-gas/stor/wkly/"

    def __init__(self, settings: Settings):
        self.settings = settings
        eia_cfg = settings.raw["data"]["eia"]
        self.api_key = eia_cfg["api_key"] or ""
        self.report_day = eia_cfg["report_day"]
        self.report_time = eia_cfg["report_time"]
        self._history: Optional[pd.DataFrame] = None

    def fetch_storage_history(self, periods: int = 104) -> pd.DataFrame:
        """Fetch weekly storage data (last N weeks)."""
        if self._history is not None and len(self._history) >= periods:
            return self._history.tail(periods)

        if not self.api_key:
            logger.warning("No EIA API key — using synthetic storage data")
            return self._synthetic_storage(periods)

        try:
            params = {
                "api_key": self.api_key,
                "frequency": "weekly",
                "data[0]": "value",
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "length": periods,
            }
            resp = requests.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            rows = data.get("response", {}).get("data", [])
            if not rows:
                logger.warning("EIA returned empty data")
                return self._synthetic_storage(periods)

            df = pd.DataFrame(rows)
            df["period"] = pd.to_datetime(df["period"])
            df = df.rename(columns={"value": "storage_bcf"})
            df["storage_bcf"] = pd.to_numeric(df["storage_bcf"], errors="coerce")
            df = df.set_index("period").sort_index()

            # Compute weekly change
            df["change_bcf"] = df["storage_bcf"].diff()
            self._history = df
            logger.info(f"Fetched {len(df)} weeks of EIA storage data")
            return df

        except Exception as e:
            logger.error(f"EIA fetch error: {e}")
            return self._synthetic_storage(periods)

    def get_latest_report(self) -> dict:
        """Returns the latest storage report as a dict."""
        df = self.fetch_storage_history(periods=10)
        if df.empty:
            return {}
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest
        return {
            "date": str(df.index[-1].date()),
            "storage_bcf": float(latest["storage_bcf"]),
            "change_bcf": float(latest.get("change_bcf", 0)),
            "prev_storage_bcf": float(prev["storage_bcf"]),
        }

    def compute_surprise(self, actual_change: float, consensus: float) -> float:
        """Compute surprise = actual - consensus (in BCF).
        Positive surprise → bearish (more injected than expected).
        Negative surprise → bullish (less injected / more withdrawn).
        """
        return actual_change - consensus

    def is_report_imminent(self, now: Optional[dt.datetime] = None) -> bool:
        """Check if EIA report is about to be released."""
        if now is None:
            now = dt.datetime.now()
        # Thursday 10:30 ET
        if now.weekday() != 3:  # Thursday
            return False
        report_hour, report_min = 10, 30
        minutes_until = (report_hour * 60 + report_min) - (now.hour * 60 + now.minute)
        pause_before = self.settings.risk.eia_pause_before
        return 0 <= minutes_until <= pause_before

    def is_report_just_released(self, now: Optional[dt.datetime] = None) -> bool:
        """Check if EIA report was just released (within pause window)."""
        if now is None:
            now = dt.datetime.now()
        if now.weekday() != 3:
            return False
        report_hour, report_min = 10, 30
        minutes_since = (now.hour * 60 + now.minute) - (report_hour * 60 + report_min)
        pause_after = self.settings.risk.eia_pause_after
        return 0 <= minutes_since <= pause_after

    @staticmethod
    def _synthetic_storage(periods: int) -> pd.DataFrame:
        """Generate synthetic storage data for testing."""
        import numpy as np

        dates = pd.date_range(end=dt.datetime.now(), periods=periods, freq="W-THU")
        # Seasonal pattern: inject Apr-Oct, withdraw Nov-Mar
        storage = 2000.0  # BCF baseline
        values = []
        for d in dates:
            month = d.month
            if 4 <= month <= 10:
                change = np.random.uniform(50, 120)  # injection
            else:
                change = np.random.uniform(-180, -50)  # withdrawal
            storage += change
            storage = max(500, min(4500, storage))
            values.append(storage)

        df = pd.DataFrame({"storage_bcf": values}, index=dates)
        df.index.name = "period"
        df["change_bcf"] = df["storage_bcf"].diff()
        return df
