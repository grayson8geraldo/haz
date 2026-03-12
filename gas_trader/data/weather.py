"""Weather / HDD (Heating Degree Days) feed via Open-Meteo / NOAA."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from loguru import logger

from gas_trader.config import Settings


@dataclass
class HDDData:
    date: dt.date
    hdd_actual: float
    hdd_normal: float
    hdd_deviation_pct: float
    forecast_14d_hdd: float


class WeatherFeed:
    """Fetches temperature data and computes HDD for key US gas-demand cities."""

    OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_HIST_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, settings: Settings):
        self.settings = settings
        weather_cfg = settings.raw["data"]["weather"]
        self.cities = weather_cfg["cities"]
        self.hdd_base = weather_cfg["hdd_base_temp"]  # 65°F
        self.forecast_days = weather_cfg["forecast_days"]

    def fetch_current_hdd(self) -> HDDData:
        """Fetch current HDD across tracked cities and compare to normals."""
        try:
            hdd_today = self._fetch_hdd_from_api()
        except Exception as e:
            logger.warning(f"Weather API error: {e}. Using synthetic data.")
            hdd_today = self._synthetic_hdd_today()

        # Historical normal (simplified — use 30-year avg in production)
        month = dt.date.today().month
        hdd_normal = self._seasonal_normal_hdd(month)
        deviation = ((hdd_today - hdd_normal) / max(hdd_normal, 1)) * 100

        forecast = self._fetch_forecast_hdd()

        return HDDData(
            date=dt.date.today(),
            hdd_actual=hdd_today,
            hdd_normal=hdd_normal,
            hdd_deviation_pct=deviation,
            forecast_14d_hdd=forecast,
        )

    def _fetch_hdd_from_api(self) -> float:
        """Fetch today's temperature from Open-Meteo and compute population-weighted HDD."""
        total_hdd = 0.0
        # Population weights for gas demand
        weights = {"New York": 0.30, "Chicago": 0.25, "Boston": 0.20, "Houston": 0.25}

        for city in self.cities:
            params = {
                "latitude": city["lat"],
                "longitude": city["lon"],
                "daily": "temperature_2m_mean",
                "temperature_unit": "fahrenheit",
                "timezone": "America/New_York",
                "past_days": 1,
                "forecast_days": 1,
            }
            resp = requests.get(self.OPEN_METEO_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            temps = data.get("daily", {}).get("temperature_2m_mean", [])
            if temps:
                avg_temp = temps[-1]
                hdd = max(0, self.hdd_base - avg_temp)
                weight = weights.get(city["name"], 0.25)
                total_hdd += hdd * weight

        return round(total_hdd, 1)

    def _fetch_forecast_hdd(self) -> float:
        """Fetch 14-day forecast HDD (sum)."""
        try:
            total = 0.0
            for city in self.cities:
                params = {
                    "latitude": city["lat"],
                    "longitude": city["lon"],
                    "daily": "temperature_2m_mean",
                    "temperature_unit": "fahrenheit",
                    "timezone": "America/New_York",
                    "forecast_days": self.forecast_days,
                }
                resp = requests.get(self.OPEN_METEO_URL, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                temps = data.get("daily", {}).get("temperature_2m_mean", [])
                for t in temps:
                    if t is not None:
                        total += max(0, self.hdd_base - t) * 0.25
            return round(total, 1)
        except Exception:
            return self._synthetic_forecast_hdd()

    @staticmethod
    def _seasonal_normal_hdd(month: int) -> float:
        """Approximate 30-year normal HDD by month (population-weighted US)."""
        normals = {
            1: 35.0, 2: 30.0, 3: 22.0, 4: 12.0, 5: 4.0, 6: 0.5,
            7: 0.0, 8: 0.0, 9: 2.0, 10: 10.0, 11: 20.0, 12: 32.0,
        }
        return normals.get(month, 15.0)

    @staticmethod
    def _synthetic_hdd_today() -> float:
        month = dt.date.today().month
        base = {
            1: 35, 2: 30, 3: 22, 4: 12, 5: 4, 6: 0.5,
            7: 0, 8: 0, 9: 2, 10: 10, 11: 20, 12: 32,
        }
        return base.get(month, 15) + np.random.uniform(-5, 5)

    @staticmethod
    def _synthetic_forecast_hdd() -> float:
        month = dt.date.today().month
        if month in (12, 1, 2):
            return np.random.uniform(300, 500)
        if month in (6, 7, 8):
            return np.random.uniform(0, 20)
        return np.random.uniform(50, 200)

    def get_hdd_history(self, days: int = 90) -> pd.DataFrame:
        """Get historical daily HDD data."""
        try:
            end = dt.date.today()
            start = end - dt.timedelta(days=days)
            # Use first city as representative
            city = self.cities[0]
            params = {
                "latitude": city["lat"],
                "longitude": city["lon"],
                "daily": "temperature_2m_mean",
                "temperature_unit": "fahrenheit",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            }
            resp = requests.get(self.OPEN_METEO_HIST_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            dates = data["daily"]["time"]
            temps = data["daily"]["temperature_2m_mean"]
            df = pd.DataFrame({"date": dates, "temp_f": temps})
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df["hdd"] = (self.hdd_base - df["temp_f"]).clip(lower=0)
            return df
        except Exception as e:
            logger.warning(f"HDD history fetch failed: {e}")
            return self._synthetic_hdd_history(days)

    @staticmethod
    def _synthetic_hdd_history(days: int) -> pd.DataFrame:
        dates = pd.date_range(end=dt.datetime.now(), periods=days, freq="D")
        months = dates.month
        base_temps = np.array([
            {1: 30, 2: 32, 3: 42, 4: 55, 5: 65, 6: 75,
             7: 82, 8: 80, 9: 70, 10: 58, 11: 45, 12: 33}[m] for m in months
        ])
        temps = base_temps + np.random.randn(days) * 5
        hdd = np.maximum(0, 65 - temps)
        return pd.DataFrame({"temp_f": temps, "hdd": hdd}, index=dates)
