"""Seasonality signal — injection/withdrawal seasons, calendar spreads."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class SeasonalityIndicators:
    season: str  # "injection" | "withdrawal" | "transition"
    seasonal_bias: float  # -1 to 1 (positive = bullish)
    month_of_year_signal: float
    calendar_spread_signal: float
    composite_signal: float
    confidence: float


class SeasonalitySignal:
    """Generates signals based on natural gas seasonal patterns."""

    # Historical monthly returns bias (simplified from 20+ years of NG data)
    MONTHLY_BIAS = {
        1: 0.15,   # Jan: moderately bullish (cold)
        2: 0.10,   # Feb: slightly bullish (cold tail)
        3: -0.20,  # Mar: bearish (winter ending)
        4: -0.25,  # Apr: bearish (injection starts)
        5: -0.15,  # May: slightly bearish
        6: -0.05,  # Jun: neutral-bearish
        7: 0.10,   # Jul: slightly bullish (cooling demand)
        8: 0.05,   # Aug: neutral
        9: 0.05,   # Sep: neutral
        10: 0.15,  # Oct: bullish (winter hedge buying)
        11: 0.25,  # Nov: bullish (withdrawal begins)
        12: 0.20,  # Dec: bullish (peak heating)
    }

    def __init__(self, config: dict):
        self.calendar_spread = config.get("calendar_spread", True)

    def compute(self, current_date: dt.date | None = None) -> SeasonalityIndicators:
        """Compute seasonality signal for the given date."""
        if current_date is None:
            current_date = dt.date.today()

        month = current_date.month
        day = current_date.day

        # Determine season
        season = self._get_season(month)

        # Monthly bias
        month_signal = self.MONTHLY_BIAS.get(month, 0.0)

        # Seasonal bias
        if season == "withdrawal":
            seasonal_bias = 0.20  # Bullish during withdrawal
        elif season == "injection":
            seasonal_bias = -0.15  # Bearish during injection
        else:
            seasonal_bias = 0.0

        # Calendar spread signal (front month vs deferred)
        calendar_signal = self._calendar_spread_signal(month, day)

        # Composite
        composite = month_signal * 0.4 + seasonal_bias * 0.35 + calendar_signal * 0.25
        confidence = 0.4  # Seasonality is a weak but consistent signal

        return SeasonalityIndicators(
            season=season,
            seasonal_bias=seasonal_bias,
            month_of_year_signal=month_signal,
            calendar_spread_signal=calendar_signal,
            composite_signal=float(np.clip(composite, -1, 1)),
            confidence=confidence,
        )

    @staticmethod
    def _get_season(month: int) -> str:
        if month in (4, 5, 6, 7, 8, 9, 10):
            return "injection"
        elif month in (11, 12, 1, 2, 3):
            return "withdrawal"
        return "transition"

    def _calendar_spread_signal(self, month: int, day: int) -> float:
        """Calendar spread tendency: front month premium in winter,
        deferred premium in summer (contango)."""
        if not self.calendar_spread:
            return 0.0

        # Winter: front > deferred (backwardation tendency) → bullish
        if month in (11, 12, 1, 2):
            return 0.15
        # Summer: front < deferred (contango) → bearish
        if month in (5, 6, 7, 8):
            return -0.10
        # Transition: roll effects
        if month in (3, 4):
            return -0.05
        if month in (9, 10):
            return 0.10
        return 0.0
