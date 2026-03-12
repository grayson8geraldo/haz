"""Fundamental analysis signals — EIA storage surprise, weather/HDD, supply-demand."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from gas_trader.data.eia_storage import EIAStorageFeed
from gas_trader.data.weather import WeatherFeed, HDDData
from gas_trader.data.lng_production import LNGProductionFeed


@dataclass
class FundamentalIndicators:
    storage_surprise_bcf: float  # Actual - consensus
    storage_signal: float  # -1 to 1
    hdd_deviation_pct: float
    hdd_signal: float  # -1 to 1
    supply_demand_signal: float  # -1 to 1
    composite_signal: float  # -1 to 1
    confidence: float  # 0 to 1


class FundamentalSignal:
    """Generates fundamental signals from storage, weather, and supply data."""

    def __init__(self, config: dict):
        self.storage_threshold = config.get("storage_surprise_threshold", 5)
        self.hdd_threshold = config.get("hdd_deviation_threshold", 10)

    def compute(
        self,
        eia: EIAStorageFeed,
        weather: WeatherFeed,
        lng: LNGProductionFeed,
        consensus_change: float = 0.0,
    ) -> FundamentalIndicators:
        """Compute fundamental indicators from all data sources."""
        # 1. Storage surprise
        report = eia.get_latest_report()
        actual_change = report.get("change_bcf", 0.0)
        surprise = eia.compute_surprise(actual_change, consensus_change)
        storage_signal = self._storage_to_signal(surprise)

        # 2. HDD / Weather
        hdd_data: HDDData = weather.fetch_current_hdd()
        hdd_signal = self._hdd_to_signal(hdd_data.hdd_deviation_pct)

        # 3. Supply/demand balance
        sd_balance = lng.get_supply_demand_balance()
        sd_signal = sd_balance["signal"]

        # Composite
        weights = [0.45, 0.35, 0.20]  # storage, weather, supply-demand
        composite = (
            storage_signal * weights[0]
            + hdd_signal * weights[1]
            + sd_signal * weights[2]
        )
        confidence = min(1.0, (abs(storage_signal) + abs(hdd_signal) + abs(sd_signal)) / 2)

        return FundamentalIndicators(
            storage_surprise_bcf=surprise,
            storage_signal=storage_signal,
            hdd_deviation_pct=hdd_data.hdd_deviation_pct,
            hdd_signal=hdd_signal,
            supply_demand_signal=sd_signal,
            composite_signal=float(np.clip(composite, -1, 1)),
            confidence=confidence,
        )

    def _storage_to_signal(self, surprise_bcf: float) -> float:
        """Convert storage surprise to signal.
        Negative surprise (less injection / more withdrawal) → bullish (positive signal).
        """
        if abs(surprise_bcf) < self.storage_threshold:
            return 0.0
        # Scale: ±20 BCF → ±1.0 signal
        signal = -surprise_bcf / 20.0
        return float(np.clip(signal, -1, 1))

    def _hdd_to_signal(self, deviation_pct: float) -> float:
        """Convert HDD deviation to signal.
        Colder than normal (positive deviation) → bullish (more gas demand).
        """
        if abs(deviation_pct) < self.hdd_threshold:
            return 0.0
        signal = deviation_pct / 30.0
        return float(np.clip(signal, -1, 1))
