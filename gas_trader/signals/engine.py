"""Signal engine — combines all signal sources into a unified trade decision."""

from __future__ import annotations

import datetime as dt
import traceback
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from loguru import logger

from gas_trader.config import Settings
from gas_trader.data.market_data import MarketDataFeed
from gas_trader.data.eia_storage import EIAStorageFeed
from gas_trader.data.weather import WeatherFeed
from gas_trader.data.lng_production import LNGProductionFeed
from gas_trader.signals.technical import TechnicalSignal, TechIndicators
from gas_trader.signals.fundamental import FundamentalSignal, FundamentalIndicators
from gas_trader.signals.ml_model import MLSignal, MLPrediction
from gas_trader.signals.seasonality import SeasonalitySignal, SeasonalityIndicators


class TradeDirection(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class TradeSignal:
    direction: TradeDirection
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    timestamp: dt.datetime = field(default_factory=dt.datetime.now)
    # Component signals for logging
    technical: float = 0.0
    fundamental: float = 0.0
    ml: float = 0.0
    seasonality: float = 0.0
    reason: str = ""


class SignalEngine:
    """Master signal engine — combines technical, fundamental, ML, and seasonality."""

    MIN_SIGNAL_THRESHOLD = 0.25  # Minimum composite signal to trigger trade
    MIN_CONFIDENCE = 0.40

    def __init__(self, settings: Settings):
        self.settings = settings
        sig_cfg = settings.raw["signals"]

        self.technical = TechnicalSignal(sig_cfg["technical"])
        self.fundamental = FundamentalSignal(sig_cfg["fundamental"])
        self.ml = MLSignal(sig_cfg["ml"], settings)
        self.seasonality = SeasonalitySignal(sig_cfg["seasonality"])

        self.weights = settings.signal_weights

    def generate_signal(
        self,
        market: MarketDataFeed,
        eia: EIAStorageFeed,
        weather: WeatherFeed,
        lng: LNGProductionFeed,
    ) -> TradeSignal:
        """Generate a composite trade signal from all sources."""
        # Get market data
        df = market.fetch_bars(timeframe="1h", days=30)
        if df.empty or len(df) < 30:
            return self._flat_signal(reason="Insufficient market data")

        # Add indicators for ML
        df_with_indicators = self.technical.add_indicators_to_df(df)
        if df_with_indicators.empty or len(df_with_indicators) < 30:
            return self._flat_signal(reason="Insufficient data after indicators")

        # 1. Technical analysis
        tech: TechIndicators = self.technical.compute(df)

        # 2. Fundamental analysis
        try:
            fund: FundamentalIndicators = self.fundamental.compute(eia, weather, lng)
        except Exception as e:
            logger.warning(f"Fundamental signal error: {e}")
            fund = FundamentalIndicators(
                storage_surprise_bcf=0, storage_signal=0, hdd_deviation_pct=0,
                hdd_signal=0, supply_demand_signal=0, composite_signal=0, confidence=0,
            )

        # 3. ML prediction
        try:
            ml_pred: MLPrediction = self.ml.predict(df_with_indicators)
        except Exception as e:
            logger.warning(f"ML signal error: {e}")
            ml_pred = MLPrediction(
                direction=0, probability=0.5, signal=0, confidence=0,
                model_name=self.ml.model_type,
            )

        # 4. Seasonality
        seas: SeasonalityIndicators = self.seasonality.compute()

        # Weighted composite signal
        composite = (
            tech.signal * self.weights.technical
            + fund.composite_signal * self.weights.fundamental
            + ml_pred.signal * self.weights.ml
            + seas.composite_signal * self.weights.seasonality
        )

        # Weighted confidence
        confidence = (
            tech.confidence * self.weights.technical
            + fund.confidence * self.weights.fundamental
            + ml_pred.confidence * self.weights.ml
            + seas.confidence * self.weights.seasonality
        )

        # Determine direction
        if composite > self.MIN_SIGNAL_THRESHOLD and confidence > self.MIN_CONFIDENCE:
            direction = TradeDirection.LONG
        elif composite < -self.MIN_SIGNAL_THRESHOLD and confidence > self.MIN_CONFIDENCE:
            direction = TradeDirection.SHORT
        else:
            direction = TradeDirection.FLAT

        # Entry, SL, TP using ATR
        current_price = df["close"].iloc[-1]
        atr = tech.atr
        risk_cfg = self.settings.risk

        if direction == TradeDirection.LONG:
            stop_loss = current_price - atr * risk_cfg.atr_multiplier_sl
            take_profit = current_price + atr * risk_cfg.atr_multiplier_tp
        elif direction == TradeDirection.SHORT:
            stop_loss = current_price + atr * risk_cfg.atr_multiplier_sl
            take_profit = current_price - atr * risk_cfg.atr_multiplier_tp
        else:
            stop_loss = 0.0
            take_profit = 0.0

        reason_parts = []
        if abs(tech.signal) > 0.2:
            reason_parts.append(f"Tech={tech.signal:+.2f}")
        if abs(fund.composite_signal) > 0.1:
            reason_parts.append(f"Fund={fund.composite_signal:+.2f}")
        if abs(ml_pred.signal) > 0.1:
            reason_parts.append(f"ML={ml_pred.signal:+.2f}({ml_pred.probability:.0%})")
        if abs(seas.composite_signal) > 0.05:
            reason_parts.append(f"Season={seas.season}")

        signal = TradeSignal(
            direction=direction,
            strength=abs(composite),
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
            technical=tech.signal,
            fundamental=fund.composite_signal,
            ml=ml_pred.signal,
            seasonality=seas.composite_signal,
            reason=" | ".join(reason_parts) or "No strong signal",
        )

        logger.info(
            f"Signal: {direction.name} strength={abs(composite):.3f} "
            f"conf={confidence:.3f} | {signal.reason}"
        )
        return signal

    def _flat_signal(self, reason: str = "") -> TradeSignal:
        return TradeSignal(
            direction=TradeDirection.FLAT,
            strength=0.0,
            confidence=0.0,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            atr=0.0,
            reason=reason,
        )
