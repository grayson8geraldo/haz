"""Technical analysis signals — RSI, VWAP, ATR, Bollinger Bands, EMA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class TechIndicators:
    rsi: float
    atr: float
    bb_upper: float
    bb_lower: float
    bb_width: float
    vwap: float
    vwap_distance: float  # % distance from VWAP
    ema_fast: float
    ema_slow: float
    ema_crossover: int  # 1=bullish, -1=bearish, 0=neutral
    signal: float  # -1 to 1
    confidence: float  # 0 to 1


class TechnicalSignal:
    """Computes technical analysis indicators and generates a composite signal."""

    def __init__(self, config: dict):
        self.rsi_period = config.get("rsi", {}).get("period", 14)
        self.rsi_ob = config.get("rsi", {}).get("overbought", 70)
        self.rsi_os = config.get("rsi", {}).get("oversold", 30)
        self.atr_period = config.get("atr", {}).get("period", 14)
        self.bb_period = config.get("bollinger", {}).get("period", 20)
        self.bb_std = config.get("bollinger", {}).get("std_dev", 2.0)
        self.ema_fast = config.get("ema", {}).get("fast", 9)
        self.ema_slow = config.get("ema", {}).get("slow", 21)

    def compute(self, df: pd.DataFrame) -> TechIndicators:
        """Compute all technical indicators on OHLCV DataFrame."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI
        rsi = self._rsi(close, self.rsi_period)
        current_rsi = rsi.iloc[-1]

        # ATR
        atr = self._atr(high, low, close, self.atr_period)
        current_atr = atr.iloc[-1]

        # Bollinger Bands
        bb_mid = close.rolling(self.bb_period).mean()
        bb_std = close.rolling(self.bb_period).std()
        bb_upper = (bb_mid + self.bb_std * bb_std).iloc[-1]
        bb_lower = (bb_mid - self.bb_std * bb_std).iloc[-1]
        bb_width = (bb_upper - bb_lower) / bb_mid.iloc[-1] if bb_mid.iloc[-1] != 0 else 0

        # VWAP
        if "vwap" in df.columns:
            current_vwap = df["vwap"].iloc[-1]
        else:
            cum_vol = df["volume"].cumsum()
            cum_vwap = (close * df["volume"]).cumsum()
            current_vwap = (cum_vwap / cum_vol).iloc[-1]

        vwap_distance = ((close.iloc[-1] - current_vwap) / current_vwap * 100
                         if current_vwap != 0 else 0)

        # EMA crossover
        ema_f = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_s = close.ewm(span=self.ema_slow, adjust=False).mean()
        ema_cross = 0
        if ema_f.iloc[-1] > ema_s.iloc[-1] and ema_f.iloc[-2] <= ema_s.iloc[-2]:
            ema_cross = 1  # Bullish crossover
        elif ema_f.iloc[-1] < ema_s.iloc[-1] and ema_f.iloc[-2] >= ema_s.iloc[-2]:
            ema_cross = -1  # Bearish crossover

        # Composite signal
        signal, confidence = self._composite_signal(
            current_rsi, close.iloc[-1], bb_upper, bb_lower,
            vwap_distance, ema_f.iloc[-1], ema_s.iloc[-1], ema_cross,
        )

        return TechIndicators(
            rsi=current_rsi,
            atr=current_atr,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
            bb_width=bb_width,
            vwap=current_vwap,
            vwap_distance=vwap_distance,
            ema_fast=ema_f.iloc[-1],
            ema_slow=ema_s.iloc[-1],
            ema_crossover=ema_cross,
            signal=signal,
            confidence=confidence,
        )

    def _composite_signal(
        self, rsi, price, bb_upper, bb_lower,
        vwap_dist, ema_f, ema_s, ema_cross,
    ) -> tuple[float, float]:
        """Combine indicators into a single signal [-1, 1] with confidence."""
        scores = []

        # RSI signal
        if rsi < self.rsi_os:
            scores.append(1.0)  # Oversold → buy
        elif rsi > self.rsi_ob:
            scores.append(-1.0)  # Overbought → sell
        else:
            # Linear scale between oversold and overbought
            scores.append((50 - rsi) / 50)

        # Bollinger Band signal
        if price < bb_lower:
            scores.append(1.0)  # Below lower band → buy
        elif price > bb_upper:
            scores.append(-1.0)  # Above upper band → sell
        else:
            bb_mid = (bb_upper + bb_lower) / 2
            scores.append((bb_mid - price) / (bb_upper - bb_mid) if bb_upper != bb_mid else 0)

        # VWAP signal
        scores.append(np.clip(-vwap_dist / 2, -1, 1))

        # EMA crossover signal
        if ema_cross != 0:
            scores.append(float(ema_cross))
        else:
            # Trend direction based on EMA position
            if ema_f > ema_s:
                scores.append(0.3)
            else:
                scores.append(-0.3)

        signal = float(np.mean(scores))
        confidence = min(1.0, abs(signal) * 1.5)
        return np.clip(signal, -1, 1), confidence

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def add_indicators_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all indicator columns to DataFrame for ML features."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        df = df.copy()
        df["rsi_14"] = self._rsi(close, self.rsi_period)
        df["atr_14"] = self._atr(high, low, close, self.atr_period)

        bb_mid = close.rolling(self.bb_period).mean()
        bb_std = close.rolling(self.bb_period).std()
        df["bb_upper"] = bb_mid + self.bb_std * bb_std
        df["bb_lower"] = bb_mid - self.bb_std * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid

        if "vwap" in df.columns:
            df["vwap_distance"] = (close - df["vwap"]) / df["vwap"] * 100
        else:
            df["vwap_distance"] = 0.0

        df["ema_fast"] = close.ewm(span=self.ema_fast, adjust=False).mean()
        df["ema_slow"] = close.ewm(span=self.ema_slow, adjust=False).mean()
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

        return df.dropna()
