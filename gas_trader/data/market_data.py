"""Market data feed — NYMEX NG / QG via IBKR or Polygon."""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from gas_trader.config import InstrumentConfig, Settings


class MarketDataFeed:
    """Fetches and caches OHLCV bars for natural gas futures."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache: dict[str, pd.DataFrame] = {}
        self._ib = None

    # ------------------------------------------------------------------
    # IBKR connection
    # ------------------------------------------------------------------
    def connect_ibkr(self) -> None:
        try:
            from ib_insync import IB

            self._ib = IB()
            broker = self.settings.raw["broker"]
            self._ib.connect(broker["host"], broker["port"], clientId=broker["client_id"])
            logger.info("Connected to IBKR gateway")
        except Exception as e:
            logger.warning(f"IBKR connection failed: {e}. Using offline mode.")
            self._ib = None

    def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()

    # ------------------------------------------------------------------
    # Contract helpers
    # ------------------------------------------------------------------
    def _make_contract(self, inst: InstrumentConfig):
        from ib_insync import Future

        if inst.contract_month == "auto":
            # Use continuous front-month
            return Future(
                symbol=inst.symbol,
                exchange=inst.exchange,
                lastTradeDateOrContractMonth="",
            )
        return Future(
            symbol=inst.symbol,
            exchange=inst.exchange,
            lastTradeDateOrContractMonth=inst.contract_month,
        )

    # ------------------------------------------------------------------
    # Fetch historical bars
    # ------------------------------------------------------------------
    def fetch_bars(
        self,
        timeframe: str = "1h",
        days: int = 90,
        instrument: Optional[InstrumentConfig] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars. Returns DataFrame with columns:
        [open, high, low, close, volume, vwap].
        """
        inst = instrument or self.settings.instrument
        cache_key = f"{inst.symbol}_{timeframe}_{days}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._ib and self._ib.isConnected():
            df = self._fetch_ibkr(inst, timeframe, days)
        else:
            logger.info("No live connection — generating synthetic data for testing")
            df = self._generate_synthetic(days, timeframe)

        self._cache[cache_key] = df
        return df

    def _fetch_ibkr(
        self, inst: InstrumentConfig, timeframe: str, days: int
    ) -> pd.DataFrame:
        bar_size_map = {
            "1m": "1 min",
            "5m": "5 mins",
            "15m": "15 mins",
            "1h": "1 hour",
            "4h": "4 hours",
            "1d": "1 day",
        }
        bar_size = bar_size_map.get(timeframe, "1 hour")
        contract = self._make_contract(inst)
        self._ib.qualifyContracts(contract)

        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=f"{days} D",
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
        )

        records = []
        for b in bars:
            records.append(
                {
                    "datetime": b.date,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
            )

        df = pd.DataFrame(records)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
            df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

        logger.info(f"Fetched {len(df)} bars ({inst.symbol} {timeframe})")
        return df

    # ------------------------------------------------------------------
    # Synthetic data for backtesting / paper without live feed
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_synthetic(days: int, timeframe: str) -> pd.DataFrame:
        """Generates realistic NG price data using geometric Brownian motion
        with mean-reversion (Ornstein-Uhlenbeck) to mimic natural gas behavior."""
        tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        minutes = tf_minutes.get(timeframe, 60)
        n_bars = (days * 24 * 60) // minutes

        np.random.seed(42)
        # NG typically ranges $1.50-$9.00, mean ~$3.00
        mu = 3.0  # Long-term mean
        theta = 0.02  # Mean-reversion speed
        sigma = 0.04  # Volatility

        prices = np.zeros(n_bars)
        prices[0] = 2.80
        for i in range(1, n_bars):
            drift = theta * (mu - prices[i - 1])
            shock = sigma * np.random.randn()
            prices[i] = max(0.50, prices[i - 1] + drift + shock)

        # Add seasonality: higher in winter
        t = np.arange(n_bars)
        season = 0.3 * np.sin(2 * np.pi * t / (365 * 24 * 60 / minutes) - np.pi / 2)
        prices = prices + season

        # Build OHLCV
        opens = prices
        noise = np.abs(np.random.randn(n_bars)) * 0.02
        highs = prices + noise
        lows = prices - noise
        closes = prices + np.random.randn(n_bars) * 0.01
        volumes = np.random.randint(1000, 50000, n_bars).astype(float)

        end = dt.datetime.now()
        start = end - dt.timedelta(minutes=minutes * n_bars)
        idx = pd.date_range(start=start, periods=n_bars, freq=f"{minutes}min")

        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            },
            index=idx,
        )
        df.index.name = "datetime"
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

        logger.info(f"Generated {n_bars} synthetic bars ({timeframe})")
        return df

    def get_latest_price(self, instrument: Optional[InstrumentConfig] = None) -> float:
        """Returns the latest close price."""
        df = self.fetch_bars(timeframe="1m", days=1, instrument=instrument)
        if df.empty:
            return 0.0
        return float(df["close"].iloc[-1])
