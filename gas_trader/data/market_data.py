"""Market data feed — NYMEX NG / QG via IBKR, Yahoo Finance, or synthetic."""

from __future__ import annotations

import datetime as dt
import time
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from gas_trader.config import InstrumentConfig, Settings


class MarketDataFeed:
    """Fetches and caches OHLCV bars for natural gas futures.

    Data source priority (when provider=auto):
        1. IBKR (if connected) — real-time, no delay
        2. Yahoo Finance (yfinance) — free, 15-min delay for intraday
        3. Synthetic — fallback for offline testing
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        market_cfg = settings.raw["data"]["market"]
        self.provider = market_cfg.get("provider", "auto")
        self.yf_ticker = market_cfg.get("yfinance_ticker", "NG=F")
        self._cache: dict[str, tuple[pd.DataFrame, float]] = {}
        self._cache_ttl = 60  # seconds
        self._ib = None
        self._data_source = "none"  # Track which source is active

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect_ibkr(self) -> bool:
        """Try to connect to IBKR TWS/Gateway."""
        try:
            from ib_insync import IB

            self._ib = IB()
            broker = self.settings.raw["broker"]
            self._ib.connect(broker["host"], broker["port"], clientId=broker["client_id"])
            self._data_source = "ibkr"
            logger.info("Connected to IBKR gateway — using real-time data")
            return True
        except Exception as e:
            logger.warning(f"IBKR connection failed: {e}")
            self._ib = None
            return False

    def connect(self) -> str:
        """Connect to the best available data source. Returns source name."""
        if self.provider == "ibkr":
            if self.connect_ibkr():
                return "ibkr"
            logger.error("IBKR required but unavailable")
            self._data_source = "synthetic"
            return "synthetic"

        if self.provider == "yfinance":
            if self._check_yfinance():
                self._data_source = "yfinance"
                return "yfinance"
            self._data_source = "synthetic"
            return "synthetic"

        if self.provider == "synthetic":
            self._data_source = "synthetic"
            return "synthetic"

        # auto — try in order
        if self.connect_ibkr():
            return "ibkr"

        if self._check_yfinance():
            self._data_source = "yfinance"
            logger.info(f"Using Yahoo Finance ({self.yf_ticker}) — free real data, 15-min delay")
            return "yfinance"

        self._data_source = "synthetic"
        logger.warning("No real data source available — using synthetic data")
        return "synthetic"

    def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()

    @property
    def source(self) -> str:
        return self._data_source

    @property
    def is_real_data(self) -> bool:
        return self._data_source in ("ibkr", "yfinance")

    # ------------------------------------------------------------------
    # Fetch bars
    # ------------------------------------------------------------------
    def fetch_bars(
        self,
        timeframe: str = "1h",
        days: int = 90,
        instrument: Optional[InstrumentConfig] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars. Returns DataFrame with columns:
        [open, high, low, close, volume, vwap].
        """
        inst = instrument or self.settings.instrument
        cache_key = f"{inst.symbol}_{timeframe}_{days}"

        # Check cache (TTL-based)
        if not force_refresh and cache_key in self._cache:
            cached_df, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_df

        # Fetch from best available source
        if self._data_source == "ibkr" and self._ib and self._ib.isConnected():
            df = self._fetch_ibkr(inst, timeframe, days)
        elif self._data_source == "yfinance":
            df = self._fetch_yfinance(timeframe, days)
        else:
            df = self._generate_synthetic(days, timeframe)

        if not df.empty:
            self._cache[cache_key] = (df, time.time())

        return df

    def get_latest_price(self, instrument: Optional[InstrumentConfig] = None) -> float:
        """Returns the latest close price."""
        if self._data_source == "yfinance":
            df = self.fetch_bars(timeframe="5m", days=1, instrument=instrument, force_refresh=True)
        else:
            df = self.fetch_bars(timeframe="1m", days=1, instrument=instrument, force_refresh=True)
        if df.empty:
            return 0.0
        return float(df["close"].iloc[-1])

    def get_realtime_quote(self) -> dict:
        """Get a real-time (or near real-time) quote."""
        if self._data_source == "yfinance":
            try:
                import yfinance as yf

                ticker = yf.Ticker(self.yf_ticker)
                info = ticker.fast_info
                return {
                    "price": float(info.last_price),
                    "prev_close": float(info.previous_close),
                    "open": float(info.open),
                    "day_high": float(info.day_high),
                    "day_low": float(info.day_low),
                    "volume": int(info.last_volume),
                    "source": "yfinance",
                    "timestamp": dt.datetime.now().isoformat(),
                }
            except Exception as e:
                logger.warning(f"yfinance quote error: {e}")

        if self._data_source == "ibkr" and self._ib:
            return self._ibkr_quote()

        return {"price": self.get_latest_price(), "source": self._data_source}

    # ------------------------------------------------------------------
    # Yahoo Finance (бесплатные реальные данные)
    # ------------------------------------------------------------------
    def _check_yfinance(self) -> bool:
        """Check if yfinance is available and can fetch NG data."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(self.yf_ticker)
            hist = ticker.history(period="5d")
            if hist.empty:
                logger.warning(f"yfinance: no data for {self.yf_ticker}")
                return False
            logger.info(
                f"yfinance OK: {self.yf_ticker} last price = "
                f"${hist['Close'].iloc[-1]:.4f}"
            )
            return True
        except ImportError:
            logger.warning("yfinance not installed: pip install yfinance")
            return False
        except Exception as e:
            logger.warning(f"yfinance check failed: {e}")
            return False

    def _fetch_yfinance(self, timeframe: str, days: int) -> pd.DataFrame:
        """Fetch data from Yahoo Finance (free, 15-min delay for intraday)."""
        try:
            import yfinance as yf

            # Map timeframe to yfinance interval + period
            # yfinance limits: 1m=7d, 5m=60d, 15m=60d, 1h=730d, 1d=unlimited
            yf_map = {
                "1m":  {"interval": "1m",  "max_days": 7},
                "5m":  {"interval": "5m",  "max_days": 60},
                "15m": {"interval": "15m", "max_days": 60},
                "1h":  {"interval": "1h",  "max_days": 730},
                "4h":  {"interval": "1h",  "max_days": 730},  # Resample from 1h
                "1d":  {"interval": "1d",  "max_days": 3650},
            }

            cfg = yf_map.get(timeframe, {"interval": "1h", "max_days": 730})
            actual_days = min(days, cfg["max_days"])

            ticker = yf.Ticker(self.yf_ticker)
            hist = ticker.history(period=f"{actual_days}d", interval=cfg["interval"])

            if hist.empty:
                logger.warning(f"yfinance returned empty data for {self.yf_ticker}")
                return self._generate_synthetic(days, timeframe)

            # Normalize column names
            df = pd.DataFrame({
                "open": hist["Open"],
                "high": hist["High"],
                "low": hist["Low"],
                "close": hist["Close"],
                "volume": hist["Volume"].astype(float),
            })
            df.index.name = "datetime"

            # Resample 4h from 1h if needed
            if timeframe == "4h" and cfg["interval"] == "1h":
                df = df.resample("4h").agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }).dropna()

            # Add VWAP
            cum_vol = df["volume"].cumsum()
            cum_vwap = (df["close"] * df["volume"]).cumsum()
            df["vwap"] = cum_vwap / cum_vol.replace(0, np.nan)
            df["vwap"] = df["vwap"].ffill()

            logger.info(
                f"Fetched {len(df)} bars from Yahoo Finance "
                f"({self.yf_ticker} {timeframe}, {actual_days}d)"
            )
            return df

        except Exception as e:
            logger.error(f"yfinance fetch error: {e}")
            return self._generate_synthetic(days, timeframe)

    # ------------------------------------------------------------------
    # IBKR
    # ------------------------------------------------------------------
    def _make_contract(self, inst: InstrumentConfig):
        from ib_insync import Future

        if inst.contract_month == "auto":
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

    def _fetch_ibkr(
        self, inst: InstrumentConfig, timeframe: str, days: int,
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
            records.append({
                "datetime": b.date,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
            df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

        logger.info(f"Fetched {len(df)} bars from IBKR ({inst.symbol} {timeframe})")
        return df

    def _ibkr_quote(self) -> dict:
        try:
            inst = self.settings.instrument
            contract = self._make_contract(inst)
            self._ib.qualifyContracts(contract)
            self._ib.reqMktData(contract)
            self._ib.sleep(2)
            ticker = self._ib.ticker(contract)
            return {
                "price": float(ticker.last or ticker.close or 0),
                "bid": float(ticker.bid or 0),
                "ask": float(ticker.ask or 0),
                "volume": int(ticker.volume or 0),
                "source": "ibkr",
                "timestamp": dt.datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"IBKR quote error: {e}")
            return {"price": 0, "source": "ibkr_error"}

    # ------------------------------------------------------------------
    # Synthetic data (fallback)
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_synthetic(days: int, timeframe: str) -> pd.DataFrame:
        """Generates realistic NG price data using mean-reversion (Ornstein-Uhlenbeck)."""
        tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        minutes = tf_minutes.get(timeframe, 60)
        n_bars = (days * 24 * 60) // minutes

        np.random.seed(42)
        mu = 3.0
        theta = 0.02
        sigma = 0.04

        prices = np.zeros(n_bars)
        prices[0] = 2.80
        for i in range(1, n_bars):
            drift = theta * (mu - prices[i - 1])
            shock = sigma * np.random.randn()
            prices[i] = max(0.50, prices[i - 1] + drift + shock)

        t = np.arange(n_bars)
        season = 0.3 * np.sin(2 * np.pi * t / (365 * 24 * 60 / minutes) - np.pi / 2)
        prices = prices + season

        noise = np.abs(np.random.randn(n_bars)) * 0.02
        opens = prices
        highs = prices + noise
        lows = prices - noise
        closes = prices + np.random.randn(n_bars) * 0.01
        volumes = np.random.randint(1000, 50000, n_bars).astype(float)

        end = dt.datetime.now()
        start = end - dt.timedelta(minutes=minutes * n_bars)
        idx = pd.date_range(start=start, periods=n_bars, freq=f"{minutes}min")

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes,
        }, index=idx)
        df.index.name = "datetime"
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

        logger.info(f"Generated {n_bars} synthetic bars ({timeframe})")
        return df
