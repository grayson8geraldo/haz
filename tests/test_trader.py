"""Tests for the Gas Futures Trading System."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gas_trader.config import Settings
from gas_trader.data.market_data import MarketDataFeed
from gas_trader.data.eia_storage import EIAStorageFeed
from gas_trader.data.weather import WeatherFeed
from gas_trader.data.lng_production import LNGProductionFeed
from gas_trader.signals.technical import TechnicalSignal
from gas_trader.signals.fundamental import FundamentalSignal
from gas_trader.signals.seasonality import SeasonalitySignal
from gas_trader.signals.engine import SignalEngine, TradeDirection
from gas_trader.risk.manager import RiskManager, PositionSize
from gas_trader.execution.broker import OrderExecutor
from gas_trader.backtest.engine import BacktestEngine


@pytest.fixture
def settings():
    return Settings.load()


@pytest.fixture
def sample_df():
    """Create a sample OHLCV DataFrame."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range(end=dt.datetime.now(), periods=n, freq="1h")
    prices = np.cumsum(np.random.randn(n) * 0.02) + 3.0
    return pd.DataFrame({
        "open": prices,
        "high": prices + np.abs(np.random.randn(n)) * 0.02,
        "low": prices - np.abs(np.random.randn(n)) * 0.02,
        "close": prices + np.random.randn(n) * 0.01,
        "volume": np.random.randint(1000, 50000, n).astype(float),
        "vwap": prices,
    }, index=dates)


# ------------------------------------------------------------------
# Config tests
# ------------------------------------------------------------------
class TestConfig:
    def test_load_settings(self, settings):
        assert settings.initial_capital == 200.0
        assert settings.target_capital == 100000.0
        assert settings.mode == "paper"

    def test_instrument_selection_micro(self, settings):
        # With $200, should select micro (QG)
        inst = settings.select_instrument(200.0)
        assert inst.symbol == "QG"

    def test_instrument_selection_full(self, settings):
        # With $5000, should select full (NG)
        inst = settings.select_instrument(5000.0)
        assert inst.symbol == "NG"


# ------------------------------------------------------------------
# Data layer tests
# ------------------------------------------------------------------
class TestMarketData:
    def test_synthetic_data(self, settings):
        feed = MarketDataFeed(settings)
        df = feed.fetch_bars(timeframe="1h", days=30)
        assert not df.empty
        assert "close" in df.columns
        assert "volume" in df.columns
        assert len(df) > 100

    def test_latest_price(self, settings):
        feed = MarketDataFeed(settings)
        price = feed.get_latest_price()
        assert price > 0


class TestEIAStorage:
    def test_synthetic_storage(self, settings):
        feed = EIAStorageFeed(settings)
        df = feed.fetch_storage_history(periods=52)
        assert not df.empty
        assert "storage_bcf" in df.columns

    def test_latest_report(self, settings):
        feed = EIAStorageFeed(settings)
        report = feed.get_latest_report()
        assert "storage_bcf" in report

    def test_surprise_calc(self, settings):
        feed = EIAStorageFeed(settings)
        surprise = feed.compute_surprise(actual_change=-80, consensus=-70)
        assert surprise == -10  # More withdrawn than expected → bullish


class TestWeather:
    def test_hdd(self, settings):
        feed = WeatherFeed(settings)
        hdd = feed.fetch_current_hdd()
        assert hdd.date == dt.date.today()
        assert hdd.hdd_actual >= 0


class TestLNGProduction:
    def test_supply_demand(self, settings):
        feed = LNGProductionFeed(settings)
        balance = feed.get_supply_demand_balance()
        assert "signal" in balance
        assert -1 <= balance["signal"] <= 1


# ------------------------------------------------------------------
# Signal tests
# ------------------------------------------------------------------
class TestTechnicalSignal:
    def test_compute(self, settings, sample_df):
        tech = TechnicalSignal(settings.raw["signals"]["technical"])
        indicators = tech.compute(sample_df)
        assert 0 <= indicators.rsi <= 100
        assert indicators.atr > 0
        assert -1 <= indicators.signal <= 1

    def test_add_indicators(self, settings, sample_df):
        tech = TechnicalSignal(settings.raw["signals"]["technical"])
        df = tech.add_indicators_to_df(sample_df)
        assert "rsi_14" in df.columns
        assert "atr_14" in df.columns
        assert "bb_width" in df.columns


class TestFundamentalSignal:
    def test_compute(self, settings):
        fund = FundamentalSignal(settings.raw["signals"]["fundamental"])
        eia = EIAStorageFeed(settings)
        weather = WeatherFeed(settings)
        lng = LNGProductionFeed(settings)
        indicators = fund.compute(eia, weather, lng)
        assert -1 <= indicators.composite_signal <= 1


class TestSeasonality:
    def test_winter_bullish(self):
        seas = SeasonalitySignal({"calendar_spread": True})
        result = seas.compute(dt.date(2025, 1, 15))
        assert result.season == "withdrawal"
        assert result.seasonal_bias > 0

    def test_summer_bearish(self):
        seas = SeasonalitySignal({"calendar_spread": True})
        result = seas.compute(dt.date(2025, 6, 15))
        assert result.season == "injection"
        assert result.seasonal_bias < 0


class TestSignalEngine:
    def test_generate_signal(self, settings):
        engine = SignalEngine(settings)
        market = MarketDataFeed(settings)
        eia = EIAStorageFeed(settings)
        weather = WeatherFeed(settings)
        lng = LNGProductionFeed(settings)
        signal = engine.generate_signal(market, eia, weather, lng)
        assert signal.direction in (TradeDirection.LONG, TradeDirection.SHORT, TradeDirection.FLAT)
        assert 0 <= signal.confidence <= 1


# ------------------------------------------------------------------
# Risk management tests
# ------------------------------------------------------------------
class TestRiskManager:
    def test_position_sizing(self, settings):
        from gas_trader.signals.engine import TradeSignal

        risk_mgr = RiskManager(settings)
        signal = TradeSignal(
            direction=TradeDirection.LONG,
            strength=0.5,
            confidence=0.7,
            entry_price=3.00,
            stop_loss=2.90,
            take_profit=3.15,
            atr=0.05,
        )
        inst = settings.select_instrument(200.0)
        pos = risk_mgr.calculate_position_size(signal, 200.0, inst)
        assert isinstance(pos, PositionSize)
        # With $200, should get at most 1 contract micro
        assert pos.contracts <= 1 or not pos.approved

    def test_daily_loss_halt(self, settings):
        risk_mgr = RiskManager(settings)
        # Simulate losing more than daily limit
        risk_mgr.update_pnl(-20.0, 180.0)  # -10% of $200
        assert risk_mgr.daily_stats.is_halted

    def test_eia_kill_switch(self, settings):
        risk_mgr = RiskManager(settings)
        eia = EIAStorageFeed(settings)
        # Should not be active on non-Thursday
        monday = dt.datetime(2025, 1, 6, 10, 0)  # Monday
        assert not eia.is_report_imminent(monday)


# ------------------------------------------------------------------
# Backtest tests
# ------------------------------------------------------------------
class TestBacktest:
    def test_run(self, settings, sample_df):
        engine = BacktestEngine(settings)
        result = engine.run(sample_df)
        assert result.initial_capital == 200.0
        assert result.total_trades >= 0
        assert len(result.equity_curve) > 0

    def test_walk_forward(self, settings):
        market = MarketDataFeed(settings)
        df = market.fetch_bars(timeframe="1h", days=365)
        engine = BacktestEngine(settings)
        results = engine.walk_forward(df, train_months=3, test_months=1)
        assert isinstance(results, list)
