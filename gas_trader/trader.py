"""Main trading orchestrator — ties all components together."""

from __future__ import annotations

import datetime as dt
import time

from loguru import logger

from gas_trader.config import Settings
from gas_trader.data.market_data import MarketDataFeed
from gas_trader.data.eia_storage import EIAStorageFeed
from gas_trader.data.weather import WeatherFeed
from gas_trader.data.lng_production import LNGProductionFeed
from gas_trader.signals.engine import SignalEngine, TradeDirection
from gas_trader.risk.manager import RiskManager
from gas_trader.execution.broker import OrderExecutor
from gas_trader.monitoring.telegram_alerts import TelegramAlerter
from gas_trader.monitoring.dashboard import DashboardData


class GasFuturesTrader:
    """Master orchestrator for the Natural Gas Futures Trading System."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        self.equity = self.settings.initial_capital
        self.equity_curve: list[float] = [self.equity]

        # Initialize all components
        self.market = MarketDataFeed(self.settings)
        self.eia = EIAStorageFeed(self.settings)
        self.weather = WeatherFeed(self.settings)
        self.lng = LNGProductionFeed(self.settings)
        self.signal_engine = SignalEngine(self.settings)
        self.risk_mgr = RiskManager(self.settings)
        self.executor = OrderExecutor(self.settings)
        self.telegram = TelegramAlerter(self.settings)

        self._running = False

    def start(self) -> None:
        """Start the trading system."""
        logger.info("=" * 60)
        logger.info("GAS FUTURES TRADER — Starting")
        logger.info(f"Mode: {self.settings.mode.upper()}")
        logger.info(f"Capital: ${self.equity:.2f}")
        logger.info(f"Target: ${self.settings.target_capital:.2f}")
        logger.info("=" * 60)

        # Connect to broker
        if self.settings.mode == "live":
            self.market.connect_ibkr()
            self.executor.connect()

        self.telegram.send(
            f"\U0001f680 <b>Trading System Started</b>\n"
            f"Mode: {self.settings.mode.upper()}\n"
            f"Capital: ${self.equity:.2f}"
        )

        self._running = True
        self._run_loop()

    def stop(self) -> None:
        """Stop the trading system gracefully."""
        self._running = False
        # Close all open positions
        inst = self.settings.select_instrument(self.equity)
        price = self.market.get_latest_price(inst)
        if price > 0:
            pnl = self.executor.close_all_positions(price, inst)
            self.equity += pnl

        self.market.disconnect()
        self.executor.disconnect()

        self.telegram.alert_daily_pnl(
            self.risk_mgr.daily_stats.pnl_today,
            self.equity,
            self.risk_mgr.daily_stats.trades_today,
        )
        logger.info(f"Trading stopped. Final equity: ${self.equity:.2f}")

    def _run_loop(self) -> None:
        """Main trading loop."""
        loop_interval = 60  # seconds between signal checks

        while self._running:
            try:
                self._tick()
                time.sleep(loop_interval)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt — stopping")
                self.stop()
                break
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                self.telegram.alert_error(str(e))
                time.sleep(30)

    def _tick(self) -> None:
        """Single iteration of the trading loop."""
        inst = self.settings.select_instrument(self.equity)

        # 1. Check EIA Kill Switch
        if self.risk_mgr.check_eia_kill_switch(self.eia):
            logger.info("EIA Kill Switch active — skipping this tick")
            return

        # 2. Check if trading is allowed
        if not self.risk_mgr.is_trading_allowed:
            logger.info(f"Trading halted: {self.risk_mgr.daily_stats.halt_reason}")
            return

        # 3. Update open positions
        current_price = self.market.get_latest_price(inst)
        if current_price <= 0:
            return

        for pos in self.executor.get_open_positions():
            # Update trailing stop
            pos.current_stop = self.risk_mgr.calculate_trailing_stop(
                pos.entry_price, current_price, pos.current_stop, pos.direction,
            )
            # Check SL/TP
            pnl = self.executor.update_position(pos.id, current_price, inst)
            if pnl is not None:
                self.equity += pnl
                self.risk_mgr.update_pnl(pnl, self.equity)
                self.risk_mgr.remove_position(pos.id)
                self.telegram.alert_trade_close(pos, "SL/TP hit")

                # Check if target reached
                if self.equity >= self.settings.target_capital:
                    logger.info(f"TARGET REACHED! Equity: ${self.equity:.2f}")
                    self.telegram.send(
                        f"\U0001f3c6 <b>TARGET REACHED!</b>\n"
                        f"Equity: ${self.equity:.2f}\n"
                        f"Target: ${self.settings.target_capital:.2f}"
                    )
                    self.stop()
                    return

        self.equity_curve.append(self.equity)

        # 4. Generate new signal (only if no open positions)
        if not self.executor.get_open_positions():
            signal = self.signal_engine.generate_signal(
                self.market, self.eia, self.weather, self.lng,
            )

            if signal.direction != TradeDirection.FLAT:
                # Calculate position size
                pos_size = self.risk_mgr.calculate_position_size(signal, self.equity, inst)

                if pos_size.approved:
                    position = self.executor.execute_signal(signal, pos_size, inst)
                    if position:
                        self.risk_mgr.register_position({"id": position.id})
                        self.telegram.alert_trade_open(position, signal)

        # 5. Periodic logging
        if dt.datetime.now().minute == 0:
            logger.info(
                f"Status: Equity=${self.equity:.2f} | "
                f"Open={len(self.executor.get_open_positions())} | "
                f"Daily P&L=${self.risk_mgr.daily_stats.pnl_today:.2f}"
            )

    def get_dashboard_data(self) -> DashboardData:
        """Get current state for dashboard."""
        return DashboardData.from_state(
            self.executor, self.risk_mgr,
            self.equity, self.settings.initial_capital,
            self.equity_curve,
        )

    def run_single_tick(self) -> dict:
        """Run a single tick (useful for testing)."""
        self._tick()
        return {
            "equity": self.equity,
            "open_positions": len(self.executor.get_open_positions()),
            "daily_pnl": self.risk_mgr.daily_stats.pnl_today,
            "is_halted": self.risk_mgr.daily_stats.is_halted,
        }
