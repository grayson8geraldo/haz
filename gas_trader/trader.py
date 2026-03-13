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
from gas_trader.paper_state import PaperState


class GasFuturesTrader:
    """Master orchestrator for the Natural Gas Futures Trading System.

    In paper mode with real data:
    - Connects to Yahoo Finance (free) or IBKR for real NG prices
    - Tracks virtual balance starting from initial_capital
    - Saves state to disk between sessions (data/paper_state.json)
    - All trades are simulated against real market prices
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()

        # Load persistent paper state (or create new)
        self.paper_state = PaperState.load(self.settings.initial_capital)
        self.equity = self.paper_state.equity
        self.equity_curve: list[float] = self.paper_state.equity_curve

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
        self._tick_count = 0

    def start(self) -> None:
        """Start the trading system."""
        logger.info("=" * 60)
        logger.info("GAS FUTURES TRADER — Starting")
        logger.info(f"Mode: {self.settings.mode.upper()}")
        logger.info(f"Capital: ${self.equity:.2f}")
        logger.info(f"Target: ${self.settings.target_capital:.2f}")
        logger.info("=" * 60)

        # Connect to data source (works in BOTH paper and live mode)
        data_source = self.market.connect()
        self.paper_state.data_source = data_source

        if self.market.is_real_data:
            logger.info(f"REAL DATA active via {data_source}")
            # Show current NG price
            quote = self.market.get_realtime_quote()
            if quote.get("price", 0) > 0:
                logger.info(f"Current NG price: ${quote['price']:.4f}")
        else:
            logger.warning(
                "Using SYNTHETIC data — install yfinance or connect IBKR for real prices:\n"
                "  pip install yfinance"
            )

        # Connect order executor (only for live mode)
        if self.settings.mode == "live":
            self.executor.connect()

        self.telegram.send(
            f"\U0001f680 <b>Trading System Started</b>\n"
            f"Mode: {self.settings.mode.upper()}\n"
            f"Data: {data_source}\n"
            f"Capital: ${self.equity:.2f}\n"
            f"Real data: {'YES' if self.market.is_real_data else 'NO'}"
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
            for pos in self.executor.get_open_positions():
                pnl = self.executor.update_position(pos.id, price, inst)
                if pnl is None:
                    # Force close
                    pnl_forced = self.executor.close_all_positions(price, inst)
                    self.equity += pnl_forced
                    break

        self.market.disconnect()
        self.executor.disconnect()

        # Save paper state
        self.paper_state.update_equity(self.equity)
        self.paper_state.save()

        self.telegram.alert_daily_pnl(
            self.risk_mgr.daily_stats.pnl_today,
            self.equity,
            self.risk_mgr.daily_stats.trades_today,
        )
        logger.info(f"Trading stopped. Equity: ${self.equity:.2f}")
        logger.info(f"\n{self.paper_state.summary()}")

    def _run_loop(self) -> None:
        """Main trading loop."""
        # Shorter interval for real data, longer for synthetic
        if self.market.is_real_data:
            loop_interval = 60  # 1 minute with real data
        else:
            loop_interval = 5   # 5 seconds for synthetic (simulation speed)

        logger.info(f"Trading loop started (interval: {loop_interval}s)")

        while self._running:
            try:
                self._tick()
                # Auto-save every 10 ticks
                self._tick_count += 1
                if self._tick_count % 10 == 0:
                    self.paper_state.update_equity(self.equity)
                    self.paper_state.save()

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

        # 3. Get current price
        current_price = self.market.get_latest_price(inst)
        if current_price <= 0:
            logger.debug("No price available — skipping tick")
            return

        # 4. Update open positions
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

                # Record trade in paper state
                self.paper_state.record_trade({
                    "id": pos.id,
                    "direction": pos.direction.name,
                    "instrument": pos.instrument,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "entry_time": str(pos.entry_time),
                    "exit_price": current_price,
                    "exit_time": dt.datetime.now().isoformat(),
                    "pnl": pnl,
                    "equity_after": self.equity,
                })
                self.paper_state.update_equity(self.equity)

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

        # 5. Generate new signal (only if no open positions)
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

        # 6. Periodic logging (every 5 minutes)
        if self._tick_count % 5 == 0:
            open_count = len(self.executor.get_open_positions())
            unrealized = sum(p.pnl for p in self.executor.get_open_positions())
            logger.info(
                f"[{self.market.source.upper()}] "
                f"NG=${current_price:.4f} | "
                f"Equity=${self.equity:.2f} | "
                f"Open={open_count} (${unrealized:+.2f}) | "
                f"Day P&L=${self.risk_mgr.daily_stats.pnl_today:+.2f} | "
                f"Trades={self.paper_state.total_trades}"
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
        if self.market.source == "none":
            self.market.connect()
        self._tick()
        return {
            "equity": self.equity,
            "open_positions": len(self.executor.get_open_positions()),
            "daily_pnl": self.risk_mgr.daily_stats.pnl_today,
            "is_halted": self.risk_mgr.daily_stats.is_halted,
            "data_source": self.market.source,
        }
