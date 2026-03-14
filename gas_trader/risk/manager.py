"""Risk manager — position sizing, stop/target, daily limits, EIA kill switch."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from gas_trader.config import Settings, InstrumentConfig
from gas_trader.data.eia_storage import EIAStorageFeed
from gas_trader.signals.engine import TradeSignal, TradeDirection


@dataclass
class PositionSize:
    contracts: int
    risk_amount: float  # USD at risk
    margin_required: float
    risk_reward_ratio: float
    approved: bool
    rejection_reason: str = ""


@dataclass
class DailyStats:
    date: dt.date = field(default_factory=dt.date.today)
    trades_today: int = 0
    pnl_today: float = 0.0
    peak_equity: float = 0.0
    drawdown_pct: float = 0.0
    is_halted: bool = False
    halt_reason: str = ""


class RiskManager:
    """Comprehensive risk management for gas futures trading."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.risk = settings.risk
        self.paper_mode = settings.mode == "paper"
        self._daily = DailyStats()
        self._equity_peak = settings.initial_capital
        self._open_positions: list = []

    # ------------------------------------------------------------------
    # Position sizing (ATR-based)
    # ------------------------------------------------------------------
    def calculate_position_size(
        self,
        signal: TradeSignal,
        current_equity: float,
        instrument: InstrumentConfig,
    ) -> PositionSize:
        """Calculate position size using ATR-based risk management."""
        # Check daily limits first
        if self._daily.is_halted:
            return PositionSize(
                contracts=0, risk_amount=0, margin_required=0,
                risk_reward_ratio=0, approved=False,
                rejection_reason=f"Trading halted: {self._daily.halt_reason}",
            )

        if signal.direction == TradeDirection.FLAT:
            return PositionSize(
                contracts=0, risk_amount=0, margin_required=0,
                risk_reward_ratio=0, approved=False,
                rejection_reason="No trade signal",
            )

        # Max positions check
        if len(self._open_positions) >= self.risk.max_open_positions:
            return PositionSize(
                contracts=0, risk_amount=0, margin_required=0,
                risk_reward_ratio=0, approved=False,
                rejection_reason="Max open positions reached",
            )

        # Trades per day check
        if self._daily.trades_today >= self.risk.max_trades_per_day:
            return PositionSize(
                contracts=0, risk_amount=0, margin_required=0,
                risk_reward_ratio=0, approved=False,
                rejection_reason="Max daily trades reached",
            )

        # ATR-based position sizing
        risk_pct = self.risk.risk_per_trade_pct / 100
        max_risk_pct = self.risk.max_risk_per_trade_pct / 100
        risk_amount = current_equity * risk_pct

        # Risk per contract = |entry - stop| × point_value
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        if stop_distance == 0:
            return PositionSize(
                contracts=0, risk_amount=0, margin_required=0,
                risk_reward_ratio=0, approved=False,
                rejection_reason="Invalid stop distance (zero)",
            )

        risk_per_contract = stop_distance * instrument.point_value
        contracts = int(risk_amount / risk_per_contract)
        contracts = max(1, contracts)  # At least 1 contract

        # Verify margin
        margin_required = contracts * instrument.margin_initial
        if self.paper_mode:
            # Paper mode: allow 1 contract regardless of margin
            # Scale P&L proportionally but let the strategy trade
            contracts = 1
            margin_required = instrument.margin_initial
            if margin_required > current_equity:
                logger.debug(
                    f"Paper mode: margin ${margin_required:.0f} > equity ${current_equity:.0f} "
                    f"— allowing 1 contract for strategy testing"
                )
        else:
            if margin_required > current_equity * 0.80:
                contracts = max(1, int((current_equity * 0.80) / instrument.margin_initial))
                margin_required = contracts * instrument.margin_initial

            if margin_required > current_equity:
                return PositionSize(
                    contracts=0, risk_amount=0, margin_required=margin_required,
                    risk_reward_ratio=0, approved=False,
                    rejection_reason=f"Insufficient margin: need ${margin_required:.0f}, have ${current_equity:.0f}",
                )

        # Actual risk
        actual_risk = contracts * risk_per_contract
        actual_risk_pct = actual_risk / current_equity

        if actual_risk_pct > max_risk_pct:
            contracts = max(1, int(current_equity * max_risk_pct / risk_per_contract))
            actual_risk = contracts * risk_per_contract

        # Risk/reward ratio
        reward_distance = abs(signal.entry_price - signal.take_profit)
        rr_ratio = reward_distance / stop_distance if stop_distance > 0 else 0

        approved = rr_ratio >= 1.5  # Minimum 1.5:1 risk/reward

        logger.info(
            f"Position size: {contracts} contracts | "
            f"Risk: ${actual_risk:.2f} ({actual_risk / current_equity * 100:.1f}%) | "
            f"R:R = 1:{rr_ratio:.1f} | Approved: {approved}"
        )

        return PositionSize(
            contracts=contracts,
            risk_amount=actual_risk,
            margin_required=margin_required,
            risk_reward_ratio=rr_ratio,
            approved=approved,
            rejection_reason="" if approved else f"R:R too low ({rr_ratio:.1f}:1, need 1.5:1)",
        )

    # ------------------------------------------------------------------
    # Daily limits / drawdown
    # ------------------------------------------------------------------
    def update_pnl(self, pnl: float, current_equity: float) -> None:
        """Update daily P&L and check limits."""
        today = dt.date.today()
        if self._daily.date != today:
            self._reset_daily(current_equity)

        self._daily.pnl_today += pnl
        self._daily.trades_today += 1

        # Update equity peak
        if current_equity > self._equity_peak:
            self._equity_peak = current_equity

        # Check daily loss limit
        daily_loss_limit = current_equity * (self.risk.max_daily_loss_pct / 100)
        if self._daily.pnl_today < -daily_loss_limit:
            self._daily.is_halted = True
            self._daily.halt_reason = (
                f"Daily loss limit hit: ${self._daily.pnl_today:.2f} "
                f"(limit: -${daily_loss_limit:.2f})"
            )
            logger.warning(f"HALT: {self._daily.halt_reason}")

        # Check max drawdown
        drawdown = (self._equity_peak - current_equity) / self._equity_peak * 100
        self._daily.drawdown_pct = drawdown
        if drawdown > self.risk.max_drawdown_pct:
            self._daily.is_halted = True
            self._daily.halt_reason = (
                f"Max drawdown hit: {drawdown:.1f}% "
                f"(limit: {self.risk.max_drawdown_pct}%)"
            )
            logger.warning(f"HALT: {self._daily.halt_reason}")

    def _reset_daily(self, equity: float) -> None:
        self._daily = DailyStats(peak_equity=equity)

    # ------------------------------------------------------------------
    # EIA Kill Switch
    # ------------------------------------------------------------------
    def check_eia_kill_switch(self, eia: EIAStorageFeed) -> bool:
        """Returns True if trading should be paused due to EIA report."""
        if not self.risk.eia_kill_switch:
            return False

        now = dt.datetime.now()
        if eia.is_report_imminent(now):
            logger.info("EIA Kill Switch: Report imminent — pausing trading")
            return True
        if eia.is_report_just_released(now):
            logger.info("EIA Kill Switch: Report just released — pausing trading")
            return True
        return False

    # ------------------------------------------------------------------
    # Trailing stop
    # ------------------------------------------------------------------
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        current_stop: float,
        direction: TradeDirection,
    ) -> float:
        """Update trailing stop if conditions met."""
        if not self.risk.trailing_stop_enabled:
            return current_stop

        if direction == TradeDirection.LONG:
            pnl_pct = (current_price - entry_price) / entry_price * 100
            if pnl_pct >= self.risk.trailing_activation_pct:
                new_stop = current_price * (1 - self.risk.trailing_trail_pct / 100)
                return max(current_stop, new_stop)

        elif direction == TradeDirection.SHORT:
            pnl_pct = (entry_price - current_price) / entry_price * 100
            if pnl_pct >= self.risk.trailing_activation_pct:
                new_stop = current_price * (1 + self.risk.trailing_trail_pct / 100)
                return min(current_stop, new_stop) if current_stop > 0 else new_stop

        return current_stop

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def register_position(self, position: dict) -> None:
        self._open_positions.append(position)

    def remove_position(self, position_id: str) -> None:
        self._open_positions = [p for p in self._open_positions if p.get("id") != position_id]

    @property
    def daily_stats(self) -> DailyStats:
        return self._daily

    @property
    def is_trading_allowed(self) -> bool:
        return not self._daily.is_halted
