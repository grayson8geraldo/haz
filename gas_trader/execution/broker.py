"""Order execution — IBKR broker API with smart routing."""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger

from gas_trader.config import Settings, InstrumentConfig
from gas_trader.signals.engine import TradeSignal, TradeDirection
from gas_trader.risk.manager import PositionSize


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    id: str
    instrument: str
    direction: str  # "BUY" | "SELL"
    quantity: int
    order_type: str  # "LMT" | "MKT" | "STP" | "STP_LMT"
    price: float
    stop_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float = 0.0
    fill_time: Optional[dt.datetime] = None
    parent_id: str = ""
    oca_group: str = ""


@dataclass
class Position:
    id: str
    instrument: str
    direction: TradeDirection
    quantity: int
    entry_price: float
    entry_time: dt.datetime
    stop_loss: float
    take_profit: float
    current_stop: float  # May differ from stop_loss if trailing
    pnl: float = 0.0
    is_open: bool = True


class OrderExecutor:
    """Executes orders via IBKR with bracket orders (entry + SL + TP)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        exec_cfg = settings.raw["execution"]
        self.order_type = exec_cfg["order_type"]
        self.limit_offset_ticks = exec_cfg["limit_offset_ticks"]
        self.timeout = exec_cfg["timeout_seconds"]
        self.max_slippage = exec_cfg["slippage_max_ticks"]
        self._ib = None
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._paper_mode = settings.mode == "paper"

        # Paper mode P&L scaling: when equity < margin, scale P&L so
        # a full stop loss risks at most risk_per_trade% of equity
        inst = settings.select_instrument(settings.initial_capital)
        margin = inst.margin_initial
        equity = settings.initial_capital
        if self._paper_mode and equity < margin:
            self._pnl_scale = equity / margin
        else:
            self._pnl_scale = 1.0

    def connect(self) -> None:
        if self._paper_mode:
            logger.info("Running in PAPER mode — no live broker connection")
            return

        try:
            from ib_insync import IB

            self._ib = IB()
            broker = self.settings.raw["broker"]
            self._ib.connect(broker["host"], broker["port"], clientId=broker["client_id"])
            logger.info("OrderExecutor connected to IBKR")
        except Exception as e:
            logger.error(f"Broker connection failed: {e}")
            self._paper_mode = True

    def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()

    # ------------------------------------------------------------------
    # Execute trade
    # ------------------------------------------------------------------
    def execute_signal(
        self,
        signal: TradeSignal,
        position_size: PositionSize,
        instrument: InstrumentConfig,
    ) -> Optional[Position]:
        """Execute a trade signal with bracket orders (entry + SL + TP)."""
        if signal.direction == TradeDirection.FLAT:
            return None
        if not position_size.approved:
            logger.warning(f"Position not approved: {position_size.rejection_reason}")
            return None

        pos_id = str(uuid.uuid4())[:8]
        direction_str = "BUY" if signal.direction == TradeDirection.LONG else "SELL"

        logger.info(
            f"Executing {direction_str} {position_size.contracts}x {instrument.symbol} "
            f"@ {signal.entry_price:.4f} | SL={signal.stop_loss:.4f} TP={signal.take_profit:.4f}"
        )

        if self._paper_mode:
            return self._paper_execute(pos_id, signal, position_size, instrument)

        return self._live_execute(pos_id, signal, position_size, instrument)

    def _paper_execute(
        self,
        pos_id: str,
        signal: TradeSignal,
        size: PositionSize,
        instrument: InstrumentConfig,
    ) -> Position:
        """Simulate order execution for paper trading."""
        position = Position(
            id=pos_id,
            instrument=instrument.symbol,
            direction=signal.direction,
            quantity=size.contracts,
            entry_price=signal.entry_price,
            entry_time=dt.datetime.now(),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            current_stop=signal.stop_loss,
        )
        self._positions[pos_id] = position

        scale_info = ""
        if self._pnl_scale < 1.0:
            scale_info = f" (P&L scaled {self._pnl_scale:.0%})"
        logger.info(f"[PAPER] Opened position {pos_id}: {position.direction.name} "
                     f"{position.quantity}x @ {position.entry_price:.4f}{scale_info}")
        return position

    def _live_execute(
        self,
        pos_id: str,
        signal: TradeSignal,
        size: PositionSize,
        instrument: InstrumentConfig,
    ) -> Optional[Position]:
        """Execute via IBKR API with bracket orders."""
        try:
            from ib_insync import Future, LimitOrder, StopOrder, Order as IBOrder

            contract = Future(
                symbol=instrument.symbol,
                exchange=instrument.exchange,
            )
            self._ib.qualifyContracts(contract)

            action = "BUY" if signal.direction == TradeDirection.LONG else "SELL"
            reverse = "SELL" if action == "BUY" else "BUY"

            # Smart routing: use limit order with offset
            if self.order_type == "limit":
                offset = instrument.tick_size * self.limit_offset_ticks
                if action == "BUY":
                    limit_price = signal.entry_price + offset
                else:
                    limit_price = signal.entry_price - offset

                # Round to tick size
                limit_price = round(limit_price / instrument.tick_size) * instrument.tick_size

                entry_order = LimitOrder(action, size.contracts, limit_price)
            else:
                entry_order = IBOrder(action=action, totalQuantity=size.contracts, orderType="MKT")

            # Bracket: SL and TP as OCO
            oca = f"bracket_{pos_id}"

            sl_price = round(signal.stop_loss / instrument.tick_size) * instrument.tick_size
            tp_price = round(signal.take_profit / instrument.tick_size) * instrument.tick_size

            sl_order = StopOrder(reverse, size.contracts, sl_price)
            sl_order.ocaGroup = oca
            sl_order.ocaType = 1  # Cancel others on fill

            tp_order = LimitOrder(reverse, size.contracts, tp_price)
            tp_order.ocaGroup = oca
            tp_order.ocaType = 1

            # Submit entry
            entry_trade = self._ib.placeOrder(contract, entry_order)
            self._ib.sleep(1)

            if entry_trade.orderStatus.status == "Filled":
                fill_price = entry_trade.orderStatus.avgFillPrice

                # Submit SL and TP
                self._ib.placeOrder(contract, sl_order)
                self._ib.placeOrder(contract, tp_order)

                position = Position(
                    id=pos_id,
                    instrument=instrument.symbol,
                    direction=signal.direction,
                    quantity=size.contracts,
                    entry_price=fill_price,
                    entry_time=dt.datetime.now(),
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    current_stop=signal.stop_loss,
                )
                self._positions[pos_id] = position
                logger.info(f"[LIVE] Opened position {pos_id}: filled @ {fill_price:.4f}")
                return position

            logger.warning(f"Entry order not filled: {entry_trade.orderStatus.status}")
            self._ib.cancelOrder(entry_order)
            return None

        except Exception as e:
            logger.error(f"Live execution error: {e}")
            return None

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------
    def update_position(
        self, pos_id: str, current_price: float, instrument: InstrumentConfig,
    ) -> Optional[float]:
        """Update position P&L and check SL/TP. Returns P&L if closed, None otherwise."""
        pos = self._positions.get(pos_id)
        if not pos or not pos.is_open:
            return None

        # Calculate unrealized P&L (scaled for paper mode small accounts)
        if pos.direction == TradeDirection.LONG:
            pos.pnl = (current_price - pos.entry_price) * instrument.point_value * pos.quantity * self._pnl_scale
        else:
            pos.pnl = (pos.entry_price - current_price) * instrument.point_value * pos.quantity * self._pnl_scale

        # Check stop loss
        if pos.direction == TradeDirection.LONG and current_price <= pos.current_stop:
            return self._close_position(pos_id, pos.current_stop, instrument, "Stop loss hit")
        if pos.direction == TradeDirection.SHORT and current_price >= pos.current_stop:
            return self._close_position(pos_id, pos.current_stop, instrument, "Stop loss hit")

        # Check take profit
        if pos.direction == TradeDirection.LONG and current_price >= pos.take_profit:
            return self._close_position(pos_id, pos.take_profit, instrument, "Take profit hit")
        if pos.direction == TradeDirection.SHORT and current_price <= pos.take_profit:
            return self._close_position(pos_id, pos.take_profit, instrument, "Take profit hit")

        return None

    def _close_position(
        self, pos_id: str, exit_price: float,
        instrument: InstrumentConfig, reason: str,
    ) -> float:
        pos = self._positions[pos_id]
        if pos.direction == TradeDirection.LONG:
            pnl = (exit_price - pos.entry_price) * instrument.point_value * pos.quantity * self._pnl_scale
        else:
            pnl = (pos.entry_price - exit_price) * instrument.point_value * pos.quantity * self._pnl_scale

        # Subtract commission
        commission = self.settings.raw.get("backtest", {}).get("commission_per_trade", 2.50)
        pnl -= commission

        pos.pnl = pnl
        pos.is_open = False

        logger.info(
            f"Closed position {pos_id}: {reason} | "
            f"Exit @ {exit_price:.4f} | P&L: ${pnl:.2f}"
        )
        return pnl

    def close_all_positions(self, current_price: float, instrument: InstrumentConfig) -> float:
        """Emergency close all open positions."""
        total_pnl = 0.0
        for pos_id, pos in list(self._positions.items()):
            if pos.is_open:
                pnl = self._close_position(pos_id, current_price, instrument, "Force close")
                total_pnl += pnl
        return total_pnl

    def get_open_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.is_open]

    def get_all_positions(self) -> list[Position]:
        return list(self._positions.values())
