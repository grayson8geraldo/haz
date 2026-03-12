"""Telegram alerts — P&L, signals, errors, EIA reports."""

from __future__ import annotations

import datetime as dt
from typing import Optional

import requests
from loguru import logger

from gas_trader.config import Settings
from gas_trader.signals.engine import TradeSignal
from gas_trader.execution.broker import Position


class TelegramAlerter:
    """Sends trading alerts via Telegram bot."""

    API_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, settings: Settings):
        tg_cfg = settings.raw["monitoring"]["telegram"]
        self.enabled = tg_cfg["enabled"]
        self.token = tg_cfg["bot_token"] or ""
        self.chat_id = tg_cfg["chat_id"] or ""
        self._alerts = tg_cfg.get("alerts", [])

    def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send a message to Telegram."""
        if not self.enabled or not self.token or not self.chat_id:
            logger.debug(f"[TG disabled] {message}")
            return False

        try:
            url = self.API_URL.format(token=self.token)
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
            }
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    # ------------------------------------------------------------------
    # Formatted alerts
    # ------------------------------------------------------------------
    def alert_trade_open(self, position: Position, signal: TradeSignal) -> None:
        if "trade_open" not in self._alerts:
            return
        emoji = "\U0001f7e2" if position.direction.name == "LONG" else "\U0001f534"
        msg = (
            f"{emoji} <b>TRADE OPENED</b>\n"
            f"Direction: {position.direction.name}\n"
            f"Instrument: {position.instrument}\n"
            f"Quantity: {position.quantity}\n"
            f"Entry: ${position.entry_price:.4f}\n"
            f"SL: ${position.stop_loss:.4f}\n"
            f"TP: ${position.take_profit:.4f}\n"
            f"Signal: {signal.reason}\n"
            f"Confidence: {signal.confidence:.0%}"
        )
        self.send(msg)

    def alert_trade_close(self, position: Position, reason: str) -> None:
        if "trade_close" not in self._alerts:
            return
        emoji = "\U0001f4b0" if position.pnl >= 0 else "\U0001f4a5"
        msg = (
            f"{emoji} <b>TRADE CLOSED</b>\n"
            f"Direction: {position.direction.name}\n"
            f"Instrument: {position.instrument}\n"
            f"P&L: ${position.pnl:+.2f}\n"
            f"Reason: {reason}"
        )
        self.send(msg)

    def alert_daily_pnl(self, pnl: float, equity: float, trades: int) -> None:
        if "daily_pnl" not in self._alerts:
            return
        emoji = "\U0001f4c8" if pnl >= 0 else "\U0001f4c9"
        msg = (
            f"{emoji} <b>DAILY SUMMARY</b>\n"
            f"Date: {dt.date.today()}\n"
            f"P&L: ${pnl:+.2f}\n"
            f"Equity: ${equity:.2f}\n"
            f"Trades: {trades}"
        )
        self.send(msg)

    def alert_drawdown(self, drawdown_pct: float, equity: float) -> None:
        if "drawdown_warning" not in self._alerts:
            return
        msg = (
            f"\u26a0\ufe0f <b>DRAWDOWN WARNING</b>\n"
            f"Drawdown: {drawdown_pct:.1f}%\n"
            f"Equity: ${equity:.2f}"
        )
        self.send(msg)

    def alert_eia_report(self, storage_bcf: float, change_bcf: float) -> None:
        if "eia_report" not in self._alerts:
            return
        msg = (
            f"\U0001f4ca <b>EIA STORAGE REPORT</b>\n"
            f"Storage: {storage_bcf:.0f} BCF\n"
            f"Change: {change_bcf:+.0f} BCF"
        )
        self.send(msg)

    def alert_error(self, error: str) -> None:
        if "error" not in self._alerts:
            return
        msg = f"\u274c <b>ERROR</b>\n{error}"
        self.send(msg)
