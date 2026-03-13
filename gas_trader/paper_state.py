"""Persistent paper trading state — saves/restores equity, positions, trade history to JSON."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from loguru import logger

STATE_DIR = Path(__file__).resolve().parent.parent / "data"
STATE_FILE = STATE_DIR / "paper_state.json"


class PaperState:
    """Saves and restores paper trading state between sessions."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.equity_curve: list[float] = [initial_capital]
        self.trades: list[dict] = []
        self.open_positions: list[dict] = []
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.peak_equity = initial_capital
        self.max_drawdown_pct = 0.0
        self.started_at: str = dt.datetime.now().isoformat()
        self.last_updated: str = dt.datetime.now().isoformat()
        self.data_source: str = ""
        self.session_count = 0

    def record_trade(self, trade: dict) -> None:
        """Record a completed trade."""
        self.trades.append(trade)
        self.total_trades += 1
        pnl = trade.get("pnl", 0)
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

    def update_equity(self, equity: float) -> None:
        """Update equity and track curve / drawdown."""
        self.equity = equity
        self.equity_curve.append(equity)
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = (self.peak_equity - equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0
        if dd > self.max_drawdown_pct:
            self.max_drawdown_pct = dd
        self.last_updated = dt.datetime.now().isoformat()

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades * 100

    @property
    def return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.equity - self.initial_capital) / self.initial_capital * 100

    def save(self) -> None:
        """Save state to JSON file."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.last_updated = dt.datetime.now().isoformat()
        data = {
            "initial_capital": self.initial_capital,
            "equity": self.equity,
            "equity_curve": self.equity_curve[-10000:],  # Keep last 10k points
            "trades": self.trades[-500:],  # Keep last 500 trades
            "open_positions": self.open_positions,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": self.total_pnl,
            "peak_equity": self.peak_equity,
            "max_drawdown_pct": self.max_drawdown_pct,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "data_source": self.data_source,
            "session_count": self.session_count,
        }
        with open(STATE_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug(f"Paper state saved: equity=${self.equity:.2f}")

    @classmethod
    def load(cls, initial_capital: float) -> PaperState:
        """Load state from JSON file, or create new if not exists."""
        if not STATE_FILE.exists():
            logger.info("No saved paper state — starting fresh")
            state = cls(initial_capital)
            return state

        try:
            with open(STATE_FILE) as f:
                data = json.load(f)

            state = cls(data.get("initial_capital", initial_capital))
            state.equity = data.get("equity", initial_capital)
            state.equity_curve = data.get("equity_curve", [initial_capital])
            state.trades = data.get("trades", [])
            state.open_positions = data.get("open_positions", [])
            state.total_trades = data.get("total_trades", 0)
            state.wins = data.get("wins", 0)
            state.losses = data.get("losses", 0)
            state.total_pnl = data.get("total_pnl", 0.0)
            state.peak_equity = data.get("peak_equity", initial_capital)
            state.max_drawdown_pct = data.get("max_drawdown_pct", 0.0)
            state.started_at = data.get("started_at", dt.datetime.now().isoformat())
            state.last_updated = data.get("last_updated", "")
            state.data_source = data.get("data_source", "")
            state.session_count = data.get("session_count", 0) + 1

            logger.info(
                f"Restored paper state: equity=${state.equity:.2f} | "
                f"trades={state.total_trades} | "
                f"P&L=${state.total_pnl:+.2f} | "
                f"session #{state.session_count}"
            )
            return state

        except Exception as e:
            logger.warning(f"Could not load paper state: {e} — starting fresh")
            return cls(initial_capital)

    @staticmethod
    def reset() -> None:
        """Delete saved state to start fresh."""
        if STATE_FILE.exists():
            STATE_FILE.unlink()
            logger.info("Paper state reset — starting fresh next run")

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Paper Trading Summary\n"
            f"{'=' * 40}\n"
            f"Started:       {self.started_at[:19]}\n"
            f"Sessions:      {self.session_count}\n"
            f"Data Source:   {self.data_source}\n"
            f"Initial:       ${self.initial_capital:.2f}\n"
            f"Equity:        ${self.equity:.2f}\n"
            f"Total P&L:     ${self.total_pnl:+.2f} ({self.return_pct:+.1f}%)\n"
            f"Trades:        {self.total_trades}\n"
            f"Win Rate:      {self.win_rate:.1f}%\n"
            f"Max Drawdown:  {self.max_drawdown_pct:.1f}%\n"
            f"Peak Equity:   ${self.peak_equity:.2f}\n"
        )
