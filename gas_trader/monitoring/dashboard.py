"""Dashboard data provider — feeds Streamlit dashboard with real-time P&L."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from gas_trader.execution.broker import OrderExecutor, Position
from gas_trader.risk.manager import RiskManager


@dataclass
class DashboardData:
    """Aggregated data for the monitoring dashboard."""

    equity: float = 0.0
    initial_capital: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    daily_pnl: float = 0.0
    drawdown_pct: float = 0.0
    open_positions: list = field(default_factory=list)
    closed_trades: list = field(default_factory=list)
    win_rate: float = 0.0
    total_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    equity_curve: list = field(default_factory=list)
    timestamp: dt.datetime = field(default_factory=dt.datetime.now)

    @classmethod
    def from_state(
        cls,
        executor: OrderExecutor,
        risk_mgr: RiskManager,
        equity: float,
        initial_capital: float,
        equity_curve: list,
    ) -> DashboardData:
        open_pos = executor.get_open_positions()
        all_pos = executor.get_all_positions()
        closed = [p for p in all_pos if not p.is_open]

        wins = [p.pnl for p in closed if p.pnl > 0]
        losses = [p.pnl for p in closed if p.pnl <= 0]

        total_trades = len(closed)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Simplified Sharpe
        if len(equity_curve) > 1:
            import numpy as np

            returns = np.diff(equity_curve) / equity_curve[:-1]
            if returns.std() > 0:
                sharpe = returns.mean() / returns.std() * (252 ** 0.5)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        return cls(
            equity=equity,
            initial_capital=initial_capital,
            total_pnl=equity - initial_capital,
            total_pnl_pct=(equity - initial_capital) / initial_capital * 100,
            daily_pnl=risk_mgr.daily_stats.pnl_today,
            drawdown_pct=risk_mgr.daily_stats.drawdown_pct,
            open_positions=[_pos_to_dict(p) for p in open_pos],
            closed_trades=[_pos_to_dict(p) for p in closed[-50:]],
            win_rate=win_rate,
            total_trades=total_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            equity_curve=equity_curve,
        )


def _pos_to_dict(p: Position) -> dict:
    return {
        "id": p.id,
        "instrument": p.instrument,
        "direction": p.direction.name,
        "quantity": p.quantity,
        "entry_price": p.entry_price,
        "entry_time": str(p.entry_time),
        "stop_loss": p.stop_loss,
        "take_profit": p.take_profit,
        "pnl": p.pnl,
        "is_open": p.is_open,
    }
