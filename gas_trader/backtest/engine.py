"""Backtesting engine — event-driven backtest with walk-forward validation."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from gas_trader.config import Settings
from gas_trader.signals.technical import TechnicalSignal
from gas_trader.signals.seasonality import SeasonalitySignal


@dataclass
class BacktestTrade:
    entry_time: dt.datetime
    exit_time: dt.datetime
    direction: str  # "LONG" | "SHORT"
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    initial_capital: float
    final_equity: float
    total_return_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    equity_curve: list[float]
    trades: list[BacktestTrade]
    monthly_returns: dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """Event-driven backtester for natural gas futures strategies."""

    def __init__(self, settings: Settings):
        self.settings = settings
        bt_cfg = settings.raw["backtest"]
        self.initial_capital = bt_cfg["initial_capital"]
        self.commission = bt_cfg["commission_per_trade"]
        self.slippage_ticks = bt_cfg["slippage_ticks"]

    def run(
        self,
        df: pd.DataFrame,
        signal_threshold: float = 0.25,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        risk_per_trade_pct: float = 1.0,
    ) -> BacktestResult:
        """Run backtest on OHLCV DataFrame with technical signals."""
        tech = TechnicalSignal(self.settings.raw["signals"]["technical"])
        seas = SeasonalitySignal(self.settings.raw["signals"]["seasonality"])

        # Add indicators
        df = tech.add_indicators_to_df(df)
        if len(df) < 50:
            logger.warning("Not enough data for backtest")
            return self._empty_result()

        equity = self.initial_capital
        equity_curve = [equity]
        trades: list[BacktestTrade] = []
        position = None  # Active position

        inst = self.settings.select_instrument(equity)

        for i in range(50, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            price = row["close"]
            atr = row.get("atr_14", 0.01)

            # Update instrument based on current equity
            inst = self.settings.select_instrument(equity)

            # Check if position should be closed
            if position is not None:
                exit_price = None
                exit_reason = ""

                if position["direction"] == "LONG":
                    if price <= position["stop_loss"]:
                        exit_price = position["stop_loss"]
                        exit_reason = "Stop loss"
                    elif price >= position["take_profit"]:
                        exit_price = position["take_profit"]
                        exit_reason = "Take profit"
                elif position["direction"] == "SHORT":
                    if price >= position["stop_loss"]:
                        exit_price = position["stop_loss"]
                        exit_reason = "Stop loss"
                    elif price <= position["take_profit"]:
                        exit_price = position["take_profit"]
                        exit_reason = "Take profit"

                if exit_price is not None:
                    # Close position
                    if position["direction"] == "LONG":
                        pnl = (exit_price - position["entry_price"]) * inst.point_value * position["qty"]
                    else:
                        pnl = (position["entry_price"] - exit_price) * inst.point_value * position["qty"]
                    pnl -= self.commission

                    equity += pnl
                    trades.append(BacktestTrade(
                        entry_time=position["entry_time"],
                        exit_time=df.index[i],
                        direction=position["direction"],
                        entry_price=position["entry_price"],
                        exit_price=exit_price,
                        quantity=position["qty"],
                        pnl=pnl,
                        pnl_pct=pnl / equity * 100 if equity > 0 else 0,
                        exit_reason=exit_reason,
                    ))
                    position = None

            # Generate signal (simplified for backtest speed)
            if position is None and equity > inst.margin_initial:
                indicators = tech.compute(df.iloc[max(0, i - 50):i + 1])
                season = seas.compute(df.index[i].date() if hasattr(df.index[i], 'date') else None)

                composite = indicators.signal * 0.6 + season.composite_signal * 0.4

                if composite > signal_threshold:
                    direction = "LONG"
                elif composite < -signal_threshold:
                    direction = "SHORT"
                else:
                    equity_curve.append(equity)
                    continue

                # Position sizing
                risk_amount = equity * (risk_per_trade_pct / 100)
                stop_distance = atr * atr_sl_mult
                risk_per_contract = stop_distance * inst.point_value
                qty = max(1, int(risk_amount / risk_per_contract))

                # Check margin
                if qty * inst.margin_initial > equity * 0.8:
                    qty = max(1, int(equity * 0.8 / inst.margin_initial))

                # Slippage
                slippage = inst.tick_size * self.slippage_ticks

                if direction == "LONG":
                    entry = price + slippage
                    sl = entry - stop_distance
                    tp = entry + atr * atr_tp_mult
                else:
                    entry = price - slippage
                    sl = entry + stop_distance
                    tp = entry - atr * atr_tp_mult

                position = {
                    "direction": direction,
                    "entry_price": entry,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "qty": qty,
                    "entry_time": df.index[i],
                }

            equity_curve.append(equity)

        # Close any remaining position
        if position is not None:
            last_price = df["close"].iloc[-1]
            if position["direction"] == "LONG":
                pnl = (last_price - position["entry_price"]) * inst.point_value * position["qty"]
            else:
                pnl = (position["entry_price"] - last_price) * inst.point_value * position["qty"]
            pnl -= self.commission
            equity += pnl
            trades.append(BacktestTrade(
                entry_time=position["entry_time"],
                exit_time=df.index[-1],
                direction=position["direction"],
                entry_price=position["entry_price"],
                exit_price=last_price,
                quantity=position["qty"],
                pnl=pnl,
                pnl_pct=pnl / equity * 100 if equity > 0 else 0,
                exit_reason="End of data",
            ))
            equity_curve.append(equity)

        return self._compute_stats(equity_curve, trades)

    def walk_forward(
        self,
        df: pd.DataFrame,
        train_months: int = 6,
        test_months: int = 2,
    ) -> list[BacktestResult]:
        """Walk-forward analysis: train on N months, test on M months, step forward."""
        results = []
        total_bars = len(df)
        bars_per_month = total_bars // max(1, (df.index[-1] - df.index[0]).days // 30)
        train_bars = train_months * bars_per_month
        test_bars = test_months * bars_per_month
        step_bars = test_bars

        i = 0
        while i + train_bars + test_bars <= total_bars:
            test_start = i + train_bars
            test_end = test_start + test_bars
            test_data = df.iloc[test_start:test_end]

            if len(test_data) > 50:
                result = self.run(test_data)
                results.append(result)
                logger.info(
                    f"WF fold: {test_data.index[0].date()} to {test_data.index[-1].date()} | "
                    f"Return: {result.total_return_pct:.1f}% | Sharpe: {result.sharpe_ratio:.2f}"
                )
            i += step_bars

        return results

    def _compute_stats(
        self, equity_curve: list[float], trades: list[BacktestTrade],
    ) -> BacktestResult:
        eq = np.array(equity_curve)
        initial = self.initial_capital
        final = eq[-1] if len(eq) > 0 else initial

        # Returns
        total_return = (final - initial) / initial * 100

        # Trades stats
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        n_trades = len(trades)
        win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = gross_profit / len(wins) if wins else 0
        avg_loss = -gross_loss / len(losses) if losses else 0

        # Drawdown
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak * 100
        max_dd = dd.max()

        # Sharpe & Sortino
        if len(eq) > 1:
            returns = np.diff(eq) / eq[:-1]
            sharpe = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0
            downside = returns[returns < 0]
            sortino = (
                returns.mean() / downside.std() * (252 ** 0.5)
                if len(downside) > 0 and downside.std() > 0 else 0
            )
        else:
            sharpe = sortino = 0

        # Calmar
        annual_return = total_return / max(1, len(eq) / (252 * 24))
        calmar = annual_return / max_dd if max_dd > 0 else 0

        # Consecutive wins/losses
        max_con_wins = max_con_losses = cur_wins = cur_losses = 0
        for t in trades:
            if t.pnl > 0:
                cur_wins += 1
                cur_losses = 0
                max_con_wins = max(max_con_wins, cur_wins)
            else:
                cur_losses += 1
                cur_wins = 0
                max_con_losses = max(max_con_losses, cur_losses)

        return BacktestResult(
            initial_capital=initial,
            final_equity=final,
            total_return_pct=total_return,
            total_trades=n_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            avg_trade_pnl=sum(t.pnl for t in trades) / n_trades if n_trades > 0 else 0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_wins=max_con_wins,
            max_consecutive_losses=max_con_losses,
            equity_curve=equity_curve,
            trades=trades,
        )

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            initial_capital=self.initial_capital,
            final_equity=self.initial_capital,
            total_return_pct=0,
            total_trades=0,
            win_rate=0,
            profit_factor=0,
            max_drawdown_pct=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            avg_trade_pnl=0,
            avg_win=0,
            avg_loss=0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            equity_curve=[self.initial_capital],
            trades=[],
        )
