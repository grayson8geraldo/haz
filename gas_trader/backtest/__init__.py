"""Backtesting and optimization — vectorized backtest, walk-forward, Optuna."""

from gas_trader.backtest.engine import BacktestEngine
from gas_trader.backtest.optimizer import StrategyOptimizer

__all__ = ["BacktestEngine", "StrategyOptimizer"]
