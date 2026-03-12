"""Strategy optimizer — Optuna-based Bayesian optimization."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from gas_trader.config import Settings
from gas_trader.backtest.engine import BacktestEngine, BacktestResult


class StrategyOptimizer:
    """Optimizes strategy parameters using Optuna (Bayesian optimization)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = BacktestEngine(settings)
        opt_cfg = settings.raw["backtest"]["optimization"]
        self.n_trials = opt_cfg["n_trials"]
        self.metric = opt_cfg["metric"]  # sharpe | sortino | calmar

    def optimize(
        self,
        df: pd.DataFrame,
        n_trials: Optional[int] = None,
    ) -> dict:
        """Run Bayesian optimization to find best strategy parameters."""
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed — running grid search fallback")
            return self._grid_search(df)

        trials = n_trials or self.n_trials

        def objective(trial):
            params = {
                "signal_threshold": trial.suggest_float("signal_threshold", 0.10, 0.50),
                "atr_sl_mult": trial.suggest_float("atr_sl_mult", 1.0, 4.0),
                "atr_tp_mult": trial.suggest_float("atr_tp_mult", 1.5, 6.0),
                "risk_per_trade_pct": trial.suggest_float("risk_per_trade_pct", 0.5, 3.0),
            }

            result = self.engine.run(df, **params)

            if result.total_trades < 10:
                return -999  # Not enough trades

            if self.metric == "sharpe":
                return result.sharpe_ratio
            elif self.metric == "sortino":
                return result.sortino_ratio
            elif self.metric == "calmar":
                return result.calmar_ratio
            else:
                return result.sharpe_ratio

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials, show_progress_bar=True)

        best = study.best_params
        best_result = self.engine.run(df, **best)

        logger.info(
            f"Optimization complete ({trials} trials)\n"
            f"Best {self.metric}: {study.best_value:.3f}\n"
            f"Params: {best}\n"
            f"Return: {best_result.total_return_pct:.1f}% | "
            f"Win rate: {best_result.win_rate:.1f}% | "
            f"Max DD: {best_result.max_drawdown_pct:.1f}%"
        )

        return {
            "best_params": best,
            "best_metric": study.best_value,
            "result": best_result,
            "study": study,
        }

    def _grid_search(self, df: pd.DataFrame) -> dict:
        """Fallback grid search when Optuna is not available."""
        import itertools

        param_grid = {
            "signal_threshold": [0.15, 0.25, 0.35],
            "atr_sl_mult": [1.5, 2.0, 3.0],
            "atr_tp_mult": [2.0, 3.0, 4.5],
            "risk_per_trade_pct": [0.5, 1.0, 2.0],
        }

        keys = list(param_grid.keys())
        best_score = -999
        best_params = {}
        best_result = None

        combos = list(itertools.product(*param_grid.values()))
        logger.info(f"Grid search: {len(combos)} combinations")

        for combo in combos:
            params = dict(zip(keys, combo))
            result = self.engine.run(df, **params)

            if result.total_trades < 10:
                continue

            score = getattr(result, f"{self.metric}_ratio", result.sharpe_ratio)
            if score > best_score:
                best_score = score
                best_params = params
                best_result = result

        if best_result:
            logger.info(
                f"Grid search best {self.metric}: {best_score:.3f} | "
                f"Return: {best_result.total_return_pct:.1f}%"
            )

        return {
            "best_params": best_params,
            "best_metric": best_score,
            "result": best_result,
        }
