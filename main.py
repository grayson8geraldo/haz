#!/usr/bin/env python3
"""Gas Futures Trader — Entry point.

Usage:
    python main.py trade          # Start live/paper trading
    python main.py backtest       # Run backtest
    python main.py optimize       # Optimize strategy parameters
    python main.py dashboard      # Launch Streamlit dashboard
    python main.py status         # Show current status
"""

from __future__ import annotations

import sys

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("logs/trader_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days", level="DEBUG")


def cmd_trade():
    """Start the trading system."""
    from gas_trader.config import Settings
    from gas_trader.trader import GasFuturesTrader

    settings = Settings.load()
    trader = GasFuturesTrader(settings)

    print("\n" + "=" * 60)
    print("  NATURAL GAS FUTURES TRADER")
    print(f"  Mode: {settings.mode.upper()}")
    print(f"  Capital: ${settings.initial_capital:.2f}")
    print(f"  Target:  ${settings.target_capital:.2f}")
    inst = settings.select_instrument(settings.initial_capital)
    print(f"  Instrument: {inst.symbol} (margin: ${inst.margin_initial:.0f})")
    print("=" * 60 + "\n")

    if settings.mode == "live":
        print("WARNING: Running in LIVE mode!")
        confirm = input("Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Aborted.")
            return

    trader.start()


def cmd_backtest():
    """Run backtest on historical data."""
    from gas_trader.config import Settings
    from gas_trader.data.market_data import MarketDataFeed
    from gas_trader.backtest.engine import BacktestEngine

    settings = Settings.load()
    market = MarketDataFeed(settings)
    engine = BacktestEngine(settings)

    print("\nGenerating historical data for backtest...")
    df = market.fetch_bars(timeframe="1h", days=365)
    print(f"Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    print("\nRunning backtest...")
    result = engine.run(df)

    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Initial Capital:  ${result.initial_capital:.2f}")
    print(f"  Final Equity:     ${result.final_equity:.2f}")
    print(f"  Total Return:     {result.total_return_pct:+.1f}%")
    print(f"  Total Trades:     {result.total_trades}")
    print(f"  Win Rate:         {result.win_rate:.1f}%")
    print(f"  Profit Factor:    {result.profit_factor:.2f}")
    print(f"  Max Drawdown:     {result.max_drawdown_pct:.1f}%")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:    {result.sortino_ratio:.2f}")
    print(f"  Avg Win:          ${result.avg_win:.2f}")
    print(f"  Avg Loss:         ${result.avg_loss:.2f}")
    print(f"  Max Consec Wins:  {result.max_consecutive_wins}")
    print(f"  Max Consec Losses:{result.max_consecutive_losses}")
    print("=" * 60)

    # Walk-forward
    print("\nRunning walk-forward analysis...")
    wf_results = engine.walk_forward(df, train_months=6, test_months=2)
    if wf_results:
        avg_return = sum(r.total_return_pct for r in wf_results) / len(wf_results)
        avg_sharpe = sum(r.sharpe_ratio for r in wf_results) / len(wf_results)
        print(f"  Walk-Forward Folds: {len(wf_results)}")
        print(f"  Avg Return:         {avg_return:+.1f}%")
        print(f"  Avg Sharpe:         {avg_sharpe:.2f}")


def cmd_optimize():
    """Optimize strategy parameters."""
    from gas_trader.config import Settings
    from gas_trader.data.market_data import MarketDataFeed
    from gas_trader.backtest.optimizer import StrategyOptimizer

    settings = Settings.load()
    market = MarketDataFeed(settings)
    optimizer = StrategyOptimizer(settings)

    print("\nGenerating data for optimization...")
    df = market.fetch_bars(timeframe="1h", days=365)
    print(f"Data: {len(df)} bars")

    print(f"\nOptimizing ({optimizer.n_trials} trials, metric: {optimizer.metric})...")
    result = optimizer.optimize(df)

    print("\n" + "=" * 60)
    print("  OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"  Best {optimizer.metric}: {result['best_metric']:.3f}")
    print(f"  Parameters:")
    for k, v in result["best_params"].items():
        print(f"    {k}: {v:.3f}")
    if result.get("result"):
        r = result["result"]
        print(f"  Return: {r.total_return_pct:+.1f}%")
        print(f"  Win Rate: {r.win_rate:.1f}%")
        print(f"  Max DD: {r.max_drawdown_pct:.1f}%")
    print("=" * 60)


def cmd_dashboard():
    """Launch Streamlit dashboard."""
    import subprocess

    dashboard_path = "scripts/dashboard_app.py"
    print(f"Launching dashboard: streamlit run {dashboard_path}")
    subprocess.run(["streamlit", "run", dashboard_path, "--server.port", "8501"])


def cmd_status():
    """Show current configuration status."""
    from gas_trader.config import Settings

    settings = Settings.load()
    inst = settings.select_instrument(settings.initial_capital)

    print("\n" + "=" * 60)
    print("  GAS FUTURES TRADER — STATUS")
    print("=" * 60)
    print(f"  Mode:           {settings.mode}")
    print(f"  Capital:        ${settings.initial_capital:.2f}")
    print(f"  Target:         ${settings.target_capital:.2f}")
    print(f"  Instrument:     {inst.symbol} ({inst.exchange})")
    print(f"  Margin:         ${inst.margin_initial:.0f}")
    print(f"  Risk/Trade:     {settings.risk.risk_per_trade_pct}%")
    print(f"  Max Daily Loss: {settings.risk.max_daily_loss_pct}%")
    print(f"  Max Drawdown:   {settings.risk.max_drawdown_pct}%")
    print(f"  EIA Kill Switch:{settings.risk.eia_kill_switch}")
    print(f"  Signal Weights:")
    print(f"    Technical:    {settings.signal_weights.technical:.0%}")
    print(f"    Fundamental:  {settings.signal_weights.fundamental:.0%}")
    print(f"    ML:           {settings.signal_weights.ml:.0%}")
    print(f"    Seasonality:  {settings.signal_weights.seasonality:.0%}")
    print("=" * 60)


COMMANDS = {
    "trade": cmd_trade,
    "backtest": cmd_backtest,
    "optimize": cmd_optimize,
    "dashboard": cmd_dashboard,
    "status": cmd_status,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print(f"Available commands: {', '.join(COMMANDS)}")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
