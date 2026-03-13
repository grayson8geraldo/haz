#!/usr/bin/env python3
"""Gas Futures Trader — Entry point.

Usage:
    python main.py trade          # Start paper trading with real data
    python main.py backtest       # Run backtest on historical data
    python main.py optimize       # Optimize strategy parameters
    python main.py dashboard      # Launch Streamlit dashboard
    python main.py status         # Show current status + paper trading P&L
    python main.py reset          # Reset paper trading state (start fresh)
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

# Ensure logs dir exists
Path("logs").mkdir(exist_ok=True)

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("logs/trader_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days", level="DEBUG")


def cmd_trade():
    """Start the trading system with real data + virtual balance."""
    from gas_trader.config import Settings
    from gas_trader.trader import GasFuturesTrader

    settings = Settings.load()
    trader = GasFuturesTrader(settings)

    inst = settings.select_instrument(trader.equity)

    print("\n" + "=" * 60)
    print("  NATURAL GAS FUTURES TRADER")
    print("=" * 60)
    print(f"  Mode:        {settings.mode.upper()}")
    print(f"  Balance:     ${trader.equity:.2f}")
    if trader.equity != settings.initial_capital:
        pnl = trader.equity - settings.initial_capital
        print(f"  Total P&L:   ${pnl:+.2f} ({pnl / settings.initial_capital * 100:+.1f}%)")
        print(f"  Trades:      {trader.paper_state.total_trades}")
    print(f"  Target:      ${settings.target_capital:,.2f}")
    print(f"  Instrument:  {inst.symbol} (margin: ${inst.margin_initial:.0f})")
    print(f"  Data Source:  auto (IBKR -> Yahoo Finance -> Synthetic)")
    print("=" * 60)

    if settings.mode == "live":
        print("\nWARNING: Running in LIVE mode with REAL money!")
        confirm = input("Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Aborted.")
            return
    else:
        print("\n  Paper trading — real prices, virtual balance.")
        print("  Press Ctrl+C to stop. State auto-saves.\n")

    trader.start()


def cmd_backtest():
    """Run backtest on historical data (real if yfinance available)."""
    from gas_trader.config import Settings
    from gas_trader.data.market_data import MarketDataFeed
    from gas_trader.backtest.engine import BacktestEngine

    settings = Settings.load()
    market = MarketDataFeed(settings)

    # Connect to get real data if possible
    source = market.connect()

    print(f"\nFetching data ({source})...")
    df = market.fetch_bars(timeframe="1h", days=365)
    print(f"Data: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
    if source != "synthetic":
        print(f"Latest NG price: ${df['close'].iloc[-1]:.4f}")

    engine = BacktestEngine(settings)
    print("\nRunning backtest...")
    result = engine.run(df)

    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS (data: {source})")
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
    """Optimize strategy parameters using real data."""
    from gas_trader.config import Settings
    from gas_trader.data.market_data import MarketDataFeed
    from gas_trader.backtest.optimizer import StrategyOptimizer

    settings = Settings.load()
    market = MarketDataFeed(settings)
    source = market.connect()

    print(f"\nFetching data ({source}) for optimization...")
    df = market.fetch_bars(timeframe="1h", days=365)
    print(f"Data: {len(df)} bars")

    optimizer = StrategyOptimizer(settings)
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
    """Show current configuration + paper trading state."""
    from gas_trader.config import Settings
    from gas_trader.data.market_data import MarketDataFeed
    from gas_trader.paper_state import PaperState

    settings = Settings.load()
    inst = settings.select_instrument(settings.initial_capital)

    print("\n" + "=" * 60)
    print("  GAS FUTURES TRADER — STATUS")
    print("=" * 60)

    # Config
    print(f"  Mode:           {settings.mode}")
    print(f"  Initial Capital:${settings.initial_capital:.2f}")
    print(f"  Target:         ${settings.target_capital:,.2f}")
    print(f"  Instrument:     {inst.symbol} ({inst.exchange})")
    print(f"  Margin:         ${inst.margin_initial:.0f}")
    print(f"  Risk/Trade:     {settings.risk.risk_per_trade_pct}%")
    print(f"  Max Daily Loss: {settings.risk.max_daily_loss_pct}%")
    print(f"  Max Drawdown:   {settings.risk.max_drawdown_pct}%")
    print(f"  EIA Kill Switch:{settings.risk.eia_kill_switch}")

    # Data source check
    print(f"\n  Data Sources:")
    market = MarketDataFeed(settings)
    source = market.connect()
    print(f"    Market data:  {source.upper()}", end="")
    if market.is_real_data:
        quote = market.get_realtime_quote()
        if quote.get("price", 0) > 0:
            print(f" — NG = ${quote['price']:.4f}")
        else:
            print(" (connected)")
    else:
        print(" (no real data — install yfinance)")

    eia_ok = bool(settings.raw["data"]["eia"]["api_key"])
    print(f"    EIA Storage:  {'OK' if eia_ok else 'NO KEY (using synthetic)'}")
    print(f"    Weather/HDD:  Open-Meteo (free, no key needed)")

    tg_ok = bool(settings.raw["monitoring"]["telegram"]["bot_token"])
    print(f"    Telegram:     {'OK' if tg_ok else 'NOT CONFIGURED'}")

    # Paper trading state
    state = PaperState.load(settings.initial_capital)
    if state.total_trades > 0:
        print(f"\n  Paper Trading Progress:")
        print(f"    Balance:      ${state.equity:.2f}")
        print(f"    Total P&L:    ${state.total_pnl:+.2f} ({state.return_pct:+.1f}%)")
        print(f"    Trades:       {state.total_trades}")
        print(f"    Win Rate:     {state.win_rate:.1f}%")
        print(f"    Max Drawdown: {state.max_drawdown_pct:.1f}%")
        print(f"    Sessions:     {state.session_count}")
        print(f"    Started:      {state.started_at[:19]}")
        print(f"    Last Update:  {state.last_updated[:19]}")
    else:
        print(f"\n  No paper trading history yet. Run: python main.py trade")

    print("=" * 60)

    market.disconnect()


def cmd_reset():
    """Reset paper trading state."""
    from gas_trader.paper_state import PaperState

    state = PaperState.load(200)
    if state.total_trades > 0:
        print(f"\nCurrent paper trading state:")
        print(f"  Balance:  ${state.equity:.2f}")
        print(f"  Trades:   {state.total_trades}")
        print(f"  P&L:      ${state.total_pnl:+.2f}")
        print()
        confirm = input("Reset all paper trading data? Type 'RESET' to confirm: ")
        if confirm == "RESET":
            PaperState.reset()
            print("Paper trading state cleared. Next run starts fresh.")
        else:
            print("Cancelled.")
    else:
        print("No paper trading state to reset.")


COMMANDS = {
    "trade": cmd_trade,
    "backtest": cmd_backtest,
    "optimize": cmd_optimize,
    "dashboard": cmd_dashboard,
    "status": cmd_status,
    "reset": cmd_reset,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print(f"Available commands: {', '.join(COMMANDS)}")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
