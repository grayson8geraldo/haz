#!/usr/bin/env python3
"""Streamlit dashboard for real-time P&L monitoring.

Run: streamlit run scripts/dashboard_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import streamlit as st
    import plotly.graph_objects as go
except ImportError:
    print("Install streamlit and plotly: pip install streamlit plotly")
    sys.exit(1)

from gas_trader.config import Settings
from gas_trader.data.market_data import MarketDataFeed
from gas_trader.backtest.engine import BacktestEngine


def main():
    st.set_page_config(page_title="Gas Futures Trader", layout="wide")
    st.title("Natural Gas Futures Trader — Dashboard")

    settings = Settings.load()

    tab1, tab2, tab3 = st.tabs(["Live Status", "Backtest", "Settings"])

    with tab1:
        st.header("Trading Status")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mode", settings.mode.upper())
        col2.metric("Initial Capital", f"${settings.initial_capital:.0f}")
        col3.metric("Target", f"${settings.target_capital:,.0f}")
        inst = settings.select_instrument(settings.initial_capital)
        col4.metric("Instrument", f"{inst.symbol} ({inst.exchange})")

        st.info(
            "Connect to live trading to see real-time P&L. "
            "Use `python main.py trade` to start."
        )

    with tab2:
        st.header("Backtest")

        col1, col2 = st.columns(2)
        days = col1.slider("History (days)", 30, 730, 365)
        timeframe = col2.selectbox("Timeframe", ["1h", "4h", "1d"], index=0)

        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                market = MarketDataFeed(settings)
                df = market.fetch_bars(timeframe=timeframe, days=days)
                engine = BacktestEngine(settings)
                result = engine.run(df)

            # Metrics
            st.subheader("Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Final Equity", f"${result.final_equity:.2f}")
            c2.metric("Return", f"{result.total_return_pct:+.1f}%")
            c3.metric("Win Rate", f"{result.win_rate:.1f}%")
            c4.metric("Sharpe", f"{result.sharpe_ratio:.2f}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Trades", result.total_trades)
            c6.metric("Profit Factor", f"{result.profit_factor:.2f}")
            c7.metric("Max DD", f"{result.max_drawdown_pct:.1f}%")
            c8.metric("Sortino", f"{result.sortino_ratio:.2f}")

            # Equity curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=result.equity_curve,
                mode="lines",
                name="Equity",
                line=dict(color="#00cc66", width=2),
            ))
            fig.update_layout(
                title="Equity Curve",
                yaxis_title="USD",
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Trade table
            if result.trades:
                import pandas as pd

                trades_data = []
                for t in result.trades:
                    trades_data.append({
                        "Entry": str(t.entry_time)[:16],
                        "Exit": str(t.exit_time)[:16],
                        "Dir": t.direction,
                        "Entry$": f"{t.entry_price:.4f}",
                        "Exit$": f"{t.exit_price:.4f}",
                        "P&L": f"${t.pnl:.2f}",
                        "Reason": t.exit_reason,
                    })
                st.dataframe(pd.DataFrame(trades_data), use_container_width=True)

    with tab3:
        st.header("Configuration")
        st.json(settings.raw)


if __name__ == "__main__":
    main()
