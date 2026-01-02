"""
EDiOS Visualization Module.

Provides Streamlit-based visualization for backtest attribution analysis.
"""

import streamlit as st
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nq.config import DatabaseConfig, load_config
from nq.repo.edios_repo import EdiosRepo


class EdiosVisualization:
    """Streamlit visualization for EDiOS backtest attribution."""

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        """
        Initialize visualization.

        Args:
            db_config: Database configuration. If None, loads from config file.
        """
        if db_config is None:
            config = load_config()
            db_config = config.database

        self.repo = EdiosRepo(db_config)

    def render_experiment_selector(self) -> Optional[str]:
        """
        Render experiment selector.

        Returns:
            Selected experiment ID or None.
        """
        # Get all experiments
        try:
            experiments = self.repo.experiment.list_experiments(limit=100)
        except Exception as e:
            st.error(f"Failed to load experiments: {e}")
            return None

        if not experiments:
            st.warning("No experiments found. Please create an experiment first by running a backtest with --enable_edios.")
            return None

        # Format experiment names
        exp_options = {}
        for exp in experiments:
            exp_id = str(exp.get("exp_id", ""))
            exp_name = exp.get("name", "Unnamed")
            exp_date = exp.get("start_date", "")
            display_name = f"{exp_name} ({exp_date})" if exp_date else exp_name
            exp_options[display_name] = exp_id

        if not exp_options:
            st.warning("No experiments found.")
            return None

        selected_name = st.selectbox("Select Experiment", list(exp_options.keys()))
        
        if selected_name:
            exp_id = exp_options[selected_name]
            # Display experiment info
            selected_exp = next((e for e in experiments if str(e.get("exp_id")) == exp_id), None)
            if selected_exp:
                with st.expander("Experiment Details", expanded=False):
                    st.json(selected_exp)
            return exp_id

        return None

    def render_performance_panel(self, exp_id: str) -> None:
        """
        Render performance metrics panel (Qlib style).

        Args:
            exp_id: Experiment ID.
        """
        st.header("Performance Metrics")

        # Get ledger data
        ledger_data = self.repo.ledger.get_ledger(exp_id)
        if not ledger_data:
            st.warning("No ledger data found.")
            return

        df_ledger = pd.DataFrame(ledger_data)

        # Calculate metrics
        if len(df_ledger) > 0:
            first_nav = df_ledger.iloc[0]["nav"]
            last_nav = df_ledger.iloc[-1]["nav"]
            total_return = (last_nav - first_nav) / first_nav if first_nav > 0 else 0

            # Calculate max drawdown
            df_ledger["cummax"] = df_ledger["nav"].cummax()
            df_ledger["drawdown"] = (
                (df_ledger["cummax"] - df_ledger["nav"]) / df_ledger["cummax"]
            )
            max_drawdown = df_ledger["drawdown"].max()

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{total_return:.2%}")
            with col2:
                st.metric("Max Drawdown", f"{max_drawdown:.2%}")
            with col3:
                st.metric("Final NAV", f"{last_nav:,.2f}")
            with col4:
                st.metric("Trading Days", len(df_ledger))

    def render_trade_stats_panel(self, exp_id: str) -> None:
        """
        Render trade statistics panel.

        Args:
            exp_id: Experiment ID.
        """
        st.header("Trade Statistics")

        # Get trades data
        trades_data = self.repo.trades.get_trades(exp_id)
        if not trades_data:
            st.warning("No trades data found.")
            return

        df_trades = pd.DataFrame(trades_data)

        # Filter sell trades with PnL
        df_sells = df_trades[
            (df_trades["side"] == -1) & (df_trades["pnl_ratio"].notna())
        ]

        if len(df_sells) > 0:
            winning_trades = len(df_sells[df_sells["pnl_ratio"] > 0])
            total_trades = len(df_sells)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            total_profit = df_sells[df_sells["pnl_ratio"] > 0]["pnl_ratio"].sum()
            total_loss = abs(df_sells[df_sells["pnl_ratio"] < 0]["pnl_ratio"].sum())
            profit_factor = total_profit / total_loss if total_loss > 0 else 0

            avg_profit = (
                df_sells[df_sells["pnl_ratio"] > 0]["pnl_ratio"].mean()
                if winning_trades > 0
                else 0
            )
            avg_loss = (
                abs(df_sells[df_sells["pnl_ratio"] < 0]["pnl_ratio"].mean())
                if total_trades - winning_trades > 0
                else 0
            )
            profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Win Rate", f"{win_rate:.2%}")
            with col2:
                st.metric("Profit Factor", f"{profit_factor:.2f}")
            with col3:
                st.metric("Profit/Loss Ratio", f"{profit_loss_ratio:.2f}")

    def render_nav_chart(self, exp_id: str) -> None:
        """
        Render NAV chart with trade markers.

        Args:
            exp_id: Experiment ID.
        """
        st.header("NAV Chart")

        # Get ledger data
        ledger_data = self.repo.ledger.get_ledger(exp_id)
        if not ledger_data:
            st.warning("No ledger data found.")
            return

        df_ledger = pd.DataFrame(ledger_data)
        df_ledger["date"] = pd.to_datetime(df_ledger["date"])
        # Normalize to timezone-naive for comparison
        if df_ledger["date"].dt.tz is not None:
            df_ledger["date"] = df_ledger["date"].dt.tz_localize(None)

        # Get trades data
        trades_data = self.repo.trades.get_trades(exp_id)
        df_trades = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()
        if not df_trades.empty:
            df_trades["deal_time"] = pd.to_datetime(df_trades["deal_time"])
            # Normalize to timezone-naive for comparison
            if df_trades["deal_time"].dt.tz is not None:
                df_trades["deal_time"] = df_trades["deal_time"].dt.tz_localize(None)

        # Create chart
        fig = make_subplots(specs=[[{"secondary_y": False}]])

        # Add NAV line
        fig.add_trace(
            go.Scatter(
                x=df_ledger["date"],
                y=df_ledger["nav"],
                mode="lines",
                name="NAV",
                line=dict(color="blue", width=2),
            )
        )

        # Add buy markers
        if not df_trades.empty:
            df_buys = df_trades[df_trades["side"] == 1]
            if not df_buys.empty:
                # Get NAV at buy time
                buy_navs = []
                for buy_time in df_buys["deal_time"]:
                    # Ensure buy_time is timezone-naive for comparison
                    if isinstance(buy_time, pd.Timestamp) and buy_time.tz is not None:
                        buy_time = buy_time.tz_localize(None)
                    nav_at_time = df_ledger[df_ledger["date"] <= buy_time]["nav"]
                    if not nav_at_time.empty:
                        buy_navs.append(nav_at_time.iloc[-1])
                    else:
                        buy_navs.append(None)

                fig.add_trace(
                    go.Scatter(
                        x=df_buys["deal_time"],
                        y=buy_navs,
                        mode="markers",
                        name="Buy",
                        marker=dict(color="green", size=10, symbol="triangle-up"),
                    )
                )

            # Add sell markers
            df_sells = df_trades[df_trades["side"] == -1]
            if not df_sells.empty:
                sell_navs = []
                for sell_time in df_sells["deal_time"]:
                    # Ensure sell_time is timezone-naive for comparison
                    if isinstance(sell_time, pd.Timestamp) and sell_time.tz is not None:
                        sell_time = sell_time.tz_localize(None)
                    nav_at_time = df_ledger[df_ledger["date"] <= sell_time]["nav"]
                    if not nav_at_time.empty:
                        sell_navs.append(nav_at_time.iloc[-1])
                    else:
                        sell_navs.append(None)

                fig.add_trace(
                    go.Scatter(
                        x=df_sells["deal_time"],
                        y=sell_navs,
                        mode="markers",
                        name="Sell",
                        marker=dict(color="red", size=10, symbol="triangle-down"),
                    )
                )

        fig.update_layout(
            title="Net Asset Value (NAV) Chart",
            xaxis_title="Date",
            yaxis_title="NAV",
            hovermode="x unified",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_rank_trajectory(self, exp_id: str, symbol: str) -> None:
        """
        Render rank trajectory chart (GNN feature).

        Args:
            exp_id: Experiment ID.
            symbol: Symbol to analyze.
        """
        st.header(f"Rank Trajectory: {symbol}")

        # Get model outputs for symbol
        outputs_data = self.repo.model_outputs.get_outputs(
            exp_id, symbol=symbol
        )
        if not outputs_data:
            st.warning("No model outputs found for this symbol.")
            return

        df_outputs = pd.DataFrame(outputs_data)
        df_outputs["date"] = pd.to_datetime(df_outputs["date"])

        # Create chart
        fig = go.Figure()

        # Add rank line
        fig.add_trace(
            go.Scatter(
                x=df_outputs["date"],
                y=df_outputs["rank"],
                mode="lines+markers",
                name="Rank",
                line=dict(color="purple", width=2),
            )
        )

        fig.update_layout(
            title=f"Rank Trajectory for {symbol}",
            xaxis_title="Date",
            yaxis_title="Rank",
            yaxis=dict(autorange="reversed"),  # Lower rank is better
            hovermode="x unified",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    def run_streamlit_app(self) -> None:
        """Run the Streamlit application."""
        st.set_page_config(page_title="EDiOS Backtest Attribution", layout="wide")

        st.title("EDiOS: Universal Backtest Attribution System")

        # Experiment selector
        exp_id = self.render_experiment_selector()

        if exp_id:
            # Performance panel
            self.render_performance_panel(exp_id)

            # Trade stats panel
            self.render_trade_stats_panel(exp_id)

            # NAV chart
            self.render_nav_chart(exp_id)

            # Rank trajectory (if symbol selected)
            symbol = st.text_input("Enter symbol for rank trajectory analysis")
            if symbol:
                self.render_rank_trajectory(exp_id, symbol)


def main():
    """Main entry point for Streamlit app."""
    viz = EdiosVisualization()
    viz.run_streamlit_app()


if __name__ == "__main__":
    main()

