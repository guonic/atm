"""
Plotter for Teapot pattern recognition signals.

Generates visualization charts for individual signals.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import polars as pl

logger = logging.getLogger(__name__)

# Set Chinese font for matplotlib
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class TeapotPlotter:
    """
    Plotter for Teapot signals.

    Generates K-line charts with box lines, trap zones, and signal annotations.
    """

    def __init__(
        self,
        lookback_days: int = 100,
        forward_days: int = 20,
        figsize: Tuple[int, int] = (12, 8),
    ):
        """
        Initialize plotter.

        Args:
            lookback_days: Number of days to show before signal.
            forward_days: Number of days to show after signal.
            figsize: Figure size (width, height).
        """
        self.lookback_days = lookback_days
        self.forward_days = forward_days
        self.figsize = figsize

    def plot_signal(
        self,
        ts_code: str,
        signal_date: str,
        market_data: pl.DataFrame,
        signal_info: Dict,
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot single signal chart.

        Args:
            ts_code: Stock code.
            signal_date: Signal date (YYYY-MM-DD).
            market_data: Market data DataFrame.
            signal_info: Signal information dictionary.

        Returns:
            matplotlib Figure object.
        """
        # Filter stock data
        stock_data = market_data.filter(pl.col("ts_code") == ts_code).sort(
            "trade_date"
        )

        if stock_data.is_empty():
            logger.warning(f"No data found for {ts_code}")
            return None

        # Find signal date index
        signal_idx = None
        dates = stock_data["trade_date"].to_list()
        for idx, date in enumerate(dates):
            if str(date) == signal_date:
                signal_idx = idx
                break

        if signal_idx is None:
            logger.warning(f"Signal date {signal_date} not found in data")
            return None

        # Extract data window
        start_idx = max(0, signal_idx - self.lookback_days)
        end_idx = min(len(stock_data), signal_idx + self.forward_days + 1)

        plot_data = stock_data[start_idx:end_idx]
        plot_dates = plot_data["trade_date"].to_list()
        plot_closes = plot_data["close"].to_list()
        plot_highs = plot_data["high"].to_list()
        plot_lows = plot_data["low"].to_list()
        plot_volumes = plot_data["volume"].to_list()

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=self.figsize, height_ratios=[3, 1]
        )
        fig.suptitle(
            f"{ts_code} - Teapot Signal ({signal_date})",
            fontsize=14,
            fontweight="bold",
        )

        # Plot 1: Price chart
        ax1.plot(
            range(len(plot_dates)),
            plot_closes,
            label="Close Price",
            color="black",
            linewidth=1.5,
        )

        # Draw box lines
        box_h = signal_info.get("box_h")
        box_l = signal_info.get("box_l")
        if box_h and box_l:
            signal_plot_idx = signal_idx - start_idx
            ax1.axhline(
                y=box_h,
                color="green",
                linestyle="--",
                linewidth=2,
                label="Box Upper Bound",
                alpha=0.7,
            )
            ax1.axhline(
                y=box_l,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Box Lower Bound",
                alpha=0.7,
            )

            # Highlight box area
            ax1.axhspan(
                box_l,
                box_h,
                alpha=0.1,
                color="blue",
                label="Box Range",
            )

        # Mark signal point
        signal_plot_idx = signal_idx - start_idx
        if 0 <= signal_plot_idx < len(plot_closes):
            ax1.scatter(
                signal_plot_idx,
                plot_closes[signal_plot_idx],
                color="red",
                s=200,
                marker="^",
                label="Signal Point",
                zorder=5,
            )

        ax1.set_ylabel("Price (CNY)", fontsize=10)
        ax1.set_title("Price Chart with Box Lines", fontsize=12)
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Volume chart
        ax2.bar(
            range(len(plot_dates)),
            plot_volumes,
            color="blue",
            alpha=0.6,
            label="Volume",
        )

        # Highlight signal day volume
        if 0 <= signal_plot_idx < len(plot_volumes):
            ax2.bar(
                signal_plot_idx,
                plot_volumes[signal_plot_idx],
                color="red",
                alpha=0.8,
                label="Signal Day Volume",
            )

        ax2.set_xlabel("Trading Days", fontsize=10)
        ax2.set_ylabel("Volume", fontsize=10)
        ax2.set_title("Volume Chart", fontsize=12)
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        n_ticks = min(10, len(plot_dates))
        tick_indices = [
            int(i * (len(plot_dates) - 1) / (n_ticks - 1))
            for i in range(n_ticks)
        ]
        tick_labels = [str(plot_dates[i])[:10] for i in tick_indices]
        ax1.set_xticks(tick_indices)
        ax1.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax2.set_xticks(tick_indices)
        ax2.set_xticklabels(tick_labels, rotation=45, ha="right")

        plt.tight_layout()

        # Save if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Chart saved to {output_path}")

        return fig

    def plot_comparison(
        self,
        success_signals: list,
        failure_signals: list,
        market_data: pl.DataFrame,
        output_dir: Path,
        n_samples: int = 10,
    ) -> None:
        """
        Plot comparison charts (success vs failure patterns).

        Args:
            success_signals: List of successful signal dictionaries.
            failure_signals: List of failed signal dictionaries.
            market_data: Market data DataFrame.
            output_dir: Output directory for charts.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot top success signals
        success_dir = output_dir / "success"
        success_dir.mkdir(exist_ok=True)
        for i, signal in enumerate(success_signals[:n_samples]):
            ts_code = signal["ts_code"]
            signal_date = signal["signal_date"]
            output_path = success_dir / f"{ts_code}_{signal_date}.png"
            self.plot_signal(
                ts_code=ts_code,
                signal_date=signal_date,
                market_data=market_data,
                signal_info=signal,
                output_path=output_path,
            )
            plt.close()

        # Plot top failure signals
        failure_dir = output_dir / "failure"
        failure_dir.mkdir(exist_ok=True)
        for i, signal in enumerate(failure_signals[:n_samples]):
            ts_code = signal["ts_code"]
            signal_date = signal["signal_date"]
            output_path = failure_dir / f"{ts_code}_{signal_date}.png"
            self.plot_signal(
                ts_code=ts_code,
                signal_date=signal_date,
                market_data=market_data,
                signal_info=signal,
                output_path=output_path,
            )
            plt.close()

        logger.info(
            f"Comparison charts saved to {output_dir} "
            f"({len(success_signals[:n_samples])} success, "
            f"{len(failure_signals[:n_samples])} failure)"
        )
