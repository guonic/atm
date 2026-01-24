"""
Batch plotter for Teapot pattern recognition signals.

Generates visualization charts for multiple signals in parallel.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

from tools.visualization.teapot.plotter import TeapotPlotter

logger = logging.getLogger(__name__)


class BatchPlotter:
    """
    Batch plotter for Teapot signals.

    Generates charts for multiple signals in parallel.
    """

    def __init__(
        self,
        n_workers: int = 4,
        output_dir: Path = Path("outputs/teapot/visualizations"),
    ):
        """
        Initialize batch plotter.

        Args:
            n_workers: Number of worker processes.
            output_dir: Output directory for charts.
        """
        self.n_workers = n_workers
        self.output_dir = Path(output_dir)
        self.plotter = TeapotPlotter()

    def generate_all_plots(
        self,
        signals: pl.DataFrame,
        market_data: pl.DataFrame,
        evaluation_results: Optional[pl.DataFrame] = None,
    ) -> Dict[str, int]:
        """
        Generate all signal plots.

        Args:
            signals: Signals DataFrame.
            market_data: Market data DataFrame.
            evaluation_results: Optional evaluation results for classification.

        Returns:
            Dictionary with statistics (success_count, failure_count, etc.).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Classify signals if evaluation results provided
        success_signals = []
        failure_signals = []

        if evaluation_results is not None and not evaluation_results.is_empty():
            # Merge signals with evaluation results
            merged = signals.join(
                evaluation_results,
                on=["ts_code", "signal_date"],
                how="left",
            )

            # Classify by return_t20
            if "return_t20" in merged.columns:
                success_df = merged.filter(pl.col("return_t20") > 0)
                failure_df = merged.filter(
                    (pl.col("return_t20") <= 0)
                    | pl.col("return_t20").is_null()
                )

                success_signals = success_df.to_dicts()
                failure_signals = failure_df.to_dicts()
            else:
                # No evaluation results, treat all as unclassified
                signals_list = signals.to_dicts()
                success_signals = signals_list
        else:
            # No evaluation results, treat all as unclassified
            signals_list = signals.to_dicts()
            success_signals = signals_list

        # Create directories
        success_dir = self.output_dir / "success"
        failure_dir = self.output_dir / "failure"
        success_dir.mkdir(exist_ok=True)
        failure_dir.mkdir(exist_ok=True)

        # Generate plots in parallel
        success_count = self._generate_plots_parallel(
            success_signals, market_data, success_dir
        )
        failure_count = self._generate_plots_parallel(
            failure_signals, market_data, failure_dir
        )

        logger.info(
            f"Generated {success_count} success plots and "
            f"{failure_count} failure plots"
        )

        return {
            "success_count": success_count,
            "failure_count": failure_count,
            "total_count": success_count + failure_count,
        }

    def _generate_plots_parallel(
        self,
        signals: List[Dict],
        market_data: pl.DataFrame,
        output_dir: Path,
    ) -> int:
        """
        Generate plots in parallel.

        Args:
            signals: List of signal dictionaries.
            market_data: Market data DataFrame.
            output_dir: Output directory.

        Returns:
            Number of plots generated.
        """
        if not signals:
            return 0

        count = 0
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for signal in signals:
                ts_code = signal["ts_code"]
                signal_date = signal["signal_date"]
                output_path = output_dir / f"{ts_code}_{signal_date}.png"

                # Skip if already exists
                if output_path.exists():
                    continue

                future = executor.submit(
                    self._plot_single_signal,
                    ts_code,
                    signal_date,
                    signal,
                    market_data,
                    output_path,
                )
                futures.append(future)

            # Wait for completion
            for future in as_completed(futures):
                try:
                    success = future.result()
                    if success:
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to generate plot: {e}")

        return count

    def _plot_single_signal(
        self,
        ts_code: str,
        signal_date: str,
        signal_info: Dict,
        market_data: pl.DataFrame,
        output_path: Path,
    ) -> bool:
        """
        Plot single signal (for parallel processing).

        Args:
            ts_code: Stock code.
            signal_date: Signal date.
            signal_info: Signal information.
            market_data: Market data DataFrame.
            output_path: Output path.

        Returns:
            True if successful.
        """
        try:
            self.plotter.plot_signal(
                ts_code=ts_code,
                signal_date=signal_date,
                market_data=market_data,
                signal_info=signal_info,
                output_path=output_path,
            )
            return True
        except Exception as e:
            logger.warning(
                f"Failed to plot {ts_code} {signal_date}: {e}"
            )
            return False

    def generate_sample_plots(
        self,
        signals: pl.DataFrame,
        market_data: pl.DataFrame,
        n_samples: int = 100,
        random_seed: int = 42,
    ) -> None:
        """
        Generate sample plots for quick preview.

        Args:
            signals: Signals DataFrame.
            market_data: Market data DataFrame.
            n_samples: Number of samples.
            random_seed: Random seed.
        """
        # Sample signals
        if len(signals) > n_samples:
            sampled = signals.sample(n=n_samples, seed=random_seed)
        else:
            sampled = signals

        signals_list = sampled.to_dicts()

        sample_dir = self.output_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        self._generate_plots_parallel(signals_list, market_data, sample_dir)

        logger.info(f"Generated {len(signals_list)} sample plots")
