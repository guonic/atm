#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_structure_expert.py

Description:
    Backtest script for Structure Expert GNN model.
    This script demonstrates how to:
    1. Load a trained Structure Expert model (.pth file)
    2. Generate predictions for each trading day
    3. Convert predictions to Qlib format signals
    4. Run backtest using Qlib's backtest framework
    5. Display backtest results and metrics

Usage:
    # Basic usage
    python backtest_structure_expert.py \
        --model_path models/structure_expert.pth \
        --start_date 2024-01-01 \
        --end_date 2024-06-30

    # Custom portfolio strategy
    python backtest_structure_expert.py \
        --model_path models/structure_expert.pth \
        --start_date 2024-01-01 \
        --end_date 2024-06-30 \
        --top_k 30 \
        --initial_cash 1000000

Arguments:
    --model_path         Path to trained model file (.pth)
    --start_date         Start date of backtest period (YYYY-MM-DD)
    --end_date           End date of backtest period (YYYY-MM-DD)
    --top_k              Number of top stocks to select (default: 30)
    --strategy           Portfolio strategy class name (default: TopkDropoutStrategy)
    --initial_cash       Initial cash amount (default: 1000000)
    --save_results       Save backtest results to file (default: False)
    --qlib_dir           Qlib data directory (default: ~/.qlib/qlib_data/cn_data)
    --region             Qlib region (default: cn)
    --n_feat             Number of input features (default: 158 for Alpha158)
    --n_hidden           Hidden layer size (default: 128)
    --n_heads            Number of attention heads (default: 8)
    --device             Device to use (default: cuda if available, else cpu)
    --config_path        Path to config file (optional, for database config)

Output:
    - Prints backtest metrics and portfolio performance
    - Optionally saves results to file
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Gymnasium compatibility patch for Qlib
# Qlib's RL module imports 'gym', but gym is unmaintained.
# Gymnasium is the maintained drop-in replacement.
# This patch allows Qlib to use gymnasium if installed.
try:
    import gymnasium as gym
    sys.modules['gym'] = gym
    # Also patch gym.spaces if it's accessed directly
    import gymnasium.spaces as spaces
    sys.modules['gym.spaces'] = spaces
    logging.getLogger(__name__).info("Patched 'gym' import to use 'gymnasium'.")
except ImportError:
    # If gymnasium is not installed, try to import gym as fallback
    try:
        import gym
        logging.getLogger(__name__).warning(
            "Using deprecated 'gym' package. "
            "Please install 'gymnasium' (pip install gymnasium) for better compatibility."
        )
    except ImportError:
        raise ImportError(
            "Neither 'gymnasium' nor 'gym' is installed. "
            "Qlib's backtest module requires one of them. "
            "Please install gymnasium (recommended):\n"
            "  pip install gymnasium\n"
            "Or install gym (deprecated):\n"
            "  pip install gym"
        )

import numpy as np
import pandas as pd
import qlib
import torch
from qlib.backtest import backtest, executor
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.data import D
from sqlalchemy import text

# Matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import font_manager
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nq.config import DatabaseConfig, load_config
from nq.repo.stock_repo import StockIndustryMemberRepo

# Import structure expert model
tools_train_dir = Path(__file__).parent.parent / "tools" / "qlib" / "train"
if str(tools_train_dir) not in sys.path:
    sys.path.insert(0, str(tools_train_dir))

from structure_expert import (
    GraphDataBuilder,
    StructureExpertGNN,
    StructureTrainer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_TOP_K = 30
DEFAULT_INITIAL_CASH = 1000000
DEFAULT_QLIB_DIR = "~/.qlib/qlib_data/cn_data"
DEFAULT_N_FEAT = 158  # Alpha158 features
DEFAULT_N_HIDDEN = 128
DEFAULT_N_HEADS = 8


def convert_ts_code_to_qlib_format(ts_code: str) -> str:
    """Convert ts_code to Qlib format."""
    return ts_code


def normalize_benchmark_code(benchmark: str) -> str:
    """
    Normalize benchmark code to Qlib format.
    
    Qlib uses format: code.(sh|sz|bj), e.g., '000300.SH', '399001.SZ'
    Common input formats:
    - 'SH000300' -> '000300.SH'
    - '000300.SH' -> '000300.SH' (already correct)
    - 'CSI300' -> try to convert or return as is
    
    Args:
        benchmark: Benchmark code in various formats.
    
    Returns:
        Normalized benchmark code in Qlib format.
    """
    if not benchmark:
        return benchmark
    
    benchmark = benchmark.strip().upper()
    
    # If already in correct format (code.EXCHANGE), return as is
    if "." in benchmark:
        parts = benchmark.split(".")
        if len(parts) == 2 and parts[1] in ["SH", "SZ", "BJ"]:
            return benchmark
    
    # Try to convert formats like 'SH000300' to '000300.SH'
    if benchmark.startswith("SH"):
        code = benchmark[2:]  # Remove 'SH' prefix
        return f"{code}.SH"
    elif benchmark.startswith("SZ"):
        code = benchmark[2:]  # Remove 'SZ' prefix
        return f"{code}.SZ"
    elif benchmark.startswith("BJ"):
        code = benchmark[2:]  # Remove 'BJ' prefix
        return f"{code}.BJ"
    
    # If starts with number, assume it's a code and try to determine exchange
    if benchmark[0].isdigit():
        if benchmark.startswith("6"):
            return f"{benchmark}.SH"
        elif benchmark.startswith(("0", "3")):
            return f"{benchmark}.SZ"
    
    # Return as is if cannot determine
    return benchmark


def load_industry_map(
    db_config: DatabaseConfig, target_date: Optional[datetime] = None, schema: str = "quant"
) -> Dict[str, int]:
    """
    Load industry mapping from database.

    Args:
        db_config: Database configuration.
        target_date: Target date for industry membership (default: current date).
        schema: Database schema name.

    Returns:
        Dictionary mapping stock codes (in Qlib format) to industry IDs.
        Format: {stock_code: industry_id}
    """
    repo = StockIndustryMemberRepo(db_config, schema=schema)

    if target_date is None:
        target_date = datetime.now()

    # Get all current industry members
    engine = repo._get_engine()
    table_name = repo._get_full_table_name()

    sql = f"""
    SELECT DISTINCT ts_code, l3_code
    FROM {table_name}
    WHERE (out_date IS NULL OR out_date > :target_date)
      AND in_date <= :target_date
    ORDER BY ts_code
    """

    with engine.connect() as conn:
        result = conn.execute(
            text(sql),
            {"target_date": target_date.date()},
        )
        rows = result.fetchall()

    # Convert to Qlib format and create mapping
    industry_codes = sorted(set(row[1] for row in rows))
    industry_id_map = {code: idx for idx, code in enumerate(industry_codes)}

    # Create stock_code -> industry_id mapping
    industry_map = {}
    for ts_code, l3_code in rows:
        qlib_code = convert_ts_code_to_qlib_format(ts_code)
        industry_map[qlib_code] = industry_id_map[l3_code]

    logger.info(f"Loaded industry mapping: {len(industry_map)} stocks, {len(industry_codes)} industries")
    return industry_map


def load_model(
    model_path: str,
    n_feat: int = DEFAULT_N_FEAT,
    n_hidden: int = DEFAULT_N_HIDDEN,
    n_heads: int = DEFAULT_N_HEADS,
    device: str = "cuda",
) -> StructureExpertGNN:
    """
    Load trained Structure Expert model.

    Args:
        model_path: Path to model file (.pth).
        n_feat: Number of input features.
        n_hidden: Hidden layer size.
        n_heads: Number of attention heads.
        device: Device to load model on.

    Returns:
        Loaded model in evaluation mode.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from {model_path}")
    logger.info(f"Model parameters: n_feat={n_feat}, n_hidden={n_hidden}, n_heads={n_heads}")

    # Initialize model
    model = StructureExpertGNN(n_feat=n_feat, n_hidden=n_hidden, n_heads=n_heads)

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move to device and set to eval mode (inference only, no training)
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(device_obj)
    model.eval()  # Set to evaluation mode - no gradient computation, no training

    logger.info(f"Model loaded successfully on {device_obj} (evaluation mode - inference only)")
    return model


def generate_predictions(
    model: StructureExpertGNN,
    builder: GraphDataBuilder,
    start_date: str,
    end_date: str,
    device: str = "cuda",
    save_embeddings: bool = False,
    embeddings_storage_dir: Optional[str] = None,
    industry_label_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Generate predictions for all trading days in the date range.

    Args:
        model: Trained Structure Expert model.
        builder: GraphDataBuilder instance.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        device: Device to run inference on.

    Returns:
        DataFrame with predictions in Qlib format.
        Index: MultiIndex (datetime, instrument)
        Column: 'score' (prediction score)
    """
    logger.info(f"Generating predictions from {start_date} to {end_date}")
    logger.info("Note: This is inference only, not training. Model weights are frozen.")

    # Get trading calendar
    calendar = D.calendar(start_time=start_date, end_time=end_date)
    if len(calendar) == 0:
        raise ValueError(f"No trading days found between {start_date} and {end_date}")

    logger.info(f"Found {len(calendar)} trading days to process")
    
    # Get full calendar to determine data range for fit period
    full_calendar = D.calendar()
    # Convert to date objects for consistent comparison
    if len(full_calendar) > 0:
        data_start_ts = full_calendar[0]
        data_end_ts = full_calendar[-1]
        # Convert Timestamp to date
        if isinstance(data_start_ts, pd.Timestamp):
            data_start_date = data_start_ts.date()
        elif hasattr(data_start_ts, 'date'):
            data_start_date = data_start_ts.date()
        else:
            data_start_date = data_start_ts
        
        if isinstance(data_end_ts, pd.Timestamp):
            data_end_date = data_end_ts.date()
        elif hasattr(data_end_ts, 'date'):
            data_end_date = data_end_ts.date()
        else:
            data_end_date = data_end_ts
    else:
        data_start_date = None
        data_end_date = None
    logger.debug(f"Qlib data range: {data_start_date} to {data_end_date}")

    # Alpha158 requires historical data to calculate technical indicators (MA, RSI, etc.)
    # We need to provide a lookback window (e.g., 60 days) before each prediction date
    LOOKBACK_DAYS = 60  # Number of historical days needed for feature calculation
    
    all_predictions = []
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    
    # Track statistics
    skipped_days = []
    successful_days = []

    for i, trade_date in enumerate(calendar):
        trade_date_str = trade_date.strftime("%Y-%m-%d")
        logger.info(f"Processing {trade_date_str} ({i+1}/{len(calendar)})")

        try:
            # Calculate lookback start date (need historical data for Alpha158 features)
            # Alpha158 needs ~30-60 days of history to calculate technical indicators
            from datetime import timedelta
            lookback_start = trade_date - timedelta(days=LOOKBACK_DAYS)
            lookback_start_str = lookback_start.strftime("%Y-%m-%d")
            
            logger.debug(f"Loading data from {lookback_start_str} to {trade_date_str} for feature calculation...")
            
            # Get available instruments from file
            # D.instruments() returns a dict, so we read from instruments/all.txt
            # Format: stock_code\tstart_date\tend_date (tab-separated)
            instruments = None
            try:
                from pathlib import Path
                # Get qlib_dir from the function's closure or use default
                qlib_dir_path = Path(qlib_dir if 'qlib_dir' in globals() else '~/.qlib/qlib_data/cn_data').expanduser()
                instruments_file = qlib_dir_path / 'instruments' / 'all.txt'
                if instruments_file.exists():
                    with open(instruments_file, 'r') as f:
                        # Parse tab-separated format: stock_code\tstart_date\tend_date
                        instruments = []
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            # Split by tab and take first part (stock code)
                            parts = line.split('\t')
                            if len(parts) >= 1:
                                stock_code = parts[0].strip()
                                if stock_code:
                                    instruments.append(stock_code)
                    logger.debug(f"Loaded {len(instruments)} instruments from file")
                else:
                    logger.warning(f"Instruments file not found: {instruments_file}")
            except Exception as e:
                logger.debug(f"Could not load instruments: {e}")
                instruments = None
            
            # Alpha158 needs fit_start_time and fit_end_time to initialize processors
            # These are used to calculate statistics (mean, std) for normalization
            # We'll use a period before the prediction date for fitting
            # Calculate fit period: use data from 1 year before lookback_start to lookback_start
            # Convert Timestamp to date for timedelta operations
            if isinstance(lookback_start, pd.Timestamp):
                lookback_start_date = lookback_start.date()
            elif hasattr(lookback_start, 'date'):
                lookback_start_date = lookback_start.date()
            else:
                lookback_start_date = lookback_start
            
            fit_end_date = lookback_start_date - timedelta(days=1)  # End before lookback starts
            fit_start_date = fit_end_date - timedelta(days=365)  # 1 year of data for fitting
            
            # Ensure fit period is within available data range
            if data_start_date is not None:
                # data_start_date is already a date object (from .date() call above)
                if fit_start_date < data_start_date:
                    logger.debug(f"Fit start {fit_start_date} is before data start {data_start_date}, adjusting...")
                    fit_start_date = data_start_date
                if fit_end_date < data_start_date:
                    # If even fit_end is before data start, use a minimal fit period
                    logger.warning(f"Fit end {fit_end_date} is before data start {data_start_date}, using minimal fit period")
                    fit_start_date = data_start_date
                    # Use at least 30 days for fit if possible
                    fit_calendar = D.calendar(start_time=data_start_date.strftime("%Y-%m-%d"), end_time=lookback_start_str)
                    if len(fit_calendar) > 30:
                        fit_end_ts = fit_calendar[-30]
                        fit_end_date = fit_end_ts.date() if hasattr(fit_end_ts, 'date') else fit_end_ts
                    else:
                        if len(fit_calendar) > 0:
                            fit_end_ts = fit_calendar[-1]
                            fit_end_date = fit_end_ts.date() if hasattr(fit_end_ts, 'date') else fit_end_ts
                        else:
                            fit_end_date = lookback_start_date
            
            fit_start_str = fit_start_date.strftime("%Y-%m-%d")
            fit_end_str = fit_end_date.strftime("%Y-%m-%d")
            
            logger.debug(f"Alpha158 fit period: {fit_start_str} to {fit_end_str}")
            logger.debug(f"Alpha158 inference period: {lookback_start_str} to {trade_date_str}")
            
            # Try with lookback first
            try:
                handler_kwargs = {
                    "start_time": lookback_start_str,
                    "end_time": trade_date_str,
                    "fit_start_time": fit_start_str,
                    "fit_end_time": fit_end_str,
                }
                # Add instruments if available (may help with data loading)
                if instruments is not None:
                    handler_kwargs["instruments"] = instruments
                
                date_handler = Alpha158(**handler_kwargs)
                date_handler.setup_data()
                
                # Get features - use handler.data or fetch
                try:
                    df_x = date_handler.data
                    if df_x is None or df_x.empty:
                        df_x = date_handler.fetch(col_set="feature")
                except Exception:
                    df_x = date_handler.fetch(col_set="feature")
            except Exception as e:
                logger.debug(f"Failed to load with lookback ({lookback_start_str} to {trade_date_str}): {e}")
                # If lookback fails, try without lookback (use only the target date)
                # But still need fit period
                logger.debug(f"Trying without lookback (using only {trade_date_str})...")
                handler_kwargs = {
                    "start_time": trade_date_str,
                    "end_time": trade_date_str,
                    "fit_start_time": fit_start_str,
                    "fit_end_time": fit_end_str,
                }
                if instruments is not None:
                    handler_kwargs["instruments"] = instruments
                
                date_handler = Alpha158(**handler_kwargs)
                date_handler.setup_data()
                try:
                    df_x = date_handler.data
                    if df_x is None or df_x.empty:
                        df_x = date_handler.fetch(col_set="feature")
                except Exception:
                    df_x = date_handler.fetch(col_set="feature")
            
            if df_x.empty:
                # Try to get more diagnostic information
                try:
                    # Check if there are any instruments
                    instruments = D.instruments()
                    logger.debug(f"Total instruments available: {len(instruments)}")
                    
                    # Try to load raw data for a sample stock
                    if len(instruments) > 0:
                        sample_stock = instruments[0]
                        sample_data = D.features(
                            [sample_stock],
                            ["$close"],
                            start_time=trade_date_str,
                            end_time=trade_date_str,
                            freq="day",
                        )
                        logger.debug(f"Sample stock {sample_stock} data for {trade_date_str}: {sample_data.shape}")
                except Exception as diag_e:
                    logger.debug(f"Diagnostic check failed: {diag_e}")
                
                logger.warning(
                    f"No data for {trade_date_str} (with lookback from {lookback_start_str}). "
                    f"Alpha158 returned empty DataFrame. This may indicate: "
                    f"1) No data available for this date, "
                    f"2) Insufficient lookback data, or "
                    f"3) Alpha158 feature calculation failed."
                )
                skipped_days.append(trade_date_str)
                continue
            
            # Filter to only the target date (Alpha158 may return multiple dates)
            if isinstance(df_x.index, pd.MultiIndex):
                date_level = df_x.index.get_level_values(0)
                # Filter to target date
                if isinstance(date_level[0], pd.Timestamp):
                    df_x = df_x.loc[date_level.date == trade_date.date()]
                else:
                    # Try string comparison
                    date_strs = pd.to_datetime(date_level).dt.strftime("%Y-%m-%d")
                    df_x = df_x.loc[date_strs == trade_date_str]
            
            if df_x.empty:
                logger.warning(f"No data for target date {trade_date_str} after filtering, skipping")
                skipped_days.append(trade_date_str)
                continue

            # Clean NaN/Inf
            df_x = df_x.fillna(0.0).replace([np.inf, -np.inf], 0.0)

            # Build graph
            daily_graph = builder.get_daily_graph(df_x, None)

            # Skip if no stocks
            if daily_graph.x.shape[0] == 0:
                logger.warning(f"No stocks for {trade_date_str}, skipping")
                continue

            # Get predictions and embeddings (inference only - no gradients, no training)
            logger.debug(f"Running inference for {trade_date_str}...")
            with torch.no_grad():  # Disable gradient computation for inference
                data = daily_graph.to(device_obj)
                pred, embedding = model(data.x, data.edge_index)  # Forward pass only
                pred = pred.cpu().numpy().flatten()
                embedding_np = embedding.cpu().numpy()

            # Get stock symbols
            if hasattr(daily_graph, "symbols"):
                symbols = daily_graph.symbols
            else:
                # Fallback: use index from df_x
                symbols = df_x.index.get_level_values("instrument").unique().tolist()
            
            # Save embeddings if requested
            if save_embeddings:
                try:
                    from pathlib import Path
                    import pandas as pd
                    
                    storage_dir = Path(embeddings_storage_dir) if embeddings_storage_dir else Path("storage/structure_expert_cache")
                    storage_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create DataFrame with embeddings
                    df_emb = pd.DataFrame({
                        'symbol': symbols,
                        'score': pred,
                        'industry': [industry_label_map.get(s, "Unknown") if industry_label_map else "Unknown" for s in symbols],
                    })
                    
                    # Add embedding columns
                    for i in range(embedding_np.shape[1]):
                        df_emb[f'embedding_{i}'] = embedding_np[:, i]
                    
                    # Save to parquet
                    date_str = trade_date_str.replace("-", "")
                    output_path = storage_dir / f"embeddings_{date_str}.parquet"
                    df_emb.to_parquet(output_path, index=False)
                    logger.info(f"Saved embeddings to {output_path}")
                except Exception as e:
                    logger.warning(f"Failed to save embeddings for {trade_date_str}: {e}")

            # Create prediction DataFrame for this date
            if len(symbols) != len(pred):
                logger.warning(
                    f"Symbol count ({len(symbols)}) != prediction count ({len(pred)}) "
                    f"for {trade_date_str}, skipping"
                )
                continue

            pred_df = pd.DataFrame(
                {
                    "score": pred,
                },
                index=pd.MultiIndex.from_product(
                    [[trade_date], symbols],
                    names=["datetime", "instrument"],
                ),
            )
            all_predictions.append(pred_df)
            successful_days.append(trade_date_str)
            logger.debug(f"✓ Successfully generated predictions for {trade_date_str}: {len(symbols)} stocks")

        except Exception as e:
            logger.error(f"Error processing {trade_date_str}: {e}", exc_info=True)
            skipped_days.append(trade_date_str)
            continue

    # Provide detailed error information
    if not all_predictions:
        error_msg = (
            f"No predictions generated for date range {start_date} to {end_date}.\n"
            f"  - Total trading days: {len(calendar)}\n"
            f"  - Successfully processed: {len(successful_days)}\n"
            f"  - Skipped/Failed: {len(skipped_days)}\n"
        )
        if skipped_days:
            error_msg += f"  - Skipped days: {', '.join(skipped_days[:10])}"
            if len(skipped_days) > 10:
                error_msg += f" ... and {len(skipped_days) - 10} more"
            error_msg += "\n"
        
        error_msg += (
            f"\nPossible reasons:\n"
            f"  1. Date range is in the future (no data available yet)\n"
            f"     → Use historical dates (e.g., 2024-01-01 to 2024-12-31)\n"
            f"  2. Qlib data is incomplete or missing\n"
            f"     → Check data availability: python -c \"import qlib; qlib.init(); from qlib.data import D; print(D.calendar())\"\n"
            f"  3. No stocks have data in the specified date range\n"
            f"     → Check stock list: python -c \"import qlib; qlib.init(); from qlib.data import D; print(D.instruments())\"\n"
        )
        
        raise ValueError(error_msg)

    # Concatenate all predictions
    predictions = pd.concat(all_predictions)
    logger.info(f"Generated predictions for {len(predictions)} stock-days")

    return predictions


def create_portfolio_strategy(
    predictions: pd.DataFrame,
    strategy_class: type = TopkDropoutStrategy,
    top_k: int = DEFAULT_TOP_K,
) -> TopkDropoutStrategy:
    """
    Create portfolio strategy from predictions.

    Args:
        predictions: DataFrame with predictions (MultiIndex: datetime, instrument).
        strategy_class: Strategy class to use.
        top_k: Number of top stocks to select.

    Returns:
        Strategy instance.
    """
    logger.info(f"Creating {strategy_class.__name__} with top_k={top_k}")

    # Create signal DataFrame in Qlib format
    # Qlib expects a DataFrame with MultiIndex (datetime, instrument) and a 'score' column
    signal = predictions.copy()

    # Create strategy config
    strategy_config = {
        "signal": signal,
        "topk": top_k,
        "n_drop": 5,  # Drop bottom 5 to reduce turnover
    }

    # Initialize strategy
    strategy = strategy_class(**strategy_config)

    return strategy


def run_backtest(
    strategy: TopkDropoutStrategy,
    start_date: str,
    end_date: str,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    benchmark: Optional[str] = None,
    skip_auto_detect: bool = False,
) -> dict:
    """
    Run backtest using Qlib's backtest framework.

    Args:
        strategy: Portfolio strategy instance.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        initial_cash: Initial cash amount.
        benchmark: Optional benchmark code (e.g., 'SH000300').

    Returns:
        Dictionary containing backtest results and metrics.
    """
    logger.info(f"Running backtest from {start_date} to {end_date}")
    logger.info(f"Initial cash: {initial_cash:,.2f}")

    # Normalize benchmark: convert empty string to None
    if benchmark == "":
        benchmark = None

    # Normalize benchmark format if provided (in case not normalized in main)
    if benchmark is not None and benchmark.strip() != "":
        original_benchmark = benchmark
        benchmark = normalize_benchmark_code(benchmark)
        if original_benchmark != benchmark:
            logger.debug(f"Normalized benchmark code: {original_benchmark} -> {benchmark}")

    # IMPORTANT: Qlib appears to require a benchmark for backtesting.
    # Even when --no_benchmark is used, we need to find a valid benchmark code.
    # We'll find one but note that it's auto-selected, not user-specified.
    
    if benchmark is None:
        if skip_auto_detect:
            logger.info(
                "Note: --no_benchmark was specified, but Qlib requires a benchmark. "
                "Automatically finding a valid benchmark to use..."
            )
        else:
            logger.info("No benchmark specified. Auto-detecting a valid benchmark...")
        
        # Try common benchmark formats (in Qlib format: code.EXCHANGE)
        common_benchmarks = ["000300.SH", "399001.SZ", "000001.SH", "000905.SH"]
        for bm in common_benchmarks:
            try:
                # Try to load benchmark data
                test_data = D.features([bm], ["$close"], start_time=start_date, end_time=end_date, freq="day")
                if not test_data.empty:
                    benchmark = bm
                    if skip_auto_detect:
                        logger.info(f"Auto-selected benchmark (required by Qlib): {benchmark}")
                    else:
                        logger.info(f"Auto-detected benchmark: {benchmark}")
                    break
            except Exception as e:
                logger.debug(f"Failed to load benchmark {bm}: {e}")
                continue

        if benchmark is None:
            logger.warning(
                "Could not find a valid benchmark in the data. "
                "Qlib requires a benchmark for backtesting. "
                "Please provide a valid benchmark code using --benchmark option."
            )
    else:
        # Validate provided benchmark (should already be normalized)
        logger.debug(f"Validating benchmark: {benchmark}")
        try:
            test_data = D.features([benchmark], ["$close"], start_time=start_date, end_time=end_date, freq="day")
            if test_data.empty:
                logger.warning(f"Benchmark {benchmark} has no data in the date range. Running without benchmark.")
                benchmark = None
            else:
                logger.info(f"Validated benchmark: {benchmark} (has data)")
        except Exception as e:
            logger.warning(f"Failed to validate benchmark {benchmark}: {e}. Running without benchmark.")
            benchmark = None

    # Create backtest configuration
    backtest_config = {
        "start_time": start_date,
        "end_time": end_date,
        "account": initial_cash,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0015,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    }

    # Handle benchmark configuration
    # IMPORTANT: Qlib appears to require a benchmark. If user wants to disable it,
    # we need to find a valid benchmark code first, then handle it in error recovery.
    # If benchmark is None and skip_auto_detect is True, we'll try without benchmark first,
    # but will automatically find one if Qlib requires it (handled in exception handler).
    
    if benchmark is not None and benchmark.strip() != "":
        # Final check: ensure benchmark is in correct format
        final_benchmark = normalize_benchmark_code(benchmark)
        if final_benchmark != benchmark:
            logger.warning(f"Benchmark format corrected: {benchmark} -> {final_benchmark}")
            benchmark = final_benchmark
        
        # Ensure benchmark is in correct format before passing to Qlib
        backtest_config["benchmark"] = benchmark
        logger.info(f"Final benchmark for backtest: {benchmark} (Qlib format: code.EXCHANGE)")
    else:
        # User wants to disable benchmark, but Qlib may require it
        # We'll try without benchmark first, and if it fails, automatically find one
        logger.info("Attempting to run without benchmark (Qlib may require one, will auto-detect if needed)")
        # Do NOT add benchmark key - let Qlib error handler find one if needed

    # Create executor config
    exec_config = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
        "verbose": True,
    }

    # Run backtest
    logger.info("Executing backtest...")
    logger.debug(f"Backtest config keys: {list(backtest_config.keys())}")
    if "benchmark" in backtest_config:
        logger.debug(f"Benchmark in config: {backtest_config['benchmark']}")
    else:
        logger.debug("No benchmark key in config")
    
    try:
        portfolio_metric, indicator = backtest(
            executor=executor.SimulatorExecutor(**exec_config),
            strategy=strategy,
            **backtest_config,
        )
    except ValueError as e:
        # If error is about benchmark, automatically find and use a valid one
        error_msg = str(e)
        
        # Check if this is a benchmark-related error
        # Qlib may show "The benchmark []" or "The benchmark ['SH000300']" etc.
        is_benchmark_error = (
            "benchmark" in error_msg.lower() and 
            ("does not exist" in error_msg.lower() or "provide the right benchmark" in error_msg.lower())
        )
        
        if is_benchmark_error:
            logger.warning(
                f"Qlib requires a valid benchmark. "
                f"Error: {error_msg}. "
                f"Automatically finding a valid benchmark..."
            )
            # Try to find a valid benchmark (in Qlib format: code.EXCHANGE)
            common_benchmarks = ["000300.SH", "399001.SZ", "000001.SH", "000905.SH"]
            found_benchmark = None
            for bm in common_benchmarks:
                try:
                    test_data = D.features([bm], ["$close"], start_time=start_date, end_time=end_date, freq="day")
                    if not test_data.empty:
                        found_benchmark = bm
                        logger.info(f"✓ Found valid benchmark: {found_benchmark}. Using it for backtest.")
                        break
                except Exception as ex:
                    logger.debug(f"Failed to load benchmark {bm}: {ex}")
                    continue
            
            if found_benchmark:
                # Retry with found benchmark
                backtest_config["benchmark"] = found_benchmark
                logger.info("Retrying backtest with auto-detected benchmark...")
                try:
                    portfolio_metric, indicator = backtest(
                        executor=executor.SimulatorExecutor(**exec_config),
                        strategy=strategy,
                        **backtest_config,
                    )
                    logger.info("✓ Backtest completed successfully with auto-detected benchmark")
                except Exception as retry_error:
                    logger.error(f"Backtest still failed after adding benchmark: {retry_error}")
                    raise
            else:
                logger.error(
                    f"❌ Could not find a valid benchmark in the data. "
                    f"Qlib requires a benchmark for backtesting. "
                    f"Please:\n"
                    f"  1. Provide a valid benchmark code using --benchmark option (e.g., --benchmark 000300.SH)\n"
                    f"  2. Or ensure your Qlib data contains benchmark index data\n"
                    f"  3. Original error: {error_msg}"
                )
                raise
        else:
            # Re-raise if it's not a benchmark-related error
            raise
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise

    # Extract results
    results = {
        "portfolio_metric": portfolio_metric,
        "indicator": indicator,
    }

    logger.info("Backtest completed successfully")
    return results


def print_results(results: dict) -> None:
    """Print backtest results in a readable format."""
    portfolio_metric = results.get("portfolio_metric")
    indicator = results.get("indicator")

    print("\n" + "=" * 80)
    print("Backtest Results")
    print("=" * 80)

    # Portfolio metrics
    if portfolio_metric is not None:
        print("\n📊 Portfolio Metrics:")
        
        # Handle dict format where key is frequency (e.g., '1day') and value is (DataFrame, dict)
        if isinstance(portfolio_metric, dict):
            # Extract the DataFrame from the dict (usually key is '1day')
            metric_df = None
            for key, value in portfolio_metric.items():
                if isinstance(value, tuple) and len(value) >= 1:
                    if isinstance(value[0], pd.DataFrame):
                        metric_df = value[0]
                        break
                elif isinstance(value, pd.DataFrame):
                    metric_df = value
                    break
            
            if metric_df is not None and len(metric_df) > 0:
                # Get the last row (final metrics)
                final_row = metric_df.iloc[-1]
                first_row = metric_df.iloc[0]
                
                # Extract return from the 'return' column (cumulative return)
                total_return = final_row.get('return', None)
                if total_return is None or pd.isna(total_return):
                    # Calculate from account value if return column is not available
                    initial_account = first_row.get('account', None)
                    final_account = final_row.get('account', None)
                    if initial_account is not None and final_account is not None and initial_account > 0:
                        total_return = (final_account - initial_account) / initial_account
                    else:
                        total_return = None
                
                # Calculate annualized return
                annualized_return = None
                if total_return is not None and not pd.isna(total_return):
                    # Get number of trading days
                    num_days = len(metric_df)
                    if num_days > 0:
                        # Assume ~252 trading days per year
                        years = num_days / 252.0
                        if years > 0:
                            annualized_return = (1 + total_return) ** (1.0 / years) - 1
                
                # Calculate volatility from return series
                volatility = None
                if 'return' in metric_df.columns:
                    returns = metric_df['return'].dropna()
                    if len(returns) > 1:
                        # Calculate daily returns (difference)
                        daily_returns = returns.diff().dropna()
                        if len(daily_returns) > 0:
                            # Annualized volatility
                            volatility = daily_returns.std() * (252 ** 0.5)
                
                # Calculate Sharpe ratio
                sharpe = None
                if annualized_return is not None and volatility is not None and volatility > 0:
                    # Assume risk-free rate is 0 for simplicity
                    sharpe = annualized_return / volatility
                
                # Calculate max drawdown
                max_drawdown = None
                if 'account' in metric_df.columns:
                    account_values = metric_df['account'].dropna()
                    if len(account_values) > 0:
                        # Calculate running maximum
                        running_max = account_values.expanding().max()
                        # Calculate drawdown
                        drawdown = (account_values - running_max) / running_max
                        max_drawdown = drawdown.min()
                
                # Display metrics
                print(f"  Total Return: {total_return * 100:.2f}%" if total_return is not None and not pd.isna(total_return) else "  Total Return: N/A")
                print(f"  Annualized Return: {annualized_return * 100:.2f}%" if annualized_return is not None and not pd.isna(annualized_return) else "  Annualized Return: N/A")
                print(f"  Volatility: {volatility * 100:.2f}%" if volatility is not None and not pd.isna(volatility) else "  Volatility: N/A")
                print(f"  Sharpe Ratio: {sharpe:.4f}" if sharpe is not None and not pd.isna(sharpe) else "  Sharpe Ratio: N/A")
                print(f"  Max Drawdown: {max_drawdown * 100:.2f}%" if max_drawdown is not None and not pd.isna(max_drawdown) else "  Max Drawdown: N/A")
                
                # Show additional info
                if 'account' in metric_df.columns:
                    initial_account = metric_df['account'].iloc[0]
                    final_account = metric_df['account'].iloc[-1]
                    print(f"\n  Initial Account Value: {initial_account:,.2f}")
                    print(f"  Final Account Value: {final_account:,.2f}")
                    print(f"  Net P&L: {final_account - initial_account:,.2f}")
                
                if len(metric_df) > 1:
                    print(f"\n  Time Series: {len(metric_df)} trading days")
                    print(f"    First date: {metric_df.index[0]}")
                    print(f"    Last date: {metric_df.index[-1]}")
                
                # Extract trading statistics from position details
                position_details = None
                for key, value in portfolio_metric.items():
                    if isinstance(value, tuple) and len(value) >= 2:
                        if isinstance(value[1], dict):
                            position_details = value[1]
                            break
                
                if position_details:
                    # Calculate trading statistics
                    trades = []  # List of (buy_date, sell_date, stock, buy_price, sell_price, pnl, pnl_pct)
                    holdings = {}  # {stock: (buy_date, buy_price, amount)}
                    
                    # Sort dates
                    dates = sorted([pd.Timestamp(d) for d in position_details.keys() if isinstance(d, pd.Timestamp)])
                    
                    for date in dates:
                        if date not in position_details:
                            continue
                        pos_info = position_details[date]
                        
                        # Handle both dict and object types
                        if isinstance(pos_info, dict):
                            if 'position' not in pos_info:
                                continue
                            position = pos_info['position']
                        else:
                            # Try to access as object attribute
                            if not hasattr(pos_info, 'position'):
                                continue
                            position = pos_info.position
                        
                        # Convert position to dict if it's an object
                        if not isinstance(position, dict):
                            # Try to convert Position object to dict
                            try:
                                if hasattr(position, '__dict__'):
                                    position = position.__dict__
                                elif hasattr(position, 'to_dict'):
                                    position = position.to_dict()
                                else:
                                    # Skip if we can't convert
                                    continue
                            except:
                                continue
                        
                        current_holdings = {}
                        
                        # Extract current holdings
                        if isinstance(position, dict):
                            for key, value in position.items():
                                if key in ['cash', 'now_account_value']:
                                    continue
                                if isinstance(value, dict) and 'price' in value and 'amount' in value:
                                    stock = key
                                    price = float(value['price'])
                                    amount = float(value['amount'])
                                    if amount > 0:
                                        current_holdings[stock] = (price, amount)
                        
                        # Check for new buys (stocks in current but not in previous)
                        prev_date_idx = dates.index(date) - 1
                        if prev_date_idx >= 0:
                            prev_date = dates[prev_date_idx]
                            prev_holdings = {}
                            if prev_date in position_details:
                                prev_pos_info = position_details[prev_date]
                                # Handle both dict and object types
                                if isinstance(prev_pos_info, dict) and 'position' in prev_pos_info:
                                    prev_position = prev_pos_info['position']
                                elif hasattr(prev_pos_info, 'position'):
                                    prev_position = prev_pos_info.position
                                else:
                                    prev_position = None
                                
                                if prev_position is not None:
                                    # Convert to dict if needed
                                    if not isinstance(prev_position, dict):
                                        try:
                                            if hasattr(prev_position, '__dict__'):
                                                prev_position = prev_position.__dict__
                                            elif hasattr(prev_position, 'to_dict'):
                                                prev_position = prev_position.to_dict()
                                        except:
                                            prev_position = None
                                    
                                    if isinstance(prev_position, dict):
                                        for key, value in prev_position.items():
                                            if key in ['cash', 'now_account_value']:
                                                continue
                                            if isinstance(value, dict) and 'price' in value and 'amount' in value:
                                                stock = key
                                                amount = float(value['amount'])
                                                if amount > 0:
                                                    prev_holdings[stock] = amount
                            
                            # Find new positions (bought)
                            for stock in current_holdings:
                                if stock not in prev_holdings or prev_holdings[stock] == 0:
                                    # New buy
                                    price, amount = current_holdings[stock]
                                    holdings[stock] = (date, price, amount)
                            
                            # Find closed positions (sold)
                            for stock in list(holdings.keys()):
                                buy_date, buy_price, buy_amount = holdings[stock]
                                if stock not in current_holdings:
                                    # Completely sold
                                    # Use previous day's price as sell price (or current if available)
                                    sell_price = buy_price  # Fallback
                                    if prev_date in position_details:
                                        prev_pos_info = position_details[prev_date]
                                        prev_pos = None
                                        if isinstance(prev_pos_info, dict) and 'position' in prev_pos_info:
                                            prev_pos = prev_pos_info['position']
                                        elif hasattr(prev_pos_info, 'position'):
                                            prev_pos = prev_pos_info.position
                                        
                                        if prev_pos is not None:
                                            # Convert to dict if needed
                                            if not isinstance(prev_pos, dict):
                                                try:
                                                    if hasattr(prev_pos, '__dict__'):
                                                        prev_pos = prev_pos.__dict__
                                                    elif hasattr(prev_pos, 'to_dict'):
                                                        prev_pos = prev_pos.to_dict()
                                                except:
                                                    prev_pos = None
                                            
                                            if isinstance(prev_pos, dict) and stock in prev_pos:
                                                stock_info = prev_pos[stock]
                                                if isinstance(stock_info, dict) and 'price' in stock_info:
                                                    sell_price = float(stock_info['price'])
                                    pnl = (sell_price - buy_price) * buy_amount
                                    pnl_pct = (sell_price - buy_price) / buy_price if buy_price > 0 else 0
                                    trades.append((buy_date, date, stock, buy_price, sell_price, pnl, pnl_pct))
                                    del holdings[stock]
                                elif current_holdings[stock][1] < buy_amount:
                                    # Partially sold - treat as full sale for simplicity
                                    sell_price = current_holdings[stock][0]
                                    pnl = (sell_price - buy_price) * buy_amount
                                    pnl_pct = (sell_price - buy_price) / buy_price if buy_price > 0 else 0
                                    trades.append((buy_date, date, stock, buy_price, sell_price, pnl, pnl_pct))
                                    # Update holding with remaining amount
                                    remaining_amount = current_holdings[stock][1]
                                    if remaining_amount > 0:
                                        holdings[stock] = (date, sell_price, remaining_amount)  # Update to new buy date
                                    else:
                                        del holdings[stock]
                        else:
                            # First day - all are new buys
                            for stock, (price, amount) in current_holdings.items():
                                holdings[stock] = (date, price, amount)
                    
                    # Calculate statistics from trades
                    if trades:
                        winning_trades = [t for t in trades if t[5] > 0]  # pnl > 0
                        losing_trades = [t for t in trades if t[5] < 0]   # pnl < 0
                        
                        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
                        
                        # Average holding period
                        holding_periods = [(t[1] - t[0]).days for t in trades]
                        avg_holding_days = sum(holding_periods) / len(holding_periods) if holding_periods else 0
                        
                        # Average return per trade
                        avg_return = sum([t[6] for t in trades]) / len(trades) * 100 if trades else 0
                        
                        # Profit factor (total profit / total loss)
                        total_profit = sum([t[5] for t in winning_trades]) if winning_trades else 0
                        total_loss = abs(sum([t[5] for t in losing_trades])) if losing_trades else 0
                        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
                        
                        # Average win/loss
                        avg_win = sum([t[5] for t in winning_trades]) / len(winning_trades) if winning_trades else 0
                        avg_loss = sum([t[5] for t in losing_trades]) / len(losing_trades) if losing_trades else 0
                        avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf') if avg_win > 0 else 0
                        
                        print(f"\n📈 Trading Statistics:")
                        print(f"  Total Trades: {len(trades)}")
                        print(f"  Winning Trades: {len(winning_trades)}")
                        print(f"  Losing Trades: {len(losing_trades)}")
                        print(f"  Win Rate: {win_rate:.2f}%")
                        print(f"  Average Holding Period: {avg_holding_days:.1f} days")
                        print(f"  Average Return per Trade: {avg_return:.2f}%")
                        print(f"  Profit Factor: {profit_factor:.2f}" if profit_factor != float('inf') else "  Profit Factor: ∞")
                        print(f"  Avg Win / Avg Loss: {avg_win_loss_ratio:.2f}" if avg_win_loss_ratio != float('inf') else "  Avg Win / Avg Loss: ∞")
                        print(f"  Total Profit: {total_profit:,.2f}")
                        print(f"  Total Loss: {total_loss:,.2f}")
                    
                    # Show turnover statistics
                    if 'total_turnover' in metric_df.columns:
                        final_turnover = metric_df['total_turnover'].iloc[-1]
                        avg_daily_turnover = metric_df['turnover'].mean() if 'turnover' in metric_df.columns else None
                        print(f"\n💰 Turnover Statistics:")
                        print(f"  Total Turnover: {final_turnover:,.2f}")
                        if avg_daily_turnover is not None:
                            print(f"  Average Daily Turnover: {avg_daily_turnover:.2%}")
            else:
                print("  ⚠ Could not extract DataFrame from portfolio_metric dict")
                print(f"  Dict keys: {list(portfolio_metric.keys())}")
        # Handle DataFrame format (direct)
        elif isinstance(portfolio_metric, pd.DataFrame):
            # Qlib returns DataFrame with time series of metrics
            if len(portfolio_metric) > 0:
                # Get the last row (final metrics)
                final_metrics = portfolio_metric.iloc[-1]
                
                # Extract metrics
                return_val = final_metrics.get('return', None)
                annualized_return = final_metrics.get('return.annualized', final_metrics.get('return_annualized', None))
                volatility = final_metrics.get('volatility', None)
                sharpe = final_metrics.get('sharpe', final_metrics.get('Sharpe', None))
                max_drawdown = final_metrics.get('max_drawdown', final_metrics.get('Max Drawdown', final_metrics.get('maxdrawdown', None)))
                
                # Display metrics
                print(f"  Return: {return_val * 100:.2f}%" if return_val is not None and not pd.isna(return_val) else "  Return: N/A")
                print(f"  Annualized Return: {annualized_return * 100:.2f}%" if annualized_return is not None and not pd.isna(annualized_return) else "  Annualized Return: N/A")
                print(f"  Volatility: {volatility * 100:.2f}%" if volatility is not None and not pd.isna(volatility) else "  Volatility: N/A")
                print(f"  Sharpe Ratio: {sharpe:.4f}" if sharpe is not None and not pd.isna(sharpe) else "  Sharpe Ratio: N/A")
                print(f"  Max Drawdown: {max_drawdown * 100:.2f}%" if max_drawdown is not None and not pd.isna(max_drawdown) else "  Max Drawdown: N/A")
            else:
                print("  ⚠ Portfolio metrics DataFrame is empty")
        else:
            print(f"  ⚠ Portfolio metrics format not recognized: {type(portfolio_metric)}")
            print(f"  Value: {portfolio_metric}")
    else:
        print("\n📊 Portfolio Metrics:")
        print("  ⚠ Portfolio metrics is None")
        print("  This may indicate:")
        print("    1. generate_portfolio_metrics was not enabled in executor config")
        print("    2. Backtest period is too short to calculate metrics")
        print("    3. Qlib backtest function did not return portfolio metrics")

    # Indicators
    if indicator is not None:
        print("\n📈 Performance Indicators:")
        if isinstance(indicator, dict):
            for key, value in indicator.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                elif isinstance(value, pd.DataFrame):
                    print(f"  {key}: (DataFrame with shape {value.shape})")
                    # Show summary if small
                    if len(value) <= 10:
                        print(f"    {value}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  ⚠ Indicator format not recognized: {type(indicator)}")
            print(f"  Value: {indicator}")

    print("\n" + "=" * 80)


def plot_results(results: dict, output_path: Optional[str] = None) -> None:
    """
    Plot backtest results visualization.
    
    Args:
        results: Backtest results dictionary containing portfolio_metric and indicator.
        output_path: Optional path to save the plot. If None, saves to 'backtest_results.png'.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available. Skipping visualization.")
        return
    
    portfolio_metric = results.get("portfolio_metric")
    
    if portfolio_metric is None:
        logger.warning("No portfolio_metric available for plotting.")
        return
    
    # Extract DataFrame from portfolio_metric
    metric_df = None
    if isinstance(portfolio_metric, dict):
        for key, value in portfolio_metric.items():
            if isinstance(value, tuple) and len(value) >= 1:
                if isinstance(value[0], pd.DataFrame):
                    metric_df = value[0]
                    break
            elif isinstance(value, pd.DataFrame):
                metric_df = value
                break
    elif isinstance(portfolio_metric, pd.DataFrame):
        metric_df = portfolio_metric
    
    if metric_df is None or len(metric_df) == 0:
        logger.warning("No valid DataFrame found in portfolio_metric for plotting.")
        return
    
    # Set up Chinese font support (if available)
    try:
        # Try to find a Chinese font
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'STHeiti']
        font_found = False
        for font_name in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                font_found = True
                break
            except:
                continue
        if not font_found:
            # Use default font
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    except:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Convert index to datetime if needed
    if not isinstance(metric_df.index, pd.DatetimeIndex):
        try:
            metric_df.index = pd.to_datetime(metric_df.index)
        except:
            pass
    
    dates = metric_df.index
    
    # 1. Account Value Over Time
    ax1 = fig.add_subplot(gs[0, :])
    if 'account' in metric_df.columns:
        account_values = metric_df['account']
        ax1.plot(dates, account_values, linewidth=2, label='Account Value', color='#2E86AB')
        ax1.axhline(y=account_values.iloc[0], color='gray', linestyle='--', alpha=0.5, label='Initial Value')
        ax1.set_title('Account Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Account Value (CNY)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Cumulative Return
    ax2 = fig.add_subplot(gs[1, 0])
    if 'return' in metric_df.columns:
        returns = metric_df['return'] * 100  # Convert to percentage
        ax2.plot(dates, returns, linewidth=2, label='Cumulative Return', color='#A23B72')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Cumulative Return (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Drawdown
    ax3 = fig.add_subplot(gs[1, 1])
    if 'account' in metric_df.columns:
        account_values = metric_df['account']
        running_max = account_values.expanding().max()
        drawdown = (account_values - running_max) / running_max * 100
        ax3.fill_between(dates, drawdown, 0, alpha=0.3, color='#F18F01', label='Drawdown')
        ax3.plot(dates, drawdown, linewidth=1.5, color='#F18F01')
        ax3.set_title('Drawdown (%)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Daily Returns
    ax4 = fig.add_subplot(gs[2, 0])
    if 'return' in metric_df.columns:
        returns = metric_df['return']
        daily_returns = returns.diff().dropna() * 100  # Convert to percentage
        if len(daily_returns) > 0:
            colors = ['#06A77D' if x >= 0 else '#D00000' for x in daily_returns]
            ax4.bar(daily_returns.index, daily_returns.values, color=colors, alpha=0.7, width=0.8)
            ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax4.set_title('Daily Returns (%)', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Daily Return (%)')
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 5. Turnover
    ax5 = fig.add_subplot(gs[2, 1])
    if 'turnover' in metric_df.columns:
        turnover = metric_df['turnover'] * 100  # Convert to percentage
        ax5.plot(dates, turnover, linewidth=2, label='Daily Turnover', color='#6A4C93', marker='o', markersize=3)
        ax5.set_title('Daily Turnover (%)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Turnover (%)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add overall title
    fig.suptitle('Backtest Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        logger.info("Plot saved to backtest_results.png")
    
    plt.close()


def save_results(results: dict, output_path: str) -> None:
    """Save backtest results to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    portfolio_metric = results["portfolio_metric"]
    indicator = results["indicator"]

    # Combine all metrics
    all_metrics = {}
    if portfolio_metric is not None:
        all_metrics.update(portfolio_metric)
    if indicator is not None:
        all_metrics.update(indicator)

    # Save to CSV
    df = pd.DataFrame([all_metrics])
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backtest Structure Expert GNN model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model file (.pth)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date of backtest period (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="End date of backtest period (YYYY-MM-DD)",
    )

    # Optional arguments
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top stocks to select (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="TopkDropoutStrategy",
        help="Portfolio strategy class name (default: TopkDropoutStrategy)",
    )
    parser.add_argument(
        "--initial_cash",
        type=float,
        default=DEFAULT_INITIAL_CASH,
        help=f"Initial cash amount (default: {DEFAULT_INITIAL_CASH})",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save backtest results to file",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots for backtest results",
    )
    parser.add_argument(
        "--plot_output",
        type=str,
        default=None,
        help="Path to save the plot image (default: outputs/structure_expert_backtest_<dates>.png)",
    )
    parser.add_argument(
        "--qlib_dir",
        type=str,
        default=DEFAULT_QLIB_DIR,
        help=f"Qlib data directory (default: {DEFAULT_QLIB_DIR})",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="cn",
        help="Qlib region (default: cn)",
    )
    parser.add_argument(
        "--n_feat",
        type=int,
        default=DEFAULT_N_FEAT,
        help=f"Number of input features (default: {DEFAULT_N_FEAT})",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=DEFAULT_N_HIDDEN,
        help=f"Hidden layer size (default: {DEFAULT_N_HIDDEN})",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=DEFAULT_N_HEADS,
        help=f"Number of attention heads (default: {DEFAULT_N_HEADS})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config file (for database config)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Benchmark code (e.g., 'SH000300'). If not provided, will try to auto-detect.",
    )
    parser.add_argument(
        "--no_benchmark",
        action="store_true",
        help="Explicitly disable benchmark comparison",
    )

    args = parser.parse_args()

    # Handle benchmark
    skip_auto_detect = args.no_benchmark
    if args.no_benchmark:
        benchmark = None
        logger.info("Benchmark comparison disabled by --no_benchmark flag")
    else:
        benchmark = args.benchmark
        # Normalize benchmark format early if provided
        if benchmark is not None and benchmark.strip() != "":
            benchmark = normalize_benchmark_code(benchmark)
            logger.debug(f"Normalized benchmark code in main: {benchmark}")

    # Validate dates
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
    today = datetime.now()
    
    if start_dt >= end_dt:
        logger.error(f"Start date ({args.start_date}) must be before end date ({args.end_date})")
        return 1
    
    # Check for future dates - warn but don't error (data might exist in test/simulated scenarios)
    if start_dt > today or end_dt > today:
        logger.warning(
            f"⚠️  Warning: Date range contains future dates.\n"
            f"  Start date: {args.start_date} {'(FUTURE)' if start_dt > today else ''}\n"
            f"  End date: {args.end_date} {'(FUTURE)' if end_dt > today else ''}\n"
            f"  Today: {today.strftime('%Y-%m-%d')}\n"
            f"Proceeding anyway - Qlib data may contain future dates (test/simulated data)."
        )

    # Initialize Qlib
    qlib_dir = Path(args.qlib_dir).expanduser()
    logger.info(f"Initializing Qlib with data directory: {qlib_dir}")
    qlib.init(provider_uri=str(qlib_dir), region=args.region)
    
    # Check Qlib data availability and date range
    try:
        full_calendar = D.calendar()
        if len(full_calendar) == 0:
            logger.error("❌ No trading days found in Qlib data. Please check your Qlib data installation.")
            return 1
        
        data_start = full_calendar[0].date()
        data_end = full_calendar[-1].date()
        
        logger.info(f"Qlib data date range: {data_start} to {data_end} ({len(full_calendar)} trading days)")
        
        # Check if requested dates are within data range
        if start_dt < data_start:
            logger.warning(
                f"⚠️  Start date {args.start_date} is before Qlib data start date ({data_start}). "
                f"Adjusting to {data_start}."
            )
            args.start_date = data_start.strftime("%Y-%m-%d")
            start_dt = data_start
        
        if end_dt > data_end:
            logger.warning(
                f"⚠️  End date {args.end_date} is after Qlib data end date ({data_end}). "
                f"Adjusting to {data_end}."
            )
            args.end_date = data_end.strftime("%Y-%m-%d")
            end_dt = data_end
        
        # Check if date range (with lookback) is available
        LOOKBACK_DAYS = 60  # Same as in generate_predictions
        lookback_start = start_dt - timedelta(days=LOOKBACK_DAYS)
        if lookback_start < data_start:
            logger.warning(
                f"⚠️  Start date {args.start_date} requires lookback to {lookback_start}, "
                f"but Qlib data only starts from {data_start}. "
                f"Some dates may not have enough historical data for Alpha158 features."
            )
        
        # Verify that requested date range has data
        requested_calendar = D.calendar(start_time=args.start_date, end_time=args.end_date)
        if len(requested_calendar) == 0:
            logger.error(
                f"❌ No trading days found in Qlib data for date range {args.start_date} to {args.end_date}.\n"
                f"Available data range: {data_start} to {data_end}\n"
                f"Please adjust your date range to be within the available data."
            )
            return 1
        
        logger.info(f"✓ Found {len(requested_calendar)} trading days in requested range")
        
    except Exception as e:
        logger.warning(f"Could not verify Qlib data availability: {e}")
        logger.warning("Proceeding anyway, but data may not be available for requested dates.")

    # Load database config if provided
    if args.config_path:
        config = load_config(args.config_path)
        db_config = config.database
    else:
        # Try to load from default location
        try:
            config = load_config("config/config.yaml")
            db_config = config.database
        except Exception:
            logger.warning(
                "Could not load database config. "
                "Industry mapping will be limited. "
                "Provide --config_path to specify config file."
            )
            db_config = None

    try:
        # Load model
        model = load_model(
            model_path=args.model_path,
            n_feat=args.n_feat,
            n_hidden=args.n_hidden,
            n_heads=args.n_heads,
            device=args.device,
        )

        # Load industry mapping
        if db_config is not None:
            # Use end_date for industry mapping (most recent)
            end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
            industry_map = load_industry_map(db_config, target_date=end_dt)
            logger.info(f"Loaded industry mapping: {len(industry_map)} stocks with industry info")
            logger.info(
                f"Note: Stocks not in industry_map will still be processed, "
                f"but won't have industry-based graph edges"
            )
        else:
            logger.warning("No database config, using empty industry map")
            logger.warning("All stocks will be processed, but without industry-based graph edges")
            industry_map = {}

        # Create graph builder
        builder = GraphDataBuilder(industry_map)

        # Generate predictions
        predictions = generate_predictions(
            model=model,
            builder=builder,
            start_date=args.start_date,
            end_date=args.end_date,
            device=args.device,
        )

        # Create portfolio strategy
        strategy = create_portfolio_strategy(
            predictions=predictions,
            strategy_class=TopkDropoutStrategy,
            top_k=args.top_k,
        )

        # Run backtest
        results = run_backtest(
            strategy=strategy,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_cash=args.initial_cash,
            benchmark=benchmark,
            skip_auto_detect=skip_auto_detect,
        )

        # Print results
        print_results(results)

        # Plot results if requested
        if args.plot:
            plot_output = args.plot_output if args.plot_output else f"outputs/structure_expert_backtest_{args.start_date}_{args.end_date}.png"
            plot_results(results, plot_output)

        # Save results if requested
        if args.save_results:
            output_path = f"outputs/structure_expert_backtest_{args.start_date}_{args.end_date}.csv"
            save_results(results, output_path)

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

