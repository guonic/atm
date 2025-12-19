# -*- coding: utf-8 -*-
"""
kronos_cn_predict_example.py

Description:
    Predicts future daily K-line (1D) data for A-share markets using Kronos model.
    The script loads historical data from database and runs model inference.

Usage:
    python kronos_cn_predict_example.py --ts_code 000001.SZ --pred_len 120

Arguments:
    --ts_code      Stock code in Tushare format (e.g., 000001.SZ, 600000.SH)
    --pred_len     Number of trading days to predict (default: 120)
    --lookback     Number of historical days to use (default: 400)
    --device       Device to use (default: cpu, can be cuda:0)
    --model_name   Pretrained model name (default: NeoQuasar/Kronos-base)
    --tokenizer_name Pretrained tokenizer name (default: NeoQuasar/Kronos-Tokenizer-base)

Output:
    - Saves the prediction results to ./outputs/pred_<ts_code>_data.csv
    - Saves the prediction chart to ./outputs/pred_<ts_code>_chart.png
    - Logs and progress are printed to console

Example:
    python kronos_cn_predict_example.py --ts_code 000001.SZ --pred_len 120
    python kronos_cn_predict_example.py --ts_code 600000.SH --pred_len 60 --lookback 300
"""

import argparse
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from atm.ai.kronos import Kronos, KronosPredictor, KronosTokenizer
from atm.config import load_config
from atm.repo.kline_repo import StockKlineDayRepo
from atm.repo.trading_calendar_repo import TradingCalendarRepo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_SAVE_DIR = "./outputs"
DEFAULT_TOKENIZER_PRETRAINED = "NeoQuasar/Kronos-Tokenizer-base"
DEFAULT_MODEL_PRETRAINED = "NeoQuasar/Kronos-base"
DEFAULT_DEVICE = "cpu"
DEFAULT_MAX_CONTEXT = 512
DEFAULT_LOOKBACK = 400
DEFAULT_PRED_LEN = 20
DEFAULT_T = 1.0
DEFAULT_TOP_P = 0.9
DEFAULT_SAMPLE_COUNT = 1


def load_data_from_db(
    db_config,
    ts_code: str,
    lookback: int,
    schema: str = "quant",
) -> pd.DataFrame:
    """
    Load historical K-line data from database.

    Args:
        db_config: Database configuration.
        ts_code: Stock code in Tushare format (e.g., 000001.SZ).
        lookback: Number of historical days to retrieve.
        schema: Database schema name.

    Returns:
        DataFrame with OHLCV data.
    """
    logger.info(f"📥 Loading {ts_code} daily data from database ...")

    repo = StockKlineDayRepo(db_config, schema)

    # Get the latest data, sorted by trade_date descending
    klines = repo.get_by_ts_code(
        ts_code=ts_code,
        limit=lookback,
    )

    if not klines:
        raise ValueError(f"No data found for {ts_code} in database")

    # Convert to DataFrame
    data_list = []
    for kline in klines:
        data_list.append({
            "date": kline.trade_date,
            "open": float(kline.open) if kline.open else None,
            "high": float(kline.high) if kline.high else None,
            "low": float(kline.low) if kline.low else None,
            "close": float(kline.close) if kline.close else None,
            "volume": int(kline.volume) if kline.volume else 0,
            "amount": float(kline.amount) if kline.amount else None,
        })

    df = pd.DataFrame(data_list)

    # Sort by date ascending
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure date is datetime
    if not isinstance(df["date"].dtype, pd.DatetimeIndex):
        df["date"] = pd.to_datetime(df["date"])

    # Fix missing amount
    if df["amount"].isna().all() or (df["amount"] == 0).all():
        df["amount"] = df["close"] * df["volume"]

    # Fix invalid open values
    open_bad = (df["open"] == 0) | (df["open"].isna())
    if open_bad.any():
        logger.warning(f"⚠️  Fixed {open_bad.sum()} invalid open values.")
        df.loc[open_bad, "open"] = df["close"].shift(1)
        df["open"].fillna(df["close"], inplace=True)

    # Fill any remaining NaN values
    df["open"] = df["open"].ffill()
    df["high"] = df["high"].ffill()
    df["low"] = df["low"].ffill()
    df["close"] = df["close"].ffill()
    df["volume"] = df["volume"].fillna(0)
    df["amount"] = df["amount"].fillna(df["close"] * df["volume"])

    logger.info(
        f"✅ Data loaded: {len(df)} rows, range: {df['date'].min()} ~ {df['date'].max()}"
    )

    logger.debug("Data Head:")
    logger.debug(f"\n{df.head()}")

    return df


def infer_exchange_from_ts_code(ts_code: str) -> str:
    """
    Infer exchange code from ts_code.

    Args:
        ts_code: Stock code in Tushare format (e.g., 000001.SZ, 600000.SH).

    Returns:
        Exchange code (SSE/SZSE/BSE).
    """
    if ts_code.endswith(".SH"):
        return "SSE"
    elif ts_code.endswith(".SZ"):
        return "SZSE"
    elif ts_code.endswith(".BJ"):
        return "BSE"
    else:
        # Default to SSE if cannot infer
        logger.warning(f"Cannot infer exchange from {ts_code}, defaulting to SSE")
        return "SSE"


def get_future_trading_days(
    db_config,
    exchange: str,
    start_date: date,
    pred_len: int,
    schema: str = "quant",
) -> pd.Series:
    """
    Get future trading days from trading calendar.

    Args:
        db_config: Database configuration.
        exchange: Exchange code (SSE/SZSE/BSE).
        start_date: Start date (next trading day after historical data).
        pred_len: Number of trading days to get.
        schema: Database schema name.

    Returns:
        Series of trading day dates.
    """
    repo = TradingCalendarRepo(db_config, schema)

    # Get trading days up to a reasonable future date
    # Assuming we need at least pred_len trading days, we'll query up to 2 years ahead
    end_date = start_date + timedelta(days=pred_len * 2)

    trading_days = repo.get_trading_days(
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
    )

    if not trading_days:
        logger.warning(
            f"No trading days found in calendar for {exchange}. Using business days as fallback."
        )
        # Fallback to business days if calendar is not available
        # Convert DatetimeIndex to Series
        bdate_range = pd.bdate_range(start=start_date, periods=pred_len)
        return pd.Series(bdate_range)

    # Extract dates and sort
    dates = sorted([cal.cal_date for cal in trading_days])

    # Take only the required number of trading days
    if len(dates) < pred_len:
        logger.warning(
            f"Only {len(dates)} trading days found, requested {pred_len}. "
            f"Using all available trading days."
        )
        return pd.Series(dates[:pred_len])

    return pd.Series(dates[:pred_len])


def prepare_inputs(
    df: pd.DataFrame,
    lookback: int,
    db_config,
    pred_len: int,
    ts_code: str,
    schema: str = "quant",
):
    """
    Prepare inputs for prediction.

    Args:
        df: Historical data DataFrame.
        lookback: Number of historical days to use.
        db_config: Database configuration.
        pred_len: Number of days to predict.
        ts_code: Stock code in Tushare format (used to infer exchange).
        schema: Database schema name.

    Returns:
        Tuple of (x_df, x_timestamp, y_timestamp).
    """
    # Get the last lookback days
    x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]].copy()
    x_timestamp = df.iloc[-lookback:]["date"].copy()

    # Determine exchange from ts_code
    exchange = infer_exchange_from_ts_code(ts_code)

    # Get the last date and find next trading day
    last_date = df["date"].iloc[-1].date()
    next_date = last_date + timedelta(days=1)

    # Get future trading days
    y_timestamp = get_future_trading_days(
        db_config=db_config,
        exchange=exchange,
        start_date=next_date,
        pred_len=pred_len,
        schema=schema,
    )

    return x_df, pd.Series(x_timestamp), y_timestamp


def apply_price_limits(pred_df: pd.DataFrame, last_close: float, limit_rate: float = 0.1) -> pd.DataFrame:
    """
    Apply price limits to predictions (A-share market has ±10% daily limit).

    Args:
        pred_df: Prediction DataFrame.
        last_close: Last close price from historical data.
        limit_rate: Price limit rate (default: 0.1 for ±10%).

    Returns:
        DataFrame with price limits applied.
    """
    logger.info(f"🔒 Applying ±{limit_rate*100:.0f}% price limit ...")

    # Ensure integer index
    pred_df = pred_df.reset_index(drop=True)

    # Ensure float64 dtype for safe assignment
    cols = ["open", "high", "low", "close"]
    pred_df[cols] = pred_df[cols].astype("float64")

    current_close = last_close

    for i in range(len(pred_df)):
        limit_up = current_close * (1 + limit_rate)
        limit_down = current_close * (1 - limit_rate)

        for col in cols:
            value = pred_df.at[i, col]
            if pd.notna(value):
                clipped = max(min(value, limit_up), limit_down)
                pred_df.at[i, col] = float(clipped)

        # Update current_close for next iteration
        current_close = float(pred_df.at[i, "close"])

    return pred_df


def plot_result(df_hist: pd.DataFrame, df_pred: pd.DataFrame, ts_code: str, save_dir: str):
    """
    Plot prediction results.

    Args:
        df_hist: Historical data DataFrame.
        df_pred: Prediction DataFrame.
        ts_code: Stock code.
        save_dir: Directory to save the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df_hist["date"], df_hist["close"], label="Historical", color="blue")
    plt.plot(df_pred["date"], df_pred["close"], label="Predicted", color="red", linestyle="--")
    plt.title(f"Kronos Prediction for {ts_code}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"pred_{ts_code.replace('.', '_')}_chart.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"📊 Chart saved: {plot_path}")


def predict_future(
    ts_code: str,
    pred_len: int = DEFAULT_PRED_LEN,
    lookback: int = DEFAULT_LOOKBACK,
    device: str = DEFAULT_DEVICE,
    model_name: str = DEFAULT_MODEL_PRETRAINED,
    tokenizer_name: str = DEFAULT_TOKENIZER_PRETRAINED,
    max_context: int = DEFAULT_MAX_CONTEXT,
    T: float = DEFAULT_T,
    top_p: float = DEFAULT_TOP_P,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    save_dir: str = DEFAULT_SAVE_DIR,
    local_files_only: bool = False,
):
    """
    Predict future stock prices using Kronos model.

    Args:
        ts_code: Stock code in Tushare format (e.g., 000001.SZ).
        pred_len: Number of trading days to predict.
        lookback: Number of historical days to use.
        device: Device to use (cpu or cuda:0).
        model_name: Pretrained model name.
        tokenizer_name: Pretrained tokenizer name.
        max_context: Maximum context length.
        T: Sampling temperature.
        top_p: Top-p (nucleus sampling) threshold.
        sample_count: Number of parallel samples.
        save_dir: Directory to save outputs.
    """
    # Load configuration
    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load model and tokenizer
    logger.info(f"🚀 Loading Kronos tokenizer: {tokenizer_name}, model: {model_name} ...")
    
    # Check cache status
    try:
        from huggingface_hub import HfFolder
        cache_dir = HfFolder.get_cache_dir()
        logger.debug(f"HuggingFace cache directory: {cache_dir}")
        
        # Check if models are already cached
        tokenizer_cache_path = Path(cache_dir) / "models--" + tokenizer_name.replace("/", "--")
        model_cache_path = Path(cache_dir) / "models--" + model_name.replace("/", "--")
        
        if tokenizer_cache_path.exists():
            logger.info(f"✅ Tokenizer cache found at: {tokenizer_cache_path}")
        else:
            logger.info(f"📥 Tokenizer will be downloaded and cached")
            
        if model_cache_path.exists():
            logger.info(f"✅ Model cache found at: {model_cache_path}")
        else:
            logger.info(f"📥 Model will be downloaded and cached")
    except Exception:
        # If we can't check cache, just continue
        pass
    
    try:
        # Load from cache if available, otherwise download
        tokenizer = KronosTokenizer.from_pretrained(
            tokenizer_name,
            local_files_only=local_files_only,
        )
        model = Kronos.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        predictor = KronosPredictor(
            model, tokenizer, device=device, max_context=max_context
        )
        logger.info("✅ Models loaded successfully")
    except ImportError as e:
        error_msg = str(e).lower()
        if "safetensors" in error_msg:
            logger.error(
                "safetensors library is required for loading Kronos models."
            )
            logger.info(
                "Please install it with: pip install safetensors>=0.4.3"
            )
            logger.info(
                "Or install all Kronos dependencies: pip install -e '.[kronos]'"
            )
        else:
            logger.error(f"Missing required dependency: {e}")
            logger.info(
                "To install all Kronos dependencies, run: "
                "pip install -e '.[kronos]'"
            )
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Load historical data
    df = load_data_from_db(
        db_config=db_config,
        ts_code=ts_code,
        lookback=lookback,
    )

    # Prepare inputs
    x_df, x_timestamp, y_timestamp = prepare_inputs(
        df=df,
        lookback=lookback,
        db_config=db_config,
        pred_len=pred_len,
        ts_code=ts_code,
    )

    logger.info("🔮 Generating predictions ...")

    # Generate predictions
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=T,
        top_p=top_p,
        sample_count=sample_count,
        verbose=True,
    )

    # Add date column
    pred_df["date"] = y_timestamp.values

    # Apply ±10% price limit
    last_close = df["close"].iloc[-1]
    pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.1)

    # Merge historical and predicted data
    df_out = pd.concat([
        df[["date", "open", "high", "low", "close", "volume", "amount"]],
        pred_df[["date", "open", "high", "low", "close", "volume", "amount"]]
    ]).reset_index(drop=True)

    # Save CSV
    out_file = os.path.join(save_dir, f"pred_{ts_code.replace('.', '_')}_data.csv")
    df_out.to_csv(out_file, index=False)
    logger.info(f"✅ Prediction completed and saved: {out_file}")

    # Plot
    plot_result(df, pred_df, ts_code, save_dir)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Kronos stock prediction script using database data"
    )
    parser.add_argument(
        "--ts_code",
        type=str,
        # required=True,
        default="000001.SZ",
        help="Stock code in Tushare format (e.g., 000001.SZ, 600000.SH)"
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=DEFAULT_PRED_LEN,
        help=f"Number of trading days to predict (default: {DEFAULT_PRED_LEN})"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK,
        help=f"Number of historical days to use (default: {DEFAULT_LOOKBACK})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Device to use (default: {DEFAULT_DEVICE})"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_PRETRAINED,
        help=f"Pretrained model name (default: {DEFAULT_MODEL_PRETRAINED})"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=DEFAULT_TOKENIZER_PRETRAINED,
        help=f"Pretrained tokenizer name (default: {DEFAULT_TOKENIZER_PRETRAINED})"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help=f"Directory to save outputs (default: {DEFAULT_SAVE_DIR})"
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only use local cached models, do not download from HuggingFace"
    )

    args = parser.parse_args()

    predict_future(
        ts_code=args.ts_code,
        pred_len=args.pred_len,
        lookback=args.lookback,
        device=args.device,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        save_dir=args.save_dir,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()

