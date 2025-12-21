"""
Kronos predictor implementation.

Uses Kronos model for stock price prediction.
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from huggingface_hub import constants

from atm.ai.kronos import Kronos, KronosPredictor as KronosModelPredictor, KronosTokenizer
from atm.repo.kline_repo import StockKlineDayRepo
from atm.repo.trading_calendar_repo import TradingCalendarRepo

from .base import BasePredictor, PredictionResult

logger = logging.getLogger(__name__)


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


def apply_price_limits(
    pred_df: pd.DataFrame, last_close: float, limit_rate: float = 0.1
) -> pd.DataFrame:
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


class KronosPredictor(BasePredictor):
    """
    Kronos-based predictor for stock price prediction.

    Uses Kronos model from HuggingFace to predict future stock prices.
    """

    DEFAULT_LOOKBACK = 400
    DEFAULT_DEVICE = "cpu"
    DEFAULT_MODEL_NAME = "NeoQuasar/Kronos-base"
    DEFAULT_TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
    DEFAULT_MAX_CONTEXT = 512
    DEFAULT_T = 1.0
    DEFAULT_TOP_P = 0.9
    DEFAULT_SAMPLE_COUNT = 1

    def __init__(
        self,
        db_config,
        schema: str = "quant",
        device: str = DEFAULT_DEVICE,
        model_name: str = DEFAULT_MODEL_NAME,
        tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
        max_context: int = DEFAULT_MAX_CONTEXT,
        local_files_only: bool = False,
    ):
        """
        Initialize Kronos predictor.

        Args:
            db_config: Database configuration.
            schema: Database schema name.
            device: Device to use (cpu or cuda:0).
            model_name: Pretrained model name.
            tokenizer_name: Pretrained tokenizer name.
            max_context: Maximum context length.
            local_files_only: Only use local cached models.
        """
        super().__init__(db_config, schema)
        self.device = device
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.max_context = max_context
        self.local_files_only = local_files_only

        self.model: Optional[Kronos] = None
        self.tokenizer: Optional[KronosTokenizer] = None
        self.predictor: Optional[KronosModelPredictor] = None

    def load_model(self, **kwargs) -> None:
        """
        Load Kronos model and tokenizer.

        Args:
            **kwargs: Additional loading parameters.
        """
        logger.info(
            f"🚀 Loading Kronos tokenizer: {self.tokenizer_name}, model: {self.model_name} ..."
        )

        # Check cache status
        try:
            cache_dir = constants.default_cache_path()
            logger.debug(f"HuggingFace cache directory: {cache_dir}")

            tokenizer_cache_path = Path(cache_dir) / "models--" + self.tokenizer_name.replace("/", "--")
            model_cache_path = Path(cache_dir) / "models--" + self.model_name.replace("/", "--")

            if tokenizer_cache_path.exists():
                logger.info(f"✅ Tokenizer cache found at: {tokenizer_cache_path}")
            else:
                logger.info(f"📥 Tokenizer will be downloaded and cached")

            if model_cache_path.exists():
                logger.info(f"✅ Model cache found at: {model_cache_path}")
            else:
                logger.info(f"📥 Model will be downloaded and cached")
        except Exception:
            pass

        try:
            self.tokenizer = KronosTokenizer.from_pretrained(
                self.tokenizer_name,
                local_files_only=self.local_files_only,
            )
            self.model = Kronos.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self.predictor = KronosModelPredictor(
                self.model, self.tokenizer, device=self.device, max_context=self.max_context
            )
            logger.info("✅ Models loaded successfully")
        except ImportError as e:
            error_msg = str(e).lower()
            if "safetensors" in error_msg:
                logger.error(
                    "safetensors library is required for loading Kronos models."
                )
                logger.info("Please install it with: pip install safetensors>=0.4.3")
                logger.info("Or install all Kronos dependencies: pip install -e '.[kronos]'")
            else:
                logger.error(f"Missing required dependency: {e}")
                logger.info("To install all Kronos dependencies, run: pip install -e '.[kronos]'")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_historical_data(
        self,
        ts_code: str,
        lookback: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load historical K-line data from database.

        Args:
            ts_code: Stock code in Tushare format.
            lookback: Number of historical days to retrieve. Only used if start_date/end_date not specified.
            start_date: Start date for historical data (inclusive). If None, uses latest data.
            end_date: End date for historical data (inclusive). If None, uses latest data.

        Returns:
            DataFrame with OHLCV data.
        """
        logger.info(f"📥 Loading {ts_code} daily data from database ...")

        repo = StockKlineDayRepo(self.db_config, self.schema)

        # Load data based on time range or lookback
        if start_date is not None or end_date is not None:
            # Use time range
            klines = repo.get_by_ts_code(
                ts_code=ts_code,
                start_time=start_date,
                end_time=end_date,
            )
            if start_date:
                logger.info(f"   Using time range: {start_date.date()} to {end_date.date() if end_date else 'latest'}")
        else:
            # Use lookback (latest N days)
            klines = repo.get_by_ts_code(ts_code=ts_code, limit=lookback)
            logger.info(f"   Using latest {lookback} days")

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

        return df

    def _prepare_inputs(
        self,
        df: pd.DataFrame,
        lookback: int,
        pred_len: int,
        ts_code: str,
    ):
        """
        Prepare inputs for prediction.

        Args:
            df: Historical data DataFrame.
            lookback: Number of historical days to use (from the end of df).
            pred_len: Number of days to predict.
            ts_code: Stock code in Tushare format.

        Returns:
            Tuple of (x_df, x_timestamp, y_timestamp).
        """
        # Get the last lookback days from the dataframe
        # If df has fewer rows than lookback, use all available data
        actual_lookback = min(lookback, len(df))
        x_df = df.iloc[-actual_lookback:][["open", "high", "low", "close", "volume", "amount"]].copy()
        x_timestamp = df.iloc[-actual_lookback:]["date"].copy()

        # Determine exchange from ts_code
        exchange = infer_exchange_from_ts_code(ts_code)

        # Get the last date and find next trading day
        last_date = df["date"].iloc[-1].date()
        next_date = last_date + timedelta(days=1)

        # Get future trading days
        y_timestamp = get_future_trading_days(
            db_config=self.db_config,
            exchange=exchange,
            start_date=next_date,
            pred_len=pred_len,
            schema=self.schema,
        )

        return x_df, pd.Series(x_timestamp), y_timestamp

    def predict(
        self,
        ts_code: str,
        pred_len: int,
        lookback: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        T: float = DEFAULT_T,
        top_p: float = DEFAULT_TOP_P,
        sample_count: int = DEFAULT_SAMPLE_COUNT,
        apply_price_limit: bool = True,
        price_limit_rate: float = 0.1,
        verbose: bool = True,
        **kwargs,
    ) -> PredictionResult:
        """
        Predict future stock prices using Kronos model.

        Args:
            ts_code: Stock code in Tushare format (e.g., 000001.SZ).
            pred_len: Number of trading days to predict.
            lookback: Number of historical days to use. If None, use default.
                     Only used when start_date and end_date are not specified.
            start_date: Start date for historical data (inclusive). If specified,
                       loads data from this date. If None, uses latest data.
                       For backtesting, specify the start of training period.
            end_date: End date for historical data (inclusive). If specified,
                     loads data up to this date. If None, uses latest data.
                     For backtesting, this is the cutoff date before prediction period.
            T: Sampling temperature.
            top_p: Top-p (nucleus sampling) threshold.
            sample_count: Number of parallel samples.
            apply_price_limit: Whether to apply A-share price limits.
            price_limit_rate: Price limit rate (default: 0.1 for ±10%).
            verbose: Whether to show progress.
            **kwargs: Additional prediction parameters.

        Returns:
            PredictionResult containing historical and predicted data.
        """
        self.validate_inputs(ts_code, pred_len, lookback, start_date, end_date)

        # Use default lookback if not provided and no time range specified
        if lookback is None and start_date is None and end_date is None:
            lookback = self.DEFAULT_LOOKBACK

        # Load model if not already loaded
        if self.predictor is None:
            self.load_model()

        # Load historical data
        df = self._load_historical_data(
            ts_code=ts_code,
            lookback=lookback,
            start_date=start_date,
            end_date=end_date,
        )

        # Determine actual lookback to use
        # If time range is specified, use all loaded data
        # Otherwise, use specified lookback or default
        if start_date is not None or end_date is not None:
            # When time range is specified, use all loaded data
            actual_lookback = len(df)
            logger.info(
                f"Using all {actual_lookback} days from specified time range"
            )
        else:
            # When using latest data, use lookback (or all if lookback > available)
            actual_lookback = min(lookback, len(df)) if lookback is not None else len(df)
            if lookback is not None and len(df) < lookback:
                logger.warning(
                    f"Requested {lookback} days but only {len(df)} days available. "
                    f"Using all available data."
                )

        # Prepare inputs
        x_df, x_timestamp, y_timestamp = self._prepare_inputs(
            df=df,
            lookback=actual_lookback,
            pred_len=pred_len,
            ts_code=ts_code,
        )

        logger.info("🔮 Generating predictions ...")

        # Generate predictions
        pred_df = self.predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )

        # Add date column
        pred_df["date"] = y_timestamp.values

        # Apply price limits if requested
        if apply_price_limit:
            last_close = df["close"].iloc[-1]
            pred_df = apply_price_limits(pred_df, last_close, limit_rate=price_limit_rate)

        # Round numerical columns to 2 decimal places
        price_cols = ["open", "high", "low", "close"]
        pred_df[price_cols] = pred_df[price_cols].round(2)
        # Round amount to 2 decimal places
        if "amount" in pred_df.columns:
            pred_df["amount"] = pred_df["amount"].round(2)
        # Volume should be integer, but we'll round it anyway for consistency
        if "volume" in pred_df.columns:
            pred_df["volume"] = pred_df["volume"].round(0).astype(int)

        # Also round historical data for consistency
        hist_df = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
        hist_df[price_cols] = hist_df[price_cols].round(2)
        if "amount" in hist_df.columns:
            hist_df["amount"] = hist_df["amount"].round(2)
        if "volume" in hist_df.columns:
            hist_df["volume"] = hist_df["volume"].round(0).astype(int)

        # Create result
        result = PredictionResult(
            ts_code=ts_code,
            historical_data=hist_df,
            predicted_data=pred_df[["date", "open", "high", "low", "close", "volume", "amount"]].copy(),
            prediction_date=datetime.now(),
            lookback_days=actual_lookback,
            pred_len=pred_len,
            start_date=start_date,
            end_date=end_date,
            metadata={
                "model_name": self.model_name,
                "tokenizer_name": self.tokenizer_name,
                "device": self.device,
                "T": T,
                "top_p": top_p,
                "sample_count": sample_count,
                "apply_price_limit": apply_price_limit,
                "price_limit_rate": price_limit_rate,
                "requested_lookback": lookback,
                "actual_lookback_used": actual_lookback,
                "actual_historical_range": {
                    "start": df["date"].min().isoformat() if not df.empty else None,
                    "end": df["date"].max().isoformat() if not df.empty else None,
                },
            },
        )

        logger.info("✅ Prediction completed")

        return result

    def get_info(self) -> dict:
        """
        Get predictor information.

        Returns:
            Dictionary containing predictor information.
        """
        info = super().get_info()
        info.update({
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "device": self.device,
            "max_context": self.max_context,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
        })
        return info

