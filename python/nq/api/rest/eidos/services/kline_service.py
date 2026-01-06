"""
K-line service for handling K-line data retrieval and indicator calculation.

This service layer orchestrates between repositories and business logic,
following the handler -> service -> repo -> model architecture.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import qlib
from qlib.data import D

from nq.config import DatabaseConfig
from nq.repo.kline_repo import StockKlineDayRepo
from nq.repo.eidos_repo import EidosRepo
from nq.trading.indicators.technical_indicators import calculate_indicators

logger = logging.getLogger(__name__)


class KlineService:
    """
    Service for retrieving K-line data and calculating technical indicators.
    
    This service handles:
    - K-line data retrieval from database or Qlib
    - Date range extension for indicator calculation
    - Technical indicator calculation
    - Data format conversion
    """
    
    def __init__(
        self,
        eidos_repo: EidosRepo,
        db_config: Optional[DatabaseConfig] = None,
    ):
        """
        Initialize K-line service.
        
        Args:
            eidos_repo: Eidos repository for experiment data.
            db_config: Optional database configuration for K-line repository.
        """
        self.eidos_repo = eidos_repo
        self.db_config = db_config
    
    def get_stock_kline(
        self,
        exp_id: str,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        indicators: Optional[Dict[str, bool]] = None,
        extend_days: int = 60,
    ) -> Dict[str, Any]:
        """
        Get K-line data for a stock with optional technical indicators.
        
        Args:
            exp_id: Experiment ID.
            symbol: Stock symbol (e.g., "000001.SZ").
            start_date: Optional start date filter.
            end_date: Optional end date filter.
            indicators: Dictionary of indicator names to boolean flags.
            extend_days: Number of trading days to extend backward (default: 60).
        
        Returns:
            Dictionary with 'kline_data', 'indicators', 'backtest_start', 'backtest_end'.
        
        Raises:
            ValueError: If experiment not found.
            Exception: If data retrieval fails.
        """
        # Get experiment to get date range
        exp = self.eidos_repo.experiment.get_experiment(exp_id)
        if not exp:
            raise ValueError(f"Experiment {exp_id} not found")
        
        # Use experiment date range if not provided
        original_start_date = start_date
        if not start_date:
            exp_start = exp.get("start_date")
            if isinstance(exp_start, str):
                start_date = date.fromisoformat(exp_start)
            elif exp_start:
                start_date = exp_start
            original_start_date = start_date
        
        if not end_date:
            exp_end = exp.get("end_date")
            if isinstance(exp_end, str):
                end_date = date.fromisoformat(exp_end)
            elif exp_end:
                end_date = exp_end
        
        # Store original end date for backtest period marking
        original_end_date = end_date
        
        # Extend start_date backward to ensure sufficient data for indicators
        if start_date:
            extended_start_date = start_date - timedelta(days=int(extend_days * 1.4))
            logger.info(f"Extending K-line data range: original start={start_date}, extended start={extended_start_date}")
        else:
            extended_start_date = None
        
        # Extend end_date forward to allow viewing data beyond backtest period
        if end_date:
            extended_end_date = end_date + timedelta(days=int(30 * 1.4))
            logger.info(f"Extending K-line data range: original end={end_date}, extended end={extended_end_date}")
        else:
            extended_end_date = None
        
        try:
            # Try to get data from database first
            kline_data = self._get_kline_from_database(
                symbol=symbol,
                start_date=extended_start_date,
                end_date=extended_end_date,
                original_start_date=original_start_date,
                original_end_date=original_end_date,
            )
            
            if kline_data:
                # Calculate indicators if requested
                indicator_data = {}
                if indicators:
                    try:
                        indicator_data = calculate_indicators(kline_data, indicators)
                        logger.info(f"Calculated indicators: {list(indicator_data.keys())}")
                    except Exception as e:
                        logger.warning(f"Failed to calculate indicators: {e}", exc_info=True)
                
                return {
                    "kline_data": kline_data,
                    "indicators": indicator_data,
                    "backtest_start": original_start_date.isoformat() if original_start_date else None,
                    "backtest_end": original_end_date.isoformat() if original_end_date else None,
                }
            
            # Fallback to Qlib if database doesn't have data
            logger.info(f"No database K-line data found for {symbol}, trying Qlib...")
            kline_data = self._get_kline_from_qlib(
                symbol=symbol,
                start_date=extended_start_date,
                end_date=extended_end_date,
                original_start_date=original_start_date,
                original_end_date=original_end_date,
            )
            
            # Calculate indicators if requested
            indicator_data = {}
            if indicators:
                try:
                    indicator_data = calculate_indicators(kline_data, indicators)
                    logger.info(f"Calculated indicators: {list(indicator_data.keys())}")
                except Exception as e:
                    logger.warning(f"Failed to calculate indicators: {e}", exc_info=True)
            
            return {
                "kline_data": kline_data,
                "indicators": indicator_data,
                "backtest_start": original_start_date.isoformat() if original_start_date else None,
                "backtest_end": original_end_date.isoformat() if original_end_date else None,
            }
            
        except Exception as e:
            logger.error(f"Failed to get K-line data for {symbol}: {e}", exc_info=True)
            raise
    
    def _get_kline_from_database(
        self,
        symbol: str,
        start_date: Optional[date],
        end_date: Optional[date],
        original_start_date: Optional[date],
        original_end_date: Optional[date],
    ) -> List[Dict[str, Any]]:
        """
        Get K-line data from database.
        
        Args:
            symbol: Stock symbol.
            start_date: Extended start date.
            end_date: Extended end date.
            original_start_date: Original start date for backtest period marking.
            original_end_date: Original end date for backtest period marking.
        
        Returns:
            List of K-line dictionaries, or empty list if not found.
        """
        if not self.db_config:
            return []
        
        try:
            kline_repo = StockKlineDayRepo(self.db_config, schema="quant")
            ts_code = symbol
            
            # Convert date to datetime for repo query
            start_datetime = datetime.combine(start_date, datetime.min.time()) if start_date else None
            end_datetime = datetime.combine(end_date, datetime.max.time()) if end_date else None
            
            klines = kline_repo.get_by_ts_code(
                ts_code=ts_code,
                start_time=start_datetime,
                end_time=end_datetime,
            )
            
            if not klines:
                return []
            
            # Convert to list of dictionaries
            result = []
            for kline in klines:
                trade_date = kline.trade_date
                if isinstance(trade_date, datetime):
                    date_str = trade_date.strftime("%Y-%m-%d")
                elif isinstance(trade_date, date):
                    date_str = trade_date.strftime("%Y-%m-%d")
                else:
                    date_str = str(trade_date)
                
                item_date = date.fromisoformat(date_str) if len(date_str) == 10 else None
                
                result.append({
                    "date": date_str,
                    "open": float(kline.open) if kline.open else 0.0,
                    "high": float(kline.high) if kline.high else 0.0,
                    "low": float(kline.low) if kline.low else 0.0,
                    "close": float(kline.close) if kline.close else 0.0,
                    "volume": float(kline.volume) if kline.volume else 0.0,
                    "is_backtest_period": (
                        item_date is not None and
                        (original_start_date is None or item_date >= original_start_date) and
                        (original_end_date is None or item_date <= original_end_date)
                    ) if item_date else False,
                })
            
            # Sort by date
            result.sort(key=lambda x: x["date"])
            
            logger.info(f"Retrieved {len(result)} K-line data points for {symbol} from database")
            return result
            
        except Exception as e:
            logger.warning(f"Failed to get K-line data from database: {e}", exc_info=True)
            return []
    
    def _get_kline_from_qlib(
        self,
        symbol: str,
        start_date: Optional[date],
        end_date: Optional[date],
        original_start_date: Optional[date],
        original_end_date: Optional[date],
    ) -> List[Dict[str, Any]]:
        """
        Get K-line data from Qlib.
        
        Args:
            symbol: Stock symbol.
            start_date: Extended start date.
            end_date: Extended end date.
            original_start_date: Original start date for backtest period marking.
            original_end_date: Original end date for backtest period marking.
        
        Returns:
            List of K-line dictionaries.
        """
        # Initialize Qlib if not already initialized
        try:
            qlib.init()
        except:
            pass  # Already initialized
        
        # Fetch K-line data from Qlib
        start_str = start_date.strftime("%Y-%m-%d") if start_date else None
        end_str = end_date.strftime("%Y-%m-%d") if end_date else None
        
        # Get OHLCV data
        fields = ["$open", "$high", "$low", "$close", "$volume"]
        data = D.features(
            [symbol],
            fields,
            start_time=start_str,
            end_time=end_str,
            freq="day",
        )
        
        if data.empty:
            logger.warning(f"No K-line data found for {symbol} in date range {start_str} to {end_str}")
            return []
        
        # Convert to list of dictionaries
        result = []
        for idx, row in data.iterrows():
            date_obj = idx if hasattr(idx, "strftime") else pd.to_datetime(idx)
            date_str = date_obj.strftime("%Y-%m-%d") if hasattr(date_obj, "strftime") else str(date_obj)
            try:
                item_date = date.fromisoformat(date_str) if isinstance(date_str, str) and len(date_str) == 10 else None
            except:
                item_date = None
            
            result.append({
                "date": date_str,
                "open": float(row["$open"]) if pd.notna(row["$open"]) else 0.0,
                "high": float(row["$high"]) if pd.notna(row["$high"]) else 0.0,
                "low": float(row["$low"]) if pd.notna(row["$low"]) else 0.0,
                "close": float(row["$close"]) if pd.notna(row["$close"]) else 0.0,
                "volume": float(row["$volume"]) if pd.notna(row["$volume"]) else 0.0,
                "is_backtest_period": (
                    item_date is not None and
                    (original_start_date is None or item_date >= original_start_date) and
                    (original_end_date is None or item_date <= original_end_date)
                ) if item_date else False,
            })
        
        logger.info(f"Retrieved {len(result)} K-line data points for {symbol} from Qlib")
        return result

