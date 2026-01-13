"""Trading strategy module for ATM project."""

from .arg_parser import (
    create_strategy_parser,
    parse_date_args,
    parse_strategy_args,
    validate_dates,
)
from .base import BaseStrategy, StrategyConfig
from .dual_model import DualModelStrategy
from .buy_models import StructureExpertBuyModel
from .sell_models import MLExitSellModel
from .atr_strategy import ATRStrategy
from .bollinger_60min_strategy import Bollinger60MinStrategy
from .bollinger_daily_strategy import BollingerDailyStrategy
from .chip_peak_strategy import ChipPeakStrategy
from .hma_strategy import HMAStrategy
from .volume_weighted_momentum_strategy import (
    VolumeWeightedMomentumStrategy,
)
from .cci_strategy import CCIStrategy
from .cci_strategy_optimized import CCIStrategyOptimized
from .fibonacci_ma_strategy import FibonacciMAStrategy
from .improved_macd_strategy import ImprovedMACDStrategy
from .rsi_macd_resonance_strategy import RSIMACDResonanceStrategy
from .strategy_runner import StrategyRunner
from .sma_cross_strategy import SMACrossStrategy
from .dmi_strategy import DMIStrategy
from .price_entropy_strategy import PriceEntropyStrategy
from .obv_rsi_resonance_strategy import OBVRSIResonanceStrategy
from .cci_bull_launch_strategy import CCIBullLaunchStrategy
from .atr_dynamic_stop_strategy import ATRDynamicStopStrategy
from .cmf_resonance_strategy import CMFResonanceStrategy
from .macd_obv_resonance_strategy import MACDOBVResonanceStrategy
from .holy_grail_strategy import HolyGrailStrategy
from .triple_ma_135_strategy import TripleMA135Strategy
from .rsi_multi_signal_strategy import RSIMultiSignalStrategy
from .chip_concentration_strategy import ChipConcentrationStrategy

__all__ = [
    # Backtrader strategies
    "BaseStrategy",
    "StrategyConfig",
    "StrategyRunner",
    "SMACrossStrategy",
    "CCIStrategy",
    "CCIStrategyOptimized",
    "ATRStrategy",
    "Bollinger60MinStrategy",
    "BollingerDailyStrategy",
    "ChipPeakStrategy",
    "HMAStrategy",
    "FibonacciMAStrategy",
    "ImprovedMACDStrategy",
    "RSIMACDResonanceStrategy",
    "VolumeWeightedMomentumStrategy",
    "DMIStrategy",
    "PriceEntropyStrategy",
    "OBVRSIResonanceStrategy",
    "CCIBullLaunchStrategy",
    "ATRDynamicStopStrategy",
    "CMFResonanceStrategy",
    "MACDOBVResonanceStrategy",
    "HolyGrailStrategy",
    "TripleMA135Strategy",
    "RSIMultiSignalStrategy",
    "ChipConcentrationStrategy",
    # Dual-model strategy (new trading framework)
    "DualModelStrategy",
    "StructureExpertBuyModel",
    "MLExitSellModel",
    # Utilities
    "create_strategy_parser",
    "parse_date_args",
    "parse_strategy_args",
    "validate_dates",
]

