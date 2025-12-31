"""
Model utility functions for ATM project.

This module provides utility functions for model operations, including
saving embeddings and other model-related operations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_embeddings(
    symbols: List[str],
    symbols_normalized: List[str],
    predictions: np.ndarray,
    embeddings: np.ndarray,
    trade_date_str: str,
    storage_dir: Optional[str] = None,
    industry_label_map: Optional[Dict[str, str]] = None,
) -> Optional[Path]:
    """
    Save model embeddings and predictions to parquet file.

    This function saves embeddings, predictions, and associated metadata (symbols,
    industry labels) to a parquet file for later analysis and visualization.

    Args:
        symbols: List of original stock symbols (may be in various formats).
        symbols_normalized: List of normalized stock symbols (standard format).
        predictions: Prediction scores array of shape [num_stocks].
        embeddings: Embedding vectors array of shape [num_stocks, embedding_dim].
        trade_date_str: Trading date string in format 'YYYY-MM-DD'.
        storage_dir: Directory to save embeddings (default: 'storage/structure_expert_cache').
        industry_label_map: Optional dictionary mapping stock codes to industry names.

    Returns:
        Path to saved file if successful, None otherwise.

    Examples:
        >>> symbols = ['000001.SZ', '600000.SH']
        >>> symbols_norm = ['000001.SZ', '600000.SH']
        >>> pred = np.array([0.5, 0.3])
        >>> emb = np.random.randn(2, 128)
        >>> path = save_embeddings(
        ...     symbols, symbols_norm, pred, emb, '2024-01-01',
        ...     industry_label_map={'000001.SZ': '银行', '600000.SH': '银行'}
        ... )
        >>> print(path)
        Path('storage/structure_expert_cache/embeddings_20240101.parquet')
    """
    try:
        storage_dir_path = Path(storage_dir) if storage_dir else Path("storage/structure_expert_cache")
        storage_dir_path.mkdir(parents=True, exist_ok=True)

        # Debug: Check symbol format and industry label map matching
        if industry_label_map:
            # Try multiple format variants for matching
            matched_count = 0
            unmatched_symbols = []
            for idx, s_norm in enumerate(symbols_normalized):
                # Try normalized format first
                if s_norm in industry_label_map:
                    matched_count += 1
                # Try lowercase variant
                elif s_norm.lower() in industry_label_map:
                    matched_count += 1
                # Try original symbol
                elif idx < len(symbols) and symbols[idx] in industry_label_map:
                    matched_count += 1
                else:
                    unmatched_symbols.append(s_norm)

            logger.info(f"Industry label matching for {trade_date_str}: {matched_count}/{len(symbols_normalized)} symbols matched")
            if matched_count == 0 and len(symbols_normalized) > 0:
                sample_symbols = symbols_normalized[:5] if len(symbols_normalized) > 5 else symbols_normalized
                sample_map_keys = list(industry_label_map.keys())[:5]
                logger.warning(f"No symbols matched! Sample symbols: {sample_symbols}, Sample map keys: {sample_map_keys}")
            elif len(unmatched_symbols) > 0 and len(unmatched_symbols) <= 10:
                logger.debug(f"Unmatched symbols: {unmatched_symbols}")

        # Create DataFrame with embeddings
        # Build all columns at once to avoid fragmentation
        # Use normalized symbols for matching (standard format)
        industry_labels = [
            industry_label_map.get(s_norm, "Unknown") if industry_label_map else "Unknown"
            for s_norm in symbols_normalized
        ]

        data_dict = {
            'symbol': symbols,  # Keep original symbols for display
            'score': predictions,
            'industry': industry_labels,
        }

        # Add embedding columns at once
        for i in range(embeddings.shape[1]):
            data_dict[f'embedding_{i}'] = embeddings[:, i]

        df_emb = pd.DataFrame(data_dict)

        # Save to parquet
        date_str = trade_date_str.replace("-", "")
        output_path = storage_dir_path / f"embeddings_{date_str}.parquet"
        df_emb.to_parquet(output_path, index=False)
        logger.info(f"Saved embeddings to {output_path}")
        return output_path
    except Exception as e:
        logger.warning(f"Failed to save embeddings for {trade_date_str}: {e}")
        return None

