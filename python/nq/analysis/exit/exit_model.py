"""
Exit model for predicting when to sell positions.

Uses logistic regression to predict exit probability based on momentum exhaustion
and position management features.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .feature_builder import ExitFeatureBuilder

logger = logging.getLogger(__name__)


class ExitModel:
    """
    Exit prediction model using logistic regression.
    
    Predicts the probability that a position should be exited based on:
    - Momentum exhaustion indicators
    - Risk asymmetry / position management
    - Time-based features
    """

    def __init__(
        self,
        model: Optional[LogisticRegression] = None,
        scaler: Optional[StandardScaler] = None,
        feature_builder: Optional[ExitFeatureBuilder] = None,
        threshold: float = 0.65,
    ):
        """
        Initialize exit model.
        
        Args:
            model: Trained logistic regression model (if None, will be created on fit).
            scaler: Feature scaler (if None, will be created on fit).
            feature_builder: Feature builder instance (if None, will be created).
            threshold: Risk probability threshold for exit signal (default: 0.65).
        """
        self.model = model
        self.scaler = scaler
        self.feature_builder = (
            feature_builder if feature_builder is not None else ExitFeatureBuilder()
        )
        self.threshold = threshold
        self.feature_names: Optional[List[str]] = None

    def fit(
        self,
        feature_df: pd.DataFrame,
        target_col: str = "label",
        C: float = 0.1,
        class_weight: str = "balanced",
    ) -> "ExitModel":
        """
        Train the exit model.
        
        Args:
            feature_df: DataFrame with features and label column.
            target_col: Name of target column (default: 'label').
            C: Regularization strength (default: 0.1).
            class_weight: Class weight strategy (default: 'balanced').
        
        Returns:
            Self for method chaining.
        """
        if target_col not in feature_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in feature_df")

        # Drop target column and non-numeric columns (metadata like trade_id, symbol, date)
        X = feature_df.drop(columns=[target_col])
        y = feature_df[target_col]

        # Only keep numeric columns for features
        # Expected feature columns: bias_5, close_pos, vol_ratio, curr_ret, drawdown, days_held
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = set(X.columns) - set(numeric_cols)
        
        if non_numeric_cols:
            logger.info(
                f"Dropping non-numeric columns from features: {non_numeric_cols}. "
                f"These are likely metadata columns (trade_id, symbol, date, etc.)"
            )
            X = X[numeric_cols]

        # Store feature names
        self.feature_names = list(X.columns)

        # Initialize scaler if not provided
        if self.scaler is None:
            self.scaler = StandardScaler()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize model if not provided
        if self.model is None:
            self.model = LogisticRegression(
                C=C, class_weight=class_weight, random_state=42, max_iter=1000
            )

        # Train model
        self.model.fit(X_scaled, y)

        # Log feature importance
        if hasattr(self.model, "coef_") and len(self.model.coef_) > 0:
            importance = pd.DataFrame(
                {"feature": self.feature_names, "weight": self.model.coef_[0]}
            ).sort_values(by="weight", ascending=False)
            logger.info("Feature importance (sorted by weight):")
            logger.info(f"\n{importance.to_string()}")

        logger.info(
            f"Model trained: {len(X)} samples, "
            f"positive samples: {y.sum()}, "
            f"negative samples: {(y == 0).sum()}"
        )

        return self

    def predict_proba(
        self,
        daily_df: pd.DataFrame,
        entry_price: Optional[float] = None,
        highest_price_since_entry: Optional[float] = None,
        days_held: Optional[int] = None,
    ) -> np.ndarray:
        """
        Predict exit probability for given position data.
        
        Args:
            daily_df: DataFrame with daily OHLCV data.
            entry_price: Entry price for the position.
            highest_price_since_entry: Highest price since entry.
            days_held: Days held.
        
        Returns:
            Array of exit probabilities (probability of label=1).
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call fit() first or load from file.")

        # Build features
        features = self.feature_builder.build_features(
            daily_df=daily_df,
            entry_price=entry_price,
            highest_price_since_entry=highest_price_since_entry,
            days_held=days_held,
        )

        if features.empty:
            logger.warning("No features generated, returning zero probabilities")
            return np.array([])

        # Ensure feature order matches training
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(features.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            features = features[self.feature_names]

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict probabilities
        proba = self.model.predict_proba(features_scaled)

        # Return probability of positive class (label=1, i.e., should exit)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

    def predict(
        self,
        daily_df: pd.DataFrame,
        entry_price: Optional[float] = None,
        highest_price_since_entry: Optional[float] = None,
        days_held: Optional[int] = None,
    ) -> bool:
        """
        Predict whether to exit position.
        
        Args:
            daily_df: DataFrame with daily OHLCV data.
            entry_price: Entry price for the position.
            highest_price_since_entry: Highest price since entry.
            days_held: Days held.
        
        Returns:
            True if should exit, False otherwise.
        """
        proba = self.predict_proba(
            daily_df=daily_df,
            entry_price=entry_price,
            highest_price_since_entry=highest_price_since_entry,
            days_held=days_held,
        )

        if len(proba) == 0:
            return False

        # Use the latest probability
        latest_proba = proba[-1] if len(proba) > 0 else 0.0

        return latest_proba > self.threshold

    def save(self, model_path: str, scaler_path: Optional[str] = None) -> None:
        """
        Save model and scaler to files.
        
        Args:
            model_path: Path to save model.
            scaler_path: Path to save scaler (if None, uses model_path with '_scaler' suffix).
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")

        model_path_obj = Path(model_path)
        model_path_obj.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

        if self.scaler is not None:
            if scaler_path is None:
                scaler_path = str(model_path_obj.with_suffix("")) + "_scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")

    @classmethod
    def load(
        cls,
        model_path: str,
        scaler_path: Optional[str] = None,
        threshold: float = 0.65,
    ) -> "ExitModel":
        """
        Load model and scaler from files.
        
        Args:
            model_path: Path to model file.
            scaler_path: Path to scaler file (if None, uses model_path with '_scaler' suffix).
            threshold: Risk probability threshold for exit signal.
        
        Returns:
            Loaded ExitModel instance.
        """
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        if scaler_path is None:
            scaler_path = str(model_path_obj.with_suffix("")) + "_scaler.pkl"

        scaler = None
        scaler_path_obj = Path(scaler_path)
        if scaler_path_obj.exists():
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        else:
            logger.warning(f"Scaler file not found: {scaler_path}")

        return cls(model=model, scaler=scaler, threshold=threshold)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from trained model.
        
        Returns:
            DataFrame with feature names and weights, sorted by absolute weight.
        """
        if self.model is None or not hasattr(self.model, "coef_"):
            return None

        if self.feature_names is None:
            return None

        importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "weight": self.model.coef_[0],
                "abs_weight": np.abs(self.model.coef_[0]),
            }
        ).sort_values(by="abs_weight", ascending=False)

        return importance
