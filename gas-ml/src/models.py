"""
Modular single-model implementations for gas fee prediction.

This module provides lightweight regressors that can be trained and
saved independently. Each model exposes a unified interface so the
training pipeline and inference engine can orchestrate multiple
benchmarks (e.g., XGBoost, RandomForest, LSTM) without relying on the
previous hybrid stack implementation.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import joblib

# Try importing lightgbm, log warning if not found
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

logger = logging.getLogger(__name__)


class BlockSequenceDataset(Dataset):
    """PyTorch dataset that creates sliding window sequences."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, sequence_length: int):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.sequence_length = sequence_length
        self.valid_indices = list(range(sequence_length - 1, len(features)))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.sequence_length + 1
        sequence = self.features[start_idx:end_idx + 1]
        label = self.labels[end_idx]
        return sequence, label


class LSTMRegressorNet(nn.Module):
    """Multi-layer LSTM followed by a regression head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        output_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(output_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            features = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            features = h_n[-1]
        features = self.dropout(features)
        return self.head(features).squeeze(-1)


class BaseGasFeeModel(ABC, BaseEstimator, RegressorMixin):
    """Abstract base class for all regressors."""

    model_type: str = "base"
    _estimator_type: str = "regressor"
    requires_target_scaler: bool = False
    prediction_offset: int = 0

    def __init__(self, name: str, config: Dict[str, Any], sequence_length: int = 1):
        self.name = name
        self.config = config
        self.sequence_length = sequence_length

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the underlying model."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions in the same scale as the training target."""

    @abstractmethod
    def save(self, output_dir: Path) -> None:
        """Persist model weights to ``output_dir``."""

    @classmethod
    @abstractmethod
    def load(cls, model_dir: Path, metadata: Dict[str, Any]) -> "BaseGasFeeModel":
        """Instantiate model from ``model_dir`` and metadata."""


class XGBoostGasFeeModel(BaseGasFeeModel):
    model_type = "xgboost"
    requires_target_scaler = False

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name=name, config=config, sequence_length=1)
        self.model: Optional[xgb.XGBRegressor] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        params = self.config.copy()
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # Use asymmetric loss by default unless specified otherwise
        use_custom_obj = params.pop("use_custom_objective", True)
        objective = params.pop("objective", "reg:squarederror")
        
        if use_custom_obj:
            objective = self._asymmetric_mse_obj
            
        early_stopping_rounds = params.pop("early_stopping_rounds", None)
        self.model = xgb.XGBRegressor(**params, objective=objective)
        self.model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds, verbose=False)

    @staticmethod
    def _asymmetric_mse_obj(y_true, y_pred):
        """
        Custom XGBoost objective that penalizes under-estimation.
        Adapted from src/stack.py for standalone robustness.
        """
        residual = y_pred - y_true
        # Higher gradient for under-predictions
        grad = np.where(residual < 0, 2.5 * residual, residual)
        # Higher hessian for under-predictions for second-order optimization
        hess = np.where(residual < 0, 2.5 * np.ones_like(residual), np.ones_like(residual))
        return grad, hess

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained")
        return self.model.predict(X)

    def save(self, output_dir: Path) -> None:
        assert self.model is not None, "Cannot save untrained model"
        self.model.save_model(output_dir / "model.xgb.json")

    @classmethod
    def load(cls, model_dir: Path, metadata: Dict[str, Any]) -> "XGBoostGasFeeModel":
        model = cls(name=metadata["model_name"], config=metadata["model_params"])
        booster = xgb.XGBRegressor()
        booster.load_model(model_dir / "model.xgb.json")
        model.model = booster
        return model


class RandomForestGasFeeModel(BaseGasFeeModel):
    model_type = "random_forest"
    requires_target_scaler = False

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name=name, config=config)
        self.model: Optional[RandomForestRegressor] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        params = {
            "n_estimators": 400,
            "max_depth": None,
            "n_jobs": -1,
            "random_state": 42,
        }
        params.update(self.config or {})
        self.model = RandomForestRegressor(**params)
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained")
        return self.model.predict(X)

    def save(self, output_dir: Path) -> None:
        assert self.model is not None, "Cannot save untrained model"
        joblib.dump(self.model, output_dir / "model.joblib")

    @classmethod
    def load(cls, model_dir: Path, metadata: Dict[str, Any]) -> "RandomForestGasFeeModel":
        model = cls(name=metadata["model_name"], config=metadata["model_params"])
        model.model = joblib.load(model_dir / "model.joblib")
        return model


class MeanBaselineGasFeeModel(BaseGasFeeModel):
    model_type = "mean_baseline"
    requires_target_scaler = False

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name=name, config=config)
        self.mean_: Optional[float] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        if y_train.size == 0:
            raise ValueError("Mean baseline requires non-empty targets")
        self.mean_ = float(np.mean(y_train))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Model is not trained")
        return np.full(shape=(len(X),), fill_value=self.mean_, dtype=np.float32)

    def save(self, output_dir: Path) -> None:
        assert self.mean_ is not None, "Cannot save untrained model"
        payload = {"mean": self.mean_}
        with open(output_dir / "mean.json", "w", encoding="utf-8") as fp:
            json.dump(payload, fp)

    @classmethod
    def load(cls, model_dir: Path, metadata: Dict[str, Any]) -> "MeanBaselineGasFeeModel":
        model = cls(name=metadata["model_name"], config=metadata["model_params"])
        with open(model_dir / "mean.json", "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        model.mean_ = float(payload["mean"])
        return model


class LinearRegressionGasFeeModel(BaseGasFeeModel):
    model_type = "linear_regression"
    requires_target_scaler = False

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name=name, config=config)
        self.model: Optional[LinearRegression] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        params = self.config or {}
        self.model = LinearRegression(**params)
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained")
        return self.model.predict(X)

    def save(self, output_dir: Path) -> None:
        assert self.model is not None, "Cannot save untrained model"
        joblib.dump(self.model, output_dir / "model.joblib")

    @classmethod
    def load(cls, model_dir: Path, metadata: Dict[str, Any]) -> "LinearRegressionGasFeeModel":
        model = cls(name=metadata["model_name"], config=metadata["model_params"])
        model.model = joblib.load(model_dir / "model.joblib")
        return model


class RidgeGasFeeModel(BaseGasFeeModel):
    model_type = "ridge_regression"
    requires_target_scaler = False

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name=name, config=config)
        self.model: Optional[Ridge] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        params = self.config or {}
        self.model = Ridge(**params)
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained")
        return self.model.predict(X)

    def save(self, output_dir: Path) -> None:
        assert self.model is not None, "Cannot save untrained model"
        joblib.dump(self.model, output_dir / "model.joblib")

    @classmethod
    def load(cls, model_dir: Path, metadata: Dict[str, Any]) -> "RidgeGasFeeModel":
        model = cls(name=metadata["model_name"], config=metadata["model_params"])
        model.model = joblib.load(model_dir / "model.joblib")
        return model


class LightGBMGasFeeModel(BaseGasFeeModel):
    model_type = "lightgbm"
    requires_target_scaler = False

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name=name, config=config)
        self.model: Optional[Any] = None  # lgb.LGBMRegressor
        if lgb is None:
            logger.warning("LightGBM is not installed. Training will fail.")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        if lgb is None:
            raise ImportError("LightGBM not installed")
            
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "n_jobs": -1,
            "random_state": 42
        }
        params.update(self.config or {})
        
        # LightGBM supports dedicated validation sets
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            
        self.model = lgb.LGBMRegressor(**params)
        
        # Fit logic
        callbacks = []
        # Early stopping requires a list of callbacks in recent sklearn API or specialized calls
        # We'll use standard sklearn fit which handles eval_set if mapped correctly
        self.model.fit(
            X_train, 
            y_train, 
            eval_set=eval_set,
            # early_stopping_rounds=... (deprecated in favor of callbacks in newer versions, 
            # but usually supported in fit kwargs for backward compat. We'll skip complex setup for now)
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained")
        return self.model.predict(X)

    def save(self, output_dir: Path) -> None:
        assert self.model is not None, "Cannot save untrained model"
        # LightGBM sklearn API models can be pickled
        joblib.dump(self.model, output_dir / "model.joblib")

    @classmethod
    def load(cls, model_dir: Path, metadata: Dict[str, Any]) -> "LightGBMGasFeeModel":
        model = cls(name=metadata["model_name"], config=metadata["model_params"])
        model.model = joblib.load(model_dir / "model.joblib")
        return model