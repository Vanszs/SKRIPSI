"""
Benchmark models for SINTA 3 comparison.
wrappers for ARIMA, ETS, and SVR to be used in train_benchmarks.py
"""
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib

from src.models import BaseGasFeeModel

logger = logging.getLogger(__name__)

# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

class GradientBoostingGasFeeModel(BaseGasFeeModel):
    model_type = "gbr"
    requires_target_scaler = False

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name=name, config=config)
        self.model = None
        self.params = config

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        
        # Instantiate GBR
        logger.info(f"Training GradientBoostingRegressor with params: {self.params}")
        self.model = GradientBoostingRegressor(
            n_estimators=self.params.get("n_estimators", 200),
            learning_rate=self.params.get("learning_rate", 0.1),
            max_depth=self.params.get("max_depth", 5),
            random_state=42,
            verbose=1
        )
        
        # Subsample if massive data (GBR can be slow)
        limit = 100000 
        if len(y_train) > limit:
            logger.warning(f"Subsampling GBR training data from {len(y_train)} to {limit}")
            indices = np.random.choice(len(y_train), limit, replace=False)
            X_sub = X_train[indices]
            y_sub = y_train[indices]
            self.model.fit(X_sub, y_sub)
        else:
            self.model.fit(X_train, y_train)
            
        logger.info("GradientBoosting fit complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict(X)

    def save(self, output_dir: Path) -> None:
        assert self.model is not None
        joblib.dump(self.model, output_dir / "gbr_model.joblib")

    @classmethod
    def load(cls, model_dir: Path, metadata: Dict[str, Any]) -> "GradientBoostingGasFeeModel":
        model = cls(name=metadata["model_name"], config=metadata["model_params"])
        model.model = joblib.load(model_dir / "gbr_model.joblib")
        return model
