"""
Multi-model Training Script for SINTA 3 Benchmarking.
Trains: Linear Regression, Random Forest, ARIMA, ETS, Prophet.
Independent of train.py to avoid conflicts.
"""

import logging
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Models
from src.models import (
    LinearRegressionGasFeeModel, 
    RandomForestGasFeeModel, 
    RidgeGasFeeModel,
    LightGBMGasFeeModel,
    BaseGasFeeModel
)
from src.baselines import (
    GradientBoostingGasFeeModel
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenchmarkPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.scaler = StandardScaler()
        self.target_column = 'baseFee_next'

    def load_data(self, data_path: str) -> pd.DataFrame:
        logger.info(f"Loading data from {data_path}...")
        return pd.read_parquet(data_path)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Same preparation logic as train.py for fair comparison."""
        
        # NOTE: XGBoost used FULL dataset (9994 rows) and 20% split (1998 samples).
        # We ignore config['n_blocks'] and config['test_split'] to force alignment.
        
        exclude_cols = ['number', 'timestamp', 'datetime', 'baseFee_next']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols].values
        y = df[self.target_column].values
        
        # Split
        test_split = 0.2
        val_split = 0.15 # Keep val split as is, doesn't affect test size
        
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_split, shuffle=False)
        val_size = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, shuffle=False)
        
        # Normalize Features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        # We do NOT normalize targets for benchmarks to keep interpretation simple (and aligned with standard baselines)
        # XGBoost also uses raw targets in our standalone setup.
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def evaluate(self, model: BaseGasFeeModel, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        preds = model.predict(X_test)
        
        # Basic Metrics
        mae = np.mean(np.abs(preds - y_test))
        mse = np.mean((preds - y_test)**2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        
        # R2
        ss_res = np.sum((y_test - preds) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            "mae": float(mae),
            "mae_wei": float(mae),
            "mae_gwei": float(mae / 1e9),
            "rmse": float(rmse),
            "rmse_wei": float(rmse),
            "rmse_gwei": float(rmse / 1e9),
            "mape": float(mape),
            "r2": float(r2),
        }

    def run(self, data_path: str, output_root: str):
        df = self.load_data(data_path)
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(df)
        
        models_to_train = [
            (LinearRegressionGasFeeModel, "linear_regression", {}),
            (RandomForestGasFeeModel, "random_forest", {"n_estimators": 100, "max_depth": 20, "n_jobs": -1}),
            (RidgeGasFeeModel, "ridge_regression", {"alpha": 1.0}),
            (LightGBMGasFeeModel, "lightgbm", {"n_estimators": 200, "learning_rate": 0.1, "n_jobs": -1}),
            # Replaced KNN with GradientBoosting as requested for robust baseline
            (GradientBoostingGasFeeModel, "gbr", {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 5}),
        ]
        
        results = {}

        for ModelClass, name, params in models_to_train:
            logger.info(f"\nTraining {name}...")
            save_dir = Path(output_root) / name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                model = ModelClass(name=name, config=params)
                model.fit(X_train, y_train, X_val, y_val)
                
                metrics = self.evaluate(model, X_test, y_test)
                logger.info(f"Metrics ({name}): {metrics}")
                
                # Save
                model.save(save_dir)
                with open(save_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                
                # Save Predictions for Plotting (First 1000 and Last 1000 for efficiency?)
                # Or just save all test preds
                preds = model.predict(X_test)
                np.save(save_dir / "predictions.npy", preds)
                np.save(save_dir / "y_test.npy", y_test) # redundant but safe
                
                results[name] = metrics
                
                # Metadata for universal loader
                metadata = {
                    "model_type": model.model_type,
                    "model_name": name,
                    "model_params": params,
                    "metrics": metrics,
                    "created_at": datetime.now().isoformat()
                }
                with open(save_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to train {name}: {e}", exc_info=True)

        # Summary
        logger.info("\nBenchmark Summary:")
        print(json.dumps(results, indent=2))
        
        # Save aggregate
        with open(Path(output_root) / "benchmark_summary.json", "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    # Default paths based on project structure
    config_path = "cfg/exp.yaml"
    if not Path(config_path).exists():
        config_path = "config.yaml"

    pipeline = BenchmarkPipeline(config_path)
    
    # Check common data locations
    data_paths = [
        "data/features.parquet",
        "data/processed/features.parquet",
        "data/final_features.parquet"
    ]
    
    data_path = None
    for p in data_paths:
        if Path(p).exists():
            data_path = p
            break
            
    if data_path is None:
         print(f"Error: Could not find data in any of: {data_paths}")
         sys.exit(1)
         
    print(f"Using config: {config_path}")
    print(f"Using data: {data_path}")
    
    pipeline.run(data_path, "models/benchmarks")
