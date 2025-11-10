"""
XGBoost Standalone Model for Gas Fee Prediction
Experiment 5: Baseline comparison against Hybrid LSTM-XGBoost

This script trains a standalone XGBoost model using the same parameters
as the hybrid model's XGBoost component for fair comparison.
"""

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def asymmetric_mse_obj(y_true, y_pred):
    """
    Custom asymmetric MSE objective for XGBoost.
    Penalizes under-estimation more heavily than over-estimation.
    
    Args:
        y_true: Ground truth (DMatrix labels)
        y_pred: Predictions
        
    Returns:
        Tuple of (gradient, hessian)
    """
    under_penalty = 2.5
    
    residual = y_pred - y_true
    grad = np.where(residual < 0, -2 * under_penalty * residual, 2 * residual)
    hess = np.where(residual < 0, 2 * under_penalty, 2)
    
    return grad, hess


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_and_prepare_data(
    data_path: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_state: int = 42,
    sequence_length: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split data into train/val/test sets.
    MATCHES HYBRID: Drops first (sequence_length-1) samples for fair comparison.
    
    Args:
        data_path: Path to features.parquet
        val_split: Validation split ratio
        test_split: Test split ratio
        random_state: Random seed
        sequence_length: Sequence length (to match hybrid's data loss)
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Get target
    target_col = 'baseFee_next' if 'baseFee_next' in df.columns else 'baseFeePerGas'
    y = df[target_col].values
    
    # Get features (drop target columns)
    drop_cols = [col for col in df.columns if 'baseFee' in col and 'ema' not in col.lower()]
    X = df.drop(columns=drop_cols, errors='ignore').values
    
    logger.info(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    
    # Calculate split indices
    n_samples = len(X)
    n_train = int(n_samples * (1 - val_split - test_split))
    n_val = int(n_samples * val_split)
    
    # Split (time-ordered, no shuffle)
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    # CRITICAL FIX: Match hybrid's data alignment by dropping first (sequence_length-1) samples
    # Hybrid loses these samples due to sequence windowing
    skip = sequence_length - 1
    X_train = X_train[skip:]
    y_train = y_train[skip:]
    X_val = X_val[skip:]
    y_val = y_val[skip:]
    X_test = X_test[skip:]
    y_test = y_test[skip:]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"(Dropped first {skip} samples to match hybrid's sequence windowing)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """
    Normalize features and targets using StandardScaler.
    
    Returns:
        Tuple of (normalized X_train, X_val, X_test, y_train, y_val, y_test, feature_scaler, target_scaler)
    """
    logger.info("Normalizing features...")
    feature_scaler = StandardScaler()
    X_train_norm = feature_scaler.fit_transform(X_train)
    X_val_norm = feature_scaler.transform(X_val)
    X_test_norm = feature_scaler.transform(X_test)
    
    logger.info("Normalizing targets...")
    target_scaler = StandardScaler()
    y_train_norm = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_norm = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_norm = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    logger.info(f"✓ Features normalized (mean≈0, std≈1)")
    logger.info(f"✓ Targets normalized (mean={target_scaler.mean_[0]/1e9:.2f} Gwei, std={np.sqrt(target_scaler.var_[0])/1e9:.2f} Gwei)")
    
    return X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, feature_scaler, target_scaler


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any]
) -> xgb.XGBRegressor:
    """
    Train XGBoost model with asymmetric loss.
    
    Args:
        X_train: Training features (normalized)
        y_train: Training targets (normalized)
        X_val: Validation features (normalized)
        y_val: Validation targets (normalized)
        config: XGBoost configuration
        
    Returns:
        Trained XGBoost model
    """
    logger.info("\n" + "="*60)
    logger.info("Training XGBoost Model")
    logger.info("="*60)
    
    xgb_params = config['model']['xgboost'].copy()
    under_penalty = xgb_params.pop('under_penalty', 2.5)
    early_stop = xgb_params.pop('early_stopping_rounds', 70)
    
    logger.info(f"Parameters: {xgb_params}")
    logger.info(f"Using SAME asymmetric objective as hybrid (penalty={under_penalty}x)")
    
    # Custom objective that penalizes under-estimation (SAME as hybrid)
    def asymmetric_mse_obj(y_true, y_pred):
        """Custom XGBoost objective that penalizes under-estimation."""
        residual = y_pred - y_true
        # Higher gradient for under-predictions
        grad = np.where(residual < 0, under_penalty * residual, residual)
        # Higher hessian for under-predictions for second-order optimization
        hess = np.where(residual < 0, under_penalty * np.ones_like(residual), np.ones_like(residual))
        return grad, hess
    
    model = xgb.XGBRegressor(
        **xgb_params,
        objective=asymmetric_mse_obj,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )
    
    logger.info("✓ XGBoost training complete\n")
    
    return model


def evaluate_model(
    model: xgb.XGBRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler: StandardScaler
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test features (normalized)
        y_test: Test targets (normalized)
        target_scaler: Scaler for denormalization
        
    Returns:
        Dictionary of metrics
    """
    logger.info("\n" + "="*60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("="*60)
    
    # Predict (normalized)
    y_pred_norm = model.predict(X_test)
    
    # Denormalize predictions and targets
    y_pred = target_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics (in original scale - Wei)
    mae_wei = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse_wei = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Under-estimation rate
    under_est_rate = (np.sum(y_pred < y_true) / len(y_true)) * 100
    
    # Hit@epsilon (5% tolerance)
    epsilon = 0.05
    relative_error = np.abs((y_true - y_pred) / y_true)
    hit_at_epsilon = (np.sum(relative_error <= epsilon) / len(y_true)) * 100
    
    # Convert to Gwei for display
    mae_gwei = mae_wei / 1e9
    rmse_gwei = rmse_wei / 1e9
    
    metrics = {
        'mae_wei': mae_wei,
        'mae_gwei': mae_gwei,
        'mape': mape,
        'rmse_wei': rmse_wei,
        'rmse_gwei': rmse_gwei,
        'r2': r2,
        'under_estimation_rate': under_est_rate,
        'hit_at_epsilon': hit_at_epsilon
    }
    
    # Display metrics
    logger.info("\nMETRICS:")
    logger.info(f"  MAE:              {mae_gwei:.4f} Gwei")
    logger.info(f"  MAPE:             {mape:.2f}%")
    logger.info(f"  RMSE:             {rmse_gwei:.4f} Gwei")
    logger.info(f"  R²:               {r2:.4f}")
    logger.info(f"  Under-estimation: {under_est_rate:.2f}%")
    logger.info(f"  Hit@ε=5%:         {hit_at_epsilon:.2f}%")
    logger.info("\n" + "="*60 + "\n")
    
    return metrics


def save_model(
    model: xgb.XGBRegressor,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    output_dir: str
):
    """Save model, scalers, and metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to {output_dir}")
    
    # Save XGBoost model
    model.save_model(str(output_path / "xgboost_only.bin"))
    
    # Save scalers
    with open(output_path / "feature_scaler.pkl", 'wb') as f:
        pickle.dump(feature_scaler, f)
    
    with open(output_path / "target_scaler.pkl", 'wb') as f:
        pickle.dump(target_scaler, f)
    
    # Save metrics
    with open(output_path / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save metadata
    metadata = {
        'experiment': config['experiment']['name'],
        'version': config['experiment']['version'],
        'timestamp': datetime.now().isoformat(),
        'n_features': model.n_features_in_,
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'learning_rate': model.learning_rate
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("✓ Model saved successfully")


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost standalone model')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--data', type=str, required=True, help='Path to features.parquet')
    parser.add_argument('--output', type=str, required=True, help='Output directory for model')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Version: {config['experiment']['version']}")
    logger.info(f"{'='*60}\n")
    
    # Load and prepare data
    sequence_length = 20  # Match hybrid's sequence length
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(
        args.data,
        val_split=config['data']['validation_split'],
        test_split=config['data']['test_split'],
        sequence_length=sequence_length
    )
    
    # Normalize data
    X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, feature_scaler, target_scaler = normalize_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Train model
    model = train_xgboost(
        X_train_norm, y_train_norm,
        X_val_norm, y_val_norm,
        config
    )
    
    # Evaluate model
    metrics = evaluate_model(model, X_test_norm, y_test_norm, target_scaler)
    
    # Save model
    save_model(model, feature_scaler, target_scaler, metrics, config, args.output)
    
    logger.info("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
