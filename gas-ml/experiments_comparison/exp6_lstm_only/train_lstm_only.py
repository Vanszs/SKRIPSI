"""
LSTM Standalone Model for Gas Fee Prediction
Experiment 6: Pure temporal modeling baseline

This script trains a standalone LSTM model with direct prediction head,
using the same parameters as the hybrid model's LSTM component for fair comparison.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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


class BlockSequenceDataset(Dataset):
    """PyTorch Dataset for sequential block data."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, sequence_length: int = 20):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.sequence_length = sequence_length
        self.valid_indices = list(range(sequence_length - 1, len(features)))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.sequence_length + 1
        sequence = self.features[start_idx:end_idx + 1]
        label = self.labels[end_idx]
        return sequence, label


class LSTMPredictor(nn.Module):
    """
    LSTM network with direct prediction head.
    SAME architecture as hybrid's LSTM component.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 192,
        num_layers: int = 2,
        dropout: float = 0.25,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output size
        self.output_size = hidden_size * (2 if bidirectional else 1)
        
        # CRITICAL: Prediction head for direct output
        self.prediction_head = nn.Linear(self.output_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with prediction."""
        lstm_out, (hidden, cell) = self.lstm(x)
        # Use last time step
        features = lstm_out[:, -1, :]
        # Direct prediction
        prediction = self.prediction_head(features).squeeze()
        return prediction


def asymmetric_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, under_penalty: float = 2.5) -> torch.Tensor:
    """
    Asymmetric MSE loss that penalizes under-estimation.
    SAME as hybrid model's loss function.
    """
    residual = predictions - targets
    squared_residual = residual ** 2
    # Apply higher penalty for under-predictions (residual < 0)
    loss = torch.where(
        residual < 0,
        under_penalty * squared_residual,
        squared_residual
    )
    return loss.mean()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_and_prepare_data(
    data_path: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
    sequence_length: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split data - MATCHES hybrid exactly.
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
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"(Sequence windowing will reduce by {sequence_length-1} samples)")
    
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
    Normalize features and targets - SAME as hybrid.
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


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[LSTMPredictor, dict]:
    """
    Train LSTM model with asymmetric loss.
    MATCHES hybrid's LSTM training exactly.
    """
    logger.info("\n" + "="*60)
    logger.info("Training LSTM Model")
    logger.info("="*60)
    
    # Get configuration
    lstm_config = config['model']['lstm']
    training_config = config['training']
    sequence_length = config['data']['sequence_length']
    
    logger.info(f"Configuration:")
    logger.info(f"  Hidden size: {lstm_config['hidden_size']}")
    logger.info(f"  Num layers: {lstm_config['num_layers']}")
    logger.info(f"  Dropout: {lstm_config['dropout']}")
    logger.info(f"  Sequence length: {sequence_length}")
    logger.info(f"  Batch size: {training_config['batch_size']}")
    logger.info(f"  Learning rate: {training_config['learning_rate']}")
    logger.info(f"  Under-estimation penalty: {training_config['under_penalty']}x")
    
    # Create datasets
    train_dataset = BlockSequenceDataset(X_train, y_train, sequence_length)
    val_dataset = BlockSequenceDataset(X_val, y_val, sequence_length)
    
    # DataLoaders - CRITICAL: shuffle=False for time series
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False)
    
    logger.info(f"Training samples after windowing: {len(train_dataset)}")
    logger.info(f"Validation samples after windowing: {len(val_dataset)}")
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    model = LSTMPredictor(
        input_size=X_train.shape[1],
        hidden_size=lstm_config['hidden_size'],
        num_layers=lstm_config['num_layers'],
        dropout=lstm_config['dropout'],
        bidirectional=lstm_config['bidirectional']
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer - SAME as hybrid
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = training_config['early_stopping']['patience']
    min_delta = training_config['early_stopping']['min_delta']
    under_penalty = training_config['under_penalty']
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0
    }
    
    logger.info("\nStarting training...")
    
    for epoch in range(training_config['epochs']):
        # Training
        model.train()
        train_losses = []
        
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(sequences)
            loss = asymmetric_mse_loss(predictions, targets, under_penalty)
            loss.backward()
            
            # Gradient clipping - SAME as hybrid
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                predictions = model(sequences)
                loss = asymmetric_mse_loss(predictions, targets, under_penalty)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        
        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{training_config['epochs']} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            training_history['best_epoch'] = epoch + 1
            # Save best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    logger.info(f"✓ LSTM training complete. Best val loss: {best_val_loss:.6f}")
    logger.info(f"Best epoch: {training_history['best_epoch']}\n")
    
    return model, training_history


def evaluate_model(
    model: LSTMPredictor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler: StandardScaler,
    sequence_length: int,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate LSTM model on test set.
    """
    logger.info("\n" + "="*60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("="*60)
    
    # Create dataset
    test_dataset = BlockSequenceDataset(X_test, y_test, sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Predict
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for sequences, batch_targets in test_loader:
            sequences = sequences.to(device)
            batch_preds = model(sequences)
            predictions.append(batch_preds.cpu().numpy())
            targets.append(batch_targets.numpy())
    
    y_pred_norm = np.concatenate(predictions)
    y_true_norm = np.concatenate(targets)
    
    # Denormalize
    y_pred = target_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()
    
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
    model: LSTMPredictor,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    metrics: Dict[str, float],
    training_history: dict,
    config: Dict[str, Any],
    output_dir: str
):
    """Save model, scalers, and metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to {output_dir}")
    
    # Save LSTM model
    torch.save(model.state_dict(), output_path / "lstm_only.pt")
    
    # Save scalers
    with open(output_path / "feature_scaler.pkl", 'wb') as f:
        pickle.dump(feature_scaler, f)
    
    with open(output_path / "target_scaler.pkl", 'wb') as f:
        pickle.dump(target_scaler, f)
    
    # Save metrics (convert numpy types to Python types)
    metrics_serializable = {k: float(v) for k, v in metrics.items()}
    with open(output_path / "metrics.json", 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    # Save training history
    with open(output_path / "training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save metadata
    metadata = {
        'experiment': config['experiment']['name'],
        'version': config['experiment']['version'],
        'timestamp': datetime.now().isoformat(),
        'model_type': 'LSTM',
        'input_size': model.input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'output_size': model.output_size,
        'total_parameters': sum(p.numel() for p in model.parameters())
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("✓ Model saved successfully")


def main():
    parser = argparse.ArgumentParser(description='Train LSTM standalone model')
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
    sequence_length = config['data']['sequence_length']
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, training_history = train_lstm(
        X_train_norm, y_train_norm,
        X_val_norm, y_val_norm,
        config
    )
    
    # Evaluate model
    metrics = evaluate_model(model, X_test_norm, y_test_norm, target_scaler, sequence_length, device)
    
    # Save model
    save_model(model, feature_scaler, target_scaler, metrics, training_history, config, args.output)
    
    logger.info("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
