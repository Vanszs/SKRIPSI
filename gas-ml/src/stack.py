"""
Hybrid LSTM→XGBoost Model untuk Gas Fee Prediction.

Architecture:
1. LSTM: Ekstraksi temporal patterns dari sequential block data
2. XGBoost: Meta-learner untuk final prediction menggunakan LSTM features

Model ini menggabungkan kekuatan:
- LSTM: Capture temporal dependencies dan sequential patterns
- XGBoost: Non-linear feature interactions dan robust predictions
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import pickle
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlockSequenceDataset(Dataset):
    """
    PyTorch Dataset untuk sequential block data.
    
    Creates sliding windows of block sequences untuk LSTM input.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 24
    ):
        """
        Initialize dataset.
        
        Args:
            features: Feature array (n_samples, n_features)
            labels: Label array (n_samples,)
            sequence_length: Number of blocks in sequence
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.sequence_length = sequence_length
        
        # Calculate valid indices for sequences
        self.valid_indices = list(range(
            sequence_length - 1,
            len(features)
        ))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sequence and label.
        
        Returns:
            Tuple of (sequence, label)
            - sequence: (sequence_length, n_features)
            - label: scalar
        """
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.sequence_length + 1
        
        sequence = self.features[start_idx:end_idx + 1]
        label = self.labels[end_idx]
        
        return sequence, label


class LSTMFeatureExtractor(nn.Module):
    """
    LSTM network untuk ekstraksi temporal features.
    
    Architecture:
    - Multi-layer LSTM
    - Dropout untuk regularization
    - Output: encoded temporal representation
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output dimension
        self.output_size = hidden_size * (2 if bidirectional else 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Prediction head for supervised pre-training
        self.prediction_head = nn.Linear(self.output_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            LSTM features (batch_size, hidden_size * directions)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            features = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            features = h_n[-1]
        
        # Apply dropout
        features = self.dropout(features)
        
        return features
    
    def extract_features(
        self,
        dataloader: DataLoader,
        device: str = 'cpu'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract LSTM features from dataloader.
        
        Args:
            dataloader: PyTorch DataLoader
            device: Device to use ('cpu' or 'cuda')
            
        Returns:
            Tuple of (features, labels)
        """
        self.eval()
        self.to(device)
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in dataloader:
                sequences = sequences.to(device)
                
                # Extract features
                features = self.forward(sequences)
                
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
        
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        
        return features, labels


class HybridGasFeePredictor:
    """
    Hybrid LSTM→XGBoost model untuk gas fee prediction.
    
    Pipeline:
    1. LSTM ekstrak temporal features dari sequence
    2. XGBoost predict baseFee menggunakan LSTM features + static features
    """
    
    def __init__(
        self,
        lstm_config: Dict[str, Any],
        xgb_config: Dict[str, Any],
        sequence_length: int = 24
    ):
        """
        Initialize hybrid model.
        
        Args:
            lstm_config: LSTM configuration dict
            xgb_config: XGBoost configuration dict
            sequence_length: Sequence length for LSTM
        """
        self.sequence_length = sequence_length
        self.lstm_config = lstm_config
        self.xgb_config = xgb_config
        
        # Models (initialized during fit)
        self.lstm_model: Optional[LSTMFeatureExtractor] = None
        self.xgb_model: Optional[xgb.XGBRegressor] = None
        
        # Feature metadata
        self.feature_names: Optional[list] = None
        self.n_input_features: Optional[int] = None
        
        # Target scaler for denormalization
        self.target_scaler: Optional[Any] = None
    
    def _train_lstm(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        epochs: int = 100,
        learning_rate: float = 0.001
    ) -> LSTMFeatureExtractor:
        """
        Train LSTM feature extractor.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: Device to use
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Trained LSTM model
        """
        logger.info("Training LSTM feature extractor...")
        
        # Initialize model
        model = LSTMFeatureExtractor(**self.lstm_config).to(device)
        
        # Custom loss function that penalizes under-estimation more heavily
        def asymmetric_mse_loss(predictions, targets, under_penalty=2.0):
            """
            Asymmetric MSE loss that penalizes under-estimation more.
            
            Args:
                predictions: Predicted values
                targets: Target values
                under_penalty: Penalty multiplier for under-estimation (default: 2.0)
            """
            errors = predictions - targets
            # Apply higher weight to under-estimations (predictions < targets)
            weights = torch.where(errors < 0, under_penalty, 1.0)
            squared_errors = (errors ** 2) * weights
            return squared_errors.mean()
        
        criterion = asymmetric_mse_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Gradient clipping to prevent exploding gradients
        max_grad_norm = 1.0
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for sequences, labels in train_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                # Forward
                optimizer.zero_grad()
                features = model(sequences)
                
                # Use proper prediction head for supervised pre-training
                prediction = model.prediction_head(features).squeeze()
                
                # Use asymmetric loss with under-estimation penalty
                loss = criterion(prediction, labels, under_penalty=2.5)
                
                # Backward
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(device)
                    labels = labels.to(device)
                    
                    features = model(sequences)
                    prediction = model.prediction_head(features).squeeze()
                    
                    # Use asymmetric loss for validation too
                    loss = criterion(prediction, labels, under_penalty=2.5)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Logging
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        logger.info(f"✓ LSTM training complete. Best val loss: {best_val_loss:.6f}")
        
        return model
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list,
        device: str = 'cpu',
        batch_size: int = 32,
        lstm_epochs: int = 100,
        lstm_lr: float = 0.001
    ):
        """
        Fit hybrid model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            feature_names: List of feature names
            device: Device for LSTM training
            batch_size: Batch size
            lstm_epochs: LSTM training epochs
            lstm_lr: LSTM learning rate
        """
        logger.info("\n" + "="*60)
        logger.info("Training Hybrid LSTM→XGBoost Model")
        logger.info("="*60 + "\n")
        
        self.feature_names = feature_names
        self.n_input_features = X_train.shape[1]
        
        # Step 1: Create datasets
        train_dataset = BlockSequenceDataset(X_train, y_train, self.sequence_length)
        val_dataset = BlockSequenceDataset(X_val, y_val, self.sequence_length)
        
        # CRITICAL: shuffle=False to preserve temporal order and avoid data leakage
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Step 2: Train LSTM
        self.lstm_model = self._train_lstm(
            train_loader, val_loader, device, lstm_epochs, lstm_lr
        )
        
        # Step 3: Extract LSTM features
        logger.info("\nExtracting LSTM features...")
        
        lstm_train_features, _ = self.lstm_model.extract_features(train_loader, device)
        lstm_val_features, _ = self.lstm_model.extract_features(val_loader, device)
        
        logger.info(f"✓ LSTM features extracted: {lstm_train_features.shape}")
        
        # Step 4: Prepare XGBoost training data
        # Combine LSTM features with original features (aligned with sequences)
        X_train_xgb = np.hstack([
            X_train[self.sequence_length - 1:],
            lstm_train_features
        ])
        y_train_xgb = y_train[self.sequence_length - 1:]
        
        X_val_xgb = np.hstack([
            X_val[self.sequence_length - 1:],
            lstm_val_features
        ])
        y_val_xgb = y_val[self.sequence_length - 1:]
        
        # Step 5: Train XGBoost with asymmetric loss
        logger.info("\nTraining XGBoost meta-learner with under-estimation penalty...")
        
        # Custom objective that penalizes under-estimation
        def asymmetric_mse_obj(y_true, y_pred):
            """Custom XGBoost objective that penalizes under-estimation."""
            residual = y_pred - y_true
            # Higher gradient for under-predictions
            grad = np.where(residual < 0, 2.5 * residual, residual)
            # Higher hessian for under-predictions for second-order optimization
            hess = np.where(residual < 0, 2.5 * np.ones_like(residual), np.ones_like(residual))
            return grad, hess
        
        # Create XGBoost model with custom objective
        xgb_config_custom = self.xgb_config.copy()
        # Remove objective if specified (we'll use custom)
        xgb_config_custom.pop('objective', None)
        
        self.xgb_model = xgb.XGBRegressor(
            **xgb_config_custom,
            objective=asymmetric_mse_obj
        )
        
        self.xgb_model.fit(
            X_train_xgb, y_train_xgb,
            eval_set=[(X_val_xgb, y_val_xgb)],
            verbose=False
        )
        
        logger.info("✓ XGBoost training complete")
        logger.info("\n" + "="*60)
        logger.info("Hybrid Model Training Complete!")
        logger.info("="*60 + "\n")
    
    def predict(self, X: np.ndarray, device: str = 'cpu', denormalize: bool = True) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features (n_samples, n_features)
            device: Device for LSTM inference
            denormalize: Whether to denormalize predictions back to original scale
            
        Returns:
            Predictions array (denormalized if target_scaler is available)
        """
        # Create dataset and loader
        # Use zeros for labels (not used in inference)
        dummy_labels = np.zeros(len(X))
        dataset = BlockSequenceDataset(X, dummy_labels, self.sequence_length)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        # Extract LSTM features
        lstm_features, _ = self.lstm_model.extract_features(loader, device)
        
        # Prepare XGBoost input
        X_xgb = np.hstack([
            X[self.sequence_length - 1:],
            lstm_features
        ])
        
        # Predict with XGBoost (normalized predictions)
        predictions = self.xgb_model.predict(X_xgb)
        
        # Denormalize predictions back to original scale (Wei)
        if denormalize and self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions
    
    def save(self, output_dir: str):
        """
        Save hybrid model.
        
        Args:
            output_dir: Directory to save models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LSTM
        lstm_path = output_dir / 'lstm.pt'
        torch.save(self.lstm_model.state_dict(), lstm_path)
        logger.info(f"✓ LSTM saved: {lstm_path}")
        
        # Save XGBoost
        xgb_path = output_dir / 'xgb.bin'
        self.xgb_model.save_model(xgb_path)
        logger.info(f"✓ XGBoost saved: {xgb_path}")
        
        # Save metadata (including target_scaler if available)
        metadata = {
            'lstm_config': self.lstm_config,
            'xgb_config': self.xgb_config,
            'sequence_length': self.sequence_length,
            'feature_names': self.feature_names,
            'n_input_features': self.n_input_features,
            'has_target_scaler': self.target_scaler is not None,
        }
        
        metadata_path = output_dir / 'hybrid_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save target scaler separately if available
        if self.target_scaler is not None:
            target_scaler_path = output_dir / 'target_scaler.pkl'
            with open(target_scaler_path, 'wb') as f:
                pickle.dump(self.target_scaler, f)
            logger.info(f"✓ Target scaler saved: {target_scaler_path}")
        
        logger.info(f"✓ Metadata saved: {metadata_path}")
        logger.info(f"\n✓ Complete hybrid model saved to: {output_dir}")
    
    @classmethod
    def load(cls, model_dir: str, device: str = 'cpu') -> 'HybridGasFeePredictor':
        """
        Load hybrid model from directory.
        
        Args:
            model_dir: Directory containing saved models
            device: Device to load LSTM model
            
        Returns:
            Loaded HybridGasFeePredictor instance
            
        Raises:
            FileNotFoundError: If required model files not found
            ValueError: If model metadata is invalid
        """
        model_dir = Path(model_dir)
        
        # Validate required files exist
        required_files = ['lstm.pt', 'xgb.bin', 'hybrid_metadata.pkl']
        missing_files = [f for f in required_files if not (model_dir / f).exists()]
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing required model files in {model_dir}: {missing_files}\n"
                f"Please train the model first using: python -m src.train"
            )
        
        # Load metadata
        try:
            with open(model_dir / 'hybrid_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load model metadata: {e}")
        
        # Create instance
        predictor = cls(
            lstm_config=metadata['lstm_config'],
            xgb_config=metadata['xgb_config'],
            sequence_length=metadata['sequence_length']
        )
        
        # Validate metadata
        required_keys = ['lstm_config', 'xgb_config', 'sequence_length', 'feature_names', 'n_input_features']
        missing_keys = [k for k in required_keys if k not in metadata]
        if missing_keys:
            raise ValueError(f"Invalid metadata: missing keys {missing_keys}")
        
        predictor.feature_names = metadata['feature_names']
        predictor.n_input_features = metadata['n_input_features']
        
        # Load target scaler if available
        if metadata.get('has_target_scaler', False):
            target_scaler_path = model_dir / 'target_scaler.pkl'
            if target_scaler_path.exists():
                with open(target_scaler_path, 'rb') as f:
                    predictor.target_scaler = pickle.load(f)
                logger.info("✓ Target scaler loaded")
        
        # Load LSTM
        try:
            predictor.lstm_model = LSTMFeatureExtractor(**metadata['lstm_config'])
            predictor.lstm_model.load_state_dict(
                torch.load(model_dir / 'lstm.pt', map_location=device, weights_only=True)
            )
            predictor.lstm_model.to(device)
            predictor.lstm_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load LSTM model: {e}")
        
        # Load XGBoost
        try:
            predictor.xgb_model = xgb.XGBRegressor()
            predictor.xgb_model.load_model(model_dir / 'xgb.bin')
        except Exception as e:
            raise RuntimeError(f"Failed to load XGBoost model: {e}")
        
        logger.info(f"✓ Hybrid model loaded successfully from: {model_dir}")
        logger.info(f"  - LSTM: {predictor.lstm_config['hidden_size']}x{predictor.lstm_config['num_layers']}")
        logger.info(f"  - Features: {predictor.n_input_features}")
        logger.info(f"  - Sequence length: {predictor.sequence_length}")
        
        return predictor
