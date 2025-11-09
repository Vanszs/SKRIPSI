"""
Training Pipeline untuk Hybrid Gas Fee Prediction Model.

Pipeline lengkap:
1. Load dan prepare data
2. Split train/val/test
3. Train hybrid LSTM→XGBoost model
4. Evaluate performance
5. Save models dan metrics
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from stack import HybridGasFeePredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline untuk gas fee prediction model.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline dengan configuration.
        
        Args:
            config_path: Path to config YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'baseFee_next'
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load feature data.
        
        Args:
            data_path: Path to features Parquet file
            
        Returns:
            DataFrame dengan features
        """
        logger.info(f"Loading data from {data_path}...")
        
        df = pd.read_parquet(data_path)
        
        logger.info(f"✓ Loaded {len(df)} samples")
        logger.info(f"  Features: {len(df.columns)}")
        logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, selected_features: list = None) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare features untuk training.
        
        Args:
            df: Feature DataFrame
            selected_features: Optional list of features to use (feature selection)
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Preparing features...")
        
        # Select feature columns (exclude metadata dan target)
        exclude_cols = ['number', 'timestamp', 'datetime', 'baseFee_next']
        
        if selected_features:
            # Use selected features only
            self.feature_columns = [col for col in selected_features if col in df.columns]
            missing = set(selected_features) - set(df.columns)
            if missing:
                logger.warning(f"Missing selected features: {missing}")
            logger.info(f"Using {len(self.feature_columns)} selected features")
        else:
            # Use all features
            self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        
        logger.info(f"✓ Features prepared: {X.shape}")
        logger.info(f"  Feature columns: {len(self.feature_columns)}")
        
        return X, y, self.feature_columns
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data ke train/val/test sets.
        
        Args:
            X, y: Features dan labels
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Splitting data...")
        
        val_split = self.config['data']['validation_split']
        test_split = self.config['data']['test_split']
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_split,
            shuffle=False  # Preserve temporal order
        )
        
        # Second split: separate train and validation
        val_size = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            shuffle=False
        )
        
        logger.info(f"✓ Data split:")
        logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        logger.info(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize_features(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features menggunakan StandardScaler.
        
        Args:
            X_train, X_val, X_test: Feature arrays
            
        Returns:
            Normalized arrays
        """
        logger.info("Normalizing features...")
        
        # Fit on training data only
        X_train_norm = self.scaler.fit_transform(X_train)
        X_val_norm = self.scaler.transform(X_val)
        X_test_norm = self.scaler.transform(X_test)
        
        logger.info("✓ Features normalized")
        
        return X_train_norm, X_val_norm, X_test_norm
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list
    ) -> HybridGasFeePredictor:
        """
        Train hybrid model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            feature_names: List of feature names
            
        Returns:
            Trained model
        """
        logger.info("\n" + "="*60)
        logger.info("TRAINING HYBRID MODEL")
        logger.info("="*60 + "\n")
        
        # Get model config
        lstm_config = {
            'input_size': X_train.shape[1],
            **self.config['model']['lstm']
        }
        xgb_config = self.config['model']['xgboost']
        
        # Create model
        model = HybridGasFeePredictor(
            lstm_config=lstm_config,
            xgb_config=xgb_config,
            sequence_length=self.config['data']['sequence_length']
        )
        
        # Train
        training_config = self.config['training']
        model.fit(
            X_train, y_train,
            X_val, y_val,
            feature_names=feature_names,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            batch_size=training_config['batch_size'],
            lstm_epochs=training_config['epochs'],
            lstm_lr=training_config['learning_rate']
        )
        
        return model
    
    def evaluate_model(
        self,
        model: HybridGasFeePredictor,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model pada test set.
        
        Args:
            model: Trained model
            X_test, y_test: Test data
            
        Returns:
            Dictionary dengan evaluation metrics
        """
        logger.info("\nEvaluating model on test set...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Align test labels dengan predictions (account for sequence length)
        seq_len = model.sequence_length
        y_test_aligned = y_test[seq_len - 1:]
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_test_aligned, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_aligned, y_pred))
        r2 = r2_score(y_test_aligned, y_pred)
        
        # MAPE
        mape = np.mean(np.abs((y_test_aligned - y_pred) / y_test_aligned)) * 100
        
        # Under-estimation rate
        under_est_rate = np.mean(y_pred < y_test_aligned) * 100
        
        # Hit@epsilon (within 5% tolerance)
        epsilon = self.config['evaluation']['epsilon']
        within_tolerance = np.abs(y_pred - y_test_aligned) / y_test_aligned <= epsilon
        hit_at_epsilon = np.mean(within_tolerance) * 100
        
        metrics = {
            'mae': float(mae),
            'mae_gwei': float(mae / 1e9),
            'rmse': float(rmse),
            'rmse_gwei': float(rmse / 1e9),
            'mape': float(mape),
            'r2': float(r2),
            'under_estimation_rate': float(under_est_rate),
            'hit_at_epsilon': float(hit_at_epsilon),
            'epsilon': float(epsilon * 100)
        }
        
        # Log metrics
        logger.info("\n" + "="*60)
        logger.info("EVALUATION METRICS")
        logger.info("="*60)
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        logger.info("="*60 + "\n")
        
        return metrics
    
    def save_artifacts(
        self,
        model: HybridGasFeePredictor,
        metrics: Dict[str, float],
        output_dir: str = 'models'
    ):
        """
        Save model dan artifacts.
        
        Args:
            model: Trained model
            metrics: Evaluation metrics
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save(output_dir)
        
        # Save scaler
        scaler_path = output_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"✓ Scaler saved: {scaler_path}")
        
        # Save metrics
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✓ Metrics saved: {metrics_path}")
        
        # Save training info
        info = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'feature_columns': self.feature_columns,
            'metrics': metrics
        }
        
        info_path = output_dir / 'training_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"✓ Training info saved: {info_path}")
    
    def run(self, data_path: str, output_dir: str = 'models', selected_features_path: str = None):
        """
        Run complete training pipeline.
        
        Args:
            data_path: Path to features file
            output_dir: Output directory untuk models
            selected_features_path: Optional path to selected features file
        """
        try:
            # Load data
            df = self.load_data(data_path)
            
            # Load selected features if provided
            selected_features = None
            if selected_features_path:
                logger.info(f"Loading selected features from {selected_features_path}")
                with open(selected_features_path, 'r') as f:
                    selected_features = [line.strip() for line in f if line.strip()]
                logger.info(f"✓ Loaded {len(selected_features)} selected features")
            
            # Prepare features
            X, y, feature_names = self.prepare_features(df, selected_features)
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
            
            # Normalize
            X_train, X_val, X_test = self.normalize_features(X_train, X_val, X_test)
            
            # Train model
            model = self.train_model(X_train, y_train, X_val, y_val, feature_names)
            
            # Evaluate
            metrics = self.evaluate_model(model, X_test, y_test)
            
            # Save
            self.save_artifacts(model, metrics, output_dir)
            
            logger.info("\n" + "="*60)
            logger.info("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Train hybrid gas fee prediction model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--in',
        dest='input_file',
        type=str,
        required=True,
        help='Path to features Parquet file'
    )
    
    parser.add_argument(
        '--out-dir',
        type=str,
        default='models',
        help='Output directory untuk models (default: models/)'
    )
    
    parser.add_argument(
        '--selected-features',
        type=str,
        default=None,
        help='Path to selected features file (optional, filters features before training)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Check torch availability
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Running on CPU")
        
        # Run pipeline
        pipeline = TrainingPipeline(args.cfg)
        pipeline.run(args.input_file, args.out_dir, args.selected_features)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("\n\nInterrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
