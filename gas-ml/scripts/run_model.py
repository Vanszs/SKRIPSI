"""
Model Inference CLI - Run trained model for gas fee prediction

Usage:
    # Predict next block gas fee from latest data
    python scripts/run_model.py predict --blocks data/blocks_5k.csv
    
    # Predict with custom confidence threshold
    python scripts/run_model.py predict --blocks data/blocks_5k.csv --confidence 0.9
    
    # Batch prediction mode
    python scripts/run_model.py batch --input data/features.parquet --output predictions.csv
    
    # Real-time monitoring mode
    python scripts/run_model.py monitor --network mainnet --interval 12
    
    # Model info
    python scripts/run_model.py info
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import json
import time
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime
from typing import Dict, Any, Optional

from src.features import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelInference:
    """Run inference with trained hybrid LSTM-XGBoost model"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model components
        self.load_model()
        self.load_metadata()
        
    def load_model(self):
        """Load trained model and scaler"""
        logger.info(f"Loading model from {self.model_dir}...")
        
        # Load LSTM
        lstm_path = self.model_dir / "lstm.pt"
        if not lstm_path.exists():
            raise FileNotFoundError(f"LSTM model not found: {lstm_path}")
        
        self.lstm_state = torch.load(lstm_path, map_location=self.device)
        
        # Load XGBoost
        xgb_path = self.model_dir / "xgb.bin"
        if not xgb_path.exists():
            raise FileNotFoundError(f"XGBoost model not found: {xgb_path}")
        
        import xgboost as xgb
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(str(xgb_path))
        
        # Load scaler
        scaler_path = self.model_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = self.model_dir / "hybrid_metadata.pkl"
        if metadata_path.exists():
            self.metadata = joblib.load(metadata_path)
        else:
            self.metadata = {}
        
        logger.info("✓ Model loaded successfully")
        
    def load_metadata(self):
        """Load training info and metrics"""
        # Load metrics
        metrics_path = self.model_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {}
        
        # Load training info
        info_path = self.model_dir / "training_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.training_info = json.load(f)
        else:
            self.training_info = {}
    
    def prepare_input(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare input features for prediction"""
        # Engineer features
        engineer = FeatureEngineer()
        df = engineer.create_temporal_features(df)
        df = engineer.create_utilization_features(df)
        df = engineer.create_delta_features(df)
        df = engineer.create_ema_features(df)
        df = engineer.create_rolling_features(df)
        df = engineer.create_momentum_features(df)  # Add momentum
        df = engineer.create_interaction_features(df)  # Add interaction
        df = engineer.create_statistical_features(df)  # Add statistical
        
        # Select features used in training
        selected_features_path = Path("data/selected_features.txt")
        if selected_features_path.exists():
            with open(selected_features_path, 'r') as f:
                feature_names = [line.strip() for line in f if line.strip()]
        else:
            # Fallback to common features
            feature_names = [
                'baseFeePerGas', 'delta_baseFee', 'utilization', 
                'ema_baseFee_short', 'ema_baseFee_long',
                'rolling_mean_baseFee_6', 'rolling_std_baseFee_6',
                'hour_cos', 'congestion_score'
            ]
        
        # Get available features
        available_features = [f for f in feature_names if f in df.columns]
        X = df[available_features].values
        
        # Remove NaN rows
        X = X[~np.isnan(X).any(axis=1)]
        
        # Normalize
        X = self.scaler.transform(X)
        
        return X
    
    def predict_lstm_features(self, X: np.ndarray, sequence_length: int = 20) -> np.ndarray:
        """Extract LSTM features"""
        from src.stack import LSTMFeatureExtractor
        
        # Reconstruct LSTM model
        input_size = X.shape[1]
        hidden_size = self.metadata.get('lstm_hidden_size', 192)
        num_layers = self.metadata.get('lstm_num_layers', 2)
        dropout = self.metadata.get('lstm_dropout', 0.25)
        
        # Create model and load state
        lstm = LSTMFeatureExtractor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        lstm.load_state_dict(self.lstm_state)
        lstm.to(self.device)
        lstm.eval()
        
        # Create sequences
        sequences = []
        for i in range(len(X) - sequence_length + 1):
            seq = X[i:i+sequence_length]
            sequences.append(seq)
        
        if not sequences:
            raise ValueError(f"Not enough data for sequence_length={sequence_length}")
        
        sequences = np.array(sequences)
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        
        # Extract features
        with torch.no_grad():
            lstm_features = lstm(sequences_tensor)
            lstm_features = lstm_features.cpu().numpy()
        
        return lstm_features
    
    def predict(self, X: np.ndarray, sequence_length: int = 20) -> Dict[str, Any]:
        """Run full prediction pipeline"""
        # Get LSTM features
        lstm_features = self.predict_lstm_features(X, sequence_length)
        
        # Predict with XGBoost (using DMatrix)
        import xgboost as xgb
        dmatrix = xgb.DMatrix(lstm_features)
        predictions_wei = self.xgb_model.predict(dmatrix)
        predictions_gwei = predictions_wei / 1e9
        
        # Calculate statistics
        result = {
            'predictions_wei': predictions_wei.tolist(),
            'predictions_gwei': predictions_gwei.tolist(),
            'mean_gwei': float(np.mean(predictions_gwei)),
            'median_gwei': float(np.median(predictions_gwei)),
            'std_gwei': float(np.std(predictions_gwei)),
            'min_gwei': float(np.min(predictions_gwei)),
            'max_gwei': float(np.max(predictions_gwei)),
            'count': len(predictions_gwei)
        }
        
        return result
    
    def predict_next_block(self, df: pd.DataFrame, confidence: float = 0.85) -> Dict[str, Any]:
        """Predict next block gas fee with confidence interval"""
        # Prepare input
        X = self.prepare_input(df)
        
        # Get prediction
        result = self.predict(X, sequence_length=20)
        
        # Get latest prediction (most recent)
        latest_prediction = result['predictions_gwei'][-1]
        
        # Calculate confidence interval based on model MAE
        mae_gwei = self.metrics.get('mae_gwei', 0.0021)
        confidence_interval = mae_gwei * 1.96  # 95% CI
        
        # Policy recommendations
        buffer_multiplier = 1.1
        recommended_base_fee = latest_prediction * buffer_multiplier
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'predicted_base_fee_gwei': round(latest_prediction, 6),
            'confidence_interval': round(confidence_interval, 6),
            'lower_bound_gwei': round(latest_prediction - confidence_interval, 6),
            'upper_bound_gwei': round(latest_prediction + confidence_interval, 6),
            'recommended_base_fee_gwei': round(recommended_base_fee, 6),
            'model_performance': {
                'mae_gwei': self.metrics.get('mae_gwei', 0),
                'mape_percent': self.metrics.get('mape', 0),
                'r2': self.metrics.get('r2', 0),
                'hit_at_5_percent': self.metrics.get('hit_at_epsilon', 0)
            }
        }
        
        return output
    
    def show_info(self):
        """Display model information"""
        logger.info("="*70)
        logger.info("MODEL INFORMATION")
        logger.info("="*70)
        
        # Model architecture
        logger.info("\nArchitecture:")
        logger.info(f"  Type: Hybrid LSTM → XGBoost")
        logger.info(f"  LSTM Hidden Size: {self.metadata.get('lstm_hidden_size', 'N/A')}")
        logger.info(f"  LSTM Layers: {self.metadata.get('lstm_num_layers', 'N/A')}")
        logger.info(f"  XGBoost Trees: {self.metadata.get('xgb_n_estimators', 'N/A')}")
        logger.info(f"  Device: {self.device}")
        
        # Training info
        if self.training_info:
            logger.info("\nTraining Info:")
            logger.info(f"  Dataset: {self.training_info.get('dataset_file', 'N/A')}")
            logger.info(f"  Train Samples: {self.training_info.get('train_samples', 'N/A')}")
            logger.info(f"  Val Samples: {self.training_info.get('val_samples', 'N/A')}")
            logger.info(f"  Test Samples: {self.training_info.get('test_samples', 'N/A')}")
            logger.info(f"  Features: {self.training_info.get('n_features', 'N/A')}")
        
        # Performance metrics
        if self.metrics:
            logger.info("\n🎯 Performance Metrics:")
            logger.info(f"  MAE:  {self.metrics.get('mae_gwei', 0):.4f} Gwei")
            logger.info(f"  MAPE: {self.metrics.get('mape', 0):.2f}%")
            logger.info(f"  RMSE: {self.metrics.get('rmse_gwei', 0):.4f} Gwei")
            logger.info(f"  R²:   {self.metrics.get('r2', 0):.4f}")
            logger.info(f"  Hit@5%: {self.metrics.get('hit_at_epsilon', 0):.2f}%")
            logger.info(f"  Under-estimation: {self.metrics.get('under_estimation_rate', 0):.2f}%")
        
        logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Model Inference CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show model info
  python scripts/run_model.py info
  
  # Predict next block from CSV
  python scripts/run_model.py predict --blocks data/blocks_5k.csv
  
  # Batch predictions
  python scripts/run_model.py batch --input data/features.parquet --output predictions.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Info command
    subparsers.add_parser('info', help='Show model information')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict next block gas fee')
    predict_parser.add_argument('--blocks', type=str, required=True,
                                help='Path to blocks CSV file')
    predict_parser.add_argument('--confidence', type=float, default=0.85,
                                help='Confidence threshold (default: 0.85)')
    predict_parser.add_argument('--last-n', type=int, default=100,
                                help='Use last N blocks (default: 100)')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch prediction mode')
    batch_parser.add_argument('--input', type=str, required=True,
                              help='Input features parquet file')
    batch_parser.add_argument('--output', type=str, required=True,
                              help='Output predictions CSV file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize model
    inference = ModelInference()
    
    if args.command == 'info':
        inference.show_info()
    
    elif args.command == 'predict':
        logger.info(f"Loading blocks from {args.blocks}...")
        df = pd.read_csv(args.blocks)
        
        # Use last N blocks
        df = df.tail(args.last_n)
        logger.info(f"Using last {len(df)} blocks")
        
        # Predict
        logger.info("\nRunning prediction...")
        result = inference.predict_next_block(df, confidence=args.confidence)
        
        # Display result
        logger.info("\n" + "="*70)
        logger.info("PREDICTION RESULT")
        logger.info("="*70)
        logger.info(f"\n🎯 Predicted Base Fee: {result['predicted_base_fee_gwei']:.6f} Gwei")
        logger.info(f"📊 Confidence Interval: ±{result['confidence_interval']:.6f} Gwei")
        logger.info(f"   Lower Bound: {result['lower_bound_gwei']:.6f} Gwei")
        logger.info(f"   Upper Bound: {result['upper_bound_gwei']:.6f} Gwei")
        logger.info(f"\n💡 Recommended Base Fee: {result['recommended_base_fee_gwei']:.6f} Gwei")
        logger.info(f"   (with 10% safety buffer)")
        
        logger.info(f"\n📈 Model Performance:")
        perf = result['model_performance']
        logger.info(f"   MAE: {perf['mae_gwei']:.4f} Gwei")
        logger.info(f"   MAPE: {perf['mape_percent']:.2f}%")
        logger.info(f"   R²: {perf['r2']:.4f}")
        logger.info(f"   Hit@5%: {perf['hit_at_5_percent']:.2f}%")
        logger.info("="*70)
        
        # Save to JSON
        output_file = Path("predictions") / f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"\n✓ Prediction saved to: {output_file}")
    
    elif args.command == 'batch':
        logger.info(f"Loading features from {args.input}...")
        df = pd.read_parquet(args.input)
        
        # Prepare input
        X = inference.prepare_input(df)
        
        # Predict
        logger.info("Running batch predictions...")
        result = inference.predict(X)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'prediction_wei': result['predictions_wei'],
            'prediction_gwei': result['predictions_gwei']
        })
        
        predictions_df.to_csv(args.output, index=False)
        logger.info(f"✓ Saved {len(predictions_df)} predictions to: {args.output}")
        
        # Show statistics
        logger.info(f"\nPrediction Statistics:")
        logger.info(f"  Mean: {result['mean_gwei']:.6f} Gwei")
        logger.info(f"  Median: {result['median_gwei']:.6f} Gwei")
        logger.info(f"  Std: {result['std_gwei']:.6f} Gwei")
        logger.info(f"  Min: {result['min_gwei']:.6f} Gwei")
        logger.info(f"  Max: {result['max_gwei']:.6f} Gwei")


if __name__ == '__main__':
    main()
