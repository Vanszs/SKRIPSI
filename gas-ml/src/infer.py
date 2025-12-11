"""
Real-time Inference Engine untuk Gas Fee Prediction.

Fitur:
- Load trained model
- Fetch latest blocks dari network
- Generate predictions
- Provide gas fee recommendations
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pickle
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rpc import EthereumRPCClient
from models import load_model_from_dir
from policy import GasFeePolicy, create_default_policy
from features import FeatureEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GasFeeInferenceEngine:
    """
    Real-time inference engine untuk gas fee prediction.
    """
    
    def __init__(
        self,
        model_dir: str,
        network: str = '',
        rpc_url: str = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model_dir: Directory containing trained model
            network: Ethereum network
            rpc_url: Custom RPC URL (optional)
        """
        self.model_dir = Path(model_dir)
        self.network = network
        
        # Load model (detects type automatically)
        logger.info(f"Loading model from {model_dir}...")
        self.model, self.metadata = load_model_from_dir(self.model_dir)
        
        model_type = self.metadata.get('model_type', 'unknown')
        logger.info(f"Loaded model type: {model_type}")
        
        # Load scaler
        with open(self.model_dir / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load training info
        import json
        with open(self.model_dir / 'training_info.json', 'r') as f:
            self.training_info = json.load(f)
        
        self.feature_columns = self.training_info['feature_columns']
        
        # Initialize RPC client
        logger.info(f"Connecting to {network} network...")
        self.rpc_client = EthereumRPCClient(network=network, rpc_url=rpc_url)
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Initialize policy
        self.policy = create_default_policy()
        
        logger.info("✓ Inference engine ready!")
    
    def fetch_recent_blocks(self, n_blocks: int = 100) -> list:
        """
        Fetch recent blocks untuk feature engineering.
        
        Args:
            n_blocks: Number of recent blocks to fetch
            
        Returns:
            List of block data
        """
        logger.info(f"Fetching {n_blocks} recent blocks...")
        
        blocks = []
        latest = self.rpc_client.get_latest_block_number()
        
        for block_num in range(latest - n_blocks + 1, latest + 1):
            try:
                block = self.rpc_client.get_block(block_num)
                features = self.rpc_client.extract_block_features(block)
                blocks.append(features)
            except Exception as e:
                logger.warning(f"Failed to fetch block {block_num}: {e}")
                continue
        
        logger.info(f"✓ Fetched {len(blocks)} blocks")
        return blocks
    
    def prepare_features(self, blocks: list) -> np.ndarray:
        """
        Convert raw blocks to model features.
        
        Args:
            blocks: List of block data
            
        Returns:
            Feature array
        """
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame(blocks)
        
        # Apply feature engineering
        df = self.feature_engineer.engineer_features(df)
        
        # Select required features
        X = df[self.feature_columns].values
        
        # Normalize
        X_norm = self.scaler.transform(X)
        
        return X_norm
    
    def predict_next_basefee(self) -> dict:
        """
        Predict baseFee untuk next block.
        
        Returns:
            Dictionary dengan prediction dan recommendation
        """
        # Fetch recent blocks
        # Safely get sequence length (Hyrid has it, XGBoost might not - default to 1)
        seq_len = getattr(self.model, 'sequence_length', 1)
        if seq_len is None: # Just in case it's explicitly None
             seq_len = self.metadata.get('sequence_length', 1)
             
        n_blocks = seq_len + 50  # Extra for feature engineering
        
        blocks = self.fetch_recent_blocks(n_blocks)
        
        if len(blocks) < seq_len:
            raise ValueError(f"Not enough blocks: got {len(blocks)}, need {seq_len}")
        
        # Prepare features
        X = self.prepare_features(blocks)
        
        # Make prediction
        logger.info("Making prediction...")
        prediction = self.model.predict(X)
        
        # Use last prediction (most recent)
        predicted_base_fee = float(prediction[-1])
        
        logger.info(f"✓ Predicted baseFee: {predicted_base_fee / 1e9:.4f} Gwei")
        
        # Generate recommendation
        recommendation = self.policy.generate_recommendation(
            predicted_base_fee=predicted_base_fee,
            recent_blocks=blocks[-10:]  # Use last 10 blocks for priority fee
        )
        
        # Add current network info
        latest_block = blocks[-1]
        recommendation['current_block'] = latest_block['number']
        recommendation['current_base_fee_gwei'] = latest_block['baseFeePerGas'] / 1e9
        recommendation['current_utilization'] = latest_block['gasUsed'] / latest_block['gasLimit']
        
        return recommendation
    
    def run_continuous(self, interval: int = 12):
        """
        Run continuous prediction loop.
        
        Args:
            interval: Seconds between predictions (default: 12s = 1 block)
        """
        logger.info(f"\nStarting continuous prediction (interval: {interval}s)")
        logger.info("Press Ctrl+C to stop\n")
        
        try:
            while True:
                try:
                    # Make prediction
                    recommendation = self.predict_next_basefee()
                    
                    # Display
                    print(self.policy.format_recommendation(recommendation))
                    
                    # Additional info
                    print(f"Current Block: {recommendation['current_block']}")
                    print(f"Current Base Fee: {recommendation['current_base_fee_gwei']:.4f} Gwei")
                    print(f"Current Utilization: {recommendation['current_utilization']*100:.1f}%\n")
                    
                    # Wait for next interval
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    time.sleep(interval)
                    
        except KeyboardInterrupt:
            logger.info("\nStopping continuous prediction...")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Real-time gas fee prediction and recommendation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python -m src.infer --model models/
  
  # Continuous prediction (every 12 seconds)
  python -m src.infer --model models/ --continuous
  
  # Custom network
  python -m src.infer --model models/ --network mainnet --rpc-url https://your-rpc.com
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model directory'
    )
    
    parser.add_argument(
        '--network',
        type=str,
        default='',
        help='Ethereum network (default: )'
    )
    
    parser.add_argument(
        '--rpc-url',
        type=str,
        default=None,
        help='Custom RPC URL (optional)'
    )
    
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='Run continuous prediction mode'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=12,
        help='Interval between predictions in continuous mode (seconds)'
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
        # Initialize engine
        engine = GasFeeInferenceEngine(
            model_dir=args.model,
            network=args.network,
            rpc_url=args.rpc_url
        )
        
        if args.continuous:
            # Continuous mode
            engine.run_continuous(interval=args.interval)
        else:
            # Single prediction
            recommendation = engine.predict_next_basefee()
            print(engine.policy.format_recommendation(recommendation))
            
            print(f"\n📍 Network Info:")
            print(f"  Current Block: {recommendation['current_block']}")
            print(f"  Current Base Fee: {recommendation['current_base_fee_gwei']:.4f} Gwei")
            print(f"  Current Utilization: {recommendation['current_utilization']*100:.1f}%\n")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
