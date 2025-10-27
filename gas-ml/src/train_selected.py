"""
Train model with selected features only
Reads feature selection from selected_features.txt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pandas as pd
import yaml
from typing import List

# Import existing training infrastructure
from src.train import (
    prepare_sequences,
    train_lstm,
    train_xgboost,
    evaluate_model,
    save_model
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_selected_features(path: str) -> List[str]:
    """Load selected features from file"""
    with open(path, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    return features


def filter_features(df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """Keep only selected features plus target and metadata"""
    # Columns to keep
    keep_cols = ['number', 'timestamp', 'baseFee_next'] + selected_features
    
    # Filter
    available_cols = [c for c in keep_cols if c in df.columns]
    missing_cols = set(keep_cols) - set(available_cols)
    
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
    
    filtered_df = df[available_cols].copy()
    
    logger.info(f"Filtered from {len(df.columns)} to {len(filtered_df.columns)} columns")
    logger.info(f"Features: {len(filtered_df.columns) - 3}")
    
    return filtered_df


def main():
    parser = argparse.ArgumentParser(description='Train with selected features')
    parser.add_argument('--cfg', type=str, default='cfg/exp.yaml',
                        help='Path to config file')
    parser.add_argument('--in', dest='input_file', type=str, required=True,
                        help='Input features parquet')
    parser.add_argument('--features', type=str, default='data/selected_features.txt',
                        help='Selected features file')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for models')
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.cfg}")
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load selected features
    logger.info(f"Loading selected features from {args.features}")
    selected_features = load_selected_features(args.features)
    logger.info(f"Loaded {len(selected_features)} selected features")
    
    # Load data
    logger.info(f"Loading data from {args.input_file}")
    df = pd.read_parquet(args.input_file)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Filter to selected features
    logger.info("Filtering to selected features...")
    df_filtered = filter_features(df, selected_features)
    
    # Save filtered dataset for inspection
    filtered_path = Path(args.input_file).parent / 'features_selected.parquet'
    df_filtered.to_parquet(filtered_path, index=False)
    logger.info(f"Saved filtered features to {filtered_path}")
    
    # Now proceed with normal training
    logger.info("\n" + "="*60)
    logger.info("TRAINING WITH SELECTED FEATURES")
    logger.info("="*60)
    
    # Import and use the full training pipeline
    from src.train import main as train_main
    
    # Temporarily replace input file
    sys.argv = [
        'train.py',
        '--cfg', args.cfg,
        '--in', str(filtered_path)
    ]
    
    train_main()


if __name__ == '__main__':
    main()
