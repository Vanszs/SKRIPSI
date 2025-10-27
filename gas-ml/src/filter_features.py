"""
Filter features parquet to selected features only
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pandas as pd
from typing import List

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


def main():
    parser = argparse.ArgumentParser(description='Filter to selected features')
    parser.add_argument('--in', dest='input_file', type=str, required=True,
                        help='Input features parquet')
    parser.add_argument('--features', type=str, required=True,
                        help='Selected features file')
    parser.add_argument('--out', type=str, required=True,
                        help='Output filtered parquet')
    
    args = parser.parse_args()
    
    # Load selected features
    logger.info(f"Loading selected features from {args.features}")
    selected_features = load_selected_features(args.features)
    logger.info(f"Loaded {len(selected_features)} selected features")
    
    # Load data
    logger.info(f"Loading data from {args.input_file}")
    df = pd.read_parquet(args.input_file)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Keep only: metadata + selected features + target
    keep_cols = ['number', 'timestamp'] + selected_features + ['baseFee_next']
    
    # Filter
    available_cols = [c for c in keep_cols if c in df.columns]
    missing_cols = set(keep_cols) - set(available_cols)
    
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
    
    df_filtered = df[available_cols].copy()
    
    logger.info(f"\n" + "="*60)
    logger.info(f"FILTERED FEATURES")
    logger.info("="*60)
    logger.info(f"Original columns: {len(df.columns)}")
    logger.info(f"Filtered columns: {len(df_filtered.columns)}")
    logger.info(f"Features: {len(df_filtered.columns) - 3} (removed {len(df.columns) - len(df_filtered.columns)})")
    
    # Save
    df_filtered.to_parquet(args.out, index=False)
    logger.info(f"\n✅ Saved filtered features to {args.out}")
    
    # Print file size
    size_mb = Path(args.out).stat().st_size / 1024 / 1024
    logger.info(f"File size: {size_mb:.2f} MB")
    
    return df_filtered


if __name__ == '__main__':
    main()
