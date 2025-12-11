"""
Feature Engineering untuk Gas Fee Prediction.

Modul ini transform raw block data menjadi features untuk ML model:
- Temporal features (hour, day_of_week)
- Statistical features (delta, EMA, rolling stats)
- Utilization metrics
- Label generation (baseFee_next)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering untuk block data.
    
    Generates:
    - Delta features: perubahan antar blok
    - EMA features: exponential moving average
    - Rolling statistics: mean, std
    - Temporal features: hour, day_of_week
    - Utilization: gasUsed / gasLimit
    - Target label: baseFee_next
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration dict (optional)
        """
        self.config = config or {}
        
        # EMA parameters
        self.ema_short_alpha = 0.3  # Fast response
        self.ema_long_alpha = 0.1   # Slow response
        
        # Rolling window sizes
        self.rolling_window_6 = 6   # ~1 minute (6 blocks * 12s)
        self.rolling_window_24 = 24  # ~5 minutes
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load raw block data.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame dengan block data
        """
        logger.info(f"Loading data from {filepath}...")
        
        df = pd.read_csv(filepath)
        
        logger.info(f"✓ Loaded {len(df)} blocks")
        logger.info(f"  Columns: {df.columns.tolist()}")
        logger.info(f"  Block range: {df['number'].min()} - {df['number'].max()}")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features - historical values from previous blocks.
        Critical for capturing temporal dependencies.
        """
        logger.info("Creating lag features...")
        
        # Lag baseFee for 1-5 blocks back
        for lag in [1, 2, 3, 5]:
            df[f'lag_baseFee_{lag}'] = df['baseFeePerGas'].shift(lag)
            df[f'lag_utilization_{lag}'] = df['utilization'].shift(lag)
        
        # Lag delta features
        df['lag_delta_baseFee_1'] = df['delta_baseFee'].shift(1)
        df['lag_delta_baseFee_2'] = df['delta_baseFee'].shift(2)
        
        logger.info("✓ Lag features created")
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features dari timestamp.
        
        Args:
            df: Block DataFrame dengan 'timestamp' column
            
        Returns:
            DataFrame dengan temporal features
        """
        logger.info("Creating temporal features...")
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Extract temporal components
        df['hour_of_day'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df['datetime'].dt.day
        
        # Cyclical encoding untuk hour (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        
        # Cyclical encoding untuk day_of_week (7-day cycle)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        logger.info("✓ Temporal features created")
        
        return df
    
    def create_delta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create delta features (perubahan antar blok).
        
        Args:
            df: Block DataFrame
            
        Returns:
            DataFrame dengan delta features
        """
        logger.info("Creating delta features...")
        
        # Sort by block number
        df = df.sort_values('number').reset_index(drop=True)
        
        # Delta baseFee
        df['delta_baseFee'] = df['baseFeePerGas'].diff()
        df['delta_baseFee_pct'] = df['baseFeePerGas'].pct_change() * 100
        
        # Delta gasUsed
        df['delta_gasUsed'] = df['gasUsed'].diff()
        
        # Delta timestamp (block time)
        df['delta_timestamp'] = df['timestamp'].diff()
        
        # Fill first row NaN dengan 0
        df['delta_baseFee'] = df['delta_baseFee'].fillna(0)
        df['delta_baseFee_pct'] = df['delta_baseFee_pct'].fillna(0)
        df['delta_gasUsed'] = df['delta_gasUsed'].fillna(0)
        df['delta_timestamp'] = df['delta_timestamp'].fillna(12)  # default block time
        
        logger.info("✓ Delta features created")
        
        return df
    
    def create_utilization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create utilization features.
        
        Args:
            df: Block DataFrame
            
        Returns:
            DataFrame dengan utilization features
        """
        logger.info("Creating utilization features...")
        
        # Basic utilization
        df['utilization'] = df['gasUsed'] / df['gasLimit']
        
        # Clip to [0, 1] range (just in case)
        df['utilization'] = df['utilization'].clip(0, 1)
        
        # Transaction density
        df['tx_density'] = df['txCount'] / df['gasLimit'] * 1e6  # per million gas
        
        # EIP-1559 specific: Block fullness pattern (critical!)
        # Target is 50% - measure deviation
        df['fullness_deviation'] = np.abs(df['utilization'] - 0.5)
        df['above_target'] = (df['utilization'] > 0.5).astype(int)
        
        # Consecutive blocks above/below target (regime detection)
        df['above_target_streak'] = (df['above_target']
                                      .groupby((df['above_target'] != df['above_target'].shift()).cumsum())
                                      .cumcount() + 1)
        
        logger.info("✓ Utilization features created")
        
        return df
    
    def create_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Exponential Moving Average features.
        
        Args:
            df: Block DataFrame
            
        Returns:
            DataFrame dengan EMA features
        """
        logger.info("Creating EMA features...")
        
        # EMA baseFee (short term - alpha=0.3)
        df['ema_baseFee_short'] = df['baseFeePerGas'].ewm(
            alpha=self.ema_short_alpha,
            adjust=False
        ).mean()
        
        # EMA baseFee (long term - alpha=0.1)
        df['ema_baseFee_long'] = df['baseFeePerGas'].ewm(
            alpha=self.ema_long_alpha,
            adjust=False
        ).mean()
        
        # EMA utilization (short term)
        df['ema_utilization_short'] = df['utilization'].ewm(
            alpha=self.ema_short_alpha,
            adjust=False
        ).mean()
        
        # EMA utilization (long term)
        df['ema_utilization_long'] = df['utilization'].ewm(
            alpha=self.ema_long_alpha,
            adjust=False
        ).mean()
        
        # EMA delta features
        df['ema_delta_baseFee'] = df['delta_baseFee'].ewm(
            alpha=self.ema_short_alpha,
            adjust=False
        ).mean()
        
        logger.info("✓ EMA features created")
        
        return df
    
    def create_volatility_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect volatility regimes - critical for accuracy.
        High volatility = harder to predict, need to flag it.
        """
        logger.info("Creating volatility regime features...")
        
        # Rolling volatility (6 blocks)
        df['baseFee_rolling_std_6'] = df['baseFeePerGas'].rolling(6, min_periods=1).std().fillna(0)
        
        # Volatility regime: low/medium/high
        vol_quantiles = df['baseFee_rolling_std_6'].quantile([0.33, 0.67])
        q33 = vol_quantiles[0.33]
        q67 = vol_quantiles[0.67]

        # Robustness: Ensure bins are unique for low-variance data (common in stable networks)
        if q33 >= q67:
            q67 = q33 + 1e-9

        df['volatility_regime'] = pd.cut(
            df['baseFee_rolling_std_6'],
            bins=[-np.inf, q33, q67, np.inf],
            labels=[0, 1, 2]  # 0=low, 1=med, 2=high
        ).astype(int)
        
        # Price jump detection (>10% change in 1 block)
        df['price_jump'] = (np.abs(df['delta_baseFee_pct']) > 10).astype(int)
        
        # Consecutive price jumps (cascade effect)
        df['consecutive_jumps'] = (df['price_jump']
                                    .groupby((df['price_jump'] != df['price_jump'].shift()).cumsum())
                                    .cumcount())
        
        logger.info("✓ Volatility regime features created")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: Block DataFrame
            
        Returns:
            DataFrame dengan rolling features
        """
        logger.info("Creating rolling features...")
        
        # Rolling baseFee statistics (6 blocks ~1min)
        df['rolling_mean_baseFee_6'] = df['baseFeePerGas'].rolling(
            window=self.rolling_window_6,
            min_periods=1
        ).mean()
        
        df['rolling_std_baseFee_6'] = df['baseFeePerGas'].rolling(
            window=self.rolling_window_6,
            min_periods=1
        ).std().fillna(0)
        
        df['rolling_min_baseFee_6'] = df['baseFeePerGas'].rolling(
            window=self.rolling_window_6,
            min_periods=1
        ).min()
        
        df['rolling_max_baseFee_6'] = df['baseFeePerGas'].rolling(
            window=self.rolling_window_6,
            min_periods=1
        ).max()
        
        # Rolling utilization statistics
        df['rolling_mean_utilization_6'] = df['utilization'].rolling(
            window=self.rolling_window_6,
            min_periods=1
        ).mean()
        
        df['rolling_std_utilization_6'] = df['utilization'].rolling(
            window=self.rolling_window_6,
            min_periods=1
        ).std().fillna(0)
        
        # Volatility (coefficient of variation)
        df['baseFee_volatility'] = (
            df['rolling_std_baseFee_6'] / df['rolling_mean_baseFee_6']
        ).fillna(0)
        
        logger.info("✓ Rolling features created")
        
        return df
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create momentum and trend features for better prediction.
        
        Args:
            df: Block DataFrame
            
        Returns:
            DataFrame with momentum features
        """
        logger.info("Creating momentum features...")
        
        # Price momentum (rate of change over 5 blocks)
        df['baseFee_momentum_5'] = df['baseFeePerGas'].pct_change(periods=5).fillna(0)
        df['baseFee_momentum_10'] = df['baseFeePerGas'].pct_change(periods=10).fillna(0)
        
        # Utilization momentum
        df['utilization_momentum_5'] = df['utilization'].diff(periods=5).fillna(0)
        
        # Acceleration (second derivative)
        df['baseFee_acceleration'] = df['delta_baseFee'].diff().fillna(0)
        
        # Trend strength (using linear regression slope approximation)
        df['baseFee_trend_3'] = df['baseFeePerGas'].rolling(3, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) > 1 else 0,
            raw=False
        ).fillna(0)
        
        df['baseFee_trend_6'] = df['baseFeePerGas'].rolling(6, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) > 1 else 0,
            raw=False
        ).fillna(0)
        
        logger.info("✓ Momentum features created")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between utilization and baseFee.
        
        Args:
            df: Block DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        # Utilization-BaseFee interactions
        df['util_x_baseFee'] = df['utilization'] * df['baseFeePerGas']
        df['util_squared'] = df['utilization'] ** 2
        
        # High utilization indicator (above 50%)
        df['high_util_flag'] = (df['utilization'] > 0.5).astype(int)
        
        # Congestion score (weighted combination)
        df['congestion_score'] = (
            0.6 * df['utilization'] + 
            0.4 * (df['baseFeePerGas'] / df['baseFeePerGas'].max())
        )
        
        # EMA crossover signal (short > long = uptrend)
        df['ema_crossover'] = (
            df['ema_baseFee_short'] > df['ema_baseFee_long']
        ).astype(int)
        
        logger.info("✓ Interaction features created")
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced statistical features.
        
        Args:
            df: Block DataFrame
            
        Returns:
            DataFrame with statistical features
        """
        logger.info("Creating statistical features...")
        
        # Z-score (standardized baseFee)
        rolling_mean = df['baseFeePerGas'].rolling(20, min_periods=1).mean()
        rolling_std = df['baseFeePerGas'].rolling(20, min_periods=1).std()
        df['baseFee_zscore'] = (
            (df['baseFeePerGas'] - rolling_mean) / (rolling_std + 1e-8)
        ).fillna(0)
        
        # Percentile rank (where current baseFee stands in recent history)
        df['baseFee_percentile'] = df['baseFeePerGas'].rolling(
            20, min_periods=1
        ).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5,
            raw=False
        ).fillna(0.5)
        
        # Skewness and kurtosis (distribution shape)
        df['baseFee_skew_10'] = df['baseFeePerGas'].rolling(
            10, min_periods=3
        ).skew().fillna(0)
        
        logger.info("✓ Statistical features created")
        
        return df
    
    def create_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target label: baseFee_next.
        
        Args:
            df: Block DataFrame
            
        Returns:
            DataFrame dengan label column
        """
        logger.info("Creating target label...")
        
        # Shift baseFee backward by 1 (next block's baseFee)
        df['baseFee_next'] = df['baseFeePerGas'].shift(-1)
        
        # Remove last row (no next block)
        df = df[:-1].copy()
        
        logger.info(f"✓ Label created ({len(df)} samples)")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Raw block DataFrame
            
        Returns:
            DataFrame dengan all features
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Feature Engineering Pipeline")
        logger.info(f"{'='*60}\n")
        
        # Apply all feature engineering steps
        df = self.create_temporal_features(df)
        df = self.create_utilization_features(df)
        df = self.create_delta_features(df)
        df = self.create_lag_features(df)  # NEW - Elite level
        df = self.create_ema_features(df)
        df = self.create_volatility_regime_features(df)  # NEW - Elite level
        df = self.create_rolling_features(df)
        df = self.create_momentum_features(df)
        df = self.create_interaction_features(df)
        df = self.create_statistical_features(df)
        df = self.create_label(df)
        
        # Remove any remaining NaN
        initial_len = len(df)
        df = df.dropna()
        removed = initial_len - len(df)
        
        if removed > 0:
            logger.warning(f"Removed {removed} rows with NaN values")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Feature Engineering Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Total features: {len(df.columns)}")
        logger.info(f"Feature columns:")
        for col in df.columns:
            logger.info(f"  - {col}")
        
        return df
    
    def save_features(self, df: pd.DataFrame, output_path: str):
        """
        Save features to Parquet format.
        
        Args:
            df: DataFrame dengan features
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving features to {output_path}...")
        
        # Save to Parquet (efficient for large datasets)
        df.to_parquet(output_path, index=False, compression='snappy')
        
        # Save feature list
        feature_list_path = output_path.with_suffix('.txt')
        with open(feature_list_path, 'w') as f:
            f.write("Feature List:\n")
            f.write("=" * 60 + "\n")
            for col in df.columns:
                f.write(f"{col}\n")
        
        logger.info(f"✓ Features saved: {output_path}")
        logger.info(f"✓ Feature list saved: {feature_list_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate features from raw block data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m src.features --in data/blocks.csv --out data/features.parquet
  
  # With custom config
  python -m src.features --in data/blocks.csv --out data/features.parquet --config cfg/exp.yaml
        """
    )
    
    parser.add_argument(
        '--in',
        dest='input_file',
        type=str,
        required=True,
        help='Input CSV file dengan raw block data'
    )
    
    parser.add_argument(
        '--out',
        dest='output_file',
        type=str,
        required=True,
        help='Output Parquet file untuk features'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Config YAML file (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize feature engineer
        engineer = FeatureEngineer()
        
        # Load data
        df = engineer.load_data(args.input_file)
        
        # Engineer features
        df_features = engineer.engineer_features(df)
        
        # Save features
        engineer.save_features(df_features, args.output_file)
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print(f"Feature Statistics Summary")
        print(f"{'='*60}")
        print(f"\nBaseFee Statistics (Gwei):")
        print(df_features['baseFeePerGas'].describe() / 1e9)
        print(f"\nUtilization Statistics:")
        print(df_features['utilization'].describe())
        print(f"\nTarget (baseFee_next) Statistics (Gwei):")
        print(df_features['baseFee_next'].describe() / 1e9)
        print(f"{'='*60}\n")
        
        logger.info("✓ Feature engineering completed successfully!")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("\n\nInterrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
