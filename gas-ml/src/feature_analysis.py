"""
Fast Feature Analysis using Statistical Methods
Analyze feature quality without needing trained model
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StatisticalFeatureSelector:
    """Select features using statistical correlation and variance"""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'baseFee_next'):
        self.df = df
        self.target_col = target_col
        
        # Get feature columns
        exclude_cols = [target_col, 'timestamp', 'number']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        logger.info(f"Analyzing {len(self.feature_cols)} features")
        
    def analyze_correlation(self) -> pd.DataFrame:
        """Calculate correlation with target"""
        correlations = []
        
        for col in self.feature_cols:
            try:
                corr = self.df[col].corr(self.df[self.target_col])
                correlations.append({
                    'feature': col,
                    'correlation': abs(corr),  # Absolute correlation
                    'corr_sign': np.sign(corr)
                })
            except Exception as e:
                logger.warning(f"Could not calculate correlation for {col}: {e}")
        
        return pd.DataFrame(correlations).sort_values('correlation', ascending=False)
    
    def analyze_variance(self) -> pd.DataFrame:
        """Calculate variance and coefficient of variation"""
        variances = []
        
        for col in self.feature_cols:
            try:
                var = self.df[col].var()
                std = self.df[col].std()
                mean = self.df[col].mean()
                cv = std / abs(mean) if mean != 0 else np.inf
                
                variances.append({
                    'feature': col,
                    'variance': var,
                    'std': std,
                    'cv': cv
                })
            except Exception as e:
                logger.warning(f"Could not calculate variance for {col}: {e}")
        
        return pd.DataFrame(variances)
    
    def find_redundant_features(self, threshold: float = 0.95) -> List[Tuple[str, str]]:
        """Find highly correlated feature pairs (redundant)"""
        redundant = []
        
        # Calculate feature-feature correlation matrix
        corr_matrix = self.df[self.feature_cols].corr().abs()
        
        # Find pairs with high correlation
        for i in range(len(self.feature_cols)):
            for j in range(i+1, len(self.feature_cols)):
                if corr_matrix.iloc[i, j] > threshold:
                    redundant.append((
                        self.feature_cols[i],
                        self.feature_cols[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        return redundant
    
    def select_features(
        self,
        correlation_threshold: float = 0.1,
        variance_threshold: float = 1e-6,
        remove_redundant: bool = True,
        redundancy_threshold: float = 0.95
    ) -> List[str]:
        """
        Select features based on multiple criteria
        
        Returns:
            List of selected feature names
        """
        # Step 1: Remove low variance features
        var_df = self.analyze_variance()
        high_var = var_df[var_df['variance'] > variance_threshold]['feature'].tolist()
        logger.info(f"Step 1: {len(high_var)} features passed variance filter (>{variance_threshold})")
        
        # Step 2: Keep features with meaningful correlation to target
        corr_df = self.analyze_correlation()
        high_corr = corr_df[corr_df['correlation'] > correlation_threshold]['feature'].tolist()
        logger.info(f"Step 2: {len(high_corr)} features have correlation > {correlation_threshold}")
        
        # Intersection
        selected = list(set(high_var) & set(high_corr))
        logger.info(f"Step 3: {len(selected)} features passed both filters")
        
        # Step 3: Remove redundant features
        if remove_redundant:
            redundant = self.find_redundant_features(redundancy_threshold)
            if redundant:
                logger.info(f"Found {len(redundant)} redundant feature pairs")
                
                # Keep the feature with higher target correlation
                to_remove = set()
                for feat1, feat2, corr in redundant:
                    if feat1 in selected and feat2 in selected:
                        corr1 = corr_df[corr_df['feature']==feat1]['correlation'].values[0]
                        corr2 = corr_df[corr_df['feature']==feat2]['correlation'].values[0]
                        
                        # Remove the one with lower correlation
                        if corr1 < corr2:
                            to_remove.add(feat1)
                        else:
                            to_remove.add(feat2)
                
                selected = [f for f in selected if f not in to_remove]
                logger.info(f"Step 4: {len(selected)} features after removing redundancy")
        
        return selected


def main():
    parser = argparse.ArgumentParser(description='Statistical Feature Analysis')
    parser.add_argument('--features', type=str, default='data/features.parquet',
                        help='Path to features parquet')
    parser.add_argument('--corr-threshold', type=float, default=0.05,
                        help='Minimum correlation with target')
    parser.add_argument('--var-threshold', type=float, default=1e-6,
                        help='Minimum variance threshold')
    parser.add_argument('--output', type=str, default='data/selected_features.txt',
                        help='Output file for selected features')
    
    args = parser.parse_args()
    
    # Load features
    logger.info(f"Loading features from {args.features}")
    df = pd.read_parquet(args.features)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Initialize selector
    selector = StatisticalFeatureSelector(df)
    
    # Analyze correlation
    logger.info("\n" + "="*60)
    logger.info("CORRELATION ANALYSIS (Top 20)")
    logger.info("="*60)
    corr_df = selector.analyze_correlation()
    for idx, row in corr_df.head(20).iterrows():
        sign = "+" if row['corr_sign'] > 0 else "-"
        logger.info(f"  {row['feature']:30s} {sign}{row['correlation']:6.4f}")
    
    # Save full correlation report
    corr_path = Path(args.output).parent / 'feature_correlation.csv'
    corr_df.to_csv(corr_path, index=False)
    
    # Analyze variance
    logger.info("\n" + "="*60)
    logger.info("VARIANCE ANALYSIS (Bottom 10 - potential low-signal)")
    logger.info("="*60)
    var_df = selector.analyze_variance()
    for idx, row in var_df.sort_values('variance').head(10).iterrows():
        logger.info(f"  {row['feature']:30s} var={row['variance']:12.6e}")
    
    # Find redundant features
    logger.info("\n" + "="*60)
    logger.info("REDUNDANT FEATURES (correlation > 0.95)")
    logger.info("="*60)
    redundant = selector.find_redundant_features(0.95)
    if redundant:
        for feat1, feat2, corr in redundant[:10]:  # Show top 10
            logger.info(f"  {feat1:25s} <-> {feat2:25s} {corr:.3f}")
    else:
        logger.info("  No highly redundant features found")
    
    # Select features
    selected = selector.select_features(
        correlation_threshold=args.corr_threshold,
        variance_threshold=args.var_threshold,
        remove_redundant=True,
        redundancy_threshold=0.95
    )
    
    logger.info("\n" + "="*60)
    logger.info(f"SELECTED {len(selected)} FEATURES")
    logger.info("="*60)
    
    # Group by type
    groups = {
        'base': [],
        'ema': [],
        'lag': [],
        'momentum': [],
        'interaction': [],
        'statistical': [],
        'time': []
    }
    
    for feat in selected:
        if feat.startswith('ema_'):
            groups['ema'].append(feat)
        elif feat.startswith('lag_'):
            groups['lag'].append(feat)
        elif any(x in feat for x in ['momentum', 'accel', 'jerk']):
            groups['momentum'].append(feat)
        elif '_x_' in feat or 'ratio' in feat:
            groups['interaction'].append(feat)
        elif any(x in feat for x in ['std', 'cv', 'peak']):
            groups['statistical'].append(feat)
        elif any(x in feat for x in ['hour', 'day', 'sin', 'cos']):
            groups['time'].append(feat)
        else:
            groups['base'].append(feat)
    
    for group_name, features in groups.items():
        if features:
            logger.info(f"\n{group_name.upper()} ({len(features)}):")
            for feat in sorted(features):
                logger.info(f"  - {feat}")
    
    # Save selected features
    with open(args.output, 'w') as f:
        for feat in sorted(selected):
            f.write(f"{feat}\n")
    
    logger.info(f"\n✅ Saved {len(selected)} selected features to {args.output}")
    
    # Statistics
    removed = len(selector.feature_cols) - len(selected)
    logger.info(f"Removed {removed} features ({removed/len(selector.feature_cols)*100:.1f}%)")
    
    return selected


if __name__ == '__main__':
    main()
