"""
Unified Feature Selection Module

Combines:
- Statistical feature analysis (correlation, variance)
- XGBoost importance analysis
- Feature filtering

Usage:
    # Statistical analysis
    python -m src.feature_selector analyze --in data/features.parquet --out data/selected_features.txt
    
    # XGBoost importance (requires trained model)
    python -m src.feature_selector importance --model models/xgb.bin --in data/features.parquet
    
    # Filter features to selected ones
    python -m src.feature_selector filter --in data/features.parquet --features data/selected_features.txt --out data/filtered.parquet
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StatisticalFeatureAnalyzer:
    """Analyze features using statistical methods (correlation, variance)"""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'baseFee_next'):
        self.df = df
        self.target_col = target_col
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
                    'correlation': abs(corr),
                    'corr_sign': np.sign(corr)
                })
            except Exception as e:
                logger.warning(f"Could not calculate correlation for {col}: {e}")
        return pd.DataFrame(correlations).sort_values('correlation', ascending=False)
    
    def analyze_variance(self) -> pd.DataFrame:
        """Calculate variance metrics"""
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
    
    def find_redundant_features(self, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """Find highly correlated feature pairs"""
        redundant = []
        corr_matrix = self.df[self.feature_cols].corr().abs()
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
        """Select features based on multiple criteria"""
        # Remove low variance
        var_df = self.analyze_variance()
        high_var = var_df[var_df['variance'] > variance_threshold]['feature'].tolist()
        logger.info(f"✓ {len(high_var)} features passed variance filter (>{variance_threshold})")
        
        # Keep high correlation with target
        corr_df = self.analyze_correlation()
        high_corr = corr_df[corr_df['correlation'] > correlation_threshold]['feature'].tolist()
        logger.info(f"✓ {len(high_corr)} features have correlation > {correlation_threshold}")
        
        # Intersection
        selected = list(set(high_var) & set(high_corr))
        logger.info(f"✓ {len(selected)} features passed both filters")
        
        # Remove redundant
        if remove_redundant:
            redundant = self.find_redundant_features(redundancy_threshold)
            if redundant:
                logger.info(f"Found {len(redundant)} redundant pairs")
                to_remove = set()
                for feat1, feat2, _ in redundant:
                    if feat1 in selected and feat2 in selected:
                        corr1 = corr_df[corr_df['feature']==feat1]['correlation'].values[0]
                        corr2 = corr_df[corr_df['feature']==feat2]['correlation'].values[0]
                        to_remove.add(feat1 if corr1 < corr2 else feat2)
                selected = [f for f in selected if f not in to_remove]
                logger.info(f"✓ {len(selected)} features after removing redundancy")
        
        return selected


class XGBoostFeatureAnalyzer:
    """Analyze features using XGBoost importance"""
    
    def __init__(self, model_path: str, feature_names: List[str]):
        self.xgb_model = joblib.load(model_path)
        self.feature_names = feature_names
        self.importance_df = None
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance from XGBoost"""
        importance_dict = {}
        for imp_type in ['weight', 'gain', 'cover']:
            try:
                importance_dict[imp_type] = self.xgb_model.get_score(importance_type=imp_type)
            except:
                logger.warning(f"Could not get importance type: {imp_type}")
        
        # Convert to DataFrame
        importance_data = []
        for feat in self.feature_names:
            row = {'feature': feat}
            for imp_type, scores in importance_dict.items():
                row[imp_type] = scores.get(f'f{self.feature_names.index(feat)}', 0)
            importance_data.append(row)
        
        self.importance_df = pd.DataFrame(importance_data)
        
        # Normalize to percentage
        for col in ['weight', 'gain', 'cover']:
            if col in self.importance_df.columns:
                total = self.importance_df[col].sum()
                if total > 0:
                    self.importance_df[f'{col}_pct'] = (self.importance_df[col] / total * 100)
        
        # Average importance
        pct_cols = [c for c in self.importance_df.columns if c.endswith('_pct')]
        if pct_cols:
            self.importance_df['avg_importance'] = self.importance_df[pct_cols].mean(axis=1)
        
        self.importance_df = self.importance_df.sort_values('avg_importance', ascending=False)
        return self.importance_df
    
    def select_features(self, threshold_pct: float = 1.0, top_k: int = None) -> List[str]:
        """Select features by importance threshold"""
        if self.importance_df is None:
            self.get_feature_importance()
        
        if top_k is not None:
            selected = self.importance_df.head(top_k)['feature'].tolist()
            logger.info(f"✓ Selected top {top_k} features")
        else:
            selected = self.importance_df[
                self.importance_df['avg_importance'] >= threshold_pct
            ]['feature'].tolist()
            logger.info(f"✓ Selected {len(selected)} features with importance >= {threshold_pct}%")
        
        return selected


def group_features(features: List[str]) -> Dict[str, List[str]]:
    """Group features by type"""
    groups = {
        'base': [], 'ema': [], 'lag': [], 'momentum': [],
        'interaction': [], 'statistical': [], 'time': []
    }
    
    for feat in features:
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
    
    return {k: v for k, v in groups.items() if v}  # Remove empty groups


def load_selected_features(path: str) -> List[str]:
    """Load selected features from file"""
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def save_selected_features(features: List[str], path: str):
    """Save selected features to file"""
    with open(path, 'w') as f:
        for feat in sorted(features):
            f.write(f"{feat}\n")
    logger.info(f"✅ Saved {len(features)} features to {path}")


# ============================================================================
# CLI Commands
# ============================================================================

def cmd_analyze(args):
    """Statistical feature analysis"""
    logger.info(f"Loading data from {args.input}")
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    analyzer = StatisticalFeatureAnalyzer(df)
    
    # Correlation analysis
    logger.info("\n" + "="*60)
    logger.info("TOP 20 CORRELATED FEATURES")
    logger.info("="*60)
    corr_df = analyzer.analyze_correlation()
    for idx, row in corr_df.head(20).iterrows():
        sign = "+" if row['corr_sign'] > 0 else "-"
        logger.info(f"  {row['feature']:30s} {sign}{row['correlation']:6.4f}")
    
    # Save correlation report
    corr_path = Path(args.output).parent / 'feature_correlation.csv'
    corr_df.to_csv(corr_path, index=False)
    logger.info(f"Saved correlation report to {corr_path}")
    
    # Select features
    selected = analyzer.select_features(
        correlation_threshold=args.corr_threshold,
        variance_threshold=args.var_threshold,
        remove_redundant=True
    )
    
    # Group and display
    logger.info("\n" + "="*60)
    logger.info(f"SELECTED {len(selected)} FEATURES")
    logger.info("="*60)
    groups = group_features(selected)
    for group_name, features in groups.items():
        logger.info(f"\n{group_name.upper()} ({len(features)}):")
        for feat in sorted(features)[:10]:  # Show first 10
            logger.info(f"  - {feat}")
        if len(features) > 10:
            logger.info(f"  ... and {len(features)-10} more")
    
    # Save
    save_selected_features(selected, args.output)
    
    removed = len(analyzer.feature_cols) - len(selected)
    logger.info(f"\nRemoved {removed} features ({removed/len(analyzer.feature_cols)*100:.1f}%)")


def cmd_importance(args):
    """XGBoost importance analysis"""
    logger.info(f"Loading data from {args.input}")
    df = pd.read_parquet(args.input)
    
    exclude_cols = ['baseFee_next', 'timestamp', 'number']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    logger.info(f"Total features: {len(feature_cols)}")
    
    logger.info(f"Loading model from {args.model}")
    analyzer = XGBoostFeatureAnalyzer(args.model, feature_cols)
    
    importance_df = analyzer.get_feature_importance()
    
    # Save full report
    importance_path = Path(args.output).parent / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Saved importance report to {importance_path}")
    
    # Display top features
    logger.info("\n" + "="*60)
    logger.info("TOP 20 IMPORTANT FEATURES")
    logger.info("="*60)
    for idx, row in importance_df.head(20).iterrows():
        logger.info(f"  {row['feature']:30s} {row['avg_importance']:6.2f}%")
    
    # Select features
    selected = analyzer.select_features(
        threshold_pct=args.threshold,
        top_k=args.top_k
    )
    
    save_selected_features(selected, args.output)
    
    removed = len(feature_cols) - len(selected)
    logger.info(f"\nRemoved {removed} features ({removed/len(feature_cols)*100:.1f}%)")


def cmd_filter(args):
    """Filter features to selected ones only"""
    logger.info(f"Loading selected features from {args.features}")
    selected_features = load_selected_features(args.features)
    logger.info(f"Loaded {len(selected_features)} selected features")
    
    logger.info(f"Loading data from {args.input}")
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Keep: metadata + selected features + target
    keep_cols = ['number', 'timestamp'] + selected_features + ['baseFee_next']
    available_cols = [c for c in keep_cols if c in df.columns]
    missing_cols = set(keep_cols) - set(available_cols)
    
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
    
    df_filtered = df[available_cols].copy()
    
    logger.info("\n" + "="*60)
    logger.info("FILTERING RESULTS")
    logger.info("="*60)
    logger.info(f"Original: {len(df.columns)} columns")
    logger.info(f"Filtered: {len(df_filtered.columns)} columns")
    logger.info(f"Removed: {len(df.columns) - len(df_filtered.columns)} columns")
    
    df_filtered.to_parquet(args.output, index=False)
    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    logger.info(f"\n✅ Saved to {args.output} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description='Unified Feature Selection Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Statistical analysis
  python -m src.feature_selector analyze --in data/features.parquet
  
  # XGBoost importance (requires trained model)
  python -m src.feature_selector importance --model models/xgb.bin --in data/features.parquet
  
  # Filter to selected features
  python -m src.feature_selector filter --in data/features.parquet --features data/selected_features.txt --out data/filtered.parquet
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Statistical feature analysis')
    analyze_parser.add_argument('--in', dest='input', type=str, default='data/features.parquet')
    analyze_parser.add_argument('--out', dest='output', type=str, default='data/selected_features.txt')
    analyze_parser.add_argument('--corr-threshold', type=float, default=0.05,
                                help='Minimum correlation with target')
    analyze_parser.add_argument('--var-threshold', type=float, default=1e-6,
                                help='Minimum variance threshold')
    
    # Importance command
    importance_parser = subparsers.add_parser('importance', help='XGBoost importance analysis')
    importance_parser.add_argument('--model', type=str, default='models/xgb.bin')
    importance_parser.add_argument('--in', dest='input', type=str, default='data/features.parquet')
    importance_parser.add_argument('--out', dest='output', type=str, default='data/selected_features.txt')
    importance_parser.add_argument('--threshold', type=float, default=1.0,
                                   help='Importance threshold percentage')
    importance_parser.add_argument('--top-k', type=int, default=None,
                                   help='Select top K features')
    
    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Filter to selected features')
    filter_parser.add_argument('--in', dest='input', type=str, required=True)
    filter_parser.add_argument('--features', type=str, required=True)
    filter_parser.add_argument('--out', dest='output', type=str, required=True)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'importance':
        cmd_importance(args)
    elif args.command == 'filter':
        cmd_filter(args)


if __name__ == '__main__':
    main()
