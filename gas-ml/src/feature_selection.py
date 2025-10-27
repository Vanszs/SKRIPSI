"""
Feature Selection via XGBoost Importance Analysis
Target: Identify and remove noisy features that hurt generalization
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """Analyze and select important features using trained XGBoost model"""
    
    def __init__(self, model_path: str, feature_names: List[str]):
        self.xgb_model = joblib.load(model_path)
        self.feature_names = feature_names
        self.importance_df = None
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance from XGBoost"""
        importance_dict = {}
        
        # Get different importance types
        for imp_type in ['weight', 'gain', 'cover']:
            try:
                importance_dict[imp_type] = self.xgb_model.get_score(
                    importance_type=imp_type
                )
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
        
        # Normalize scores to 0-100
        for col in ['weight', 'gain', 'cover']:
            if col in self.importance_df.columns:
                total = self.importance_df[col].sum()
                if total > 0:
                    self.importance_df[f'{col}_pct'] = (
                        self.importance_df[col] / total * 100
                    )
        
        # Calculate average importance
        pct_cols = [c for c in self.importance_df.columns if c.endswith('_pct')]
        if pct_cols:
            self.importance_df['avg_importance'] = (
                self.importance_df[pct_cols].mean(axis=1)
            )
        
        # Sort by average importance
        self.importance_df = self.importance_df.sort_values(
            'avg_importance', ascending=False
        )
        
        return self.importance_df
    
    def select_features(
        self, 
        threshold_pct: float = 1.0,
        top_k: int = None
    ) -> List[str]:
        """
        Select features based on importance threshold
        
        Args:
            threshold_pct: Keep features with avg_importance >= threshold
            top_k: Alternative - keep only top K features
        
        Returns:
            List of selected feature names
        """
        if self.importance_df is None:
            self.get_feature_importance()
        
        if top_k is not None:
            selected = self.importance_df.head(top_k)['feature'].tolist()
            logger.info(f"Selected top {top_k} features")
        else:
            selected = self.importance_df[
                self.importance_df['avg_importance'] >= threshold_pct
            ]['feature'].tolist()
            logger.info(
                f"Selected {len(selected)} features with importance >= {threshold_pct}%"
            )
        
        return selected
    
    def plot_importance(self, top_n: int = 30, save_path: str = None):
        """Plot feature importance"""
        if self.importance_df is None:
            self.get_feature_importance()
        
        # Get top N features
        top_features = self.importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(
            range(len(top_features)), 
            top_features['avg_importance'].values
        )
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Average Importance (%)')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved importance plot to {save_path}")
        
        plt.close()
    
    def analyze_feature_groups(self) -> Dict[str, float]:
        """Group features by type and analyze contribution"""
        if self.importance_df is None:
            self.get_feature_importance()
        
        groups = {
            'base': [],
            'ema': [],
            'lag': [],
            'momentum': [],
            'interaction': [],
            'statistical': [],
            'time': []
        }
        
        for feat in self.importance_df['feature']:
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
        
        # Calculate group importance
        group_importance = {}
        for group_name, features in groups.items():
            if features:
                group_imp = self.importance_df[
                    self.importance_df['feature'].isin(features)
                ]['avg_importance'].sum()
                group_importance[group_name] = group_imp
        
        return group_importance


def main():
    parser = argparse.ArgumentParser(description='Feature Selection Analysis')
    parser.add_argument('--model', type=str, default='models/xgb.bin',
                        help='Path to trained XGBoost model')
    parser.add_argument('--features', type=str, default='data/features.parquet',
                        help='Path to features parquet')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Importance threshold percentage (0-100)')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Select top K features (overrides threshold)')
    parser.add_argument('--output', type=str, default='data/selected_features.txt',
                        help='Output file for selected features')
    
    args = parser.parse_args()
    
    # Load features to get column names
    logger.info(f"Loading features from {args.features}")
    df = pd.read_parquet(args.features)
    
    # Get feature columns (exclude target and metadata)
    exclude_cols = ['baseFee_next', 'timestamp', 'number']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    logger.info(f"Total features: {len(feature_cols)}")
    
    # Initialize selector
    logger.info(f"Loading model from {args.model}")
    selector = FeatureSelector(args.model, feature_cols)
    
    # Get importance
    importance_df = selector.get_feature_importance()
    
    # Save full importance report
    importance_path = Path(args.output).parent / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Saved feature importance to {importance_path}")
    
    # Print top 20 features
    logger.info("\n" + "="*60)
    logger.info("TOP 20 MOST IMPORTANT FEATURES")
    logger.info("="*60)
    for idx, row in importance_df.head(20).iterrows():
        logger.info(f"  {row['feature']:30s} {row['avg_importance']:6.2f}%")
    
    # Analyze feature groups
    logger.info("\n" + "="*60)
    logger.info("FEATURE GROUP CONTRIBUTION")
    logger.info("="*60)
    group_imp = selector.analyze_feature_groups()
    for group, imp in sorted(group_imp.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {group:15s} {imp:6.2f}%")
    
    # Select features
    selected = selector.select_features(
        threshold_pct=args.threshold,
        top_k=args.top_k
    )
    
    logger.info("\n" + "="*60)
    logger.info(f"SELECTED {len(selected)} FEATURES")
    logger.info("="*60)
    
    # Save selected features
    with open(args.output, 'w') as f:
        for feat in selected:
            f.write(f"{feat}\n")
    
    logger.info(f"Saved selected features to {args.output}")
    
    # Plot importance
    plot_path = Path(args.output).parent / 'feature_importance.png'
    selector.plot_importance(top_n=30, save_path=str(plot_path))
    
    # Print statistics
    removed = len(feature_cols) - len(selected)
    logger.info(f"\nRemoved {removed} low-importance features ({removed/len(feature_cols)*100:.1f}%)")
    
    return selected


if __name__ == '__main__':
    main()
