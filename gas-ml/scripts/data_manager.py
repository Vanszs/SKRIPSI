"""
Data Manager CLI - Manage data directory with cleanup and organization
Usage:
    python scripts/data_manager.py list              # List all files
    python scripts/data_manager.py clean              # Clean redundant files
    python scripts/data_manager.py organize           # Organize into subdirectories
    python scripts/data_manager.py info               # Show data statistics
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import shutil
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataManager:
    """Manage data directory organization and cleanup"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.archive_dir = self.data_dir / "archive"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = Path("models")
        
    def list_files(self):
        """List all files in data directory with sizes"""
        logger.info("="*60)
        logger.info("DATA DIRECTORY CONTENTS")
        logger.info("="*60)
        
        files = sorted(self.data_dir.glob("*"))
        total_size = 0
        
        for file in files:
            if file.is_file():
                size_mb = file.stat().st_size / 1024 / 1024
                total_size += size_mb
                modified = datetime.fromtimestamp(file.stat().st_mtime)
                logger.info(f"{file.name:40s} {size_mb:8.2f} MB  {modified:%Y-%m-%d %H:%M}")
        
        logger.info("="*60)
        logger.info(f"Total: {len([f for f in files if f.is_file()])} files, {total_size:.2f} MB")
        
    def get_file_info(self) -> Dict[str, dict]:
        """Analyze files and categorize them"""
        info = {
            'raw_blocks': [],
            'features': [],
            'selected_features': [],
            'analysis': [],
            'metadata': [],
            'redundant': []
        }
        
        for file in self.data_dir.glob("*"):
            if not file.is_file():
                continue
                
            name = file.name
            size_mb = file.stat().st_size / 1024 / 1024
            
            # Categorize
            if name.startswith('blocks') and (name.endswith('.csv') or name.endswith('.json')):
                info['raw_blocks'].append((name, size_mb))
            elif name.startswith('features') and name.endswith('.parquet'):
                info['features'].append((name, size_mb))
            elif 'selected_features' in name:
                info['selected_features'].append((name, size_mb))
            elif 'correlation' in name or 'importance' in name:
                info['analysis'].append((name, size_mb))
            elif name.endswith('.txt') and 'features' in name:
                info['metadata'].append((name, size_mb))
            elif name.endswith('.json'):
                info['metadata'].append((name, size_mb))
        
        return info
    
    def identify_redundant(self) -> List[str]:
        """Identify redundant files to clean up"""
        redundant = []
        
        # Keep only latest/best versions
        keep_files = {
            'blocks_5k.csv',  # Latest blocks
            'features_5k_selected.parquet',  # Best features (elite model)
            'selected_features.txt',  # Feature list for elite model
            'feature_correlation.csv',  # Analysis results
        }
        
        for file in self.data_dir.glob("*"):
            if file.is_file() and file.name not in keep_files and file.name != '.gitkeep':
                redundant.append(file.name)
        
        return sorted(redundant)
    
    def clean_redundant(self, dry_run: bool = True):
        """Clean up redundant files"""
        redundant = self.identify_redundant()
        
        if not redundant:
            logger.info("✓ No redundant files found")
            return
        
        logger.info("="*60)
        logger.info("REDUNDANT FILES TO CLEAN")
        logger.info("="*60)
        
        total_size = 0
        for filename in redundant:
            file = self.data_dir / filename
            size_mb = file.stat().st_size / 1024 / 1024
            total_size += size_mb
            logger.info(f"  - {filename:40s} {size_mb:8.2f} MB")
        
        logger.info("="*60)
        logger.info(f"Total to clean: {len(redundant)} files, {total_size:.2f} MB")
        
        if dry_run:
            logger.info("\n⚠️  DRY RUN MODE - No files deleted")
            logger.info("Run with --confirm to actually delete files")
            return
        
        # Create archive before deleting
        self.archive_dir.mkdir(exist_ok=True)
        
        logger.info("\nArchiving files before deletion...")
        for filename in redundant:
            src = self.data_dir / filename
            dst = self.archive_dir / filename
            shutil.move(str(src), str(dst))
            logger.info(f"  Archived: {filename}")
        
        logger.info(f"\n✓ Cleaned {len(redundant)} files")
        logger.info(f"✓ Archived to: {self.archive_dir}")
    
    def organize_directories(self):
        """Organize files into subdirectories"""
        logger.info("="*60)
        logger.info("ORGANIZING DATA DIRECTORY")
        logger.info("="*60)
        
        # Create subdirectories
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        moved = []
        
        # Move raw blocks to raw/
        for file in self.data_dir.glob("blocks*.csv"):
            if file.parent == self.data_dir:
                dst = self.raw_dir / file.name
                shutil.move(str(file), str(dst))
                moved.append(f"{file.name} -> raw/")
        
        # Move processed features to processed/
        for file in self.data_dir.glob("features*.parquet"):
            if file.parent == self.data_dir:
                dst = self.processed_dir / file.name
                shutil.move(str(file), str(dst))
                moved.append(f"{file.name} -> processed/")
        
        # Move analysis files to processed/
        for file in self.data_dir.glob("*correlation*.csv"):
            if file.parent == self.data_dir:
                dst = self.processed_dir / file.name
                shutil.move(str(file), str(dst))
                moved.append(f"{file.name} -> processed/")
        
        if moved:
            for item in moved:
                logger.info(f"  Moved: {item}")
            logger.info(f"\n✓ Organized {len(moved)} files")
        else:
            logger.info("✓ All files already organized")
    
    def show_stats(self):
        """Show statistics about datasets"""
        logger.info("="*60)
        logger.info("DATA STATISTICS")
        logger.info("="*60)
        
        # Find best features file
        features_file = self.data_dir / "features_5k_selected.parquet"
        if not features_file.exists():
            features_file = self.data_dir / "processed" / "features_5k_selected.parquet"
        
        if features_file.exists():
            df = pd.read_parquet(features_file)
            logger.info(f"\nDataset: {features_file.name}")
            logger.info(f"  Samples: {len(df):,}")
            logger.info(f"  Features: {len(df.columns) - 3}")  # Exclude number, timestamp, target
            logger.info(f"  File size: {features_file.stat().st_size / 1024 / 1024:.2f} MB")
            
            if 'baseFee_next' in df.columns:
                logger.info(f"\nTarget (baseFee_next) Stats:")
                logger.info(f"  Mean: {df['baseFee_next'].mean() / 1e9:.6f} Gwei")
                logger.info(f"  Std:  {df['baseFee_next'].std() / 1e9:.6f} Gwei")
                logger.info(f"  Min:  {df['baseFee_next'].min() / 1e9:.6f} Gwei")
                logger.info(f"  Max:  {df['baseFee_next'].max() / 1e9:.6f} Gwei")
        
        # Model metrics
        metrics_file = self.models_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            logger.info(f"\nCurrent Model Performance:")
            logger.info(f"  MAE:       {metrics.get('mae_gwei', 0):.4f} Gwei")
            logger.info(f"  MAPE:      {metrics.get('mape', 0):.2f}%")
            logger.info(f"  R²:        {metrics.get('r2', 0):.4f}")
            logger.info(f"  Hit@5%:    {metrics.get('hit_at_epsilon', 0):.2f}%")
        
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Data Manager CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all files
  python scripts/data_manager.py list
  
  # Show what would be cleaned (dry run)
  python scripts/data_manager.py clean
  
  # Actually clean files
  python scripts/data_manager.py clean --confirm
  
  # Organize into subdirectories
  python scripts/data_manager.py organize
  
  # Show statistics
  python scripts/data_manager.py info
        """
    )
    
    parser.add_argument('action', choices=['list', 'clean', 'organize', 'info'],
                        help='Action to perform')
    parser.add_argument('--confirm', action='store_true',
                        help='Confirm deletion (for clean action)')
    parser.add_argument('--data-dir', default='data',
                        help='Data directory path')
    
    args = parser.parse_args()
    
    manager = DataManager(args.data_dir)
    
    if args.action == 'list':
        manager.list_files()
    
    elif args.action == 'clean':
        manager.clean_redundant(dry_run=not args.confirm)
    
    elif args.action == 'organize':
        manager.organize_directories()
    
    elif args.action == 'info':
        manager.show_stats()


if __name__ == '__main__':
    main()
