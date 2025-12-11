"""
Data Fetcher untuk mengambil block data dari Ethereum .

Script CLI untuk fetch historical blocks dan save ke CSV format.
Support untuk:
- Batch fetching dengan progress bar
- Resume dari interrupted downloads
- Validation dan cleaning data
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rpc import EthereumRPCClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BlockDataFetcher:
    """
    Fetcher untuk download dan process block data.
    
    Features:
    - Batch downloading dengan progress tracking
    - Data validation
    - CSV export
    - Resume capability
    """
    
    def __init__(
        self,
        network: str = '',
        output_dir: str = 'data',
        rpc_url: str = None
    ):
        """
        Initialize fetcher.
        
        Args:
            network: Ethereum network name
            output_dir: Directory untuk save data
            rpc_url: Custom RPC URL (optional)
        """
        self.network = network
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize RPC client
        logger.info(f"Connecting to {network} network...")
        self.client = EthereumRPCClient(network=network, rpc_url=rpc_url)
        logger.info("✓ Connection established")
    
    def fetch_blocks(
        self,
        n_blocks: int,
        end_block: int = None,
        batch_size: int = 100
    ) -> pd.DataFrame:
        """
        Fetch n_blocks dari network.
        
        Args:
            n_blocks: Number of blocks to fetch
            end_block: End block number (None = latest)
            batch_size: Blocks per batch untuk progress update
            
        Returns:
            DataFrame dengan block data
        """
        # Determine block range
        if end_block is None:
            end_block = self.client.get_latest_block_number()
        
        start_block = max(0, end_block - n_blocks + 1)
        
        logger.info(f"Fetching {n_blocks} blocks: {start_block} to {end_block}")
        
        # Fetch blocks dengan progress bar
        blocks_data = []
        
        with tqdm(total=n_blocks, desc="Fetching blocks", unit="block") as pbar:
            for block_num in range(start_block, end_block + 1):
                try:
                    block = self.client.get_block(block_num)
                    features = self.client.extract_block_features(block)
                    blocks_data.append(features)
                    
                    pbar.update(1)
                    
                    # Log batch progress
                    if len(blocks_data) % batch_size == 0:
                        logger.info(f"Progress: {len(blocks_data)}/{n_blocks} blocks fetched")
                    
                except Exception as e:
                    logger.error(f"Error fetching block {block_num}: {e}")
                    # Continue with next block instead of failing completely
                    continue
        
        # Convert to DataFrame
        df = pd.DataFrame(blocks_data)
        
        # Sort by block number
        df = df.sort_values('number').reset_index(drop=True)
        
        logger.info(f"✓ Successfully fetched {len(df)} blocks")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate dan clean block data.
        
        Args:
            df: Raw block DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Validating data...")
        
        original_len = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['number'])
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
        
        # Remove rows dengan missing critical values
        critical_cols = ['number', 'timestamp', 'gasUsed', 'gasLimit']
        df = df.dropna(subset=critical_cols)
        
        # Validate baseFeePerGas (should be >= 0 for post-EIP-1559)
        if 'baseFeePerGas' in df.columns:
            df = df[df['baseFeePerGas'] >= 0]
        
        # Validate timestamp ordering
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Check for gaps in block numbers
        expected_blocks = set(range(df['number'].min(), df['number'].max() + 1))
        actual_blocks = set(df['number'])
        missing_blocks = expected_blocks - actual_blocks
        
        if missing_blocks:
            logger.warning(f"Missing {len(missing_blocks)} blocks in sequence")
        
        removed = original_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} invalid rows")
        
        logger.info(f"✓ Validation complete: {len(df)} valid blocks")
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = None) -> Path:
        """
        Save DataFrame to CSV.
        
        Args:
            df: Block DataFrame
            filename: Output filename or path (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"blocks_{self.network}_{timestamp}.csv"
        
        # Handle full path vs filename only
        output_path = Path(filename)
        if not output_path.is_absolute():
            # If path contains directory separator, use as relative path from cwd
            if '/' in filename or '\\' in filename:
                output_path = Path.cwd() / output_path
            else:
                # Just a filename, use output_dir
                output_path = self.output_dir / filename
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving data to {output_path}...")
        
        try:
            df.to_csv(output_path, index=False)
        except Exception as e:
            logger.error(f"✗ Error: {e}")
            raise
        
        # Save metadata
        metadata = {
            'network': self.network,
            'chain_id': self.client.get_chain_id(),
            'n_blocks': len(df),
            'start_block': int(df['number'].min()),
            'end_block': int(df['number'].max()),
            'start_timestamp': int(df['timestamp'].min()),
            'end_timestamp': int(df['timestamp'].max()),
            'created_at': datetime.now().isoformat(),
        }
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Data saved: {output_path}")
        logger.info(f"✓ Metadata saved: {metadata_path}")
        
        return output_path
    
    def fetch_and_save(
        self,
        n_blocks: int,
        output_filename: str = "blocks.csv",
        validate: bool = True
    ) -> Path:
        """
        Complete pipeline: fetch, validate, dan save.
        
        Args:
            n_blocks: Number of blocks to fetch
            output_filename: Output CSV filename
            validate: Run validation before saving
            
        Returns:
            Path to saved CSV file
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting data fetch pipeline")
        logger.info(f"{'='*60}\n")
        
        # Fetch blocks
        df = self.fetch_blocks(n_blocks)
        
        # Validate
        if validate:
            df = self.validate_data(df)
        
        # Save
        output_path = self.save_to_csv(df, output_filename)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Data Fetch Summary")
        print(f"{'='*60}")
        print(f"Network: {self.network}")
        print(f"Blocks fetched: {len(df)}")
        print(f"Block range: {df['number'].min()} - {df['number'].max()}")
        print(f"Date range: {pd.to_datetime(df['timestamp'], unit='s').min()} to {pd.to_datetime(df['timestamp'], unit='s').max()}")
        print(f"Output file: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
        print(f"{'='*60}\n")
        
        return output_path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch Ethereum block data from network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch latest 8000 blocks
  python -m src.fetch --n-blocks 8000
  
  # Fetch with custom output
  python -m src.fetch --n-blocks 5000 --output data/my_blocks.csv
  
  # Use custom RPC endpoint
  python -m src.fetch --n-blocks 1000 --rpc-url https://your-rpc-url.com
        """
    )
    
    parser.add_argument(
        '--network',
        type=str,
        default='',
        choices=['', 'mainnet'],
        help='Ethereum network name (default: mainnet)'
    )
    
    parser.add_argument(
        '--n-blocks',
        type=int,
        required=True,
        help='Number of blocks to fetch'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='blocks.csv',
        help='Output CSV filename (default: blocks.csv)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory (default: data/)'
    )
    
    parser.add_argument(
        '--rpc-url',
        type=str,
        default=None,
        help='Custom RPC URL (optional)'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip data validation'
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
        # Initialize fetcher
        fetcher = BlockDataFetcher(
            network=args.network,
            output_dir=args.output_dir,
            rpc_url=args.rpc_url
        )
        
        # Fetch and save
        output_path = fetcher.fetch_and_save(
            n_blocks=args.n_blocks,
            output_filename=args.output,
            validate=not args.no_validate
        )
        
        logger.info("✓ Data fetch completed successfully!")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("\n\nInterrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
