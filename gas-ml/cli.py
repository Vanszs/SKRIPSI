#!/usr/bin/env python
"""
🚀 Gas Fee ML - Command Line Interface

Production-ready CLI untuk:
- Real-time gas fee prediction
- Historical backtesting
- Model information
- Data management

Usage:
    python cli.py predict --rpc <RPC_URL>
    python cli.py predict --rpc <RPC_URL> --continuous
    python cli.py info
    python cli.py backtest --data data/blocks_5k.csv
"""

import click
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from rpc import EthereumRPCClient
from models import load_model_from_dir
from features import FeatureEngineer
from policy import GasFeePolicy, create_default_policy
import numpy as np
import pickle
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimePredictor:
    """Real-time gas fee predictor with RPC integration."""
    
    def __init__(self, rpc_url: str, model_dir: str = 'models', network: str = 'mainnet'):
        """
        Initialize real-time predictor.
        
        Args:
            rpc_url: Ethereum RPC endpoint URL
            model_dir: Directory containing trained model
            network: Network name (mainnet, )
        """
        self.network = network
        
        # Initialize RPC client
        logger.info(f"Connecting to {network} via RPC...")
        self.rpc_client = EthereumRPCClient(network=network, rpc_url=rpc_url)
        
        # Load model
        logger.info(f"Loading model from {model_dir}...")
        self.model, self.metadata = load_model_from_dir(Path(model_dir))
        
        # Load scaler
        with open(Path(model_dir) / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load training info
        with open(Path(model_dir) / 'training_info.json', 'r') as f:
            self.training_info = json.load(f)
        
        self.feature_columns = self.training_info['feature_columns']
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.policy = create_default_policy()
        
        logger.info("✅ Real-time predictor initialized successfully!")
    
    def fetch_recent_blocks(self, n_blocks: int = 100) -> pd.DataFrame:
        """
        Fetch recent blocks from network.
        
        Args:
            n_blocks: Number of blocks to fetch
            
        Returns:
            DataFrame with block data
        """
        logger.info(f"Fetching {n_blocks} recent blocks...")
        
        blocks_data = []
        latest = self.rpc_client.get_latest_block_number()
        start = max(0, latest - n_blocks + 1)
        
        for block_num in range(start, latest + 1):
            try:
                block = self.rpc_client.get_block(block_num)
                features = self.rpc_client.extract_block_features(block)
                blocks_data.append(features)
            except Exception as e:
                logger.warning(f"Failed to fetch block {block_num}: {e}")
                continue
        
        df = pd.DataFrame(blocks_data)
        logger.info(f"✅ Fetched {len(df)} blocks (latest: {latest})")
        
        return df
    
    def get_recommendation(self) -> dict:
        """
        Get gas fee recommendation for next block.
        
        Returns:
            Dictionary with prediction and recommendation
        """
        try:
            # Fetch data
            seq_len = self.model.sequence_length
            n_blocks = seq_len + 50  # Extra for feature engineering
            
            df = self.fetch_recent_blocks(n_blocks)
            
            if len(df) < seq_len:
                raise ValueError(f"Not enough blocks: got {len(df)}, need {seq_len}")
            
            # Engineer features
            logger.info("Engineering features...")
            df = self.feature_engineer.engineer_features(df)
            
            # Prepare features
            X = df[self.feature_columns].values
            X_norm = self.scaler.transform(X)
            
            # Predict
            logger.info("Making prediction...")
            prediction = self.model.predict(X_norm)
            
            # Use last prediction
            predicted_base_fee = float(prediction[-1])
            
            # Calculate confidence (based on model MAE)
            mae = self.training_info.get('metrics', {}).get('mae', 2.1e6)  # Wei
            confidence_interval = mae * 1.96  # 95% CI
            
            # Generate recommendation
            recent_blocks = df.tail(10).to_dict('records')
            recommendation = self.policy.generate_recommendation(
                predicted_base_fee=predicted_base_fee,
                recent_blocks=recent_blocks,
                prediction_uncertainty=confidence_interval
            )
            
            # Add metadata
            recommendation['network'] = self.network
            recommendation['current_block'] = int(df.iloc[-1]['number'])
            recommendation['current_base_fee_gwei'] = float(df.iloc[-1]['baseFeePerGas'] / 1e9)
            recommendation['current_utilization'] = float(df.iloc[-1]['gasUsed'] / df.iloc[-1]['gasLimit'])
            recommendation['confidence_interval_gwei'] = confidence_interval / 1e9
            recommendation['timestamp'] = datetime.now().isoformat()
            
            logger.info("✅ Prediction completed successfully!")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise


@click.group()
@click.version_option(version='1.0.0', prog_name='Gas Fee ML CLI')
def cli():
    """
    🚀 Gas Fee ML - Production-Ready Gas Fee Prediction CLI
    
    Ethereum gas fee prediction using hybrid LSTM-XGBoost model.
    """
    pass


@cli.command()
@click.option('--rpc', 
              default=lambda: os.getenv('MAINNET_RPC_URL') or os.getenv('MAINNET_PUBLIC_RPC'),
              help='Ethereum RPC URL (default: from .env)')
@click.option('--model-dir', default='models', help='Model directory (default: models/)')
@click.option('--network', default='mainnet', help='Network name (default: mainnet)')
@click.option('--continuous', is_flag=True, help='Run continuous prediction mode')
@click.option('--interval', default=12, help='Interval in continuous mode (seconds, default: 12)')
@click.option('--save', is_flag=True, help='Save predictions to JSON file')
def predict(rpc, model_dir, network, continuous, interval, save):
    """
    🔮 Get real-time gas fee prediction and recommendation.
    
    Examples:
        # Single prediction (using .env RPC)
        python cli.py predict
        
        # Single prediction (custom RPC)
        python cli.py predict --rpc https://eth.llamarpc.com
        
        # Continuous monitoring
        python cli.py predict --continuous
        
        # Save predictions
        python cli.py predict --save
    """
    try:
        # Get RPC URL from env if not provided
        if rpc is None:
            rpc = os.getenv('MAINNET_RPC_URL') or os.getenv('MAINNET_PUBLIC_RPC')
            if not rpc:
                click.echo("❌ Error: No RPC URL provided. Set MAINNET_RPC_URL in .env or use --rpc option.", err=True)
                click.echo("\nExample .env:")
                click.echo("  MAINNET_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY")
                click.echo("  MAINNET_PUBLIC_RPC=https://eth.llamarpc.com")
                sys.exit(1)
        
        click.echo(f"🔗 Connecting to: {rpc[:50]}...")
        
        # Initialize predictor
        predictor = RealtimePredictor(rpc_url=rpc, model_dir=model_dir, network=network)
        
        if continuous:
            # Continuous mode
            click.echo(f"\n🔄 Starting continuous prediction (interval: {interval}s)")
            click.echo("Press Ctrl+C to stop\n")
            
            import time
            while True:
                try:
                    recommendation = predictor.get_recommendation()
                    
                    # Display
                    display_recommendation(recommendation)
                    
                    # Save if requested
                    if save:
                        save_prediction(recommendation)
                    
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    click.echo("\n\n⏹️  Stopping continuous prediction...")
                    break
                except Exception as e:
                    click.echo(f"\n❌ Error: {e}", err=True)
                    time.sleep(interval)
        
        else:
            # Single prediction
            recommendation = predictor.get_recommendation()
            
            # Display
            display_recommendation(recommendation)
            
            # Save if requested
            if save:
                filepath = save_prediction(recommendation)
                click.echo(f"\n💾 Prediction saved to: {filepath}")
        
        sys.exit(0)
        
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model-dir', default='models', help='Model directory (default: models/)')
def info(model_dir):
    """
    ℹ️  Display model information and performance metrics.
    
    Example:
        python cli.py info
    """
    try:
        model_dir = Path(model_dir)
        
        # Load metrics
        with open(model_dir / 'metrics.json', 'r') as f:
            metrics = json.load(f)
        
        # Load training info
        with open(model_dir / 'training_info.json', 'r') as f:
            training_info = json.load(f)
            
        # Load metadata (Required for architecture info)
        try:
            with open(model_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = {}
        
        # Display
        click.echo("\n" + "="*60)
        click.echo("🧠 MODEL INFORMATION")
        click.echo("="*60)
        
        click.echo("\n📊 Performance Metrics:")
        click.echo(f"   MAE:              {metrics['mae_gwei']:.4f} Gwei")
        click.echo(f"   MAPE:             {metrics['mape']:.2f}%")
        click.echo(f"   RMSE:             {metrics['rmse_gwei']:.4f} Gwei")
        click.echo(f"   R²:               {metrics['r2']:.4f}")
        if 'epsilon' in metrics and 'hit_at_epsilon' in metrics:
            val = metrics['epsilon']
            # Heuristic: if < 1 assume it's a ratio (0.05), else percentage (5)
            display_val = val * 100 if val < 1.0 else val
            click.echo(f"   Hit@{display_val:.0f}%:          {metrics['hit_at_epsilon']:.2f}%")
        click.echo(f"   Under-estimation: {metrics['under_estimation_rate']:.2f}%")
        
        click.echo("\n🏗️  Architecture:")
        click.echo("\n🏗️  Architecture:")
        
        cfg = training_info.get('config', {}).get('model', {})
        
        click.echo(f"   Model Type:       {metadata.get('model_type', 'Unknown').upper()}")
        
        if 'xgboost' in cfg:
             xgb_cfg = cfg['xgboost']
             click.echo(f"   XGBoost trees:    {xgb_cfg.get('n_estimators', 'N/A')}")
             click.echo(f"   XGBoost depth:    {xgb_cfg.get('max_depth', 'N/A')}")
             
        if 'lstm' in cfg and metadata.get('model_type') == 'lstm':
             lstm_cfg = cfg['lstm']
             click.echo(f"   LSTM hidden:      {lstm_cfg.get('hidden_size', 'N/A')}")
             click.echo(f"   LSTM layers:      {lstm_cfg.get('num_layers', 'N/A')}")

        seq_len = metadata.get('sequence_length', 1)
        click.echo(f"   Sequence length:  {seq_len} blocks")
        
        click.echo("\n📦 Training Data:")
        click.echo(f"   Features:         {len(training_info['feature_columns'])}")
        click.echo(f"   Timestamp:        {training_info['timestamp']}")
        
        click.echo("\n" + "="*60 + "\n")
        
        sys.exit(0)
        
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--data', required=True, help='Historical blocks CSV file')
@click.option('--model-dir', default='models', help='Model directory (default: models/)')
@click.option('--output', default='reports/backtest.csv', help='Output CSV file')
def backtest(data, model_dir, output):
    """
    📈 Run backtest on historical data.
    
    Example:
        python cli.py backtest --data data/blocks_5k.csv
    """
    try:
        click.echo(f"\n📊 Running backtest on {data}...")
        
        # Load model
        model, metadata = load_model_from_dir(Path(model_dir))
        
        # Load scaler
        with open(Path(model_dir) / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load training info
        with open(Path(model_dir) / 'training_info.json', 'r') as f:
            training_info = json.load(f)
        
        feature_columns = training_info['feature_columns']
        
        # Load data
        df = pd.read_csv(data)
        click.echo(f"✅ Loaded {len(df)} blocks")
        
        # Engineer features
        click.echo("🔧 Engineering features...")
        feature_engineer = FeatureEngineer()
        df = feature_engineer.engineer_features(df)
        
        # Prepare features
        X = df[feature_columns].values
        X_norm = scaler.transform(X)
        y_true = df['baseFee_next'].values
        
        # Predict
        click.echo("🔮 Making predictions...")
        y_pred = model.predict(X_norm)
        
        # Align
        seq_len = model.sequence_length
        y_true_aligned = y_true[seq_len - 1:]
        
        # Calculate metrics
        from src.evaluate import evaluate_predictions, calculate_baseline_prediction
        
        baseline_pred = calculate_baseline_prediction(y_true_aligned, method='eip1559')
        
        metrics = evaluate_predictions(
            y_true_aligned, y_pred, baseline_pred,
            epsilon=0.05, gas_limit=21000
        )
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("📊 BACKTEST RESULTS")
        click.echo("="*60)
        click.echo(f"\nAccuracy Metrics:")
        click.echo(f"   MAE:              {metrics['mae_gwei']:.4f} Gwei")
        click.echo(f"   MAPE:             {metrics['mape_pct']:.2f}%")
        click.echo(f"   RMSE:             {metrics['rmse_gwei']:.4f} Gwei")
        click.echo(f"   Hit@5%:           {metrics['hit_at_epsilon_pct']:.2f}%")
        
        if 'cost_saving_pct' in metrics:
            click.echo(f"\nEconomic Impact:")
            click.echo(f"   Cost Saving:      {metrics['cost_saving_pct']:.2f}%")
            click.echo(f"   Model Success:    {metrics['model_success_rate']:.2f}%")
            click.echo(f"   Baseline Success: {metrics['baseline_success_rate']:.2f}%")
        
        click.echo("="*60 + "\n")
        
        # Save results
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_df = pd.DataFrame({
            'block_number': df['number'].iloc[seq_len - 1:].values,
            'actual_baseFee_gwei': y_true_aligned / 1e9,
            'predicted_baseFee_gwei': y_pred / 1e9,
            'baseline_baseFee_gwei': baseline_pred / 1e9,
            'error_gwei': (y_pred - y_true_aligned) / 1e9,
            'error_pct': np.abs((y_pred - y_true_aligned) / y_true_aligned) * 100
        })
        
        results_df.to_csv(output_path, index=False)
        click.echo(f"💾 Results saved to: {output_path}")
        
        sys.exit(0)
        
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


def display_recommendation(rec: dict):
    """Display recommendation in formatted output."""
    click.echo("\n" + "="*60)
    click.echo("⛽ GAS FEE RECOMMENDATION")
    click.echo("="*60)
    
    click.echo(f"\n🌐 Network: {rec['network'].upper()}")
    click.echo(f"📍 Current Block: #{rec['current_block']}")
    click.echo(f"💰 Current Base Fee: {rec['current_base_fee_gwei']:.4f} Gwei")
    click.echo(f"📊 Current Utilization: {rec['current_utilization']*100:.1f}%")
    
    click.echo("\n🔮 Prediction:")
    click.echo(f"   Base Fee (predicted): {rec['predicted_base_fee_gwei']:.4f} Gwei")
    click.echo(f"   Base Fee (buffered):  {rec['buffered_base_fee_gwei']:.4f} Gwei")
    click.echo(f"   Confidence Interval:  ±{rec.get('confidence_interval_gwei', 0):.4f} Gwei (95% CI)")
    
    click.echo("\n💡 Recommendation:")
    click.echo(f"   Priority Fee:    {rec['priority_fee_gwei']:.4f} Gwei")
    click.echo(f"   Max Fee Per Gas: {rec['max_fee_per_gas_gwei']:.4f} Gwei")
    click.echo(f"   Buffer Applied:  {(rec['buffer_multiplier'] - 1) * 100:.1f}%")
    
    # Status with colors
    status = rec['status']
    if status == 'Optimal':
        status_icon = "✅"
        status_color = 'green'
    elif status == 'Moderate':
        status_icon = "⚠️ "
        status_color = 'yellow'
    else:
        status_icon = "❌"
        status_color = 'red'
    
    click.echo(f"\n{status_icon} Status: ", nl=False)
    click.secho(f"{status}", fg=status_color, bold=True)
    click.echo(f"   {rec['status_description']}")
    click.echo(f"   Confidence: {rec['confidence']*100:.1f}%")
    
    click.echo("="*60 + "\n")


def save_prediction(rec: dict) -> Path:
    """Save prediction to JSON file."""
    output_dir = Path('predictions')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"prediction_{timestamp}.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(rec, f, indent=2)
    
    return filepath


if __name__ == '__main__':
    cli()
