"""
Compare Hybrid LSTM-XGBoost vs XGBoost Standalone
Generates comprehensive comparison report
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_metrics(metrics_path: str) -> dict:
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def compare_models():
    """Compare hybrid and standalone XGBoost models."""
    
    print("\n" + "="*70)
    print(" MODEL COMPARISON: Hybrid vs XGBoost Standalone")
    print("="*70 + "\n")
    
    # Load metrics
    hybrid_metrics_path = Path("../../models/metrics.json")
    xgb_metrics_path = Path("model/metrics.json")
    
    if not hybrid_metrics_path.exists():
        print(f"❌ Hybrid model metrics not found: {hybrid_metrics_path}")
        return
    
    if not xgb_metrics_path.exists():
        print(f"❌ XGBoost metrics not found: {xgb_metrics_path}")
        return
    
    hybrid_metrics = load_metrics(hybrid_metrics_path)
    xgb_metrics = load_metrics(xgb_metrics_path)
    
    # Create comparison table
    comparison = []
    
    metrics_to_compare = [
        ('mae_gwei', 'MAE (Gwei)', 'lower'),
        ('mape', 'MAPE (%)', 'lower'),
        ('rmse_gwei', 'RMSE (Gwei)', 'lower'),
        ('r2', 'R²', 'higher'),
        ('under_estimation_rate', 'Under-estimation (%)', 'target'),
        ('hit_at_epsilon', 'Hit@ε=5% (%)', 'higher')
    ]
    
    print(f"{'Metric':<25} {'Hybrid':>15} {'XGBoost Only':>15} {'Δ':>12} {'Winner':>12}")
    print("-" * 70)
    
    for key, display_name, better in metrics_to_compare:
        hybrid_val = hybrid_metrics.get(key, 0)
        xgb_val = xgb_metrics.get(key, 0)
        
        # Calculate difference
        if better == 'lower':
            delta = ((xgb_val - hybrid_val) / hybrid_val * 100) if hybrid_val != 0 else 0
            winner = "Hybrid ✓" if hybrid_val < xgb_val else "XGBoost ✓"
        elif better == 'higher':
            delta = ((xgb_val - hybrid_val) / hybrid_val * 100) if hybrid_val != 0 else 0
            winner = "Hybrid ✓" if hybrid_val > xgb_val else "XGBoost ✓"
        else:  # target (closer to ideal is better)
            # For under-estimation, 10-15% is ideal
            ideal = 12.5
            hybrid_diff = abs(hybrid_val - ideal)
            xgb_diff = abs(xgb_val - ideal)
            delta = xgb_val - hybrid_val
            winner = "Hybrid ✓" if hybrid_diff < xgb_diff else "XGBoost ✓"
        
        # Format values
        if key in ['mae_gwei', 'rmse_gwei']:
            hybrid_str = f"{hybrid_val:.4f}"
            xgb_str = f"{xgb_val:.4f}"
        elif key == 'r2':
            hybrid_str = f"{hybrid_val:.4f}"
            xgb_str = f"{xgb_val:.4f}"
        else:
            hybrid_str = f"{hybrid_val:.2f}"
            xgb_str = f"{xgb_val:.2f}"
        
        delta_str = f"{delta:+.2f}%" if delta != 0 else "0.00%"
        
        print(f"{display_name:<25} {hybrid_str:>15} {xgb_str:>15} {delta_str:>12} {winner:>12}")
        
        comparison.append({
            'Metric': display_name,
            'Hybrid': hybrid_val,
            'XGBoost_Only': xgb_val,
            'Delta_%': delta,
            'Winner': winner.replace(' ✓', '')
        })
    
    print("\n" + "="*70)
    print(" KEY INSIGHTS")
    print("="*70 + "\n")
    
    # Calculate performance summary
    mae_improvement = ((hybrid_metrics['mae_gwei'] - xgb_metrics['mae_gwei']) / xgb_metrics['mae_gwei'] * 100)
    mape_improvement = ((hybrid_metrics['mape'] - xgb_metrics['mape']) / xgb_metrics['mape'] * 100)
    r2_improvement = ((hybrid_metrics['r2'] - xgb_metrics['r2']) / xgb_metrics['r2'] * 100)
    
    # Winner = lower is better for MAE/MAPE
    if mae_improvement < 0:  # Hybrid is lower (better)
        print(f"✓ Hybrid model achieves {abs(mae_improvement):.2f}% better MAE (lower error)")
    else:  # XGBoost is lower (better)
        print(f"✓ XGBoost standalone achieves {mae_improvement:.2f}% better MAE (lower error)")
    
    if mape_improvement < 0:  # Hybrid is lower (better)
        print(f"✓ Hybrid model achieves {abs(mape_improvement):.2f}% better MAPE (lower error)")
    else:  # XGBoost is lower (better)
        print(f"✓ XGBoost standalone achieves {mape_improvement:.2f}% better MAPE (lower error)")
    
    # Winner = higher is better for R²
    if r2_improvement > 0:  # Hybrid is higher (better)
        print(f"✓ Hybrid model achieves {r2_improvement:.2f}% better R² (higher score)")
    else:  # XGBoost is higher (better)
        print(f"✓ XGBoost standalone achieves {abs(r2_improvement):.2f}% better R² (higher score)")
    
    # Under-estimation analysis
    hybrid_under = hybrid_metrics['under_estimation_rate']
    xgb_under = xgb_metrics['under_estimation_rate']
    ideal_under = 12.5
    
    print(f"\n📊 Under-estimation Analysis:")
    print(f"   Ideal range: 10-15% (minimize transaction failures)")
    print(f"   Hybrid:      {hybrid_under:.2f}% (deviation: {abs(hybrid_under - ideal_under):.2f}%)")
    print(f"   XGBoost:     {xgb_under:.2f}% (deviation: {abs(xgb_under - ideal_under):.2f}%)")
    
    # Model complexity
    print(f"\n🏗️  Model Complexity:")
    print(f"   Hybrid:      LSTM (192h × 2l) + XGBoost (700 trees)")
    print(f"   XGBoost:     XGBoost only (700 trees)")
    print(f"   → Hybrid adds ~120K parameters from LSTM")
    
    # Training time (estimated)
    print(f"\n⏱️  Training Efficiency:")
    print(f"   Hybrid:      2-stage training (LSTM + XGBoost)")
    print(f"   XGBoost:     Single-stage training")
    print(f"   → XGBoost standalone is faster to train")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATION:")
    
    # Count wins for each model
    hybrid_wins = sum(1 for item in comparison if 'Hybrid' in item['Winner'])
    xgb_wins = sum(1 for item in comparison if 'XGBoost' in item['Winner'])
    
    if hybrid_wins >= 5:  # Hybrid wins most metrics
        print(f"   ✅ Use HYBRID model")
        print(f"   Reason: Superior performance across {hybrid_wins}/6 metrics")
        print(f"   Key advantages:")
        print(f"   - {abs(mae_improvement):.1f}% better MAE (more accurate predictions)")
        print(f"   - {abs(hybrid_metrics['under_estimation_rate'] - 12.5):.1f}% closer to ideal under-estimation rate")
        print(f"   - LSTM captures temporal dependencies in gas fee dynamics")
        print(f"   Trade-off: Higher model complexity (~120K additional parameters)")
    elif abs(mae_improvement) < 5 and abs(mape_improvement) < 5:
        print(f"   ⚖️  Both models are comparable")
        print(f"   Consider: XGBoost standalone for simpler deployment")
        print(f"   Consider: Hybrid if temporal patterns are important")
    else:
        print(f"   ✅ Use XGBOOST standalone")
        print(f"   Reason: Wins {xgb_wins}/6 metrics with lower complexity")
        print(f"   Benefit: Faster training and simpler inference")
    
    print("\n" + "="*70 + "\n")
    
    # Save comparison to CSV
    df = pd.DataFrame(comparison)
    df.to_csv("comparison_results.csv", index=False)
    print(f"✓ Comparison saved to: comparison_results.csv\n")


if __name__ == "__main__":
    compare_models()
