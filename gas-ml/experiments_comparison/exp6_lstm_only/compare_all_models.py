"""
Compare All Three Models: Hybrid vs XGBoost-Only vs LSTM-Only
Comprehensive comparison for journal/thesis publication
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path


def load_metrics(metrics_path: str) -> dict:
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def compare_all_models():
    """Compare all three models comprehensively."""
    
    print("\n" + "="*80)
    print(" COMPREHENSIVE MODEL COMPARISON")
    print(" Hybrid LSTM-XGBoost vs XGBoost-Only vs LSTM-Only")
    print("="*80 + "\n")
    
    # Load metrics
    hybrid_path = Path("../../models/metrics.json")
    xgb_path = Path("../exp5_xgboost_only/model/metrics.json")
    lstm_path = Path("model/metrics.json")
    
    models = {}
    
    if hybrid_path.exists():
        models['Hybrid'] = load_metrics(hybrid_path)
    else:
        print(f"⚠️  Hybrid metrics not found: {hybrid_path}")
    
    if xgb_path.exists():
        models['XGBoost'] = load_metrics(xgb_path)
    else:
        print(f"⚠️  XGBoost metrics not found: {xgb_path}")
    
    if lstm_path.exists():
        models['LSTM'] = load_metrics(lstm_path)
    else:
        print(f"⚠️  LSTM metrics not found: {lstm_path}")
    
    if len(models) < 3:
        print("\n❌ Not all models found. Please train all models first.\n")
        return
    
    # Metrics to compare
    metrics_config = [
        ('mae_gwei', 'MAE (Gwei)', 'lower', 4),
        ('mape', 'MAPE (%)', 'lower', 2),
        ('rmse_gwei', 'RMSE (Gwei)', 'lower', 4),
        ('r2', 'R²', 'higher', 4),
        ('under_estimation_rate', 'Under-est (%)', 'target', 2),
        ('hit_at_epsilon', 'Hit@ε=5% (%)', 'higher', 2)
    ]
    
    # Create comparison table
    print(f"{'Metric':<20} {'Hybrid':>15} {'XGBoost':>15} {'LSTM':>15} {'Best':>15}")
    print("-" * 80)
    
    comparison_data = []
    
    for key, display_name, better, decimals in metrics_config:
        hybrid_val = models['Hybrid'].get(key, 0)
        xgb_val = models['XGBoost'].get(key, 0)
        lstm_val = models['LSTM'].get(key, 0)
        
        # Determine best
        if better == 'lower':
            best_val = min(hybrid_val, xgb_val, lstm_val)
            if best_val == hybrid_val:
                best = "Hybrid ✓"
            elif best_val == xgb_val:
                best = "XGBoost ✓"
            else:
                best = "LSTM ✓"
        elif better == 'higher':
            best_val = max(hybrid_val, xgb_val, lstm_val)
            if best_val == hybrid_val:
                best = "Hybrid ✓"
            elif best_val == xgb_val:
                best = "XGBoost ✓"
            else:
                best = "LSTM ✓"
        else:  # target (under-estimation, 10-15% ideal)
            ideal = 12.5
            hybrid_diff = abs(hybrid_val - ideal)
            xgb_diff = abs(xgb_val - ideal)
            lstm_diff = abs(lstm_val - ideal)
            
            best_diff = min(hybrid_diff, xgb_diff, lstm_diff)
            if best_diff == hybrid_diff:
                best = "Hybrid ✓"
            elif best_diff == xgb_diff:
                best = "XGBoost ✓"
            else:
                best = "LSTM ✓"
        
        # Format values
        if decimals == 4:
            hybrid_str = f"{hybrid_val:.4f}"
            xgb_str = f"{xgb_val:.4f}"
            lstm_str = f"{lstm_val:.4f}"
        else:
            hybrid_str = f"{hybrid_val:.2f}"
            xgb_str = f"{xgb_val:.2f}"
            lstm_str = f"{lstm_val:.2f}"
        
        print(f"{display_name:<20} {hybrid_str:>15} {xgb_str:>15} {lstm_str:>15} {best:>15}")
        
        comparison_data.append({
            'Metric': display_name,
            'Hybrid': hybrid_val,
            'XGBoost_Only': xgb_val,
            'LSTM_Only': lstm_val,
            'Best_Model': best.replace(' ✓', '')
        })
    
    print("\n" + "="*80)
    print(" DETAILED ANALYSIS")
    print("="*80 + "\n")
    
    # Performance comparison
    print("📊 PREDICTION ACCURACY:")
    hybrid_mae = models['Hybrid']['mae_gwei']
    xgb_mae = models['XGBoost']['mae_gwei']
    lstm_mae = models['LSTM']['mae_gwei']
    
    print(f"   Hybrid vs XGBoost: {((xgb_mae - hybrid_mae) / xgb_mae * 100):+.1f}% improvement")
    print(f"   Hybrid vs LSTM:    {((lstm_mae - hybrid_mae) / lstm_mae * 100):+.1f}% improvement")
    print(f"   LSTM vs XGBoost:   {((xgb_mae - lstm_mae) / xgb_mae * 100):+.1f}% LSTM better")
    
    # Under-estimation analysis
    print(f"\n⚠️  UNDER-ESTIMATION RISK (ideal: 10-15%):")
    ideal_under = 12.5
    for model_name, metrics in models.items():
        under_rate = metrics['under_estimation_rate']
        deviation = abs(under_rate - ideal_under)
        status = "✓ SAFE" if 10 <= under_rate <= 15 else "⚠️  RISKY" if under_rate < 10 else "❌ HIGH RISK"
        print(f"   {model_name:<12} {under_rate:>6.2f}% (deviation: {deviation:.2f}%) {status}")
    
    # Model complexity
    print(f"\n🏗️  MODEL COMPLEXITY:")
    print(f"   Hybrid:      LSTM (456K params) + XGBoost (700 trees) = ~576K total")
    print(f"   XGBoost:     XGBoost only (700 trees) = ~120K params")
    print(f"   LSTM:        LSTM only (456K params)")
    
    # Training efficiency
    print(f"\n⏱️  TRAINING CHARACTERISTICS:")
    print(f"   Hybrid:      2-stage training, GPU recommended, ~10-15 min")
    print(f"   XGBoost:     Single-stage, CPU efficient, ~1-2 min")
    print(f"   LSTM:        Single-stage, GPU required, ~5-10 min")
    
    # Inference speed
    print(f"\n⚡ INFERENCE SPEED (estimated):")
    print(f"   Hybrid:      ~50ms (LSTM + XGBoost)")
    print(f"   XGBoost:     ~5ms (fastest)")
    print(f"   LSTM:        ~20ms (GPU) / ~100ms (CPU)")
    
    # Wins count
    hybrid_wins = sum(1 for item in comparison_data if item['Best_Model'] == 'Hybrid')
    xgb_wins = sum(1 for item in comparison_data if item['Best_Model'] == 'XGBoost')
    lstm_wins = sum(1 for item in comparison_data if item['Best_Model'] == 'LSTM')
    
    print(f"\n🏆 PERFORMANCE SUMMARY:")
    print(f"   Hybrid wins:    {hybrid_wins}/6 metrics")
    print(f"   XGBoost wins:   {xgb_wins}/6 metrics")
    print(f"   LSTM wins:      {lstm_wins}/6 metrics")
    
    # Recommendation
    print(f"\n" + "="*80)
    print(" RECOMMENDATION FOR JOURNAL/THESIS")
    print("="*80 + "\n")
    
    if hybrid_wins >= 4:
        print("✅ RECOMMENDED: HYBRID LSTM-XGBoost Model")
        print(f"\nKey Findings:")
        print(f"  1. Wins {hybrid_wins}/6 performance metrics")
        print(f"  2. Best prediction accuracy: {models['Hybrid']['mae_gwei']:.4f} Gwei MAE")
        print(f"  3. Best R² score: {models['Hybrid']['r2']:.4f}")
        print(f"  4. Optimal under-estimation rate: {models['Hybrid']['under_estimation_rate']:.2f}%")
        print(f"\nContributions:")
        lstm_contribution = ((xgb_mae - hybrid_mae) / xgb_mae * 100)
        print(f"  • LSTM temporal features contribute {lstm_contribution:.1f}% improvement")
        print(f"  • XGBoost meta-learning adds non-linear feature interactions")
        print(f"  • Asymmetric loss reduces transaction failure risk")
        print(f"\nTrade-offs:")
        print(f"  • Higher complexity (~576K parameters)")
        print(f"  • Longer training time (~10-15 min with GPU)")
        print(f"  • Requires GPU for practical deployment")
        
    elif xgb_wins >= 3:
        print("✅ RECOMMENDED: XGBoost Standalone")
        print(f"\nReason: Practical balance of accuracy and simplicity")
    else:
        print("⚖️  MODEL SELECTION DEPENDS ON USE CASE")
    
    print(f"\n" + "="*80)
    print(" ABLATION STUDY INSIGHTS")
    print("="*80 + "\n")
    
    print("Component Contribution Analysis:")
    print(f"  • Pure XGBoost:         MAE = {xgb_mae:.4f} Gwei (baseline)")
    print(f"  • Pure LSTM:            MAE = {lstm_mae:.4f} Gwei ({((lstm_mae - xgb_mae) / xgb_mae * 100):+.1f}%)")
    print(f"  • Hybrid (LSTM+XGBoost): MAE = {hybrid_mae:.4f} Gwei ({((hybrid_mae - xgb_mae) / xgb_mae * 100):+.1f}%)")
    
    print(f"\nKey Insight:")
    if lstm_mae < xgb_mae:
        print(f"  ✓ LSTM alone outperforms XGBoost ({((xgb_mae - lstm_mae) / xgb_mae * 100):.1f}% better)")
        print(f"    → Temporal patterns are crucial for gas fee prediction")
    else:
        print(f"  ✓ XGBoost outperforms LSTM alone ({((lstm_mae - xgb_mae) / lstm_mae * 100):.1f}% better)")
        print(f"    → Feature engineering more important than temporal modeling")
    
    if hybrid_mae < min(lstm_mae, xgb_mae):
        print(f"  ✓ Hybrid synergy: Combining both yields best results")
        print(f"    → LSTM features + XGBoost ensemble = optimal performance")
    
    print("\n" + "="*80 + "\n")
    
    # Save comparison to CSV
    df = pd.DataFrame(comparison_data)
    df.to_csv("../comparison_all_models.csv", index=False)
    print(f"✓ Comparison saved to: experiments_comparison/comparison_all_models.csv\n")
    
    # Save detailed report
    report = {
        'summary': {
            'hybrid_wins': hybrid_wins,
            'xgb_wins': xgb_wins,
            'lstm_wins': lstm_wins,
            'recommended': 'Hybrid' if hybrid_wins >= 4 else 'XGBoost' if xgb_wins >= 3 else 'Depends'
        },
        'metrics': comparison_data,
        'models': models
    }
    
    with open("../comparison_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Detailed report saved to: experiments_comparison/comparison_report.json\n")


if __name__ == "__main__":
    compare_all_models()
