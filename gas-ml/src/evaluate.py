"""
Evaluation Metrics untuk Gas Fee Prediction Model.

Metrics yang diimplementasikan:
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Under-estimation Rate
- Hit@ε (accuracy within tolerance)
- Cost-saving vs baseline
"""

import numpy as np
from typing import Dict, Tuple
import pandas as pd


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        MAPE value (percentage)
    """
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_under_estimation_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate rate dimana predicted < actual (transaksi bisa gagal).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Under-estimation rate (percentage)
    """
    return float(np.mean(y_pred < y_true) * 100)


def calculate_hit_at_epsilon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 0.05
) -> float:
    """
    Calculate Hit@ε: percentage of predictions within ε tolerance.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        epsilon: Tolerance threshold (default: 5%)
        
    Returns:
        Hit@ε percentage
    """
    relative_error = np.abs(y_pred - y_true) / y_true
    within_tolerance = relative_error <= epsilon
    return float(np.mean(within_tolerance) * 100)


def calculate_cost_saving(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray,
    gas_limit: int = 21000
) -> Dict[str, float]:
    """
    Calculate cost saving vs baseline.
    
    Assumptions:
    - Predicted fee digunakan as maxFeePerGas
    - Actual baseFee charged jika maxFee >= baseFee
    - Transaction fails jika maxFee < baseFee
    
    Args:
        y_true: Actual baseFee
        y_pred: Predicted baseFee (our model)
        baseline_pred: Baseline predicted baseFee (e.g., EIP-1559 default)
        gas_limit: Gas limit per transaction
        
    Returns:
        Dictionary dengan cost metrics
    """
    # Calculate costs (in Wei)
    # For successful transactions: cost = actual_baseFee * gasLimit
    # For failed transactions: assume retry cost
    
    # Model costs
    model_success = y_pred >= y_true
    model_cost = np.where(
        model_success,
        y_true * gas_limit,  # Successful: pay actual baseFee
        y_pred * gas_limit + y_true * gas_limit  # Failed: pay both attempts
    )
    
    # Baseline costs
    baseline_success = baseline_pred >= y_true
    baseline_cost = np.where(
        baseline_success,
        y_true * gas_limit,
        baseline_pred * gas_limit + y_true * gas_limit
    )
    
    # Calculate metrics
    model_total_cost = np.sum(model_cost)
    baseline_total_cost = np.sum(baseline_cost)
    
    cost_saving = baseline_total_cost - model_total_cost
    cost_saving_pct = (cost_saving / baseline_total_cost) * 100
    
    model_success_rate = np.mean(model_success) * 100
    baseline_success_rate = np.mean(baseline_success) * 100
    
    return {
        'model_total_cost_wei': float(model_total_cost),
        'baseline_total_cost_wei': float(baseline_total_cost),
        'cost_saving_wei': float(cost_saving),
        'cost_saving_pct': float(cost_saving_pct),
        'model_success_rate': float(model_success_rate),
        'baseline_success_rate': float(baseline_success_rate),
        'model_avg_cost_gwei': float(model_total_cost / len(y_true) / 1e9),
        'baseline_avg_cost_gwei': float(baseline_total_cost / len(y_true) / 1e9),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray = None,
    epsilon: float = 0.05,
    gas_limit: int = 21000
) -> Dict[str, float]:
    """
    Comprehensive evaluation of predictions.
    
    Args:
        y_true: Actual values
        y_pred: Model predictions
        baseline_pred: Baseline predictions (optional)
        epsilon: Tolerance for Hit@ε
        gas_limit: Gas limit for cost calculation
        
    Returns:
        Dictionary dengan semua metrics
    """
    metrics = {
        'mae_wei': calculate_mae(y_true, y_pred),
        'mae_gwei': calculate_mae(y_true, y_pred) / 1e9,
        'mape_pct': calculate_mape(y_true, y_pred),
        'rmse_wei': calculate_rmse(y_true, y_pred),
        'rmse_gwei': calculate_rmse(y_true, y_pred) / 1e9,
        'under_estimation_rate_pct': calculate_under_estimation_rate(y_true, y_pred),
        'hit_at_epsilon_pct': calculate_hit_at_epsilon(y_true, y_pred, epsilon),
        'epsilon': epsilon * 100,
    }
    
    # Add cost metrics if baseline provided
    if baseline_pred is not None:
        cost_metrics = calculate_cost_saving(
            y_true, y_pred, baseline_pred, gas_limit
        )
        metrics.update(cost_metrics)
    
    return metrics


def print_evaluation_report(metrics: Dict[str, float]):
    """
    Print formatted evaluation report.
    
    Args:
        metrics: Dictionary dengan evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    print("\n📊 Accuracy Metrics:")
    print(f"  MAE:  {metrics['mae_gwei']:.4f} Gwei")
    print(f"  MAPE: {metrics['mape_pct']:.2f}%")
    print(f"  RMSE: {metrics['rmse_gwei']:.4f} Gwei")
    
    print("\n🎯 Prediction Quality:")
    print(f"  Hit@ε ({metrics['epsilon']:.0f}%): {metrics['hit_at_epsilon_pct']:.2f}%")
    print(f"  Under-estimation Rate: {metrics['under_estimation_rate_pct']:.2f}%")
    
    if 'cost_saving_pct' in metrics:
        print("\n💰 Cost Analysis:")
        print(f"  Model Success Rate: {metrics['model_success_rate']:.2f}%")
        print(f"  Baseline Success Rate: {metrics['baseline_success_rate']:.2f}%")
        print(f"  Cost Saving: {metrics['cost_saving_pct']:.2f}%")
        print(f"  Avg Cost (Model): {metrics['model_avg_cost_gwei']:.4f} Gwei")
        print(f"  Avg Cost (Baseline): {metrics['baseline_avg_cost_gwei']:.4f} Gwei")
    
    print("="*60 + "\n")


def calculate_baseline_prediction(y_true: np.ndarray, method: str = 'eip1559') -> np.ndarray:
    """
    Calculate baseline predictions untuk comparison.
    
    Args:
        y_true: Actual values
        method: Baseline method ('eip1559', 'simple_buffer', 'rolling_mean')
        
    Returns:
        Baseline predictions
    """
    if method == 'eip1559':
        # EIP-1559 default: 2x current baseFee
        # Simulate dengan shift + buffer
        baseline = np.roll(y_true, 1) * 2.0
        baseline[0] = y_true[0] * 2.0  # Handle first value
        
    elif method == 'simple_buffer':
        # Simple: previous + 10% buffer
        baseline = np.roll(y_true, 1) * 1.1
        baseline[0] = y_true[0] * 1.1
        
    elif method == 'rolling_mean':
        # Rolling mean dengan buffer
        window = 6
        baseline = pd.Series(y_true).rolling(window, min_periods=1).mean().values * 1.1
        
    else:
        raise ValueError(f"Unknown baseline method: {method}")
    
    return baseline
