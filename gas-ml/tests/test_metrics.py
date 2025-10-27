"""
Unit tests untuk evaluation metrics.

Test coverage:
- MAE calculation
- MAPE calculation
- RMSE calculation
- Under-estimation rate
- Hit@epsilon
- Cost-saving calculation
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluate import (
    calculate_mae,
    calculate_mape,
    calculate_rmse,
    calculate_under_estimation_rate,
    calculate_hit_at_epsilon,
    calculate_cost_saving,
    evaluate_predictions
)


class TestMetrics:
    """Test cases untuk evaluation metrics."""
    
    def test_mae_perfect_prediction(self):
        """Test MAE dengan perfect predictions."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])
        
        mae = calculate_mae(y_true, y_pred)
        assert mae == 0.0
    
    def test_mae_calculation(self):
        """Test MAE calculation."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        
        mae = calculate_mae(y_true, y_pred)
        expected_mae = (10 + 10 + 10) / 3
        
        assert abs(mae - expected_mae) < 1e-6
    
    def test_mape_calculation(self):
        """Test MAPE calculation."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 180, 330])
        
        mape = calculate_mape(y_true, y_pred)
        # MAPE = mean of [10%, 10%, 10%] = 10%
        expected_mape = 10.0
        
        assert abs(mape - expected_mape) < 1e-6
    
    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        
        rmse = calculate_rmse(y_true, y_pred)
        # RMSE = sqrt(mean([100, 100, 100])) = 10
        expected_rmse = 10.0
        
        assert abs(rmse - expected_rmse) < 1e-6
    
    def test_under_estimation_rate(self):
        """Test under-estimation rate calculation."""
        y_true = np.array([100, 200, 300, 400])
        y_pred = np.array([90, 210, 290, 410])
        
        rate = calculate_under_estimation_rate(y_true, y_pred)
        # 2 out of 4 are under-estimated = 50%
        expected_rate = 50.0
        
        assert abs(rate - expected_rate) < 1e-6
    
    def test_hit_at_epsilon(self):
        """Test Hit@epsilon calculation."""
        y_true = np.array([100, 200, 300, 400])
        y_pred = np.array([102, 208, 295, 420])
        
        # With 5% tolerance:
        # [2%, 4%, 1.67%, 5%] - all within 5%
        rate = calculate_hit_at_epsilon(y_true, y_pred, epsilon=0.05)
        expected_rate = 100.0
        
        assert abs(rate - expected_rate) < 1e-6
        
        # With 3% tolerance:
        # [2%, 4%, 1.67%, 5%] - 3 out of 4 within 3%
        rate = calculate_hit_at_epsilon(y_true, y_pred, epsilon=0.03)
        expected_rate = 75.0
        
        assert abs(rate - expected_rate) < 1e-6
    
    def test_cost_saving_all_success(self):
        """Test cost saving dengan all successful transactions."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 210, 310])  # All >= actual
        baseline_pred = np.array([150, 250, 350])  # All >= actual but higher
        
        result = calculate_cost_saving(y_true, y_pred, baseline_pred, gas_limit=21000)
        
        # Model should have lower cost than baseline
        assert result['model_total_cost_wei'] < result['baseline_total_cost_wei']
        assert result['cost_saving_wei'] > 0
        assert result['model_success_rate'] == 100.0
        assert result['baseline_success_rate'] == 100.0
    
    def test_cost_saving_with_failures(self):
        """Test cost saving dengan failed transactions."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([90, 210, 310])  # 1 failure
        baseline_pred = np.array([110, 220, 320])  # All success
        
        result = calculate_cost_saving(y_true, y_pred, baseline_pred, gas_limit=21000)
        
        # Model has 1 failure (retry cost)
        assert result['model_success_rate'] < 100.0
        assert result['baseline_success_rate'] == 100.0
    
    def test_evaluate_predictions_complete(self):
        """Test complete evaluation pipeline."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([105, 195, 310, 395, 510])
        baseline_pred = np.array([120, 220, 330, 440, 550])
        
        metrics = evaluate_predictions(
            y_true, y_pred, baseline_pred,
            epsilon=0.05,
            gas_limit=21000
        )
        
        # Check all required metrics exist
        required_metrics = [
            'mae_wei', 'mae_gwei', 'mape_pct',
            'rmse_wei', 'rmse_gwei',
            'under_estimation_rate_pct',
            'hit_at_epsilon_pct',
            'cost_saving_wei', 'cost_saving_pct'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_zero_division_handling(self):
        """Test handling of edge cases."""
        y_true = np.array([100, 200])
        y_pred = np.array([100, 200])
        
        # Should not raise division by zero
        mae = calculate_mae(y_true, y_pred)
        assert mae == 0.0
        
        rmse = calculate_rmse(y_true, y_pred)
        assert rmse == 0.0


class TestEdgeCases:
    """Test edge cases dan boundary conditions."""
    
    def test_single_value(self):
        """Test dengan single value."""
        y_true = np.array([100])
        y_pred = np.array([110])
        
        mae = calculate_mae(y_true, y_pred)
        assert mae == 10.0
        
        mape = calculate_mape(y_true, y_pred)
        assert mape == 10.0
    
    def test_large_values(self):
        """Test dengan large values (Wei amounts)."""
        y_true = np.array([1e18, 2e18, 3e18])  # 1-3 ETH in Wei
        y_pred = np.array([1.1e18, 1.9e18, 3.1e18])
        
        mae = calculate_mae(y_true, y_pred)
        assert mae > 0
        
        mape = calculate_mape(y_true, y_pred)
        assert 0 < mape < 100
    
    def test_negative_handling(self):
        """Test handling of negative predictions (invalid)."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([-10, 210, 310])  # Invalid negative
        
        # MAE should still work
        mae = calculate_mae(y_true, y_pred)
        assert mae > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
