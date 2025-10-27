"""
Policy Recommender untuk Gas Fee Strategy.

Menghasilkan rekomendasi:
- priorityFee (tip untuk miner)
- buffer pada predicted baseFee
- maxFeePerGas final
- confidence level
"""

import numpy as np
from typing import Dict, Any, Tuple
import pandas as pd


class GasFeePolicy:
    """
    Policy untuk generate rekomendasi gas fee.
    
    Strategy:
    1. Predicted baseFee dari model
    2. Add buffer berdasarkan uncertainty
    3. Calculate priority fee dari recent blocks
    4. Combine untuk final maxFee recommendation
    """
    
    def __init__(
        self,
        priority_fee_percentile: float = 0.5,
        buffer_multiplier: float = 1.1,
        max_fee_multiplier: float = 1.5,
        confidence_threshold: float = 0.85
    ):
        """
        Initialize policy.
        
        Args:
            priority_fee_percentile: Percentile untuk priority fee (0-1)
            buffer_multiplier: Multiplier untuk baseFee buffer
            max_fee_multiplier: Max multiplier untuk safety
            confidence_threshold: Confidence threshold untuk status
        """
        self.priority_fee_percentile = priority_fee_percentile
        self.buffer_multiplier = buffer_multiplier
        self.max_fee_multiplier = max_fee_multiplier
        self.confidence_threshold = confidence_threshold
    
    def estimate_priority_fee(
        self,
        recent_blocks: list,
        percentile: float = None
    ) -> int:
        """
        Estimate priority fee dari recent blocks.
        
        Args:
            recent_blocks: List of recent block data
            percentile: Custom percentile (optional)
            
        Returns:
            Suggested priority fee (Wei)
        """
        if percentile is None:
            percentile = self.priority_fee_percentile
        
        # Extract priority fees dari recent transactions
        # Simplified: use median/percentile dari gasPrice - baseFee
        priority_fees = []
        
        for block in recent_blocks:
            if 'baseFeePerGas' in block and 'transactions' in block:
                base_fee = block['baseFeePerGas']
                
                for tx in block['transactions']:
                    if isinstance(tx, dict) and 'gasPrice' in tx:
                        priority = tx['gasPrice'] - base_fee
                        if priority > 0:
                            priority_fees.append(priority)
        
        if not priority_fees:
            # Default fallback: 1.5 Gwei
            return int(1.5 * 1e9)
        
        # Calculate percentile
        suggested_fee = int(np.percentile(priority_fees, percentile * 100))
        
        # Clamp to reasonable range (0.1 - 10 Gwei)
        min_fee = int(0.1 * 1e9)
        max_fee = int(10 * 1e9)
        
        return max(min_fee, min(suggested_fee, max_fee))
    
    def calculate_buffer(
        self,
        predicted_base_fee: float,
        prediction_uncertainty: float = None
    ) -> float:
        """
        Calculate buffer untuk predicted baseFee.
        
        Args:
            predicted_base_fee: Predicted baseFee (Wei)
            prediction_uncertainty: Model uncertainty (optional)
            
        Returns:
            Buffered baseFee (Wei)
        """
        if prediction_uncertainty is not None:
            # Dynamic buffer based on uncertainty
            # Higher uncertainty → larger buffer
            dynamic_multiplier = 1.0 + (prediction_uncertainty / predicted_base_fee)
            multiplier = min(self.max_fee_multiplier, dynamic_multiplier)
        else:
            # Static buffer
            multiplier = self.buffer_multiplier
        
        return predicted_base_fee * multiplier
    
    def generate_recommendation(
        self,
        predicted_base_fee: float,
        recent_blocks: list = None,
        prediction_uncertainty: float = None
    ) -> Dict[str, Any]:
        """
        Generate complete gas fee recommendation.
        
        Args:
            predicted_base_fee: Model prediction untuk next baseFee (Wei)
            recent_blocks: Recent block data untuk priority fee (optional)
            prediction_uncertainty: Model uncertainty (optional)
            
        Returns:
            Dictionary dengan recommendations
        """
        # Calculate buffered baseFee
        buffered_base_fee = self.calculate_buffer(
            predicted_base_fee,
            prediction_uncertainty
        )
        
        # Estimate priority fee
        if recent_blocks:
            priority_fee = self.estimate_priority_fee(recent_blocks)
        else:
            # Default: 1.5 Gwei
            priority_fee = int(1.5 * 1e9)
        
        # Calculate maxFeePerGas
        max_fee_per_gas = int(buffered_base_fee + priority_fee)
        
        # Calculate confidence
        if prediction_uncertainty is not None:
            confidence = 1.0 - min(1.0, prediction_uncertainty / predicted_base_fee)
        else:
            confidence = 0.85  # Default confidence
        
        # Determine status
        if confidence >= self.confidence_threshold:
            status = "Optimal"
            status_desc = "High confidence - Ready to broadcast"
        elif confidence >= 0.7:
            status = "Moderate"
            status_desc = "Moderate confidence - Consider waiting"
        else:
            status = "Low"
            status_desc = "Low confidence - Wait for better conditions"
        
        return {
            'predicted_base_fee_wei': int(predicted_base_fee),
            'predicted_base_fee_gwei': predicted_base_fee / 1e9,
            'buffered_base_fee_wei': int(buffered_base_fee),
            'buffered_base_fee_gwei': buffered_base_fee / 1e9,
            'priority_fee_wei': priority_fee,
            'priority_fee_gwei': priority_fee / 1e9,
            'max_fee_per_gas_wei': max_fee_per_gas,
            'max_fee_per_gas_gwei': max_fee_per_gas / 1e9,
            'buffer_multiplier': buffered_base_fee / predicted_base_fee,
            'confidence': confidence,
            'status': status,
            'status_description': status_desc,
        }
    
    def format_recommendation(self, recommendation: Dict[str, Any]) -> str:
        """
        Format recommendation untuk display.
        
        Args:
            recommendation: Recommendation dictionary
            
        Returns:
            Formatted string
        """
        output = []
        output.append("\n" + "="*60)
        output.append("⛽ GAS FEE RECOMMENDATION")
        output.append("="*60)
        
        output.append("\n📊 Prediction:")
        output.append(f"  Base Fee (predicted): {recommendation['predicted_base_fee_gwei']:.4f} Gwei")
        output.append(f"  Base Fee (buffered):  {recommendation['buffered_base_fee_gwei']:.4f} Gwei")
        output.append(f"  Buffer: {(recommendation['buffer_multiplier'] - 1) * 100:.1f}%")
        
        output.append("\n💡 Recommendation:")
        output.append(f"  Priority Fee: {recommendation['priority_fee_gwei']:.4f} Gwei")
        output.append(f"  Max Fee Per Gas: {recommendation['max_fee_per_gas_gwei']:.4f} Gwei")
        
        output.append(f"\n✓ Status: {recommendation['status']}")
        output.append(f"  {recommendation['status_description']}")
        output.append(f"  Confidence: {recommendation['confidence']*100:.1f}%")
        
        output.append("="*60 + "\n")
        
        return "\n".join(output)


def create_default_policy() -> GasFeePolicy:
    """
    Create default policy dengan recommended settings.
    
    Returns:
        GasFeePolicy instance
    """
    return GasFeePolicy(
        priority_fee_percentile=0.5,  # Median priority fee
        buffer_multiplier=1.1,  # 10% buffer
        max_fee_multiplier=1.5,  # Max 50% over prediction
        confidence_threshold=0.85  # 85% confidence threshold
    )
