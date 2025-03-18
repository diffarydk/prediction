"""
Calibration Bridge Module for Baccarat Prediction System.

This module provides a unified interface for confidence calibration functions
used across different components of the system, ensuring consistent behavior
between the model registry and stacking ensemble.

It implements a bridge pattern that forwards calls to appropriate implementations
based on availability and context.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Try importing main calibration function from proper location
try:
    from prediction.components.confidence_calculator import calibrate_confidence as component_calibrate
    _has_component_calibration = True
except ImportError:
    _has_component_calibration = False

try:
    from prediction.utils.calibration import calibrate_confidence as utils_calibrate
    _has_utils_calibration = True
except ImportError:
    _has_utils_calibration = False


def calibrate_confidence(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bridge function for confidence calibration that ensures consistent behavior
    across system components.
    
    This function determines the best available calibration implementation and
    forwards the call appropriately, with fallback mechanisms for robustness.
    
    Args:
        prediction_result: Prediction result to calibrate
        
    Returns:
        dict: Calibrated prediction result
    """
    # Track original confidence for diagnostics
    original_confidence = prediction_result.get('confidence', 50.0)
    
    # Attempt primary calibration paths in order of preference
    if _has_component_calibration:
        try:
            result = component_calibrate(prediction_result)
            result['calibration_source'] = 'component'
            return result
        except Exception as e:
            logger.warning(f"Component calibration failed: {e}, falling back")
    
    if _has_utils_calibration:
        try:
            result = utils_calibrate(prediction_result)
            result['calibration_source'] = 'utils'
            return result
        except Exception as e:
            logger.warning(f"Utils calibration failed: {e}, using fallback")
    
    # Fallback implementation if no other methods are available
    return _fallback_calibrate(prediction_result)


def _fallback_calibrate(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal fallback calibration implementation when primary methods are unavailable.
    
    Args:
        prediction_result: Prediction result to calibrate
        
    Returns:
        dict: Calibrated prediction result
    """
    # Store original confidence
    raw_confidence = prediction_result.get('confidence', 50.0)
    
    # Extract key information
    predicted = prediction_result.get('prediction', 0)
    distribution = prediction_result.get('distribution', {0: 45.0, 1: 45.0, 2: 10.0})
    
    # Convert percentage distribution to probabilities if needed
    if any(v > 1 for v in distribution.values()):
        probabilities = {k: v/100 for k, v in distribution.items()}
    else:
        probabilities = distribution
    
    # Calculate entropy if needed for calibration
    entropy = 0.0
    for outcome, prob in probabilities.items():
        if prob > 0:
            entropy -= prob * np.log2(prob)
    
    # Max entropy for 3 outcomes is log2(3) ≈ 1.58
    entropy_ratio = entropy / 1.58
    
    # Apply entropy penalty (higher entropy = lower confidence)
    entropy_penalty = 0.75 * entropy_ratio
    
    # Apply outcome-specific adjustment
    outcome_adjustments = {
        0: 1.0,    # Banker - neutral adjustment
        1: 1.0,    # Player - neutral adjustment
        2: 0.85    # Tie - reduce confidence (harder to predict)
    }
    
    outcome_factor = outcome_adjustments.get(predicted, 1.0)
    
    # Apply adjustments
    calibrated_confidence = raw_confidence * outcome_factor * (1 - entropy_penalty)
    
    # Apply confidence caps based on outcome
    confidence_caps = {
        0: 85.0,   # Banker cap
        1: 85.0,   # Player cap
        2: 70.0    # Tie cap (lower maximum confidence)
    }
    
    max_confidence = confidence_caps.get(predicted, 85.0)
    final_confidence = min(calibrated_confidence, max_confidence)
    
    # Update result
    prediction_result['raw_confidence'] = raw_confidence
    prediction_result['confidence'] = final_confidence
    prediction_result['calibrated'] = True
    prediction_result['calibration_source'] = 'fallback'
    
    # Add entropy information if not already present
    if 'entropy' not in prediction_result:
        prediction_result['entropy'] = entropy
        prediction_result['entropy_ratio'] = entropy_ratio
    
    return prediction_result


def calibrate_distribution(distribution: Dict[int, float]) -> Dict[int, float]:
    """
    Calibrate a probability distribution across outcomes.
    
    Args:
        distribution: Probability distribution to calibrate
        
    Returns:
        dict: Calibrated probability distribution
    """
    # Convert percentages to probabilities if needed
    is_percentage = any(v > 1 for v in distribution.values())
    if is_percentage:
        probabilities = {k: v/100 for k, v in distribution.items()}
    else:
        probabilities = distribution.copy()
    
    # Apply tie probability adjustment (ties are typically overestimated)
    if 2 in probabilities:
        tie_prob = probabilities[2]
        # Apply dampening for tie predictions
        if tie_prob > 0.10:
            tie_adjustment = min(0.5, tie_prob * 0.3)
            remaining = tie_adjustment
            probabilities[2] -= tie_adjustment
            
            # Redistribute to banker/player proportionally
            non_tie_sum = probabilities.get(0, 0) + probabilities.get(1, 0)
            if non_tie_sum > 0:
                probabilities[0] = probabilities.get(0, 0) + (remaining * probabilities.get(0, 0) / non_tie_sum)
                probabilities[1] = probabilities.get(1, 0) + (remaining * probabilities.get(1, 0) / non_tie_sum)
            else:
                # Even split if no banker/player probability
                probabilities[0] = probabilities.get(0, 0) + remaining / 2
                probabilities[1] = probabilities.get(1, 0) + remaining / 2
    
    # Ensure probabilities sum to 1
    total = sum(probabilities.values())
    if abs(total - 1.0) > 0.001:
        probabilities = {k: v/total for k, v in probabilities.items()}
    
    # Convert back to percentage if input was in percentage
    if is_percentage:
        return {k: v*100 for k, v in probabilities.items()}
    
    return probabilities