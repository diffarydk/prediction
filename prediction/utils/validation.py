"""
prediction/utils/validation.py

Validation utilities for Baccarat Prediction System.
"""

import numpy as np
from typing import Tuple, List, Union, Any


def validate_input(prev_rounds: Union[List[int], np.ndarray]) -> Tuple[bool, Union[List[int], np.ndarray]]:
    """
    Validate prediction input data with comprehensive checks.
    
    Args:
        prev_rounds: Previous game outcomes (list or ndarray)
        
    Returns:
        tuple: (is_valid, validated_input)
    """
    # Check if input is None
    if prev_rounds is None:
        return False, []
    
    # Convert to numpy array if it's a list
    if isinstance(prev_rounds, list):
        # Handle nested lists
        if prev_rounds and isinstance(prev_rounds[0], list):
            prev_rounds = prev_rounds[0]
            
        # Check if all elements are valid
        if not all(isinstance(x, (int, np.integer)) and 0 <= x <= 2 for x in prev_rounds):
            # Try to fix invalid elements
            fixed_rounds = []
            for x in prev_rounds:
                try:
                    val = int(x)
                    if 0 <= val <= 2:
                        fixed_rounds.append(val)
                    else:
                        fixed_rounds.append(0)  # Default to banker for invalid values
                except (ValueError, TypeError):
                    fixed_rounds.append(0)  # Default to banker for non-numeric values
            
            if len(fixed_rounds) >= 3:  # Need at least 3 values for meaningful prediction
                return True, fixed_rounds
            else:
                return False, fixed_rounds
        
        # Valid list input
        return True, prev_rounds
    
    # Handle numpy array
    elif isinstance(prev_rounds, np.ndarray):
        # Check shape
        if prev_rounds.ndim == 1:
            # Validate values
            if not np.all((prev_rounds >= 0) & (prev_rounds <= 2)):
                # Try to fix invalid values
                fixed_rounds = np.clip(prev_rounds, 0, 2)
                return True, fixed_rounds
            return True, prev_rounds
            
        elif prev_rounds.ndim == 2:
            # If 2D array, use first row
            if prev_rounds.shape[0] >= 1:
                row = prev_rounds[0]
                if not np.all((row >= 0) & (row <= 2)):
                    # Try to fix invalid values
                    fixed_rounds = np.clip(row, 0, 2)
                    return True, fixed_rounds
                return True, row
    
    # Unsupported input type
    return False, []


def format_input(validated_input: Union[List[int], np.ndarray]) -> np.ndarray:
    """
    Format validated input for consistent processing.
    
    Args:
        validated_input: Validated input data
        
    Returns:
        numpy.ndarray: Consistently formatted input
    """
    # Convert list to numpy array if needed
    if isinstance(validated_input, list):
        input_array = np.array(validated_input)
    else:
        input_array = validated_input
    
    # Ensure 2D array with shape (1, n)
    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)
    
    return input_array


"""
prediction/utils/diagnostics.py

Diagnostics utilities for Baccarat Prediction System.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


def log_prediction_error(error_context: Dict[str, Any]) -> None:
    """
    Log detailed prediction error information.
    
    Args:
        error_context: Error context information
    """
    # Log to both console and file
    error_msg = f"Prediction error in stage '{error_context.get('stage', 'unknown')}': {error_context.get('error', 'Unknown error')}"
    
    # Add traceback if available
    if 'traceback' in error_context:
        error_msg += f"\nTraceback: {error_context['traceback']}"
    
    logger.error(error_msg)
    print(f"Error: {error_msg}")


def get_error_context(
    stage: str,
    input_data: Any,
    error: Exception,
    partial_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive error context for diagnostics.
    
    Args:
        stage: Pipeline stage where error occurred
        input_data: Input data that caused the error
        error: The exception that was raised
        partial_result: Partial result if available
        
    Returns:
        dict: Error context information
    """
    context = {
        'timestamp': time.time(),
        'stage': stage,
        'error': str(error),
        'error_type': type(error).__name__,
        'traceback': traceback.format_exc()
    }
    
    # Include input data shape/type information
    if input_data is not None:
        if hasattr(input_data, 'shape'):
            context['input_shape'] = str(input_data.shape)
        elif isinstance(input_data, list):
            context['input_length'] = len(input_data)
        context['input_type'] = str(type(input_data))
    
    # Include partial result if available
    if partial_result:
        context['partial_result'] = {
            k: v for k, v in partial_result.items()
            if k not in ['distribution']  # Skip large fields
        }
    
    return context


"""
prediction/utils/calibration.py

Calibration utilities for Baccarat Prediction System.
"""

import numpy as np
from typing import Dict, Any


def calibrate_confidence(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply simple confidence calibration based on empirical rules.
    
    This is a simplified version of the ConfidenceCalculator for contexts
    where the full component isn't available.
    
    Args:
        prediction_result: Prediction result to calibrate
        
    Returns:
            dict: Calibrated prediction result
    """
    # Store original confidence
    raw_confidence = prediction_result['confidence']
    
    # Extract key information
    predicted = prediction_result['prediction']
    distribution = prediction_result['distribution']
    
    # Convert percentage distribution to probabilities
    probabilities = {k: v/100 for k, v in distribution.items()}
    
    # Calculate entropy
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
    
    # Add entropy information if not already present
    if 'entropy' not in prediction_result:
        prediction_result['entropy'] = entropy
        prediction_result['entropy_ratio'] = entropy_ratio
    
    return prediction_result