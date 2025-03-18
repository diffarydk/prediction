"""
Enhanced Monte Carlo simulation for Baccarat prediction with adaptive models.
"""

import numpy as np
from colorama import Fore

from config import MONTE_CARLO_SAMPLES
from .monte_carlo import get_pattern_insight

# In prediction/monte_carlo_adaptive.py
# Update the adaptive_monte_carlo_prediction function to leverage stacking

# Replace the existing adaptive_monte_carlo_prediction with a cleaner version
def adaptive_monte_carlo_prediction(model_registry, prev_rounds, samples=MONTE_CARLO_SAMPLES):
    """
    Make predictions using stacking as the primary method.
    
    This is now the single, consistent prediction method using only the stacking approach.
    """
    try:
        # Get prediction directly from model registry (which now uses only stacking)
        prediction_result = model_registry.get_prediction(prev_rounds)
        
        # Add pattern insight if not already present
        if 'pattern_insight' not in prediction_result:
            pattern_insight = get_pattern_insight(prev_rounds)
            prediction_result['pattern_insight'] = pattern_insight
        
        return prediction_result
        
    except Exception as e:
        print(f"Error in adaptive_monte_carlo_prediction: {e}")
        # Final fallback
        return {
            'prediction': np.random.choice([0, 1, 2]),
            'confidence': 33.3,
            'distribution': {0: 33.3, 1: 33.3, 2: 33.4},
            'fallback': True,
            'error': str(e)
        }
def extract_pattern_type(pattern_insight):
    """Extract the pattern type from insight text for model weighting"""
    if not pattern_insight:
        return "no_pattern"
    
    if "streak" in pattern_insight.lower():
        return "streak"
    elif "alternating" in pattern_insight.lower():
        return "alternating"
    elif "tie" in pattern_insight.lower():
        return "tie"
    else:
        return "other_pattern"