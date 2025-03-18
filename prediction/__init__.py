# Update the prediction/__init__.py file

# Import core prediction pipeline
from .prediction_pipeline import PredictionPipeline

# Import components for direct access
from .components.pattern_analyzer import PatternAnalyzer, get_pattern_insight, extract_pattern_type
from .components.confidence_calculator import ConfidenceCalculator, calibrate_confidence
from .components.fallback_manager import FallbackManager

from .utils.calibration_initializer import initialize_calibration

# Legacy imports for backward compatibility
from .monte_carlo import monte_carlo_prediction, safe_monte_carlo_prediction, evaluate_monte_carlo_accuracy
from .monte_carlo_adaptive import adaptive_monte_carlo_prediction

# Enhanced unified prediction function
def predict(model_registry, prev_rounds, with_profiling=False):
    """
    Unified prediction function with optional performance profiling.
    
    Args:
        model_registry: The model registry providing model access
        prev_rounds: Previous game outcomes
        with_profiling: Whether to include performance metrics
        
    Returns:
        dict: Prediction results
    """
    pipeline = PredictionPipeline(model_registry)
    
    if with_profiling:
        return pipeline.predict_with_profiling(prev_rounds)
    else:
        return pipeline.predict(prev_rounds)

# Export all public symbols
__all__ = [
    # Core components
    'PredictionPipeline',
    'PatternAnalyzer',
    'ConfidenceCalculator',
    'FallbackManager',
    
    # Unified API
    'predict',
    
    # Legacy functions for backward compatibility
    'monte_carlo_prediction',
    'safe_monte_carlo_prediction',
    'evaluate_monte_carlo_accuracy',
    'adaptive_monte_carlo_prediction',
    'get_pattern_insight',
    'extract_pattern_type',
    'initialize_calibration',
    'CALIBRATION_INITIALIZED',
]