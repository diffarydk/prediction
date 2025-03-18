# File: interface/__init__.py
# Add to the existing imports, around line 7

# Core prediction interface with enhanced functionality
from .io_predict import (
    main_prediction_loop,
    initialize_model_registry,
    display_prediction_result,
    get_five_results,
    get_actual_result,
    update_models_with_result,
    track_accuracy,
    make_optimized_prediction  # Add the new function
)

# Add to __all__ list, around line 29
__all__ = [
    # Core prediction workflow
    'main_prediction_loop',
    'initialize_model_registry',
    'display_prediction_result',
    'get_five_results',
    'get_actual_result',
    'update_models_with_result',
    'track_accuracy',
    'make_optimized_prediction',  # Add the new function
    
    # Rest of the exports remain the same
    # ...
]