"""
Learning package for Baccarat Prediction System.
Contains online learning functionality for model updates.
"""

from .online_learning import (
    OnlineBaccaratModel,
    load_or_create_online_model,
    save_online_model,
    update_after_prediction,
    analyze_learning_curve
)

__all__ = [
    'OnlineBaccaratModel',
    'load_or_create_online_model',
    'save_online_model',
    'update_after_prediction',
    'analyze_learning_curve'
]