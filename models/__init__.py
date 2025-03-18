# Modify models/__init__.py to use conditional imports
"""
Models package for Baccarat Prediction System.
Contains model definitions for prediction.
"""

from .base_model import BaseModel
from .markov_model import MarkovModel
from .baccarat_model import BaccaratModel
from .xgboost_model import XGBoostModel

# Use conditional imports to avoid circular dependencies
try:
    from .stacking_ensemble import StackingEnsemble
except ImportError:
    StackingEnsemble = None

from .model_registry import ModelRegistry

__all__ = [
    'BaseModel',
    'MarkovModel',
    'BaccaratModel', 
    'XGBoostModel',
    'StackingEnsemble',
    'ModelRegistry'
]

