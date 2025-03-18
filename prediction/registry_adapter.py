"""
Registry Adapter for Baccarat Prediction System.

This module provides a robust adapter pattern implementation to ensure 
consistent interface compatibility between system components, with
comprehensive error handling and fallback mechanisms.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

class ModelRegistryAdapter:
    """
    Adapter that ensures a consistent model registry interface regardless
    of underlying implementation or initialization state.
    
    This implements the Adapter pattern to bridge between PredictionPipeline
    expectations and ModelRegistry capabilities, with fault tolerance.
    """
    
    def __init__(self, registry=None):
        """
        Initialize adapter with reference to actual registry.
        
        Args:
            registry: The actual model registry or None
        """
        self.registry = registry
        logger.info(f"Registry adapter initialized with registry: {registry is not None}")
        
    def get_active_base_models(self) -> Dict[str, Any]:
        """
        Get active base models with comprehensive fallback mechanisms.
        
        Returns:
            dict: Mapping from model_id to model instance
        """
        # Primary implementation: Use registry's method if available
        if self.registry is not None and hasattr(self.registry, 'get_active_base_models'):
            try:
                return self.registry.get_active_base_models()
            except Exception as e:
                logger.error(f"Primary method failed: {e}")
                
        # Secondary implementation: Direct attribute access
        if self.registry is not None and hasattr(self.registry, 'models') and hasattr(self.registry, 'model_active'):
            try:
                return {model_id: model for model_id, model in self.registry.models.items() 
                       if self.registry.model_active.get(model_id, False) and model_id != "stacking_ensemble"}
            except Exception as e:
                logger.error(f"Secondary method failed: {e}")
        
        # Ultimate fallback: Return empty dictionary
        logger.warning("All methods failed, returning empty model dictionary")
        return {}
        
    def get_prediction(self, normalized_input) -> Dict[str, Any]:
        """
        Generate prediction with multi-level fallback mechanisms.
        
        Args:
            normalized_input: Normalized input data
            
        Returns:
            dict: Prediction results with standardized structure
        """
        # Primary implementation: Use registry's method if available
        if self.registry is not None and hasattr(self.registry, 'get_prediction'):
            try:
                return self.registry.get_prediction(normalized_input)
            except Exception as e:
                logger.error(f"Primary prediction method failed: {e}")
        
        # Secondary implementation: Generate prediction from active models
        try:
            active_models = self.get_active_base_models()
            if active_models:
                return self._generate_ensemble_prediction(active_models, normalized_input)
        except Exception as e:
            logger.error(f"Secondary prediction method failed: {e}")
        
        # Ultimate fallback: Return basic prediction
        return self._generate_fallback_prediction()
    
    def _generate_ensemble_prediction(self, models, normalized_input) -> Dict[str, Any]:
        """
        Generate prediction by ensemble voting across available models.
        
        Args:
            models: Dictionary of active models
            normalized_input: Input data
            
        Returns:
            dict: Prediction results
        """
        # Collect predictions from all models
        predictions = []
        probabilities = {0: 0.0, 1: 0.0, 2: 0.0}
        model_count = 0
        
        for model_id, model in models.items():
            try:
                # Get prediction
                pred = int(model.predict(normalized_input)[0])
                predictions.append(pred)
                
                # Accumulate probabilities if available
                if hasattr(model, 'predict_proba'):
                    try:
                        probs = model.predict_proba(normalized_input)
                        if isinstance(probs, dict):
                            for k, v in probs.items():
                                probabilities[k] += v
                        else:
                            for i, p in enumerate(probs[0]):
                                if i < 3:  # Ensure index is valid
                                    probabilities[i] += p
                    except Exception:
                        # Add weight to predicted class on failure
                        probabilities[pred] += 0.8
                else:
                    # Add weight to predicted class if no proba method
                    probabilities[pred] += 0.8
                
                model_count += 1
            except Exception:
                continue
        
        # If no successful predictions, use fallback
        if not predictions:
            return self._generate_fallback_prediction()
        
        # Get majority vote prediction
        from collections import Counter
        vote_count = Counter(predictions)
        prediction = vote_count.most_common(1)[0][0]
        
        # Normalize probabilities
        if model_count > 0:
            probabilities = {k: v/model_count for k, v in probabilities.items()}
        else:
            probabilities = {0: 0.45, 1: 0.45, 2: 0.1}  # Default distribution
            
        # Calculate confidence
        confidence = probabilities[prediction] * 100
        
        # Create result dictionary with adapter metadata
        return {
            'prediction': prediction,
            'confidence': confidence,
            'distribution': {k: v*100 for k, v in probabilities.items()},
            'adapter_generated': True,
            'model_count': model_count
        }
    
    def _generate_fallback_prediction(self) -> Dict[str, Any]:
        """
        Generate basic fallback prediction when all else fails.
        
        Returns:
            dict: Basic prediction result with baccarat-appropriate probabilities
        """
        return {
            'prediction': 0,  # Default to banker (slight edge in baccarat)
            'confidence': 33.3,
            'distribution': {0: 45.0, 1: 45.0, 2: 10.0},  # Standard baccarat distribution
            'fallback': True,
            'fallback_reason': 'adapter_emergency_fallback'
        }