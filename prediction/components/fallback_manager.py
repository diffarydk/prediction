"""
Fallback Management Component for Baccarat Prediction System.

This module implements a structured approach to handling prediction failures
with progressive fallback mechanisms, ensuring the system always provides
reasonable predictions even when primary methods fail.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)


class FallbackManager:
    """
    Manages prediction fallback strategies with progressive degradation.
    
    This component provides structured error recovery with multiple fallback
    levels, ensuring the system can continue to function even when primary
    prediction methods fail. It implements a chain of increasingly simple
    fallback approaches with appropriate confidence adjustment.
    """
    
    def __init__(self, model_registry):
        """
        Initialize fallback manager with reference to model registry.
        
        Args:
            model_registry: The model registry providing model access
        """
        self.model_registry = model_registry
        
        # Fallback history tracking
        self.fallback_history = []
        self.max_history = 100
        
        # Fallback strategy parameters
        self.default_distribution = {
            0: 45.0,  # Banker (slight house edge)
            1: 45.0,  # Player
            2: 10.0   # Tie (rare)
        }
        
        # Historical class distribution (updated over time)
        self.class_distribution = {
            0: 0.459,  # Standard baccarat odds
            1: 0.446,
            2: 0.095
        }
        
        # Fallback confidence levels by strategy
        self.confidence_levels = {
            "historical_voting": 60.0,
            "model_voting": 55.0,
            "weighted_random": 45.0,
            "pattern_based": 50.0,
            "historical_distribution": 40.0,
            "default_distribution": 33.3
        }
    
    def generate_fallback(
        self,
        prev_rounds: Optional[Union[List[int], np.ndarray]],
        reason: str,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate fallback prediction with progressive strategy selection.
        
        Args:
            prev_rounds: Previous game outcomes (if available)
            reason: Reason for fallback
            error: Optional error message
            
        Returns:
            dict: Fallback prediction result
        """
        # Track fallback occurrence
        self._track_fallback(reason, error)
        
        # Determine the most appropriate fallback strategy based on context
        strategy = self._select_fallback_strategy(prev_rounds, reason)
        
        # Apply the selected fallback strategy
        prediction_result = self._apply_fallback_strategy(strategy, prev_rounds)
        
        # Add fallback metadata
        prediction_result['fallback'] = True
        prediction_result['fallback_reason'] = reason
        prediction_result['fallback_strategy'] = strategy
        
        if error:
            prediction_result['error'] = error
        
        # Add timestamp
        prediction_result['timestamp'] = time.time()
        
        return prediction_result
    
    def _select_fallback_strategy(
        self,
        prev_rounds: Optional[Union[List[int], np.ndarray]],
        reason: str
    ) -> str:
        """
        Select the most appropriate fallback strategy based on context.
        
        Args:
            prev_rounds: Previous game outcomes (if available)
            reason: Reason for fallback
            
        Returns:
            str: Selected fallback strategy
        """
        # Progressive fallback selection logic
        
        # Stage 1: If we have historical data and access to model registry
        if hasattr(self.model_registry, 'models') and hasattr(self.model_registry, 'model_active'):
            # Check if we have enough historical data for pattern-based prediction
            if prev_rounds is not None:
                # Try to use pattern-based prediction if input is valid
                if isinstance(prev_rounds, (list, np.ndarray)) and len(prev_rounds) >= 3:
                    return "pattern_based"
            
            # See if we have active base models for voting
            active_models = {
                model_id: model for model_id, model in self.model_registry.models.items()
                if model_id != "stacking_ensemble" and self.model_registry.model_active.get(model_id, False)
            }
            
            if active_models:
                return "model_voting"
        
        # Stage 2: If we have some history data
        if hasattr(self.model_registry, 'meta_y') and isinstance(self.model_registry.meta_y, list):
            meta_y = self.model_registry.meta_y
            if len(meta_y) >= 10:
                return "historical_distribution"
        
        # Stage 3: Historical fallback based on input
        if prev_rounds is not None:
            # Try weighted random based on recent history
            if isinstance(prev_rounds, (list, np.ndarray)) and len(prev_rounds) >= 3:
                return "weighted_random"
        
        # Stage 4: Complete fallback
        return "default_distribution"
    
    def _apply_fallback_strategy(
        self,
        strategy: str,
        prev_rounds: Optional[Union[List[int], np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Apply the selected fallback strategy to generate a prediction.
        
        Args:
            strategy: The selected fallback strategy
            prev_rounds: Previous game outcomes (if available)
            
        Returns:
            dict: Prediction result
        """
        if strategy == "pattern_based":
            return self._pattern_based_fallback(prev_rounds)
        elif strategy == "model_voting":
            return self._model_voting_fallback(prev_rounds)
        elif strategy == "historical_distribution":
            return self._historical_distribution_fallback()
        elif strategy == "weighted_random":
            return self._weighted_random_fallback(prev_rounds)
        else:
            return self._default_distribution_fallback()
    
    def _pattern_based_fallback(
        self,
        prev_rounds: Union[List[int], np.ndarray]
    ) -> Dict[str, Any]:
        """
        Generate fallback based on pattern analysis.
        
        Args:
            prev_rounds: Previous game outcomes
            
        Returns:
            dict: Prediction result
        """
        try:
            # Import pattern analyzer
            from .pattern_analyzer import PatternAnalyzer
            
            # Analyze the pattern
            analyzer = PatternAnalyzer()
            pattern_info = analyzer.analyze_pattern(prev_rounds)
            
            # Determine next outcome based on pattern type
            pattern_type = pattern_info['pattern_type']
            
            if pattern_type == 'streak':
                # For streaks, predict continuation
                # Find the current streak value (most recent non-tie value)
                for outcome in reversed(prev_rounds):
                    if outcome != 2:  # Skip ties
                        prediction = outcome
                        break
                else:
                    prediction = 0  # Default to banker if all ties (unlikely)
                    
                distribution = {0: 20.0, 1: 20.0, 2: 10.0}
                distribution[prediction] = 50.0
                
            elif pattern_type == 'alternating':
                # For alternating patterns, predict next in sequence
                non_tie_values = [x for x in prev_rounds if x != 2]
                if len(non_tie_values) >= 2:
                    # If last two are same, predict same (continuation)
                    if non_tie_values[-1] == non_tie_values[-2]:
                        prediction = non_tie_values[-1]
                    else:
                        # If last two alternate, continue alternation
                        prediction = 1 - non_tie_values[-1]  # Toggle between 0 and 1
                else:
                    prediction = 0  # Default to banker if insufficient data
                
                distribution = {0: 25.0, 1: 25.0, 2: 10.0}
                distribution[prediction] = 40.0
                
            elif pattern_type == 'tie_influenced':
                # After ties, slightly favor banker (historical edge)
                distribution = {0: 48.0, 1: 42.0, 2: 10.0}
                prediction = 0
                
            elif pattern_type == 'banker_dominated':
                # Banker dominance tends to continue
                distribution = {0: 55.0, 1: 35.0, 2: 10.0}
                prediction = 0
                
            elif pattern_type == 'player_dominated':
                # Player dominance tends to continue
                distribution = {0: 35.0, 1: 55.0, 2: 10.0}
                prediction = 1
                
            else:
                # Default to historical distribution with banker slight edge
                distribution = {0: 45.9, 1: 44.6, 2: 9.5}
                prediction = 0
            
            confidence = self.confidence_levels.get("pattern_based", 50.0)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'distribution': distribution,
                'pattern_type': pattern_type,
                'pattern_insight': pattern_info.get('pattern_insight', '')
            }
            
        except Exception as e:
            logger.warning(f"Pattern-based fallback failed: {e}")
            return self._default_distribution_fallback()
    
    def _model_voting_fallback(
        self,
        prev_rounds: Union[List[int], np.ndarray]
    ) -> Dict[str, Any]:
        """
        Generate fallback based on model voting.
        
        Args:
            prev_rounds: Previous game outcomes
            
        Returns:
            dict: Prediction result
        """
        try:
            # Ensure prev_rounds is properly formatted
            if isinstance(prev_rounds, list):
                prev_rounds_array = np.array(prev_rounds).reshape(1, -1)
            else:
                prev_rounds_array = prev_rounds.reshape(1, -1) if prev_rounds.ndim == 1 else prev_rounds
            
            # Get active base models
            active_models = {
                model_id: model for model_id, model in self.model_registry.models.items()
                if model_id != "stacking_ensemble" and self.model_registry.model_active.get(model_id, False)
            }
            
            # Count votes and aggregate probabilities
            votes = {0: 0, 1: 0, 2: 0}
            probability_sums = {0: 0.0, 1: 0.0, 2: 0.0}
            model_count = 0
            
            for model_id, model in active_models.items():
                try:
                    # Get prediction
                    prediction = model.predict(prev_rounds_array)[0]
                    votes[prediction] += 1
                    
                    # Get probabilities if available
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(prev_rounds_array)
                        
                        # Handle different return formats
                        if isinstance(probs, dict):
                            for outcome, prob in probs.items():
                                probability_sums[outcome] += prob
                        else:
                            # Array format (most common)
                            for outcome, prob in enumerate(probs[0]):
                                if outcome < 3:  # Ensure valid outcome
                                    probability_sums[outcome] += prob
                    else:
                        # If no probabilities, add weighted vote
                        probability_sums[prediction] += 0.8
                        for outcome in range(3):
                            if outcome != prediction:
                                probability_sums[outcome] += 0.1
                    
                    model_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error getting prediction from {model_id}: {e}")
            
            # If no models could predict, fall back to default
            if model_count == 0:
                return self._default_distribution_fallback()
            
            # Calculate average probabilities
            avg_probabilities = {
                outcome: prob_sum / model_count
                for outcome, prob_sum in probability_sums.items()
            }
            
            # Normalize probabilities
            total_prob = sum(avg_probabilities.values())
            if abs(total_prob - 1.0) > 0.01:
                avg_probabilities = {
                    outcome: prob / total_prob
                    for outcome, prob in avg_probabilities.items()
                }
            
            # Get the majority vote prediction
            if votes:
                prediction = max(votes, key=votes.get)
                vote_confidence = (votes[prediction] / model_count) * 100
            else:
                # If voting fails, use highest probability
                prediction = max(avg_probabilities, key=avg_probabilities.get)
                vote_confidence = avg_probabilities[prediction] * 100
            
            # Cap confidence for fallback
            confidence = min(vote_confidence, self.confidence_levels.get("model_voting", 55.0))
            
            # Convert probabilities to percentages for distribution
            distribution = {k: v * 100 for k, v in avg_probabilities.items()}
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'distribution': distribution,
                'model_count': model_count,
                'votes': votes
            }
            
        except Exception as e:
            logger.warning(f"Model voting fallback failed: {e}")
            return self._default_distribution_fallback()
    
    def _historical_distribution_fallback(self) -> Dict[str, Any]:
        """
        Generate fallback based on historical outcome distribution.
        
        Returns:
            dict: Prediction result
        """
        try:
            # Get historical outcomes from model registry
            if hasattr(self.model_registry, 'meta_y') and isinstance(self.model_registry.meta_y, list):
                meta_y = self.model_registry.meta_y
                
                if len(meta_y) >= 10:
                    # Calculate distribution
                    counter = Counter(meta_y)
                    total = len(meta_y)
                    
                    historical_probs = {
                        outcome: count / total
                        for outcome, count in counter.items()
                    }
                    
                    # Ensure all outcomes are represented
                    for outcome in range(3):
                        if outcome not in historical_probs:
                            historical_probs[outcome] = 0.05
                    
                    # Normalize
                    total_prob = sum(historical_probs.values())
                    historical_probs = {
                        outcome: prob / total_prob
                        for outcome, prob in historical_probs.items()
                    }
                    
                    # Predict most common outcome
                    prediction = max(historical_probs, key=historical_probs.get)
                    
                    # Set confidence based on strength of distribution
                    confidence = self.confidence_levels.get("historical_distribution", 40.0)
                    
                    # Convert to percentage for distribution
                    distribution = {k: v * 100 for k, v in historical_probs.items()}
                    
                    return {
                        'prediction': prediction,
                        'confidence': confidence,
                        'distribution': distribution
                    }
            
            # Fall back to default if historical data not available
            return self._default_distribution_fallback()
            
        except Exception as e:
            logger.warning(f"Historical distribution fallback failed: {e}")
            return self._default_distribution_fallback()
    
    def _weighted_random_fallback(
        self,
        prev_rounds: Union[List[int], np.ndarray]
    ) -> Dict[str, Any]:
        """
        Generate fallback based on weighted random selection from recent history.
        
        Args:
            prev_rounds: Previous game outcomes
            
        Returns:
            dict: Prediction result
        """
        try:
            # Convert to list if needed
            if isinstance(prev_rounds, np.ndarray):
                sequence = prev_rounds.flatten().tolist()
            else:
                sequence = prev_rounds
            
            # Count frequencies
            counter = Counter(sequence)
            total = len(sequence)
            
            # Calculate weights with recency bias (more recent outcomes weighted higher)
            weights = {0: 0.0, 1: 0.0, 2: 0.0}
            
            for i, outcome in enumerate(sequence):
                # Apply recency weighting
                recency_weight = 0.5 + (0.5 * (i / (total - 1))) if total > 1 else 1.0
                weights[outcome] += recency_weight
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {
                    outcome: weight / total_weight
                    for outcome, weight in weights.items()
                }
            else:
                # If no weights (shouldn't happen), use default
                weights = {0: 0.45, 1: 0.45, 2: 0.1}
            
            # Predict most likely outcome based on weights
            prediction = max(weights, key=weights.get)
            
            # Set confidence based on weight strength
            confidence = self.confidence_levels.get("weighted_random", 45.0)
            
            # Convert to percentage for distribution
            distribution = {k: v * 100 for k, v in weights.items()}
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'distribution': distribution
            }
            
        except Exception as e:
            logger.warning(f"Weighted random fallback failed: {e}")
            return self._default_distribution_fallback()
    
    def _default_distribution_fallback(self) -> Dict[str, Any]:
        """
        Generate fallback based on default baccarat outcome distribution.
        
        This is the ultimate fallback when all other methods fail.
        
        Returns:
            dict: Prediction result
        """
        # Use standard baccarat distribution with house edge
        distribution = dict(self.default_distribution)
        
        # Predict banker (slight edge in standard baccarat)
        prediction = 0
        confidence = self.confidence_levels.get("default_distribution", 33.3)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'distribution': distribution,
            'emergency': True  # Flag this as emergency fallback
        }
    
    def _track_fallback(self, reason: str, error: Optional[str] = None) -> None:
        """
        Track fallback occurrence for monitoring.
        
        Args:
            reason: Reason for fallback
            error: Optional error message
        """
        # Create fallback record
        fallback_record = {
            'timestamp': time.time(),
            'reason': reason
        }
        
        if error:
            fallback_record['error'] = error
        
        # Add to history
        self.fallback_history.append(fallback_record)
        
        # Limit history size
        if len(self.fallback_history) > self.max_history:
            self.fallback_history = self.fallback_history[-self.max_history:]
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """
        Generate fallback statistics for monitoring.
        
        Returns:
            dict: Fallback statistics
        """
        if not self.fallback_history:
            return {
                'count': 0,
                'reasons': {},
                'recent_count': 0
            }
        
        # Count by reason
        reason_counts = Counter(record['reason'] for record in self.fallback_history)
        
        # Count recent fallbacks (last hour)
        recent_cutoff = time.time() - 3600
        recent_count = sum(1 for record in self.fallback_history 
                          if record['timestamp'] >= recent_cutoff)
        
        return {
            'count': len(self.fallback_history),
            'reasons': dict(reason_counts),
            'recent_count': recent_count,
            'recent_rate': recent_count / len(self.fallback_history)
        }