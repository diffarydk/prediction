"""
Unified Prediction Pipeline for Baccarat Prediction System.

This module implements a structured, multi-stage prediction process with
comprehensive error handling, confidence calibration, and pattern analysis.
It orchestrates the prediction workflow while maintaining separation of concerns
between prediction logic and interface components.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Import components
from .components.confidence_calculator import ConfidenceCalculator
from .components.pattern_analyzer import PatternAnalyzer
from .components.fallback_manager import FallbackManager

# Import utilities
from .utils.validation import validate_input, format_input
from .utils.calibration import calibrate_confidence
from .utils.diagnostics import log_prediction_error, get_error_context

# Configure logging
logger = logging.getLogger(__name__)


class PredictionPipeline:
    """
    Orchestrates the multi-stage prediction process with comprehensive error
    handling and recovery mechanisms.
    
    This class implements a systematic prediction workflow:
    1. Input validation and normalization
    2. Base model prediction collection
    3. Pattern recognition and analysis
    4. Meta-model prediction generation
    5. Confidence calibration and adjustment
    6. Result enrichment and validation
    
    Each stage includes robust error handling with progressive fallback mechanisms
    to ensure prediction stability under various failure conditions.
    """
    
    def __init__(self, model_registry):
        """
        Initialize the prediction pipeline with required components and validation.
        
        Args:
            model_registry: The model registry providing model management
        """
        # Import adapter here to avoid circular imports
        from prediction.registry_adapter import ModelRegistryAdapter
        
        # Wrap with adapter to ensure interface compatibility
        self.model_registry = ModelRegistryAdapter(model_registry)
        
        # Initialize components
        self.confidence_calculator = ConfidenceCalculator()
        self.pattern_analyzer = PatternAnalyzer()
        self.fallback_manager = FallbackManager(self.model_registry)
        
        # Tracking metrics
        self.prediction_count = 0
        self.error_count = 0
        self.fallback_count = 0
        
        # Performance tracking
        self.execution_times = {
            'validation': [],
            'base_prediction': [],
            'pattern_analysis': [],
            'meta_prediction': [],
            'calibration': [],
            'enrichment': [],
            'total': []
        }
        
        # Initialize caching
        self._prediction_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_max_size = 500
    
    def predict_with_profiling(self, prev_rounds: Union[List[int], np.ndarray]) -> Dict[str, Any]:
        """
        Enhanced prediction method with comprehensive performance instrumentation.
        
        This method tracks execution time of each stage in the prediction pipeline
        to identify bottlenecks and optimization opportunities.
        
        Args:
            prev_rounds: Previous game outcomes (list or ndarray)
            
        Returns:
            dict: Prediction results with performance metrics
        """
        start_time = time.time()
        self.prediction_count += 1
        
        result = {}
        current_stage = "initialization"
        performance_metrics = {}
        
        try:
            # Stage 1: Input validation and normalization
            current_stage = "validation"
            stage_start = time.time()
            
            # Check cache first for repeated patterns
            cache_key = self._get_cache_key(prev_rounds)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                cached_result['cached'] = True
                return cached_result
                
            valid, validated_input = validate_input(prev_rounds)
            if not valid:
                logger.warning(f"Input validation failed for {prev_rounds}")
                return self.fallback_manager.generate_fallback(
                    prev_rounds, "input_validation_failed"
                )
            
            normalized_input = format_input(validated_input)
            performance_metrics['validation'] = time.time() - stage_start
            
            # Stage 2: Base model predictions
            current_stage = "base_prediction"
            stage_start = time.time()
            
            base_predictions = self._collect_base_predictions(normalized_input)
            performance_metrics['base_prediction'] = time.time() - stage_start
            
            # Stage 3: Pattern analysis
            current_stage = "pattern_analysis"
            stage_start = time.time()
            
            pattern_info = self.pattern_analyzer.analyze_pattern(normalized_input)
            performance_metrics['pattern_analysis'] = time.time() - stage_start
            
            # Stage 4: Meta-model prediction
            current_stage = "meta_prediction"
            stage_start = time.time()
            
            prediction_result = self._generate_meta_prediction(
                normalized_input, base_predictions, pattern_info
            )
            
            if prediction_result.get('fallback', False):
                # Meta-prediction failed, result is already a fallback
                self.fallback_count += 1
                performance_metrics['meta_prediction'] = time.time() - stage_start
                prediction_result['performance_metrics'] = performance_metrics
                return prediction_result
            
            performance_metrics['meta_prediction'] = time.time() - stage_start
            
            # Stage 5: Confidence calibration
            current_stage = "calibration"
            stage_start = time.time()
            
            calibrated_result = self._calibrate_confidence(prediction_result)
            performance_metrics['calibration'] = time.time() - stage_start
            
            # Stage 6: Result enrichment
            current_stage = "enrichment"
            stage_start = time.time()
            
            enriched_result = self._enrich_result(
                calibrated_result, pattern_info, base_predictions
            )
            performance_metrics['enrichment'] = time.time() - stage_start
            
            # Track total execution time
            performance_metrics['total'] = time.time() - start_time
            
            # Add performance metrics to result
            enriched_result['performance_metrics'] = performance_metrics
            
            # Cache the result
            self._add_to_cache(cache_key, enriched_result)
            
            return enriched_result
            
        except Exception as e:
            self.error_count += 1
            
            # Record performance up to error
            performance_metrics['error_stage'] = current_stage
            performance_metrics['total'] = time.time() - start_time
            
            # Log detailed error information
            error_context = get_error_context(
                stage=current_stage,
                input_data=prev_rounds,
                error=e,
                partial_result=result,
                performance_metrics=performance_metrics
            )
            
            log_prediction_error(error_context)
            
            # Generate appropriate fallback based on the failed stage
            self.fallback_count += 1
            fallback_result = self.fallback_manager.generate_fallback(
                prev_rounds, f"{current_stage}_error", error=str(e)
            )
            
            # Add performance metrics to fallback result
            fallback_result['performance_metrics'] = performance_metrics
            return fallback_result
        
    def predict(self, prev_rounds: Union[List[int], np.ndarray]) -> Dict[str, Any]:
        """
        Generate prediction through multi-stage pipeline with comprehensive error handling.
        
        Args:
            prev_rounds: Previous game outcomes (list or ndarray)
            
        Returns:
            dict: Prediction results with comprehensive metadata
        """
        start_time = time.time()
        self.prediction_count += 1
        
        result = {}
        current_stage = "initialization"
        
        try:
            # Stage 1: Input validation and normalization
            current_stage = "validation"
            stage_start = time.time()
            
            valid, validated_input = validate_input(prev_rounds)
            if not valid:
                logger.warning(f"Input validation failed for {prev_rounds}")
                return self.fallback_manager.generate_fallback(
                    prev_rounds, "input_validation_failed"
                )
            
            normalized_input = format_input(validated_input)
            self.execution_times['validation'].append(time.time() - stage_start)
            
            # Stage 2: Base model predictions
            current_stage = "base_prediction"
            stage_start = time.time()
            
            base_predictions = self._collect_base_predictions(normalized_input)
            self.execution_times['base_prediction'].append(time.time() - stage_start)
            
            # Stage 3: Pattern analysis
            current_stage = "pattern_analysis"
            stage_start = time.time()
            
            pattern_info = self.pattern_analyzer.analyze_pattern(normalized_input)
            self.execution_times['pattern_analysis'].append(time.time() - stage_start)
            
            # Stage 4: Meta-model prediction
            current_stage = "meta_prediction"
            stage_start = time.time()
            
            prediction_result = self._generate_meta_prediction(
                normalized_input, base_predictions, pattern_info
            )
            
            if prediction_result.get('fallback', False):
                # Meta-prediction failed, result is already a fallback
                self.fallback_count += 1
                self.execution_times['meta_prediction'].append(time.time() - stage_start)
                return prediction_result
            
            self.execution_times['meta_prediction'].append(time.time() - stage_start)
            
            # Stage 5: Confidence calibration
            current_stage = "calibration"
            stage_start = time.time()
            
            calibrated_result = self._calibrate_confidence(prediction_result)
            self.execution_times['calibration'].append(time.time() - stage_start)
            
            # Stage 6: Result enrichment
            current_stage = "enrichment"
            stage_start = time.time()
            
            enriched_result = self._enrich_result(
                calibrated_result, pattern_info, base_predictions
            )
            self.execution_times['enrichment'].append(time.time() - stage_start)
            
            # Track total execution time
            self.execution_times['total'].append(time.time() - start_time)
            
            return enriched_result
            
        except Exception as e:
            self.error_count += 1
            
            # Log detailed error information
            error_context = get_error_context(
                stage=current_stage,
                input_data=prev_rounds,
                error=e,
                partial_result=result
            )
            
            log_prediction_error(error_context)
            
            # Generate appropriate fallback based on the failed stage
            self.fallback_count += 1
            return self.fallback_manager.generate_fallback(
                prev_rounds, f"{current_stage}_error", error=str(e)
            )
    
    def _collect_base_predictions(self, normalized_input: np.ndarray) -> Dict[str, Dict]:
        """
        Collect predictions from all active base models.
        
        Args:
            normalized_input: Validated and normalized input data
            
        Returns:
            dict: Mapping from model_id to prediction details
        """
        base_predictions = {}
        
        # Get active base models from registry
        active_models = self.model_registry.get_active_base_models()
        
        # Collect predictions from each model with individual error handling
        for model_id, model in active_models.items():
            try:
                # Get model prediction
                prediction = model.predict(normalized_input)[0]
                
                # Get probability distribution if available
                if hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(normalized_input)
                        
                        # Handle different return formats
                        if isinstance(probabilities, dict):
                            # Dictionary format (typically from Markov models)
                            probs = {k: float(v) for k, v in probabilities.items()}
                        else:
                            # Array format (typically from ML models)
                            probs = {i: float(p) for i, p in enumerate(probabilities[0])}
                    except Exception as prob_error:
                        # Fallback for probability prediction failures
                        probs = {i: 0.33 for i in range(3)}
                        probs[prediction] = 0.6  # Assign higher probability to prediction
                else:
                    # For models without probability methods
                    probs = {i: 0.1 for i in range(3)}
                    probs[prediction] = 0.8
                
                # Store prediction details
                base_predictions[model_id] = {
                    'prediction': int(prediction),
                    'probabilities': probs,
                    'confidence': float(probs[prediction]) * 100
                }
                
            except Exception as model_error:
                logger.warning(f"Error getting prediction from {model_id}: {model_error}")
                # Don't add failed models to the predictions
        
        return base_predictions
    
# Replace or modify the existing _generate_meta_prediction method in prediction_pipeline.py

    def _generate_meta_prediction(
        self, 
        normalized_input: np.ndarray,
        base_predictions: Dict[str, Dict],
        pattern_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimized meta-prediction generation with performance enhancements.
        
        Args:
            normalized_input: Validated and normalized input data
            base_predictions: Predictions from base models
            pattern_info: Pattern analysis results
            
        Returns:
            dict: Meta-model prediction results
        """
        try:
            # Apply pattern-specific optimization strategies
            pattern_type = pattern_info.get('pattern_type', 'unknown')
            
            # Check if we should use optimized path for specific pattern types
            if pattern_type in ['streak', 'alternating'] and len(base_predictions) >= 3:
                # For well-defined patterns, we can use optimized prediction paths
                return self._pattern_specific_prediction(pattern_type, base_predictions, pattern_info)
            
            # Standard path using model registry's prediction method
            result = self.model_registry.get_prediction(normalized_input)
            
            # Add base prediction agreement metrics
            result['base_model_agreement'] = self._calculate_agreement(base_predictions)
            
            return result
        except Exception as e:
            logger.error(f"Meta-prediction error: {e}")
            
            # Fall back to ensemble voting if meta-prediction fails
            return self._fallback_to_ensemble_voting(base_predictions, pattern_info)
            
    def _pattern_specific_prediction(
        self,
        pattern_type: str,
        base_predictions: Dict[str, Dict],
        pattern_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimized prediction generation for specific pattern types.
        
        This method implements pattern-specific optimization paths for common
        baccarat patterns, improving prediction accuracy and performance for
        recognized patterns while maintaining generality for unknown patterns.
        
        Args:
            pattern_type: Detected pattern type
            base_predictions: Predictions from base models
            pattern_info: Pattern analysis results
            
        Returns:
            dict: Optimized prediction results
        """
        # Define pattern-specific voting weights
        if pattern_type == 'streak':
            # For streaks, prioritize Markov models
            model_weights = {
                model_id: 2.0 if model_id.startswith('markov') else 1.0
                for model_id in base_predictions
            }
            
            # Further adjust weights based on streak length
            streak_length = pattern_info.get('streak_length', 0)
            if streak_length >= 3:
                # For longer streaks, further increase Markov model weights
                for model_id in model_weights:
                    if model_id.startswith('markov'):
                        model_weights[model_id] *= (1.0 + min(streak_length / 10, 0.5))
                        
        elif pattern_type == 'alternating':
            # For alternating patterns, prioritize XGBoost models
            model_weights = {
                model_id: 2.0 if ('xgboost' in model_id or 'xgb' in model_id) else 1.0
                for model_id in base_predictions
            }
            
            # Further adjust weights based on alternation length
            alternation_length = pattern_info.get('alternation_length', 0)
            if alternation_length >= 3:
                # For longer alternations, further increase XGBoost model weights
                for model_id in model_weights:
                    if 'xgboost' in model_id or 'xgb' in model_id:
                        model_weights[model_id] *= (1.0 + min(alternation_length / 10, 0.5))
        
        elif pattern_type == 'tie':
            # For tie patterns, prioritize specific tie-friendly models
            model_weights = {
                model_id: 2.0 if model_id.startswith('markov_3') else 1.0
                for model_id in base_predictions
            }
            
            # Reduce weight for models that typically underestimate ties
            for model_id in model_weights:
                if 'baccarat' in model_id:
                    model_weights[model_id] *= 0.8
        else:
            # Default equal weights for unknown patterns
            model_weights = {model_id: 1.0 for model_id in base_predictions}
        
        # Perform weighted voting
        weighted_votes = {0: 0.0, 1: 0.0, 2: 0.0}
        weighted_probabilities = {0: 0.0, 1: 0.0, 2: 0.0}
        total_weight = sum(model_weights.values())
        
        # Track models that contribute to predictions
        model_contributions = {}
        
        for model_id, prediction_data in base_predictions.items():
            weight = model_weights.get(model_id, 1.0)
            prediction = prediction_data['prediction']
            
            # Record contribution
            weighted_votes[prediction] += weight
            model_contributions[model_id] = {
                'prediction': prediction,
                'weight': weight,
                'confidence': prediction_data.get('confidence', 50.0)
            }
            
            # Add weighted probabilities
            for outcome, prob in prediction_data['probabilities'].items():
                weighted_probabilities[outcome] += prob * weight
        
        # Normalize probabilities
        for outcome in weighted_probabilities:
            weighted_probabilities[outcome] /= total_weight
        
        # Select winner
        predicted = max(weighted_votes, key=weighted_votes.get)
        confidence = weighted_probabilities[predicted] * 100
        
        # Apply pattern-specific confidence adjustment
        if pattern_type == 'streak' and pattern_info.get('streak_length', 0) > 3:
            # Increase confidence for long streaks
            confidence = min(confidence * 1.1, 95.0)
        elif pattern_type == 'alternating' and pattern_info.get('alternation_length', 0) > 3:
            # Increase confidence for consistent alternation
            confidence = min(confidence * 1.1, 95.0)
        elif pattern_type == 'tie':
            # Cap tie predictions at a more reasonable confidence
            if predicted == 2:  # Tie prediction
                confidence = min(confidence, 65.0)
        
        # Create result with pattern-specific optimization indicator
        return {
            'prediction': int(predicted),
            'confidence': float(confidence),
            'distribution': {k: float(v * 100) for k, v in weighted_probabilities.items()},
            'pattern_type': pattern_type,
            'pattern_insight': pattern_info.get('pattern_insight', ''),
            'optimized_path': f"pattern_specific_{pattern_type}",
            'model_contributions': model_contributions
        }
    
    def _fallback_to_ensemble_voting(
        self, 
        base_predictions: Dict[str, Dict],
        pattern_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fall back to weighted ensemble voting when meta-prediction fails.
        
        Args:
            base_predictions: Predictions from base models
            pattern_info: Pattern analysis results
            
        Returns:
            dict: Ensemble voting prediction results
        """
        if not base_predictions:
            return self.fallback_manager.generate_fallback(
                None, "no_base_predictions"
            )
        
        # Count votes for each outcome
        votes = {0: 0, 1: 0, 2: 0}
        probability_sums = {0: 0.0, 1: 0.0, 2: 0.0}
        
        # Process each base model prediction
        for model_id, prediction_data in base_predictions.items():
            prediction = prediction_data['prediction']
            votes[prediction] += 1
            
            # Add probabilities
            for outcome, prob in prediction_data['probabilities'].items():
                probability_sums[outcome] += prob
        
        # Calculate aggregated probabilities
        model_count = len(base_predictions)
        if model_count > 0:
            avg_probabilities = {
                outcome: prob_sum / model_count
                for outcome, prob_sum in probability_sums.items()
            }
            
            # Normalize probabilities to ensure they sum to 1
            total_prob = sum(avg_probabilities.values())
            if abs(total_prob - 1.0) > 0.01:
                avg_probabilities = {
                    outcome: prob / total_prob
                    for outcome, prob in avg_probabilities.items()
                }
            
            # Get the outcome with highest average probability
            predicted = max(avg_probabilities, key=avg_probabilities.get)
            confidence = avg_probabilities[predicted] * 100
            
            # Apply pattern-based adjustments if available
            if pattern_info and 'pattern_type' in pattern_info:
                confidence = self.confidence_calculator.adjust_confidence_for_pattern(
                    confidence, pattern_info['pattern_type'], predicted
                )
        else:
            # Complete fallback if no valid predictions
            predicted = 0  # Default to banker
            confidence = 33.3
            avg_probabilities = {0: 0.45, 1: 0.45, 2: 0.1}
        
        # Create result with appropriate fallback indicators
        return {
            'prediction': int(predicted),
            'confidence': float(confidence),
            'distribution': {k: float(v * 100) for k, v in avg_probabilities.items()},
            'fallback': True,
            'fallback_method': 'ensemble_voting',
            'pattern_type': pattern_info.get('pattern_type', 'unknown'),
            'pattern_insight': pattern_info.get('pattern_insight', '')
        }
    
    # Replace existing _calibrate_confidence method in PredictionPipeline class
    def _calibrate_confidence(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calibrate confidence scores with enhanced error tolerance."""
        # Check if result is already a fallback
        if prediction_result.get('fallback', False):
            return prediction_result
                
        try:
            # Use registry's calibration if available
            if hasattr(self.model_registry, 'calibration_manager'):
                try:
                    return self.model_registry.calibration_manager.apply_calibration(
                        prediction_result
                    )
                except AttributeError as attr_error:
                    # Specific handling for missing attributes in calibrators
                    logger.warning(f"Calibrator missing attribute: {attr_error}")
                    # Apply a default calibration adjustment instead
                    prediction = prediction_result.get('prediction', 0)
                    confidence = prediction_result.get('confidence', 50.0)
                    
                    if prediction == 2:  # Tie prediction
                        # Conservative calibration for ties - reduce overconfidence
                        adjusted_confidence = min(confidence * 0.8, 65.0)
                    else:
                        # Standard adjustment for banker/player
                        if confidence > 80:
                            adjusted_confidence = 80 + (confidence - 80) * 0.5
                        else:
                            adjusted_confidence = confidence
                            
                    prediction_result['confidence'] = float(adjusted_confidence)
                    prediction_result['calibration_method'] = 'fallback_adjustment'
                    return prediction_result
            
            # Otherwise use local calibration
            return calibrate_confidence(prediction_result)
                
        except Exception as e:
            logger.warning(f"Confidence calibration failed: {e}")
            
            # Return original result if calibration fails
            prediction_result['calibration_error'] = str(e)
            return prediction_result
    
    def _enrich_result(
        self,
        result: Dict[str, Any],
        pattern_info: Dict[str, Any],
        base_predictions: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Enrich prediction result with additional metadata.
        
        Args:
            result: Prediction result to enrich
            pattern_info: Pattern analysis information
            base_predictions: Predictions from base models
            
        Returns:
            dict: Enriched prediction result
        """
        # Ensure pattern information is included
        if 'pattern_type' not in result and 'pattern_type' in pattern_info:
            result['pattern_type'] = pattern_info['pattern_type']
            
        if 'pattern_insight' not in result and 'pattern_insight' in pattern_info:
            result['pattern_insight'] = pattern_info['pattern_insight']
        
        # Add entropy calculation if not present
        if 'entropy' not in result and 'distribution' in result:
            entropy = self._calculate_entropy(result['distribution'])
            result['entropy'] = entropy
            result['entropy_ratio'] = entropy / 1.58  # Max entropy for 3 outcomes
        
        # Add base model agreement information
        if 'base_model_agreement' not in result:
            result['base_model_agreement'] = self._calculate_agreement(base_predictions)
        
        # Add timestamp
        result['timestamp'] = time.time()
        
        # Add prediction performance metrics
        result['prediction_metrics'] = {
            'total_predictions': self.prediction_count,
            'error_rate': self.error_count / max(1, self.prediction_count),
            'fallback_rate': self.fallback_count / max(1, self.prediction_count)
        }
        
        return result
    
    def _calculate_agreement(self, base_predictions: Dict[str, Dict]) -> float:
        """
        Calculate agreement level among base models.
        
        Args:
            base_predictions: Predictions from base models
            
        Returns:
            float: Agreement level (0-1)
        """
        if not base_predictions:
            return 0.0
            
        # Count predictions for each outcome
        prediction_counts = {}
        for model_id, pred_data in base_predictions.items():
            prediction = pred_data['prediction']
            prediction_counts[prediction] = prediction_counts.get(prediction, 0) + 1
        
        # Calculate agreement as ratio of most common prediction
        if prediction_counts:
            max_count = max(prediction_counts.values())
            return max_count / len(base_predictions)
        
        return 0.0
    
    def _calculate_entropy(self, distribution: Dict[int, float]) -> float:
        """
        Calculate entropy of a probability distribution.
        
        Args:
            distribution: Probability distribution (as percentages)
            
        Returns:
            float: Entropy value
        """
        # Convert percentage distribution to probabilities
        probabilities = {k: v/100 for k, v in distribution.items()}
        
        # Calculate entropy: -sum(p * log2(p))
        entropy = 0.0
        for outcome, prob in probabilities.items():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for the prediction pipeline.
        
        Returns:
            dict: Performance metrics
        """
        metrics = {
            'prediction_count': self.prediction_count,
            'error_count': self.error_count,
            'fallback_count': self.fallback_count,
            'error_rate': self.error_count / max(1, self.prediction_count),
            'fallback_rate': self.fallback_count / max(1, self.prediction_count),
        }
        
        # Add timing statistics
        for stage, times in self.execution_times.items():
            if times:
                metrics[f'{stage}_avg_ms'] = np.mean(times) * 1000
                metrics[f'{stage}_max_ms'] = np.max(times) * 1000
                metrics[f'{stage}_min_ms'] = np.min(times) * 1000
        
        return metrics
    
    

    def _get_cache_key(self, prev_rounds):
        """
        Generate a deterministic cache key from input data.
        
        Args:
            prev_rounds: Previous game outcomes
            
        Returns:
            str: Cache key
        """
        if isinstance(prev_rounds, np.ndarray):
            # Convert numpy array to tuple for hashing
            return str(tuple(prev_rounds.flatten()))
        elif isinstance(prev_rounds, list):
            # Convert list to tuple for hashing
            return str(tuple(prev_rounds))
        else:
            # Fallback for other types
            return str(prev_rounds)

    def _get_from_cache(self, cache_key):
        """
        Retrieve cached prediction result if available.
        
        Args:
            cache_key: Cache key for lookup
            
        Returns:
            dict: Cached prediction result or None if not found
        """
        # Initialize cache attribute if not exists
        if not hasattr(self, '_prediction_cache'):
            self._prediction_cache = {}
            self._cache_hits = 0
            self._cache_misses = 0
            self._cache_max_size = 1000  # Maximum cache size
        
        # Lookup in cache
        if cache_key in self._prediction_cache:
            self._cache_hits += 1
            return self._prediction_cache[cache_key]
        
        self._cache_misses += 1
        return None

    def _add_to_cache(self, cache_key, result):
        """
        Add prediction result to cache with LRU eviction.
        
        Args:
            cache_key: Cache key
            result: Prediction result to cache
        """
        # Initialize cache attribute if not exists
        if not hasattr(self, '_prediction_cache'):
            self._prediction_cache = {}
            self._cache_hits = 0
            self._cache_misses = 0
            self._cache_max_size = 1000  # Maximum cache size
        
        # Simple LRU cache implementation
        if len(self._prediction_cache) >= self._cache_max_size:
            # Remove oldest cache entry
            try:
                oldest_key = next(iter(self._prediction_cache))
                del self._prediction_cache[oldest_key]
            except (StopIteration, KeyError):
                pass
        
        # Make a deep copy to prevent mutation of cached results
        import copy
        cached_result = copy.deepcopy(result)
        self._prediction_cache[cache_key] = cached_result

    def get_cache_stats(self):
        """
        Get cache performance statistics.
        
        Returns:
            dict: Cache performance metrics
        """
        if not hasattr(self, '_prediction_cache'):
            return {"enabled": False}
        
        total_lookups = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_lookups) * 100 if total_lookups > 0 else 0
        
        return {
            "enabled": True,
            "size": len(self._prediction_cache),
            "max_size": self._cache_max_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "lookups": total_lookups
        }
    
    def optimize_cache_based_on_performance(self):
        """Optimize cache size based on performance metrics."""
        if not hasattr(self, '_prediction_cache'):
            return
        
        # If we have a high hit rate, consider increasing cache size
        cache_stats = self.get_cache_stats()
        if cache_stats.get('hit_rate', 0) > 80 and len(self._prediction_cache) > 0.9 * self._cache_max_size:
            # Increase by 20% but cap at reasonable maximum
            self._cache_max_size = min(2000, int(self._cache_max_size * 1.2))
            print(f"Increased cache size to {self._cache_max_size} due to high hit rate")