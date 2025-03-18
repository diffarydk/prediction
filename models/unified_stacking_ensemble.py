"""
Unified Stacking Ensemble for Baccarat Prediction System

This module implements a sophisticated meta-learning system that combines predictions 
from multiple base models using a stacking approach with comprehensive error handling,
automatic dimension management, and performance monitoring capabilities.

Key features:
1. Transaction-based state management
2. Dynamic feature dimension validation and correction
3. Multi-level fallback mechanism for prediction failures
4. Pattern-specific performance tracking
5. Comprehensive health monitoring and diagnostics
6. Training history management with automatic pruning
"""

import numpy as np
import time
import os
import pickle
import traceback
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from collections import Counter, defaultdict

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import base model class - use absolute import with fallback
try:
    from models.base_model import BaseModel
except ImportError:
    try:
        from base_model import BaseModel
    except ImportError:
        # Create minimal BaseModel if unavailable
        class BaseModel:
            def __init__(self):
                self.model_type = "base"
                self.is_trained = False
                
            def predict(self, X):
                return np.zeros(len(X))

try:
    from prediction.prediction_pipeline import PredictionPipeline
except ImportError:
    try:
        from prediction import PredictionPipeline
    except ImportError:
        print("Warning: PredictionPipeline not available")
        # Define minimal PredictionPipeline for fallback
        class PredictionPipeline:
            def __init__(self, model_registry):
                self.model_registry = model_registry
                
            def predict(self, prev_rounds):
                print("Warning: Using minimal PredictionPipeline implementation")
                return {"prediction": 0, "confidence": 50.0}
                
            def predict_with_profiling(self, prev_rounds):
                result = self.predict(prev_rounds)
                result['performance_metrics'] = {'total': 0.001}
                return result

class FeatureDimensionManager:
    """
    Manages feature dimensions with validation, correction, and tracking capabilities.
    
    This component handles all dimension-related operations, ensuring consistent
    feature structure between training and prediction phases.
    """
    
    def __init__(self, parent_model):
        """
        Initialize with reference to parent model.
        
        Args:
            parent_model: The UnifiedStackingEnsemble instance
        """
        self.parent = parent_model
        self.expected_count = None
        self.dimension_issues = []
        self.feature_structure = {
            'base_models': 0,
            'probs_per_model': 3,   # Banker, Player, Tie probabilities
            'pattern_features': 4    # no_pattern, streak, alternating, tie
        }
        
    def update_expected_count(self, base_models_count: int) -> int:
        """
        Update expected feature count based on active base models.
        
        Args:
            base_models_count: Number of active base models
            
        Returns:
            int: Expected feature dimension
        """
        self.feature_structure['base_models'] = base_models_count
        probs_per_model = self.feature_structure['probs_per_model']
        pattern_features = self.feature_structure['pattern_features']
        
        expected_count = (base_models_count * probs_per_model) + pattern_features
        self.expected_count = expected_count
        
        return expected_count
        
    def validate_dimensions(self, features: List[float]) -> Tuple[bool, List[float]]:
        """
        Validate and correct feature dimensions when possible.
        
        Args:
            features: Feature vector to validate
            
        Returns:
            Tuple[bool, List[float]]: (validation_success, corrected_features)
        """
        if features is None:
            self._log_dimension_issue("None features provided")
            return False, []
            
        if not isinstance(features, list):
            # Try to convert to list
            try:
                if hasattr(features, 'tolist'):
                    features = features.tolist()
                else:
                    features = list(features)
            except Exception as e:
                self._log_dimension_issue(f"Could not convert features to list: {e}")
                return False, []
        
        # If no expected dimensions yet, establish baseline
        if self.expected_count is None:
            self.expected_count = len(features)
            return True, features
        
        # Check dimensions
        if len(features) == self.expected_count:
            return True, features
            
        # Try to correct dimensions
        attempted_correction = True
        corrected_features = features.copy() if hasattr(features, 'copy') else list(features)
        
        if len(features) < self.expected_count:
            # Add padding if too short
            padding_needed = self.expected_count - len(features)
            corrected_features.extend([0.33] * padding_needed)
            self._log_dimension_issue(f"Feature vector too short: added {padding_needed} padding elements")
        else:
            # Truncate if too long
            corrected_features = corrected_features[:self.expected_count]
            self._log_dimension_issue(f"Feature vector too long: truncated from {len(features)} to {self.expected_count}")
            
        return attempted_correction, corrected_features
        
    def _log_dimension_issue(self, message: str, pattern: str = None):
        """
        Log a dimension validation issue for tracking.
        
        Args:
            message: Description of the issue
            pattern: Optional pattern type where issue occurred
        """
        self.dimension_issues.append({
            'timestamp': time.time(),
            'message': message,
            'expected': self.expected_count,
            'pattern': pattern
        })
        
        # Keep issue log manageable
        if len(self.dimension_issues) > 100:
            self.dimension_issues = self.dimension_issues[-100:]
    
    def get_recent_issues(self, hours: int = 24) -> List[Dict]:
        """
        Get recent dimension issues within specified timeframe.
        
        Args:
            hours: Timeframe in hours
            
        Returns:
            List[Dict]: Recent dimension issues
        """
        cutoff = time.time() - (hours * 3600)
        return [d for d in self.dimension_issues if d['timestamp'] >= cutoff]


class PatternAnalyzer:
    """
    Analyzes and tracks pattern-specific performance and characteristics.
    
    This component handles pattern identification, performance tracking, and
    pattern dissolution detection.
    """
    
    def __init__(self):
        """Initialize pattern analyzer with default statistics."""
        self.pattern_performance = defaultdict(self._default_pattern_stats)
        
    def _default_pattern_stats(self):
        """Return default pattern statistics dictionary."""
        return {"correct": 0, "total": 0, "last_correct": 0}
        
    def update_pattern_stats(self, pattern: str, prediction: int, actual: int):
        """
        Update pattern-specific performance statistics.
        
        Args:
            pattern: Pattern type identifier
            prediction: Predicted outcome
            actual: Actual outcome
        """
        if not pattern or pattern == 'no_pattern':
            return
            
        self.pattern_performance[pattern]['total'] += 1
        
        if prediction == actual:
            self.pattern_performance[pattern]['correct'] += 1
            self.pattern_performance[pattern]['last_correct'] = time.time()
            
    def get_pattern_effectiveness(self, pattern: str) -> float:
        """
        Calculate effectiveness score for a specific pattern.
        
        Args:
            pattern: Pattern type identifier
            
        Returns:
            float: Effectiveness score (0-1)
        """
        if pattern not in self.pattern_performance:
            return 0.5  # Default for unknown patterns
            
        stats = self.pattern_performance[pattern]
        if stats['total'] < 5:  # Too few examples
            return 0.5
            
        # Calculate accuracy
        accuracy = stats['correct'] / stats['total']
        
        # Check for pattern dissolution
        recency = 1.0
        if stats.get('last_correct', 0) > 0:
            hours_since_last = (time.time() - stats['last_correct']) / 3600
            if hours_since_last > 24:
                recency = max(0.5, 1.0 - (hours_since_last - 24) / 72)
                
        # Combine metrics
        return accuracy * recency
        
    def assess_pattern_health(self) -> float:
        """
        Calculate overall pattern health score.
        
        Returns:
            float: Pattern health score (0-1)
        """
        if not self.pattern_performance:
            return 0.5
            
        # Calculate pattern scores
        pattern_scores = []
        for pattern, stats in self.pattern_performance.items():
            if stats['total'] < 5:  # Skip patterns with too few examples
                continue
                
            pattern_scores.append(self.get_pattern_effectiveness(pattern))
        
        if not pattern_scores:
            return 0.5
            
        # Return average pattern score
        return sum(pattern_scores) / len(pattern_scores)


class ErrorTracker:
    """
    Tracks and analyzes prediction and processing errors.
    
    This component handles error logging, categorization, and trending
    for diagnostic and recovery purposes.
    """
    
    def __init__(self, max_entries: int = 100):
        """
        Initialize error tracker.
        
        Args:
            max_entries: Maximum number of errors to track
        """
        self.prediction_errors = []
        self.processing_errors = []
        self.max_entries = max_entries
        
    def log_prediction_error(self, error: Exception, context: Dict = None):
        """
        Log a prediction error with context.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
        """
        entry = {
            'timestamp': time.time(),
            'error': str(error),
            'error_type': type(error).__name__,
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        self.prediction_errors.append(entry)
        
        # Keep log manageable
        if len(self.prediction_errors) > self.max_entries:
            self.prediction_errors = self.prediction_errors[-self.max_entries:]
            
    def log_processing_error(self, error: Exception, operation: str, context: Dict = None):
        """
        Log a processing error with context.
        
        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            context: Additional context about the error
        """
        entry = {
            'timestamp': time.time(),
            'error': str(error),
            'error_type': type(error).__name__,
            'operation': operation,
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        self.processing_errors.append(entry)
        
        # Keep log manageable
        if len(self.processing_errors) > self.max_entries:
            self.processing_errors = self.processing_errors[-self.max_entries:]
            
    def get_recent_errors(self, error_type: str = 'prediction', hours: int = 24) -> List[Dict]:
        """
        Get recent errors within specified timeframe.
        
        Args:
            error_type: Type of errors to retrieve ('prediction' or 'processing')
            hours: Timeframe in hours
            
        Returns:
            List[Dict]: Recent errors
        """
        cutoff = time.time() - (hours * 3600)
        
        if error_type == 'prediction':
            return [e for e in self.prediction_errors if e['timestamp'] >= cutoff]
        elif error_type == 'processing':
            return [e for e in self.processing_errors if e['timestamp'] >= cutoff]
        else:
            # Return both types
            prediction = [e for e in self.prediction_errors if e['timestamp'] >= cutoff]
            processing = [e for e in self.processing_errors if e['timestamp'] >= cutoff]
            return prediction + processing
            
    def get_error_trend(self, days: int = 7) -> Dict[str, List[int]]:
        """
        Calculate error frequency trends over time.
        
        Args:
            days: Number of days for trend analysis
            
        Returns:
            Dict[str, List[int]]: Daily error counts by type
        """
        # Calculate day boundaries
        now = time.time()
        day_seconds = 24 * 3600
        day_timestamps = [(now - (i * day_seconds)) for i in range(days)]
        day_timestamps.reverse()  # Oldest first
        
        # Initialize counts
        prediction_counts = [0] * days
        processing_counts = [0] * days
        
        # Count errors by day
        for error in self.prediction_errors:
            for i, day_start in enumerate(day_timestamps):
                if i == days - 1 or (day_start <= error['timestamp'] < day_timestamps[i+1]):
                    prediction_counts[i] += 1
                    break
                    
        for error in self.processing_errors:
            for i, day_start in enumerate(day_timestamps):
                if i == days - 1 or (day_start <= error['timestamp'] < day_timestamps[i+1]):
                    processing_counts[i] += 1
                    break
        
        return {
            'days': [time.strftime('%Y-%m-%d', time.localtime(ts)) for ts in day_timestamps],
            'prediction': prediction_counts,
            'processing': processing_counts
        }

class EnsembleModelRegistry:
    """
    Adapter class that provides model_registry interface for UnifiedStackingEnsemble.
    
    This adapter transforms the UnifiedStackingEnsemble interface into the model_registry
    interface expected by PredictionPipeline, enabling seamless integration between
    the ensemble meta-model and the prediction orchestration pipeline.
    """
    
    def __init__(self, ensemble):
        """
        Initialize adapter with reference to UnifiedStackingEnsemble instance.
        
        Args:
            ensemble: UnifiedStackingEnsemble instance
        """
        self.ensemble = ensemble
        self.model_active = {model_id: True for model_id in ensemble.base_models}
        self.models = ensemble.base_models
        
    def get_active_base_models(self):
        """
        Get active base models for prediction, implementing the registry interface.
        
        Returns:
            dict: Mapping from model_id to model
        """
        return {model_id: model for model_id, model in self.ensemble.base_models.items() 
                if self.model_active.get(model_id, True)}
                
    def get_prediction(self, normalized_input):
        """
        Generate prediction using ensemble, implementing the registry interface.
        
        Args:
            normalized_input: Normalized input data
            
        Returns:
            dict: Prediction results
        """
        try:
            # Generate prediction with ensemble
            pred = int(self.ensemble.predict(normalized_input)[0])
            probs = self.ensemble.predict_proba(normalized_input)[0]
            
            # Convert to expected format
            return {
                'prediction': pred,
                'confidence': float(probs[pred] * 100),
                'distribution': {i: float(p * 100) for i, p in enumerate(probs)}
            }
        except Exception as e:
            # Log error if error tracker is available
            if hasattr(self.ensemble, 'error_tracker'):
                self.ensemble.error_tracker.log_prediction_error(
                    error=e,
                    context={'normalized_input': str(normalized_input)}
                )
            
            # Fallback prediction
            return {
                'prediction': 0,  # Default to banker
                'confidence': 33.3,
                'distribution': {0: 45.0, 1: 45.0, 2: 10.0},
                'fallback': True,
                'fallback_reason': f'ensemble_prediction_error: {str(e)}'
            }
        
class UnifiedStackingEnsemble(BaseModel):
    """
    Advanced stacking ensemble that combines predictions from multiple base models.
    
    Implements a robust meta-learning approach with comprehensive error handling,
    dimension management, and health monitoring capabilities.
    
    Key capabilities:
    1. Dynamic feature dimension validation and correction
    2. Multi-level fallback mechanism for prediction failures
    3. Pattern-specific performance tracking
    4. Comprehensive health monitoring and diagnostics
    5. Training history management with automatic pruning
    6. Confidence correlation analysis
    """
    
    def __init__(self, base_models=None, random_state=42):
        """
        Initialize the stacking ensemble with comprehensive configuration.
        
        Args:
            base_models: Dict of base models {model_id: model}
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.base_models = base_models or {}
        self.model_type = "unified_stacking_ensemble"
        
        # Initialize component managers
        self.dimension_manager = FeatureDimensionManager(self)
        self.pattern_analyzer = PatternAnalyzer()
        self.error_tracker = ErrorTracker()
        
        # Update expected dimensions
        self.dimension_manager.update_expected_count(len(self.base_models))
        
        # Meta-model with robust configuration
        self.meta_model = VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', 
                                          max_iter=500, random_state=random_state)),
                ('rf', RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_leaf=5, 
                                           class_weight='balanced', random_state=random_state)),
            ],
            voting='soft',  # Use probability-based voting
            weights=[1, 1]  # Equal weighting initially
        )
        
        # Training data management
        self.meta_X = []
        self.meta_y = []
        self.meta_patterns = []
        self.is_trained = False
        
        # Version tracking
        self.version = 1
        self.created_at = time.time()
        self.last_updated = None
        self.last_retrain_size = 0
        
        # Feature caching to improve performance
        self.feature_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def fit(self, X, y, patterns=None):
        """
        Train meta-model with comprehensive error handling and validation.
        
        Args:
            X: Meta-features (base model predictions)
            y: Target outcomes
            patterns: Optional pattern types for each example
            
        Returns:
            self: Trained model instance
        """
        if not X or len(X) == 0:
            print("Error: Empty feature set for meta-learner training")
            return self
            
        if len(X) < 5:
            print(f"Warning: Limited data for meta-learner training ({len(X)} examples)")
            
        try:
            # Update expected feature count
            self.dimension_manager.update_expected_count(len(self.base_models))
            
            # Store feature structure from first example
            self.dimension_manager.expected_count = len(X[0])
            
            # Convert to numpy arrays
            X_array = np.array(X) if not isinstance(X, np.ndarray) else X
            y_array = np.array(y) if not isinstance(y, np.ndarray) else y
            
            # Train meta-model with error handling
            try:
                self.meta_model.fit(X_array, y_array)
            except Exception as e:
                self.error_tracker.log_processing_error(
                    error=e, 
                    operation='meta_model_fit', 
                    context={'X_shape': X_array.shape, 'y_length': len(y_array)}
                )
                
                # Fallback to simpler model
                print("Falling back to LogisticRegression for meta-model")
                self.meta_model = LogisticRegression(
                    C=1.0, class_weight='balanced', 
                    solver='lbfgs', max_iter=500, 
                    multi_class='multinomial',
                    random_state=42
                )
                self.meta_model.fit(X_array, y_array)
            
            # Store training data
            self.meta_X = list(X)
            self.meta_y = list(y)
            if patterns:
                self.meta_patterns = list(patterns)
            else:
                self.meta_patterns = ['unknown'] * len(y)
            
            # Mark as trained and update metadata
            self.is_trained = True
            self.last_updated = time.time()
            self.last_retrain_size = len(self.meta_X)
            self.version += 1
            
            print(f"Stacking meta-model v{self.version} trained with {len(X)} examples, {len(X[0])} features")
            
            # Clear feature cache after retraining
            self.feature_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
            
            return self
            
        except Exception as e:
            self.error_tracker.log_processing_error(
                error=e, 
                operation='fit', 
                context={'X_length': len(X), 'y_length': len(y)}
            )
            print(f"Error in stacking fit: {e}")
            traceback.print_exc()
            return self
            
    def predict(self, X):
        """
        Generate predictions with multi-level fallback and error handling.
        
        Args:
            X: Meta-features (base model predictions)
            
        Returns:
            numpy.ndarray: Predicted class for each example
        """
        # Validate model state
        if not self.is_trained:
            print("Warning: Stacking model not trained. Using fallback prediction.")
            return self._fallback_predict(X, 'model_not_trained')
        
        # Ensure X is a list of feature vectors
        if not isinstance(X, list):
            X = [X]
            
        # Validate and correct dimensions
        validated_X = []
        for x in X:
            valid, corrected_x = self.dimension_manager.validate_dimensions(x)
            if valid:
                validated_X.append(corrected_x)
            else:
                print(f"Warning: Invalid feature dimensions. Using fallback prediction.")
                return self._fallback_predict(X, 'dimension_error')
                
        # Convert to numpy array
        X_array = np.array(validated_X)
        
        # Primary prediction with error handling
        try:
            return self.meta_model.predict(X_array)
        except Exception as e:
            self.error_tracker.log_prediction_error(
                error=e,
                context={'X_shape': X_array.shape if hasattr(X_array, 'shape') else None}
            )
            print(f"Error in meta-model prediction: {e}")
            return self._fallback_predict(X, 'predict_error')
            
    def predict_proba(self, X):
        """
        Generate class probabilities with multi-level fallback and error handling.
        
        Args:
            X: Meta-features (base model predictions)
            
        Returns:
            numpy.ndarray: Probability distribution for each example
        """
        # Validate model state
        if not self.is_trained:
            print("Warning: Stacking model not trained. Using fallback probabilities.")
            return self._fallback_predict_proba(X, 'model_not_trained')
        
        # Ensure X is a list of feature vectors
        if not isinstance(X, list):
            X = [X]
            
        # Validate and correct dimensions
        validated_X = []
        for x in X:
            valid, corrected_x = self.dimension_manager.validate_dimensions(x)
            if valid:
                validated_X.append(corrected_x)
            else:
                print(f"Warning: Invalid feature dimensions. Using fallback probabilities.")
                return self._fallback_predict_proba(X, 'dimension_error')
                
        # Convert to numpy array
        X_array = np.array(validated_X)
        
        # Primary prediction with error handling
        try:
            return self.meta_model.predict_proba(X_array)
        except Exception as e:
            self.error_tracker.log_prediction_error(
                error=e,
                context={'X_shape': X_array.shape if hasattr(X_array, 'shape') else None}
            )
            print(f"Error in meta-model probability prediction: {e}")
            return self._fallback_predict_proba(X, 'predict_proba_error')
            
    def update_meta_data(self, meta_features, outcome, pattern=None):
        """
        Update meta-learner with new training example and manage training schedule.
        
        Args:
            meta_features: Features for meta-model (base model predictions)
            outcome: Actual outcome that occurred
            pattern: Optional pattern type for performance tracking
        """
        # Ensure meta_features is a list
        if not isinstance(meta_features, list):
            if hasattr(meta_features, 'tolist'):  # numpy array
                meta_features = meta_features.tolist()
            elif hasattr(meta_features, 'to_list'):  # pandas Series
                meta_features = meta_features.to_list()
            else:
                try:
                    meta_features = list(meta_features)
                except Exception as e:
                    self.error_tracker.log_processing_error(
                        error=e,
                        operation='convert_meta_features', 
                        context={'meta_features_type': str(type(meta_features))}
                    )
                    print(f"Error converting meta_features to list: {e}")
                    return
        
        # Validate dimensions
        valid, corrected_features = self.dimension_manager.validate_dimensions(meta_features)
        if not valid:
            print("Warning: Meta-feature dimension validation failed. Skipping update.")
            return
            
        # Update pattern statistics if pattern provided
        if self.is_trained and pattern:
            try:
                prediction = self.predict([corrected_features])[0]
                self.pattern_analyzer.update_pattern_stats(pattern, prediction, outcome)
            except Exception as e:
                self.error_tracker.log_processing_error(
                    error=e,
                    operation='update_pattern_stats', 
                    context={'pattern': pattern}
                )
        
        # Add to training data
        self.meta_X.append(corrected_features)
        self.meta_y.append(outcome)
        self.meta_patterns.append(pattern or 'unknown')
        
        # Limit training history size
        max_history = 200
        if len(self.meta_X) > max_history:
            excess = len(self.meta_X) - max_history
            self.meta_X = self.meta_X[excess:]
            self.meta_y = self.meta_y[excess:]
            self.meta_patterns = self.meta_patterns[excess:]
        
        # Determine if retraining is needed
        should_retrain = False
        
        # Case 1: Not trained yet but have enough data
        if not self.is_trained and len(self.meta_X) >= 10:
            should_retrain = True
            
        # Case 2: Have significant new examples since last training
        elif self.is_trained:
            if not hasattr(self, 'last_retrain_size') or self.last_retrain_size is None:
                self.last_retrain_size = 0
                
            new_examples = len(self.meta_X) - self.last_retrain_size
            if new_examples >= 10:
                should_retrain = True
                
        # Perform retraining if needed
        if should_retrain:
            try:
                print(f"Retraining stacking model with {len(self.meta_X)} examples")
                self.fit(self.meta_X, self.meta_y, self.meta_patterns)
            except Exception as e:
                self.error_tracker.log_processing_error(
                    error=e,
                    operation='retrain', 
                    context={'meta_X_length': len(self.meta_X)}
                )
                print(f"Error retraining stacking model: {e}")

    """
    Add these methods to the UnifiedStackingEnsemble class
    """
    def predict_with_pipeline(self, prev_rounds, with_profiling=False):
        """
        Generate prediction using the advanced pipeline architecture.
        
        This method creates a bridge between the ensemble meta-model and the
        structured prediction pipeline, combining the strengths of both systems:
        the ensemble's meta-learning capabilities with the pipeline's comprehensive
        error handling, pattern analysis, and confidence calibration.
        
        Args:
            prev_rounds: Previous game outcomes
            with_profiling: Whether to include performance metrics
            
        Returns:
            dict: Comprehensive prediction results
        """
        try:
            # Create model registry adapter
            registry = EnsembleModelRegistry(self)
            
            # Create pipeline with registry adapter
            pipeline = PredictionPipeline(registry)
            
            # Generate prediction with pipeline
            if with_profiling:
                return pipeline.predict_with_profiling(prev_rounds)
            else:
                return pipeline.predict(prev_rounds)
        except Exception as e:
            # Log error
            self.error_tracker.log_prediction_error(
                error=e,
                context={'prev_rounds': str(prev_rounds)}
            )
            print(f"Error in pipeline prediction: {e}")
            
            # Fall back to direct ensemble prediction
            try:
                pred = int(self.predict([prev_rounds])[0])
                probs = self.predict_proba([prev_rounds])[0]
                
                # Convert to expected format
                return {
                    'prediction': pred,
                    'confidence': float(probs[pred] * 100),
                    'distribution': {i: float(p * 100) for i, p in enumerate(probs)},
                    'fallback': True,
                    'fallback_reason': f'pipeline_error: {str(e)}'
                }
            except Exception as fallback_error:
                # Complete fallback
                return {
                    'prediction': 0,  # Default to banker
                    'confidence': 33.3,
                    'distribution': {0: 45.0, 1: 45.0, 2: 10.0},
                    'fallback': True,
                    'fallback_reason': 'complete_fallback',
                    'errors': [str(e), str(fallback_error)]
                }
                
    def get_pipeline_health(self):
        """
        Evaluate the health of the prediction pipeline integration.
        
        This method assesses the compatibility and performance of the pipeline
        integration by performing a test prediction and analyzing the results.
        
        Returns:
            dict: Pipeline health assessment metrics
        """
        try:
            # Create test input
            test_input = [0, 1, 0, 1, 0]  # Simple alternating pattern
            
            # Attempt pipeline prediction
            start_time = time.time()
            result = self.predict_with_pipeline(test_input, with_profiling=True)
            execution_time = time.time() - start_time
            
            # Check result structure
            has_prediction = 'prediction' in result
            has_confidence = 'confidence' in result
            has_distribution = 'distribution' in result
            has_metrics = 'performance_metrics' in result
            
            # Calculate structure completeness
            structure_score = sum([has_prediction, has_confidence, has_distribution, has_metrics]) / 4
            
            # Check for error indicators
            is_fallback = result.get('fallback', False)
            has_error = 'error' in result or 'errors' in result
            
            # Calculate overall health score
            if has_error:
                health_score = 0.2  # Serious issues
            elif is_fallback:
                health_score = 0.5  # Working with fallbacks
            else:
                # Factor in performance
                speed_score = 1.0 if execution_time < 0.05 else (0.8 if execution_time < 0.2 else 0.6)
                health_score = 0.7 * structure_score + 0.3 * speed_score
            
            return {
                'score': health_score,
                'execution_time': execution_time,
                'structure_completeness': structure_score,
                'is_fallback': is_fallback,
                'has_error': has_error,
                'pipeline_available': True
            }
        except Exception as e:
            self.error_tracker.log_processing_error(
                error=e,
                operation='get_pipeline_health'
            )
            return {
                'score': 0.0,
                'error': str(e),
                'pipeline_available': False
            }


    def _fallback_predict(self, X, reason):
        """
        Generate fallback predictions when primary method fails.
        
        Args:
            X: Meta-features (possibly invalid)
            reason: Reason for using fallback
            
        Returns:
            numpy.ndarray: Fallback predictions
        """
        # Determine examples count
        count = len(X) if isinstance(X, list) else 1
        
        # Strategy 1: Use most common class from training history
        if self.meta_y and len(self.meta_y) >= 5:
            from collections import Counter
            most_common = Counter(self.meta_y).most_common(1)[0][0]
            return np.array([most_common] * count)
            
        # Strategy 2: Use banker-biased distribution
        return np.array([np.random.choice([0, 1, 2], p=[0.45, 0.45, 0.1]) for _ in range(count)])
        
    def _fallback_predict_proba(self, X, reason):
        """
        Generate fallback probability distributions when primary method fails.
        
        Args:
            X: Meta-features (possibly invalid)
            reason: Reason for using fallback
            
        Returns:
            numpy.ndarray: Fallback probability distributions
        """
        # Determine examples count
        count = len(X) if isinstance(X, list) else 1
        
        # Strategy 1: Use historical class distribution if available
        if self.meta_y and len(self.meta_y) >= 10:
            counts = Counter(self.meta_y)
            total = sum(counts.values())
            
            # Calculate historical distribution
            hist_probs = np.array([[counts.get(i, 0) / total for i in range(3)] for _ in range(count)])
            
            # Mix with uniform distribution for more balanced output
            uniform = np.array([[1/3, 1/3, 1/3] for _ in range(count)])
            return 0.7 * hist_probs + 0.3 * uniform
            
        # Strategy 2: Use banker-biased distribution
        banker_bias = np.array([[0.45, 0.45, 0.1] for _ in range(count)])
        return banker_bias
        
    def _make_hashable(self, meta_features):
        """
        Convert meta-features to a hashable type for caching and lookup.
        
        Args:
            meta_features: Feature vector to make hashable
            
        Returns:
            tuple: Hashable representation of features
        """
        try:
            if isinstance(meta_features, np.ndarray):
                # Ensure we handle numpy arrays properly by converting to tuples of floats
                return tuple(float(x) for x in meta_features.flatten())
            elif isinstance(meta_features, list):
                # Convert list to tuple, recursively handling nested arrays/lists
                return tuple(
                    self._make_hashable(item) if isinstance(item, (list, np.ndarray)) 
                    else float(item) if isinstance(item, (np.float32, np.float64, np.int32, np.int64)) 
                    else item 
                    for item in meta_features
                )
            elif isinstance(meta_features, (np.float32, np.float64, np.int32, np.int64)):
                # Convert numpy scalar types to Python float
                return float(meta_features)
            return meta_features
        except Exception as e:
            self.error_tracker.log_processing_error(
                error=e,
                operation='make_hashable',
                context={'meta_features_type': str(type(meta_features))}
            )
            
            # Return a default tuple with correct dimensions if possible
            if hasattr(self.dimension_manager, 'expected_count') and self.dimension_manager.expected_count:
                return tuple([0.33] * self.dimension_manager.expected_count)
            return (0.33, 0.33, 0.34)  # Minimal fallback
            
    def get_strategy_health(self):
        """
        Get comprehensive health assessment of this model's prediction strategy.
        
        Returns:
            dict: Health assessment metrics
        """
        # If not trained yet, report neutral health
        if not self.is_trained or len(self.meta_X) < 20:
            return {
                'score': 0.5, 
                'accuracy': 0.5, 
                'confidence_corr': 0.5, 
                'pattern_health': 0.5, 
                'trained': False, 
                'examples': len(self.meta_X)
            }
        
        # Calculate recent accuracy (using last 20 predictions if available)
        recent_count = min(20, len(self.meta_y))
        recent_X = self.meta_X[-recent_count:]
        recent_y = self.meta_y[-recent_count:]
        
        # Calculate metrics with error handling
        try:
            # Make predictions
            recent_preds = self.predict(recent_X)
            recent_acc = sum(recent_preds == recent_y) / len(recent_y)
            
            # Calculate confidence correlation
            conf_corr = self._calculate_confidence_correlation()
            
            # Pattern-specific performance
            pattern_health = self.pattern_analyzer.assess_pattern_health()
            
            # Count recent errors (last 24 hours)
            recent_predict_errors = self.error_tracker.get_recent_errors('prediction', 24)
            recent_processing_errors = self.error_tracker.get_recent_errors('processing', 24)
            recent_dimension_issues = self.dimension_manager.get_recent_issues(24)
            
            # Combined health score
            health_score = 0.5 * recent_acc + 0.3 * conf_corr + 0.2 * pattern_health
            
            # Reduce score if there are recent errors
            if recent_predict_errors:
                health_score *= max(0.5, 1 - (len(recent_predict_errors) / 20))
                
            if recent_dimension_issues:
                health_score *= max(0.5, 1 - (len(recent_dimension_issues) / 10))
            
            return {
                'score': health_score,
                'accuracy': recent_acc,
                'confidence_corr': conf_corr,
                'pattern_health': pattern_health,
                'meta_examples': len(self.meta_X),
                'stacking_version': self.version,
                'trained': True,
                'recent_predict_errors': len(recent_predict_errors),
                'recent_processing_errors': len(recent_processing_errors),
                'recent_dimension_issues': len(recent_dimension_issues),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'last_updated': self.last_updated
            }
        except Exception as e:
            self.error_tracker.log_processing_error(
                error=e,
                operation='get_strategy_health'
            )
            return {
                'score': 0.4,  # Reduced score due to calculation error
                'error': str(e),
                'meta_examples': len(self.meta_X),
                'stacking_version': self.version,
                'trained': self.is_trained,
                'calculation_failed': True
            }
    
    def _calculate_confidence_correlation(self):
        """
        Measure correlation between prediction confidence and accuracy.
        
        Returns:
            float: Correlation coefficient normalized to 0-1 range
        """
        if not self.is_trained or len(self.meta_X) < 20:
            return 0.5  # Default when insufficient data
        
        # Calculate with last 50 examples
        recent_count = min(50, len(self.meta_X))
        recent_X = self.meta_X[-recent_count:]
        recent_y = self.meta_y[-recent_count:]
        
        try:
            # Get probabilities and predictions
            probs = self.predict_proba(recent_X)
            preds = self.predict(recent_X)
            
            # Extract confidence for each prediction (probability of predicted class)
            confidences = [probs[i][preds[i]] for i in range(len(preds))]
            correctness = [1 if preds[i] == recent_y[i] else 0 for i in range(len(preds))]
            
            # Calculate correlation
            if len(set(confidences)) <= 1 or len(set(correctness)) <= 1:
                return 0.5  # No correlation possible with constant values
            
            try:
                from scipy.stats import pearsonr
                corr, _ = pearsonr(confidences, correctness)
                return (corr + 1) / 2  # Normalize to 0-1
            except ImportError:
                # Manual calculation if scipy is not available
                conf_mean = sum(confidences) / len(confidences)
                corr_mean = sum(correctness) / len(correctness)
                
                numerator = sum((c - conf_mean) * (corr - corr_mean) for c, corr in zip(confidences, correctness))
                conf_var = sum((c - conf_mean) ** 2 for c in confidences)
                corr_var = sum((corr - corr_mean) ** 2 for corr in correctness)
                
                if conf_var == 0 or corr_var == 0:
                    return 0.5
                    
                correlation = numerator / ((conf_var * corr_var) ** 0.5)
                
                # Normalize to 0-1 range
                return (correlation + 1) / 2
        except Exception as e:
            self.error_tracker.log_processing_error(
                error=e,
                operation='calculate_confidence_correlation'
            )
            return 0.5  # Default on error
            
    def save(self, filename):
        """
        Save model to file with error handling and atomic operations.
        
        Args:
            filename: Path to save the model
            
        Returns:
            bool: True if successful
        """
        try:
            # Ensure directory exists
            dirname = os.path.dirname(filename)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)
                
            # Save to temporary file first
            temp_filename = f"{filename}.temp"
            with open(temp_filename, 'wb') as f:
                pickle.dump(self, f)
                
            # Atomic rename operation
            os.replace(temp_filename, filename)
            
            print(f"UnifiedStackingEnsemble v{self.version} saved to {filename}")
            return True
        except Exception as e:
            self.error_tracker.log_processing_error(
                error=e,
                operation='save',
                context={'filename': filename}
            )
            print(f"Error saving stacking model: {e}")
            return False
        
    @classmethod
    def load(cls, filename, fallback_random_state=42):
        """
        Load model from file with error handling and validation.
        
        Args:
            filename: Path to load the model from
            fallback_random_state: Random state to use for new model if loading fails
            
        Returns:
            UnifiedStackingEnsemble: The loaded model or a new instance if loading fails
        """
        if not os.path.exists(filename):
            print(f"Model file {filename} not found. Creating new model.")
            return cls(random_state=fallback_random_state)
            
        try:
            with open(filename, 'rb') as f:
                model = pickle.load(f)
                
            # Verify it's the correct class
            if not isinstance(model, cls):
                print(f"Warning: Loaded model is not a {cls.__name__}. Converting...")
                new_model = cls(random_state=fallback_random_state)
                
                # Try copying attributes from old model
                try:
                    # Copy core attributes
                    core_attrs = ['meta_model', 'meta_X', 'meta_y', 'meta_patterns', 'is_trained', 
                                 'version', 'created_at', 'last_updated']
                    for attr in core_attrs:
                        if hasattr(model, attr):
                            setattr(new_model, attr, getattr(model, attr))
                            
                    # Try migrating old feature structure to new dimension manager
                    if hasattr(model, 'expected_feature_count'):
                        new_model.dimension_manager.expected_count = model.expected_feature_count
                        
                    if hasattr(model, 'feature_structure'):
                        new_model.dimension_manager.feature_structure = model.feature_structure
                        
                    # Try migrating pattern performance data
                    if hasattr(model, 'pattern_performance'):
                        new_model.pattern_analyzer.pattern_performance = model.pattern_performance
                        
                    # Mark as migrated
                    new_model.last_updated = time.time()
                    new_model.version += 1  # Increment version after migration
                    
                except Exception as migration_error:
                    print(f"Error during model migration: {migration_error}")
                    
                return new_model
                        
            # Model loaded successfully
            print(f"Loaded stacking model v{model.version if hasattr(model, 'version') else 'unknown'}")
            
            # Ensure all components are properly initialized
            if not hasattr(model, 'dimension_manager'):
                model.dimension_manager = FeatureDimensionManager(model)
                model.dimension_manager.expected_count = getattr(model, 'expected_feature_count', None)
                
            if not hasattr(model, 'pattern_analyzer'):
                model.pattern_analyzer = PatternAnalyzer()
                if hasattr(model, 'pattern_performance'):
                    model.pattern_analyzer.pattern_performance = model.pattern_performance
                    
            if not hasattr(model, 'error_tracker'):
                model.error_tracker = ErrorTracker()
                
            if not hasattr(model, 'feature_cache'):
                model.feature_cache = {}
                model.cache_hits = 0
                model.cache_misses = 0
                
            return model
            
        except Exception as e:
            print(f"Error loading stacking model: {e}")
            print("Creating new stacking ensemble...")
            return cls(random_state=fallback_random_state)
    
    def get_diagnostics(self):
        """
        Get comprehensive diagnostic information about the model state.
        
        Returns:
            dict: Diagnostic information
        """
        return {
            'model_type': self.model_type,
            'version': self.version,
            'is_trained': self.is_trained,
            'age_hours': (time.time() - self.created_at) / 3600,
            'last_updated_hours': (time.time() - self.last_updated) / 3600 if self.last_updated else None,
            'meta_examples': len(self.meta_X),
            'unique_classes': len(set(self.meta_y)) if self.meta_y else 0,
            'class_distribution': Counter(self.meta_y) if self.meta_y else None,
            'expected_features': self.dimension_manager.expected_count,
            'feature_structure': self.dimension_manager.feature_structure,
            'base_models_count': len(self.base_models) if hasattr(self, 'base_models') else 0,
            'recent_errors': {
                'prediction': len(self.error_tracker.get_recent_errors('prediction')),
                'processing': len(self.error_tracker.get_recent_errors('processing')),
                'dimension': len(self.dimension_manager.get_recent_issues())
            },
            'pattern_types': list(self.pattern_analyzer.pattern_performance.keys()),
            'meta_model_type': type(self.meta_model).__name__,
            'cache_stats': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_ratio': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
                'cache_size': len(self.feature_cache)
            }
        }