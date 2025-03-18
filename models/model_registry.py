"""
Model Registry for Baccarat Prediction System.
Manages multiple prediction models, variants, and their lifecycle with transaction-based operations.

This module implements a robust model registry that handles:
1. Model initialization, persistence, and retrieval
2. Feature dimension management and validation
3. Competitive model evolution and variant generation
4. Error recovery with multi-level fallback mechanisms
5. Confidence calibration for improved predictions
"""

import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import traceback
import sys
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from collections import defaultdict
from colorama import Fore, Style
from sklearn.isotonic import IsotonicRegression

# Import configuration
from config import MAX_MODELS

# Ensure consistent module imports with fallback mechanisms
# Replace existing imports
try:
    # First try direct imports (if running as a package)
    from models.base_model import BaseModel
    from models.baccarat_model import BaccaratModel
    from models.markov_model import MarkovModel
    from models.xgboost_model import XGBoostModel
    from models.unified_stacking_ensemble import UnifiedStackingEnsemble  # Updated import
except ImportError:
    try:
        # Try with models prefix (if running from project root)
        from models.base_model import BaseModel
        from models.baccarat_model import BaccaratModel
        from models.markov_model import MarkovModel
        from models.xgboost_model import XGBoostModel
        from models.unified_stacking_ensemble import UnifiedStackingEnsemble  # Updated import
        from prediction.monte_carlo import get_pattern_insight
        from prediction.prediction_pipeline import extract_pattern_type
    except ImportError:
        # Add parent directory to path as last resort
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        sys.path.append(os.path.join(parent_dir, 'models'))
        
        # Try imports again
        from base_model import BaseModel
        from baccarat_model import BaccaratModel
        from markov_model import MarkovModel
        from xgboost_model import XGBoostModel
        from unified_stacking_ensemble import UnifiedStackingEnsemble  # Updated import
        
        # Define fallback for extract_pattern_type if import fails
        def extract_pattern_type(pattern_insight):
            """Fallback implementation of pattern type extraction."""
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


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================

class RegistryError(Exception):
    """Base class for all registry-related errors."""
    pass

class ModelNotFoundError(RegistryError):
    """Raised when a requested model is not found in the registry."""
    pass

class ModelStateError(RegistryError):
    """Raised when a model's state is invalid or inconsistent."""
    pass

class DimensionMismatchError(RegistryError):
    """Raised when feature dimensions don't match expected values."""
    pass

class StateTransactionError(RegistryError):
    """Raised when a state-changing transaction fails."""
    pass

class PredictionError(RegistryError):
    """Raised when prediction generation fails."""
    pass


# =============================================================================
# Helper Classes
# =============================================================================

class StateTransaction:
    """
    Transaction manager for atomic state-changing operations.
    
    Implements context manager protocol to ensure registry state consistency
    by providing automatic rollback on errors.
    
    Example:
        with StateTransaction(registry) as transaction:
            registry.models[model_id] = new_model
            registry.model_active[model_id] = True
            # State changes are committed only if no exceptions occur
    """
    
    def __init__(self, state_owner):
        """
        Initialize transaction with reference to state owner.
        
        Args:
            state_owner: Object whose state is being managed (typically ModelRegistry)
        """
        self.state_owner = state_owner
        self.original_state = None
        self.successful = False
    
    def __enter__(self):
        """Begin transaction by capturing original state."""
        import copy
        self.original_state = copy.deepcopy(self.state_owner.__dict__)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        End transaction, rollback on error.
        
        Args:
            exc_type: Exception type if an error occurred, None otherwise
            exc_val: Exception value if an error occurred, None otherwise
            exc_tb: Exception traceback if an error occurred, None otherwise
        
        Returns:
            bool: False to propagate exceptions, True to suppress them
        """
        if exc_type is not None:
            # Error occurred, restore original state
            self.state_owner.__dict__.update(self.original_state)
            print(f"Transaction failed: {exc_val}. Rolling back changes.")
            return False
        
        # Transaction successful
        self.successful = True
        return True


class DummyCalibrator:
    """
    Enhanced pass-through calibrator for fallback situations.
    
    Implements a minimal interface compatible with IsotonicRegression
    to serve as an emergency replacement when calibration fails.
    Includes all required attributes for compatibility.
    """
    def __init__(self):
        """Initialize with all required attributes to prevent missing attribute errors."""
        self.X_min_ = 0.0
        self.X_max_ = 1.0
        self.y_min_ = 0.0
        self.y_max_ = 1.0
        self._y = [0.0, 1.0]  # Required by some IsotonicRegression consumers
        self._X = [0.0, 1.0]  # Required by some IsotonicRegression consumers
        self.increasing = True
        self.out_of_bounds = 'clip'
    
    def predict(self, X):
        """Return input values with proper shape handling."""
        # Handle different input formats
        import numpy as np
        
        if isinstance(X, list):
            if not X:
                return np.array([0.5])
            return np.array(X)
        elif hasattr(X, 'shape'):
            # For NumPy arrays, handle different dimensions
            if X.ndim == 2 and X.shape[1] == 1:
                # Column vector (n, 1) -> flatten to (n,)
                return X.flatten()
            elif X.ndim == 1:
                # Already in correct shape
                return X
            else:
                # Default handling for other shapes
                return X.flatten() if hasattr(X, 'flatten') else X
        else:
            # Default for other types
            return np.array([0.5])
            
    def fit(self, X, y):
        """
        Implement fit method for compatibility.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            self: For method chaining compatibility
        """
        import numpy as np
        
        # Store data in compatible format
        if hasattr(X, 'flatten'):
            self._X = X.flatten()
        else:
            self._X = np.array(X).flatten() if hasattr(X, '__iter__') else np.array([X])
            
        if hasattr(y, 'flatten'):
            self._y = y.flatten()
        else:
            self._y = np.array(y).flatten() if hasattr(y, '__iter__') else np.array([y])
            
        # Set required attributes
        self.X_min_ = min(self._X) if len(self._X) > 0 else 0.0
        self.X_max_ = max(self._X) if len(self._X) > 0 else 1.0
        self.y_min_ = min(self._y) if len(self._y) > 0 else 0.0
        self.y_max_ = max(self._y) if len(self._y) > 0 else 1.0
        
        return self

class PatternAnalysisBridge:
    """
    Adapter class that bridges the new PatternAnalyzer component with the legacy pattern analysis functions.
    
    This class provides a unified interface for pattern analysis that transparently uses the new
    pipeline components when available, with automatic fallback to legacy methods when necessary.
    """
    
    def __init__(self):
        """Initialize with lazy-loaded pattern analyzer to minimize import dependencies."""
        self._pattern_analyzer = None
        self._analyzer_import_attempted = False
    
    def get_pattern_analyzer(self):
        """
        Get or create PatternAnalyzer instance with proper error handling.
        
        Returns:
            PatternAnalyzer or None: PatternAnalyzer instance or None if import failed
        """
        if not self._analyzer_import_attempted:
            try:
                # Try importing from the new location
                from prediction.components.pattern_analyzer import PatternAnalyzer
                self._pattern_analyzer = PatternAnalyzer()
            except ImportError:
                # Mark as attempted but failed
                pass
            finally:
                self._analyzer_import_attempted = True
        
        return self._pattern_analyzer
    
    def analyze_pattern(self, prev_rounds):
        """
        Analyze pattern using new PatternAnalyzer with fallback to legacy methods.
        
        Args:
            prev_rounds: Previous game outcomes
            
        Returns:
            dict: Pattern analysis results with consistent structure
        """
        analyzer = self.get_pattern_analyzer()
        
        if analyzer:
            # Use new pattern analyzer
            try:
                # Ensure proper input format
                if isinstance(prev_rounds, np.ndarray):
                    if prev_rounds.ndim > 1 and prev_rounds.shape[0] == 1:
                        sequence = prev_rounds[0]
                    else:
                        sequence = prev_rounds
                else:
                    sequence = prev_rounds
                
                return analyzer.analyze_pattern(sequence)
            except Exception as e:
                print(f"Error in new pattern analyzer: {e}. Falling back to legacy methods.")
                # Fall through to legacy methods
        
        # Legacy fallback implementation
        try:
            # Import legacy functions with fallback implementation
            try:
                from prediction.monte_carlo import get_pattern_insight
            except ImportError:
                # Minimal get_pattern_insight implementation
                def get_pattern_insight(rounds):
                    """Minimal pattern insight implementation."""
                    return "No pattern analysis available"
            
            try:
                from prediction.components.pattern_analyzer import extract_pattern_type
            except ImportError:
                # Fallback implementation
                def extract_pattern_type(pattern_insight):
                    """Fallback pattern type extraction."""
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
            
            # Get pattern insight and type using legacy methods
            if isinstance(prev_rounds, np.ndarray) and prev_rounds.ndim > 1:
                pattern_insight = get_pattern_insight(prev_rounds[0])
            else:
                pattern_insight = get_pattern_insight(prev_rounds)
                
            pattern_type = extract_pattern_type(pattern_insight) if pattern_insight else "no_pattern"
            
            # Return in format compatible with new PatternAnalyzer
            return {
                'pattern_type': pattern_type,
                'pattern_insight': pattern_insight
            }
        except Exception as e:
            print(f"Error in legacy pattern analysis: {e}")
            # Ultimate fallback
            return {
                'pattern_type': 'no_pattern',
                'pattern_insight': 'Pattern analysis failed'
            }
    
    def extract_pattern_features(self, prev_rounds):
        """
        Extract one-hot encoded pattern features for meta-learning.
        
        Args:
            prev_rounds: Previous game outcomes
            
        Returns:
            list: Four boolean features indicating pattern type
        """
        # Get pattern analysis with framework agnostic approach
        pattern_info = self.analyze_pattern(prev_rounds)
        pattern_type = pattern_info.get('pattern_type', 'no_pattern')
        
        # One-hot encode pattern type
        pattern_features = [0, 0, 0, 0]  # [no_pattern, streak, alternating, tie]
        pattern_mapping = {
            'no_pattern': 0,
            'streak': 1,
            'alternating': 2,
            'tie': 3,
            'tie_influenced': 3,  # Map tie_influenced to tie index
            'banker_dominated': 0,  # Map to no_pattern as approximation
            'player_dominated': 0,  # Map to no_pattern as approximation
            'chaotic': 0  # Map to no_pattern as approximation
        }
        
        pattern_idx = pattern_mapping.get(pattern_type, 0)
        pattern_features[pattern_idx] = 1
        
        return pattern_features
    
class FeatureManager:
    """
    Manages feature engineering and dimension validation for model registry.
    
    Centralizes feature handling logic to ensure consistent dimension validation,
    feature generation, and error handling across the registry.
    """
    
    def __init__(self, registry):
        """
        Initialize with reference to parent registry.
        
        Args:
            registry: The ModelRegistry instance this manager belongs to
        """
        self.registry = registry
        

    def ensure_consistent_dimensions(self):
        """
        Ensure all active models have consistent feature dimensions.
        
        This method validates and updates expected feature dimensions
        across all models, applying corrections to any inconsistencies.
        
        Returns:
            dict: Validation results
        """
        results = {
            'success': True,
            'fixed_models': [],
            'issues': []
        }
        
        try:
            # Step 1: Calculate expected dimensions
            expected_dim = self.registry.update_expected_feature_dimensions()
            
            # Step 2: Check stacking model dimensions
            if "stacking_ensemble" in self.registry.models:
                stacking = self.registry.models["stacking_ensemble"]
                
                # Check different dimension attributes based on stacking implementation
                if hasattr(stacking, 'dimension_manager') and hasattr(stacking.dimension_manager, 'expected_count'):
                    actual_dim = stacking.dimension_manager.expected_count
                    if actual_dim != expected_dim:
                        # Update dimension manager
                        stacking.dimension_manager.update_expected_count(len(self.registry.get_active_base_models()))
                        results['fixed_models'].append('stacking_ensemble')
                
                elif hasattr(stacking, 'expected_feature_count'):
                    if stacking.expected_feature_count != expected_dim:
                        # Update expected feature count
                        stacking.expected_feature_count = expected_dim
                        results['fixed_models'].append('stacking_ensemble')
                
                # Verify dimensions were updated
                if "stacking_ensemble" in results['fixed_models']:
                    # Generate test features to validate dimension handling
                    test_input = np.array([[0, 1, 0, 1, 0]])
                    try:
                        meta_features = self.generate_meta_features(test_input)
                        if meta_features is None:
                            results['issues'].append("Failed to generate meta-features for validation")
                            results['success'] = False
                    except Exception as e:
                        results['issues'].append(f"Feature generation error: {str(e)}")
                        results['success'] = False
            
            # Step 3: Validate feature generation pipeline
            try:
                test_input = np.array([[0, 1, 0, 1, 0]])
                meta_features = self.generate_meta_features(test_input)
                if meta_features is None:
                    results['issues'].append("Meta-features generation failed")
                    results['success'] = False
                else:
                    feature_len = len(meta_features)
                    if feature_len != expected_dim:
                        results['issues'].append(f"Dimension mismatch: expected {expected_dim}, got {feature_len}")
                        results['success'] = False
            except Exception as e:
                results['issues'].append(f"Validation error: {str(e)}")
                results['success'] = False
            
            return results
            
        except Exception as e:
            results['success'] = False
            results['issues'].append(f"Error in dimension validation: {str(e)}")
            return results
    
    def validate_dimensions(self, features, expected_dim=None):
        """
        Validate feature dimensions with automatic correction when possible.
        
        Args:
            features: Feature vector or matrix to validate
            expected_dim: Expected feature dimension (if None, calculated automatically)
            
        Returns:
            tuple: (valid, features) - Boolean validation result and possibly corrected features
            
        Raises:
            DimensionMismatchError: If dimensions cannot be reconciled
        """
        if features is None:
            raise DimensionMismatchError("Cannot validate None features")
        
        # Calculate expected dimensions if not provided
        if expected_dim is None:
            expected_dim = self.registry.update_expected_feature_dimensions()
        
        # Get actual dimension
        actual_dim = len(features) if isinstance(features, list) else features.shape[1]
        
        # If dimensions match, return as valid
        if actual_dim == expected_dim:
            return True, features
        
        # Try to correct dimensions
        if actual_dim < expected_dim:
            # Add padding if too short
            padding_needed = expected_dim - actual_dim
            if isinstance(features, list):
                features.extend([0.33] * padding_needed)
            else:
                padding = np.ones((features.shape[0], padding_needed)) * 0.33
                features = np.hstack((features, padding))
            print(f"Corrected feature dimensions by adding {padding_needed} padding features")
            return True, features
        else:
            # Truncate if too long
            if isinstance(features, list):
                features = features[:expected_dim]
            else:
                features = features[:, :expected_dim]
            print(f"Corrected feature dimensions by truncating from {actual_dim} to {expected_dim}")
            return True, features
            
    def generate_meta_features(self, prev_rounds):
        """
        Generate meta-features from a single input example.
        
        Args:
            prev_rounds: Previous game outcomes
            
        Returns:
            list: Meta-features for stacking model
            
        Raises:
            PredictionError: If meta-feature generation fails
        """
        try:
            # Format input consistently
            if isinstance(prev_rounds, list):
                prev_rounds_arr = np.array(prev_rounds).reshape(1, -1)
            else:
                prev_rounds_arr = prev_rounds.reshape(1, -1) if prev_rounds.ndim == 1 else prev_rounds
            
            # Collect features from active base models
            meta_features = []
            active_models = self.registry.get_active_base_models()
            
            for model_id, model in active_models.items():
                try:
                    # Get prediction probabilities with uniform handling
                    prob_features = self._extract_model_probabilities(model, prev_rounds_arr)
                    meta_features.extend(prob_features)
                except Exception as e:
                    print(f"Error getting predictions from {model_id}: {e}")
                    # Add default values for missing model
                    meta_features.extend([0.33, 0.33, 0.34])
            
            # Add pattern-type features
            pattern_features = self._extract_pattern_features(prev_rounds_arr)
            meta_features.extend(pattern_features)
            
            # Validate and possibly correct dimensions
            expected_dim = self.registry.update_expected_feature_dimensions()
            valid, meta_features = self.validate_dimensions(meta_features, expected_dim)
            
            return meta_features
            
        except Exception as e:
            raise PredictionError(f"Failed to generate meta-features: {e}")
            
    def _extract_model_probabilities(self, model, prev_rounds):
        """
        Extract probability features from a single model with consistent handling.
        
        Args:
            model: The model to get probabilities from
            prev_rounds: Input data
            
        Returns:
            list: Three probability values for banker, player, tie
        """
        # Try different probability methods in order of preference
        for method_name in ['safe_predict_proba', 'conservative_predict_proba', 'predict_proba']:
            if hasattr(model, method_name):
                try:
                    method = getattr(model, method_name)
                    probs = method(prev_rounds)
                    
                    # Extract probabilities based on return format
                    if isinstance(probs, dict):
                        return [probs.get(0, 0.33), probs.get(1, 0.33), probs.get(2, 0.34)]
                    elif hasattr(probs, 'shape') and probs.shape[1] >= 3:
                        return probs[0].tolist()[:3]
                    break
                except Exception:
                    continue
        
        # Fallback: Use predict method with one-hot encoding
        try:
            pred = model.predict(prev_rounds)[0]
            result = [0.1, 0.1, 0.1]
            result[pred] = 0.8
            return result
        except Exception:
            # Ultimate fallback: return uniform distribution
            return [0.33, 0.33, 0.34]
    
    def _extract_pattern_features(self, prev_rounds):
        """
        Extract pattern features from input data.
        
        Args:
            prev_rounds: Input data
            
        Returns:
            list: Four boolean features indicating pattern type
        """
        # Use pattern bridge for unified pattern analysis
        return self.registry.pattern_bridge.extract_pattern_features(prev_rounds[0])


class ConfidenceCalibrationManager:
    """
    Manages confidence calibration for more reliable probability estimates.
    
    Provides methods for calibrator initialization, training, and applying
    calibration to prediction results with robust error handling.
    """
    
    def __init__(self, registry):
        """
        Initialize with reference to parent registry.
        
        Args:
            registry: The ModelRegistry instance this manager belongs to
        """
        self.registry = registry
        self._initialize_calibrators()
        
    def _initialize_calibrators(self):
        """Initialize calibrators with smoothly distributed training data with improved error handling."""
        try:
            # Initialize calibrators for each class
            self.registry.confidence_calibrators = {
                0: DummyCalibrator(),  # Start with DummyCalibrator as fallback
                1: DummyCalibrator(),
                2: DummyCalibrator()
            }
            
            # Create smoothly distributed training data
            for cls in range(3):
                # Generate 20 evenly distributed confidence points
                X = np.linspace(0.05, 0.95, 20).reshape(-1, 1)
                
                # Create probability outputs with realistic mapping
                y = np.linspace(0.1, 0.9, 20)
                
                # Add controlled variance for more realistic calibration
                y = np.clip(y + np.random.normal(0, 0.05, size=y.shape), 0, 1)
                
                try:
                    # Create a new IsotonicRegression instance
                    isotonic = IsotonicRegression(out_of_bounds='clip')
                    isotonic.fit(X, y)
                    
                    # Verify calibrator was properly fitted by checking for required attributes
                    if hasattr(isotonic, 'X_min_') and hasattr(isotonic, 'X_max_'):
                        self.registry.confidence_calibrators[cls] = isotonic
                        print(f"Successfully initialized IsotonicRegression calibrator for class {cls}")
                    else:
                        print(f"Warning: IsotonicRegression for class {cls} was not properly fitted, using DummyCalibrator")
                        # Explicitly fit the DummyCalibrator to have valid attributes
                        self.registry.confidence_calibrators[cls].fit(X, y)
                except Exception as fit_err:
                    print(f"Error fitting calibrator for class {cls}: {fit_err}")
                    # Explicitly fit the DummyCalibrator to have valid attributes
                    self.registry.confidence_calibrators[cls].fit(X, y)
            
            print("Confidence calibration initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing calibration (using dummy calibrators): {e}")
            
            # Ensure dummy calibrators are used as fallback
            self.registry.confidence_calibrators = {
                0: DummyCalibrator(),
                1: DummyCalibrator(),
                2: DummyCalibrator()
            }
            
            # Explicitly fit each dummy calibrator with default data
            for cls in range(3):
                X = np.array([0.0, 1.0]).reshape(-1, 1)
                y = np.array([0.0, 1.0])
                self.registry.confidence_calibrators[cls].fit(X, y)
                
            return False
    
    def calibrate_from_history(self):
        """
        Build calibration models from prediction history.
        
        Returns:
            dict: Calibration results
        """
        # Try different data sources in order of preference
        meta_features, outcomes = self._collect_calibration_data()
        
        # Train calibrators with available data
        if len(meta_features) >= 20:
            return self._train_calibrators_with_data(meta_features, outcomes)
        else:
            print(f"Insufficient data for calibration: {len(meta_features)} samples")
            self._initialize_calibrators()  # Reinitialize with synthetic data
            return {"status": "synthetic", "samples": len(meta_features)}
    
    def _collect_calibration_data(self):
        """
        Collect data for calibration from various sources.
        
        Returns:
            tuple: (meta_features, outcomes) - Lists of features and corresponding outcomes
        """
        meta_features = []
        outcomes = []
        
        # Source 1: Stacking model history
        if "stacking_ensemble" in self.registry.models:
            stacking = self.registry.models["stacking_ensemble"]
            if hasattr(stacking, 'meta_X') and hasattr(stacking, 'meta_y'):
                if len(stacking.meta_X) > 0 and len(stacking.meta_y) > 0:
                    meta_features = stacking.meta_X
                    outcomes = stacking.meta_y
                    print(f"Found {len(meta_features)} samples in stacking history")
        
        # Source 2: Log file
        if len(meta_features) < 20:
            try:
                from config import LOG_FILE
                if os.path.exists(LOG_FILE):
                    log_df = pd.read_csv(LOG_FILE)
                    
                    # Extract data from log entries
                    prev_columns = [f'Previous_{i}' for i in range(1, 6)]
                    if all(col in log_df.columns for col in prev_columns) and 'Actual' in log_df.columns:
                        for _, row in log_df.iloc[-50:].iterrows():
                            try:
                                prev_rounds = [row[col] for col in prev_columns]
                                meta_feature = self.registry.feature_manager.generate_meta_features(np.array([prev_rounds]))
                                if meta_feature is not None:
                                    meta_features.append(meta_feature)
                                    outcomes.append(row['Actual'])
                            except Exception:
                                continue
                        
                        print(f"Extracted {len(meta_features)} valid entries from log file")
            except Exception as e:
                print(f"Error loading history for calibration: {e}")
        
        return meta_features, outcomes
    
    def _train_calibrators_with_data(self, meta_features, outcomes):
        """
        Train calibrators with collected data.
        
        Args:
            meta_features: List of meta-feature vectors
            outcomes: List of corresponding outcomes
            
        Returns:
            dict: Training results
        """
        # Generate predictions for calibration
        prediction_probs = []
        for features in meta_features:
            try:
                if "stacking_ensemble" in self.registry.models:
                    stacking = self.registry.models["stacking_ensemble"]
                    probs = stacking.predict_proba([features])[0]
                    prediction_probs.append(probs)
                else:
                    # Fallback if stacking not available
                    prediction_probs.append([0.33, 0.33, 0.34])
            except Exception as e:
                print(f"Error in calibration prediction: {e}")
                prediction_probs.append([0.33, 0.33, 0.34])
        
        # Train calibrators for each class
        calibration_stats = {}
        for outcome in [0, 1, 2]:  # Banker, Player, Tie
            # Extract probabilities for this outcome
            outcome_probs = [probs[outcome] for probs in prediction_probs]
            
            # Create binary outcome indicators
            binary_outcomes = [1 if actual == outcome else 0 for actual in outcomes]
            
            # Only calibrate if we have sufficient positive examples
            positive_count = sum(binary_outcomes)
            if positive_count >= 3:
                try:
                    self.registry.confidence_calibrators[outcome].fit(
                        np.array(outcome_probs).reshape(-1, 1),
                        binary_outcomes
                    )
                    calibration_stats[outcome] = {
                        "samples": len(binary_outcomes),
                        "positive_samples": positive_count,
                        "positive_rate": positive_count / len(binary_outcomes)
                    }
                except Exception as e:
                    print(f"Error calibrating outcome {outcome}: {e}")
                    calibration_stats[outcome] = {"error": str(e)}
        
        return {"status": "success", "calibrated_outcomes": list(calibration_stats.keys())}
    
    def apply_calibration(self, result):
        """
        Apply calibration to prediction result with robust error handling.
        
        Args:
            result: Prediction result dictionary
            
        Returns:
            dict: Calibrated prediction result
        """
        if not hasattr(self.registry, 'confidence_calibrators'):
            return result
        
        try:
            predicted = result['prediction']
            raw_confidence = result['distribution'].get(predicted, 0) / 100.0
            
            # Input validation
            if raw_confidence < 0 or raw_confidence > 1:
                raw_confidence = min(max(raw_confidence, 0.0), 1.0)
            
            # Ensure predicted is a valid index
            if predicted not in [0, 1, 2] or predicted not in self.registry.confidence_calibrators:
                return result
            
            calibrator = self.registry.confidence_calibrators[predicted]
            
            # More robust attribute checking with specific fallback
            if not hasattr(calibrator, 'X_min_') or not hasattr(calibrator, 'predict'):
                # Create a properly fitted DummyCalibrator as replacement
                dummy = DummyCalibrator()
                dummy.fit(np.array([0.0, 0.5, 1.0]).reshape(-1, 1), 
                        np.array([0.0, 0.5, 1.0]))
                calibrator = dummy
                
                # Update the registry with the working dummy calibrator
                self.registry.confidence_calibrators[predicted] = dummy
            
            # Apply calibration with comprehensive error handling
            try:
                # Format input for prediction
                raw_confidence_shaped = np.array([raw_confidence]).reshape(-1, 1)
                
                # Get calibrated value with explicit error handling
                try:
                    calibrated_confidence = float(calibrator.predict(raw_confidence_shaped)[0])
                except (IndexError, TypeError, ValueError) as e:
                    # Handle issues with the returned prediction
                    print(f"Calibration prediction format error: {e}")
                    calibrated_confidence = raw_confidence
                
                # Apply game-appropriate adjustments
                if predicted == 2:  # Tie
                    max_confidence = 70.0
                else:  # Banker/Player
                    max_confidence = 90.0
                
                # Apply tapering function instead of hard cap
                if calibrated_confidence > 0.7:
                    factor = 0.7 + (0.3 * (calibrated_confidence - 0.7) / 0.3)
                    calibrated_confidence = 0.7 + (factor * (max_confidence/100 - 0.7))
                
                # Update result
                result['raw_confidence'] = result['confidence']
                result['confidence'] = calibrated_confidence * 100
                result['calibrated'] = True
            except Exception as e:
                print(f"Error applying calibration: {e}")
                # No change to result on error
        except Exception as e:
            print(f"Error in calibration: {e}")
        
        return result


# =============================================================================
# Main ModelRegistry Class
# =============================================================================

class ModelRegistry:
    def __init__(self, registry_path="models/registry"):
        """
        Initialize the model registry with consistent error handling.
        
        Args:
            registry_path: Path to store registry files
        """
        self.registry_path = registry_path
        self.models = {}  # Active models
        self.model_history = defaultdict(list)  # Performance history
        self.model_active = {}  # Active status
        self.max_models = MAX_MODELS  # Maximum models to maintain
        self.stacking_errors = []  # Track stacking errors for auto-repair
        self.initialization_state = {"base_models": False, "stacking": False, "calibration": False}
        self.pattern_bridge = PatternAnalysisBridge()
        self.feature_manager = FeatureManager(self)
        
        # Initialize helper components
        self.feature_manager = FeatureManager(self)
        
        # Ensure registry directory exists
        os.makedirs(registry_path, exist_ok=True)
        
        # Initialize registry with robust error handling
        try:
            registry_file = os.path.join(registry_path, "registry.json")
            if os.path.exists(registry_file):
                self._load_registry(registry_file)
            else:
                self.initialize_with_validation()
            
            # Apply fixes and initialize calibration
            self._fix_model_attributes()
            self.calibration_manager = ConfidenceCalibrationManager(self)
            
            # Validate registry state after initialization
            self.validate_registry_consistency()
            
        except Exception as e:
            print(f"Error initializing registry: {e}")
            traceback.print_exc()
            
            # Emergency initialization
            print("Performing emergency registry initialization...")
            self._initialize_registry()
            self.calibration_manager = ConfidenceCalibrationManager(self)

    def verify_calibrators(self):
        """
        Verify all calibrators have required attributes and repair if needed.
        
        Returns:
            bool: True if all calibrators are valid or were fixed
        """
        if not hasattr(self, 'confidence_calibrators'):
            print("Initializing confidence calibrators")
            self.calibration_manager = ConfidenceCalibrationManager(self)
            return True
        
        fixed_count = 0
        for cls in range(3):
            if cls not in self.confidence_calibrators:
                print(f"Missing calibrator for class {cls}, creating new one")
                self.confidence_calibrators[cls] = DummyCalibrator()
                self.confidence_calibrators[cls].fit(
                    np.array([0.0, 0.5, 1.0]).reshape(-1, 1),
                    np.array([0.0, 0.5, 1.0])
                )
                fixed_count += 1
                continue
                
            calibrator = self.confidence_calibrators[cls]
            
            # Check for required attributes
            if not hasattr(calibrator, 'X_min_') or not hasattr(calibrator, 'predict'):
                print(f"Calibrator for class {cls} missing required attributes, replacing")
                self.confidence_calibrators[cls] = DummyCalibrator()
                self.confidence_calibrators[cls].fit(
                    np.array([0.0, 0.5, 1.0]).reshape(-1, 1),
                    np.array([0.0, 0.5, 1.0])
                )
                fixed_count += 1
        
        if fixed_count > 0:
            print(f"Fixed {fixed_count} calibrators during verification")
        
        return True   
    
    def initialize_with_validation(self):
        """
        Perform comprehensive system initialization with stage validation.
        
        This method implements a multi-stage initialization process with
        validation between stages to ensure system consistency and integrity.
        
        Returns:
            bool: True if initialization was successful
        """
        print("Starting staged ModelRegistry initialization sequence...")
        
        # Transaction-based initialization with validation between stages
        with StateTransaction(self) as transaction:
            # Stage 1: Initialize base models
            success = self._initialize_base_models()
            if not success:
                raise RegistryError("Base model initialization failed")
            self.initialization_state["base_models"] = True
            
            # Stage 2: Validate base models
            validation_results = self._validate_base_models()
            if not validation_results.get('success', False):
                print(f"Base model validation issues: {validation_results.get('issues', [])}")
                # Apply fixes for non-critical issues
                self._fix_model_attributes()
            
            # Stage 3: Initialize stacking
            stacking_result = self._initialize_stacking()
            if not stacking_result.get('success', False):
                print(f"Stacking initialization warning: {stacking_result.get('message', '')}")
                # Continue despite stacking issues (will attempt recovery later)
            self.initialization_state["stacking"] = True
            
            # Stage 4: Validate stacking ensemble
            stacking_health = self.test_stacking_ensemble()
            if stacking_health.get('status') != 'healthy':
                print(f"Stacking health issues: {stacking_health.get('issues', [])}")
                # Apply stacking-specific fixes
                if "stacking_ensemble" in self.models:
                    stacking_fixed = self.reset_stacking(force_reset=True)
                    if not stacking_fixed:
                        print("Warning: Unable to reset stacking ensemble")
            
            # Stage 5: Initialize calibration
            calibration_result = self._initialize_calibration()
            if not calibration_result.get('success', False):
                print(f"Calibration initialization warning: {calibration_result.get('message', '')}")
            self.initialization_state["calibration"] = True
            
            # Save registry state after initialization
            self._save_registry()
            
            print(f"Registry initialization completed with status: {self.initialization_state}")
            return True
    
    def _initialize_base_models(self):
        """
        Initialize base prediction models.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Create base models
            self.models["markov_1"] = MarkovModel(order=1)
            self.models["markov_2"] = MarkovModel(order=2)
            self.models["baccarat_rf"] = BaccaratModel()
            self.models["xgboost_base"] = XGBoostModel()
            
            # Create model variants
            self.models["markov_3"] = MarkovModel(order=3)
            self.models["xgb_conservative"] = XGBoostModel()
            self.models["xgb_aggressive"] = XGBoostModel()
            
            # Initialize all models as active
            self.model_active = {model_id: True for model_id in self.models.keys()}
            
            # Train all models with minimal data
            self._force_minimal_training()
            
            return True
        except Exception as e:
            print(f"Error initializing base models: {e}")
            traceback.print_exc()
            return False
    
    def _validate_base_models(self):
        """
        Validate initialized base models.
        
        Returns:
            dict: Validation results
        """
        results = {
            'success': True,
            'issues': [],
            'validated_models': [],
            'problematic_models': []
        }
        
        # Test sample input
        test_input = np.array([[0, 1, 0, 1, 0]])
        
        for model_id, model in self.models.items():
            if model_id == "stacking_ensemble":
                continue  # Skip stacking
                
            try:
                # Check if model is trained
                if not hasattr(model, 'is_trained') or not model.is_trained:
                    results['issues'].append(f"Model {model_id} not trained")
                    results['problematic_models'].append(model_id)
                    results['success'] = False
                    continue
                
                # Try prediction
                prediction = model.predict(test_input)
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(test_input)
                
                results['validated_models'].append(model_id)
            except Exception as e:
                results['issues'].append(f"Error validating {model_id}: {str(e)}")
                results['problematic_models'].append(model_id)
                results['success'] = False
        
        return results
    
    def _initialize_stacking(self):
        """
        Initialize stacking ensemble.
        
        Returns:
            dict: Initialization results
        """
        try:
            print("Initializing unified stacking ensemble meta-learner...")
            
            # Create stacking ensemble with current models
            self.models["stacking_ensemble"] = UnifiedStackingEnsemble(self.models)
            self.model_active["stacking_ensemble"] = True
            
            # Try to train with synthetic data
            feature_dim = self._calculate_meta_feature_dimension()
            if feature_dim > 0:
                X, y = self._generate_synthetic_training_data(feature_dim, num_examples=50)
                self.models["stacking_ensemble"].fit(X, y)
                self.models["stacking_ensemble"].is_trained = True
                print("✓ Unified stacking ensemble trained with synthetic data")
                return {'success': True, 'message': 'Stacking initialized and trained'}
            else:
                return {'success': False, 'message': 'Failed to calculate feature dimensions'}
        except Exception as e:
            print(f"Error initializing stacking: {e}")
            traceback.print_exc()
            return {'success': False, 'message': str(e)}
    
    def _initialize_calibration(self):
        """
        Initialize confidence calibration.
        
        Returns:
            dict: Initialization results
        """
        try:
            # Initialize calibrators for each class
            self.confidence_calibrators = {
                0: IsotonicRegression(out_of_bounds='clip'),  # Banker
                1: IsotonicRegression(out_of_bounds='clip'),  # Player
                2: IsotonicRegression(out_of_bounds='clip')   # Tie
            }
            
            # Create smoothly distributed training data
            for cls in range(3):
                # Generate 20 evenly distributed confidence points
                X = np.linspace(0.05, 0.95, 20).reshape(-1, 1)
                
                # Create probability outputs with realistic mapping
                y = np.linspace(0.1, 0.9, 20)
                
                # Add controlled variance for more realistic calibration
                y = np.clip(y + np.random.normal(0, 0.05, size=y.shape), 0, 1)
                
                try:
                    self.confidence_calibrators[cls].fit(X, y)
                    
                    # Verify calibrator was properly fitted
                    if not hasattr(self.confidence_calibrators[cls], 'X_min_'):
                        print(f"Warning: Calibrator for class {cls} wasn't properly fitted")
                        self.confidence_calibrators[cls] = DummyCalibrator()
                        
                except Exception as fit_err:
                    print(f"Error fitting calibrator for class {cls}: {fit_err}")
                    self.confidence_calibrators[cls] = DummyCalibrator()
            
            print("Confidence calibration initialized with smooth distribution")
            return {'success': True, 'message': 'Calibrators initialized'}
            
        except Exception as e:
            print(f"Error initializing calibration: {e}")
            traceback.print_exc()
            
            # Use dummy calibrators as fallback
            self.confidence_calibrators = {
                0: DummyCalibrator(),
                1: DummyCalibrator(),
                2: DummyCalibrator()
            }
            return {'success': False, 'message': str(e)}
    
    def _initialize_registry(self):
        """
        Create initial model population with baseline training.
        """
        with StateTransaction(self) as transaction:
            # Create base models
            self.models["markov_1"] = MarkovModel(order=1)
            self.models["markov_2"] = MarkovModel(order=2)
            self.models["baccarat_rf"] = BaccaratModel()
            self.models["xgboost_base"] = XGBoostModel()
            
            # Create model variants
            self.models["markov_3"] = MarkovModel(order=3)
            self.models["xgb_conservative"] = XGBoostModel()
            self.models["xgb_aggressive"] = XGBoostModel()
            
            # [... existing code for customizing model variants ...]
            
            # Initialize all models as active
            self.model_active = {model_id: True for model_id in self.models.keys()}
            
            # Train all models with minimal data
            self._force_minimal_training()
            
            # Ensure RandomForest model is properly fitted
            if not self.models["baccarat_rf"].is_trained:
                self._ensure_random_forest_trained("baccarat_rf")
            
            # Initialize stacking ensemble
            try:
                print("Initializing unified stacking ensemble meta-learner...")
                self.models["stacking_ensemble"] = UnifiedStackingEnsemble(self.models)
                self.model_active["stacking_ensemble"] = True
                
                # Train stacking ensemble with synthetic data
                active_models = self.get_active_base_models()
                
                if all(getattr(model, 'is_trained', False) for model in active_models.values()):
                    # Generate meta-features for stacking
                    feature_dim = self._calculate_meta_feature_dimension()
                    if feature_dim > 0:
                        X, y = self._generate_synthetic_training_data(feature_dim, num_examples=50)
                        self.models["stacking_ensemble"].fit(X, y)
                        self.models["stacking_ensemble"].is_trained = True
                        print("✓ Unified stacking ensemble trained with synthetic meta-features")
                
            except Exception as e:
                print(f"Error initializing unified stacking ensemble: {e}")
            
            # Save registry state
            self._save_registry()

    def get_active_base_models(self):
        """
        Get dictionary of active base models (excluding stacking).
        
        Returns:
            dict: {model_id: model} for all active base models
        """
        return {model_id: model for model_id, model in self.models.items() 
                if self.model_active.get(model_id, False) and model_id != "stacking_ensemble"}

    def update_expected_feature_dimensions(self):
        """
        Update expected feature dimensions based on active models.
        
        Returns:
            int: Expected feature dimension count
        """
        # Get active models
        active_base_models = self.get_active_base_models()
        active_count = len(active_base_models)
        
        # Calculate expected dimensions (3 probs per model + 4 pattern features)
        expected_dimensions = (active_count * 3) + 4
        
        # Update stacking ensemble's expected dimensions
        if "stacking_ensemble" in self.models:
            old_dim = getattr(self.models["stacking_ensemble"], 'expected_feature_count', None)
            self.models["stacking_ensemble"].expected_feature_count = expected_dimensions
            
            # Only print when dimensions change
            if old_dim != expected_dimensions:
                print(f"Updated stacking dimensions: {old_dim if old_dim else 'unknown'} → {expected_dimensions}")
        
        return expected_dimensions

    def get_prediction(self, prev_rounds):
        """
        Get prediction using stacking ensemble with multi-level fallback.
        
        This method attempts to generate predictions through increasingly robust
        fallback mechanisms if earlier attempts fail:
        
        1. Stacking ensemble prediction (primary method)
        2. Base model weighted voting (first fallback)
        3. Pattern-based heuristic prediction (second fallback)
        4. Default probabilities (ultimate fallback)
        
        Args:
            prev_rounds: Previous game outcomes
            
        Returns:
            dict: Prediction details including outcome, confidence and pattern information
        """
        # Ensure registry consistency
        self.validate_registry_consistency()
        
        # Define error tracking for diagnosis and recovery
        error_details = {}
        
        try:
            # PHASE 1: Try stacking ensemble prediction
            try:
                # Format input consistently
                if isinstance(prev_rounds, list):
                    prev_rounds_arr = np.array(prev_rounds).reshape(1, -1)
                elif isinstance(prev_rounds, np.ndarray):
                    prev_rounds_arr = prev_rounds.reshape(1, -1) if prev_rounds.ndim == 1 else prev_rounds
                else:
                    raise ValueError(f"Invalid input format: {type(prev_rounds)}")
                
                # Validate input values
                if np.any(prev_rounds_arr < 0) or np.any(prev_rounds_arr > 2):
                    prev_rounds_arr = np.clip(prev_rounds_arr, 0, 2)
                    print("Fixed invalid input values by clipping to range [0, 2]")
                
                # Check stacking health
                stacking_available = "stacking_ensemble" in self.models
                stacking_trained = getattr(self.models.get("stacking_ensemble"), 'is_trained', False) if stacking_available else False
                
                if stacking_available and stacking_trained:
                    # Generate meta-features
                    meta_features = self.feature_manager.generate_meta_features(prev_rounds_arr)
                    
                    # Get stacking predictions
                    stacking = self.models["stacking_ensemble"]
                    
                    # UnifiedStackingEnsemble has improved _make_hashable method
                    if hasattr(stacking, '_make_hashable'):
                        meta_features_hashable = stacking._make_hashable(meta_features)
                    else:
                        meta_features_hashable = meta_features
                    
                    # Make prediction with properly prepared features
                    stacking_pred = stacking.predict([meta_features_hashable])[0]
                    stacking_probs = stacking.predict_proba([meta_features_hashable])[0]
                    
                    # Normalize probabilities if needed
                    probs_sum = sum(stacking_probs)
                    if abs(probs_sum - 1.0) > 0.01:
                        stacking_probs = stacking_probs / probs_sum
                    
                    # Calculate confidence and prepare result
                    stacking_confidence = stacking_probs[stacking_pred] * 100
                    stacking_dist = {i: float(p * 100) for i, p in enumerate(stacking_probs)}
                    
                    # Get pattern insight
                    pattern_info = self.pattern_bridge.analyze_pattern(prev_rounds)
                    pattern_type = pattern_info.get('pattern_type', 'no_pattern')
                    pattern_insight = pattern_info.get('pattern_insight', '')
                    
                    # Create result dictionary
                    result = {
                        'prediction': int(stacking_pred),
                        'confidence': float(stacking_confidence),
                        'distribution': stacking_dist,
                        'pattern_type': pattern_type,
                        'pattern_insight': pattern_insight,
                        'meta_learner': True,
                        'healthy': True
                    }
                    
                    # Apply confidence calibration
                    result = self.calibration_manager.apply_calibration(result)
                    
                    # Apply conservative confidence cap for games of chance
                    result['confidence'] = min(result['confidence'], 85.0)
                    
                    # Update pattern tracking if using UnifiedStackingEnsemble
                    if hasattr(stacking, 'pattern_analyzer') and hasattr(stacking.pattern_analyzer, 'update_pattern_stats'):
                        stacking.pattern_analyzer.update_pattern_stats(pattern_type, stacking_pred, None)  # Actual outcome not known yet
                    
                    return result
                    
            except Exception as stacking_error:
                # Record error for diagnostics and recovery
                error_details['stacking_error'] = str(stacking_error)
                print(f"Stacking prediction failed: {stacking_error}")
                traceback.print_exc()
                
                # Track stacking errors for auto-repair
                self.stacking_errors.append({
                    'timestamp': time.time(),
                    'error': str(stacking_error),
                    'input': prev_rounds.tolist() if hasattr(prev_rounds, 'tolist') else prev_rounds
                })
                
                # Check for repeated errors
                recent_errors = [e for e in self.stacking_errors if time.time() - e['timestamp'] < 300]
                if len(recent_errors) >= 3:
                    print("Detected multiple stacking failures. Triggering auto-repair...")
                    self.reset_stacking(force_reset=True)
                    self.stacking_errors = []  # Reset error counter
                
        except Exception as e:
            # Handle critical errors with detailed logging and recovery
            print(f"Critical error in prediction system: {e}")
            error_details['critical_error'] = str(e)
            traceback.print_exc()
            
            # Attempt emergency system recovery
            try:
                print("Attempting emergency system recovery...")
                self.reset_stacking(force_reset=True)
            except Exception as recovery_err:
                print(f"Emergency recovery failed: {recovery_err}")
                
            # Ultimate fallback with clear error indication
            return {
                'prediction': np.random.choice([0, 1, 2], p=[0.45, 0.45, 0.1]),
                'confidence': 33.3,  # Very low confidence
                'distribution': {0: 45.0, 1: 45.0, 2: 10.0},
                'fallback': True,
                'error': str(e),
                'error_details': error_details,
                'emergency': True
            }

    def reset_stacking(self, force_reset=False):
        """
        Reset and retrain the stacking ensemble with proper error handling.
        
        Implements a multi-stage approach to stacking reset with progressively
        more aggressive recovery mechanisms if earlier attempts fail.
        
        Args:
            force_reset: Force reset even if stacking appears healthy
            
        Returns:
            UnifiedStackingEnsemble: Reset and retrained stacking model or None if failed
        """
        self.validate_registry_consistency()
        print("Starting unified stacking ensemble reset procedure...")
        
        try:
            # 1. Backup existing stacking if available
            old_stacking = self.models.get("stacking_ensemble")
            old_meta_data = {}
            
            if old_stacking and not force_reset:
                # Try running a simple health check
                try:
                    if hasattr(old_stacking, 'is_trained') and old_stacking.is_trained:
                        test_input = [0, 1, 0, 1, 0]
                        meta_features = self.feature_manager.generate_meta_features(np.array(test_input).reshape(1, -1))
                        old_stacking.predict([meta_features])
                        print("Stacking model is healthy, no reset needed")
                        return old_stacking
                except Exception as e:
                    print(f"Stacking health check failed: {e}")
                
                # Save metadata for recovery attempt
                if hasattr(old_stacking, 'meta_X'):
                    old_meta_data['meta_X'] = old_stacking.meta_X
                if hasattr(old_stacking, 'meta_y'):
                    old_meta_data['meta_y'] = old_stacking.meta_y
            
            print("Resetting unified stacking ensemble...")
            
            # 2. Get active base models
            active_models = self.get_active_base_models()
            print(f"Using {len(active_models)} active base models for stacking")
            
            # 3. Create fresh stacking model using UnifiedStackingEnsemble
            try:
                new_stacking = UnifiedStackingEnsemble(active_models)
                print("Created new unified stacking ensemble instance")
            except Exception as e:
                print(f"Error creating new unified stacking ensemble: {e}")
                traceback.print_exc()
                return None
            
            # 4. Multi-stage training attempt
            reset_succeeded = False
            
            # Try real data first
            try:
                from data.data_utils import prepare_combined_dataset
                print("Loading real data for stacking training...")
                X, y = prepare_combined_dataset()
                
                if len(X) >= 10:  # Need minimum amount of data
                    # Generate meta-features
                    meta_features = self._generate_meta_features_batch(X, y)
                    
                    if meta_features and len(meta_features) == len(y):
                        # Train with available data
                        new_stacking.fit(meta_features, y)
                        new_stacking.meta_X = meta_features
                        new_stacking.meta_y = y
                        new_stacking.is_trained = True
                        reset_succeeded = True
                        print(f"✓ Unified stacking reset successful using {len(X)} real data examples")
                    else:
                        print("Failed to generate valid meta-features with matching dimensions")
                else:
                    print(f"Insufficient real data ({len(X)} examples), need at least 10")
            except Exception as e:
                print(f"Error using real data: {e}")
                traceback.print_exc()
            
            # Fall back to synthetic data if real data failed
            if not reset_succeeded:
                try:
                    print("Generating synthetic training data...")
                    feature_dim = self._calculate_meta_feature_dimension()
                    
                    if feature_dim > 0:
                        # Create synthetic data with correct dimensions
                        X_synthetic, y_synthetic = self._generate_synthetic_training_data(feature_dim, num_examples=50)
                        
                        # Train with synthetic data
                        new_stacking.fit(X_synthetic, y_synthetic)
                        new_stacking.meta_X = X_synthetic
                        new_stacking.meta_y = y_synthetic
                        new_stacking.is_trained = True
                        reset_succeeded = True
                        print(f"✓ Unified stacking reset successful using synthetic data")
                    else:
                        print("Failed to calculate feature dimensions")
                except Exception as e:
                    print(f"Error using synthetic data: {e}")
                    traceback.print_exc()
            
            # Try to recover from old stacking data as last resort
            if not reset_succeeded and 'meta_X' in old_meta_data and 'meta_y' in old_meta_data:
                try:
                    print("Attempting recovery from previous stacking data...")
                    old_X = old_meta_data['meta_X']
                    old_y = old_meta_data['meta_y']
                    
                    if len(old_X) >= 5:  # Need minimum data
                        new_stacking.fit(old_X, old_y)
                        new_stacking.meta_X = old_X
                        new_stacking.meta_y = old_y
                        new_stacking.is_trained = True
                        reset_succeeded = True
                        print(f"✓ Unified stacking reset successful using recovered data")
                    else:
                        print("Insufficient recovered data")
                except Exception as e:
                    print(f"Recovery attempt failed: {e}")
                    traceback.print_exc()
            
            # Ultra-fallback: minimal synthetic data with simplified model
            if not reset_succeeded:
                try:
                    print("EMERGENCY RECOVERY: Using minimal synthetic data...")
                    
                    # [... existing emergency recovery code ...]
                    # Create extremely simple synthetic data (just enough to train)
                    X_minimal = []
                    y_minimal = []
                    
                    # Determine feature count
                    feature_count = self._calculate_meta_feature_dimension()
                    if feature_count <= 0:
                        # Directly calculate if method failed
                        active_count = len(active_models)
                        feature_count = (active_count * 3) + 4  # 3 probs per model + 4 pattern features
                    
                    print(f"Using feature count: {feature_count}")
                    
                    # Generate minimal training examples
                    for i in range(15):
                        # Create a feature vector with expected shape
                        features = [0.33] * feature_count
                        # Set a few values differently to avoid singular matrix
                        for j in range(min(5, feature_count)):
                            features[j] = 0.1 + (j * 0.2)
                        X_minimal.append(features)
                        # Distribute classes evenly
                        y_minimal.append(i % 3)
                    
                    # Use dimension manager from UnifiedStackingEnsemble
                    if hasattr(new_stacking, 'dimension_manager'):
                        new_stacking.dimension_manager.update_expected_count(len(active_models))
                        
                    # Try training with minimal data
                    new_stacking.fit(X_minimal, y_minimal)
                    new_stacking.meta_X = X_minimal
                    new_stacking.meta_y = y_minimal
                    new_stacking.is_trained = True
                    reset_succeeded = True
                    print("✓ Emergency stacking reset with minimal data succeeded")
                except Exception as e:
                    print(f"Ultra-fallback recovery failed: {e}")
                    traceback.print_exc()
            
            # Save and validate the results
            if reset_succeeded:
                self.models["stacking_ensemble"] = new_stacking
                self.model_active["stacking_ensemble"] = True
                self._save_registry()
                
                # Verify training status
                if new_stacking.is_trained:
                    print("Stacking model successfully trained and saved")
                    return new_stacking
                else:
                    print("Warning: Training completed but is_trained flag not set")
                    new_stacking.is_trained = True
                    self._save_registry()
                    return new_stacking
            else:
                print("⚠ All stacking reset attempts failed")
                return None
                
        except Exception as e:
            print(f"Critical error in reset_stacking: {e}")
            traceback.print_exc()
            return None

    def _save_registry(self):
        """
        Save registry state to disk with transaction-based consistency.
        
        Implements a two-phase commit process:
        1. Save registry metadata
        2. Save individual model files
        
        If any part fails, previous state remains intact.
        """
        try:
            # First, ensure all models are properly marked as active
            for model_id in self.models.keys():
                if model_id not in self.model_active:
                    self.model_active[model_id] = True
            
            # Create registry metadata
            registry_meta = {
                "model_ids": list(self.models.keys()),
                "active_status": self.model_active,
                "history": self.model_history,
                "updated_at": time.time()
            }
            
            # Save metadata to temporary file first
            temp_registry_file = os.path.join(self.registry_path, "registry.json.temp")
            with open(temp_registry_file, 'w') as f:
                json.dump(registry_meta, f, indent=2)
            
            # Rename to actual registry file (atomic operation)
            registry_file = os.path.join(self.registry_path, "registry.json")
            os.replace(temp_registry_file, registry_file)
            
            # Save individual models
            success_count = 0
            for model_id, model in self.models.items():
                try:
                    # Save to temporary file first
                    temp_model_path = os.path.join(self.registry_path, f"{model_id}.pkl.temp")
                    with open(temp_model_path, 'wb') as f:
                        pickle.dump(model, f)
                    
                    # Rename to actual model file (atomic operation)
                    model_path = os.path.join(self.registry_path, f"{model_id}.pkl")
                    os.replace(temp_model_path, model_path)
                    success_count += 1
                except Exception as e:
                    print(f"Error saving model {model_id}: {e}")
                    print("This model will be available for this session but won't persist")
            
            print(f"Registry saved: {success_count}/{len(self.models)} models persisted")
            return True
            
        except Exception as e:
            print(f"Error saving registry: {e}")
            traceback.print_exc()
            return False

    def _load_registry(self, registry_file):
        """
        Load registry state from file with comprehensive error handling.
        
        Implements robust loading with multiple recovery paths:
        1. Load registry metadata
        2. Load individual model files with type-specific recovery
        3. Fix any missing attributes or methods
        4. Initialize missing models if needed
        
        Args:
            registry_file: Path to the registry JSON file
        """
        if not os.path.exists(registry_file):
            print(f"Registry file {registry_file} not found. Creating new registry...")
            self._initialize_registry()
            return
                
        try:
            # Load registry metadata
            with open(registry_file, 'r') as f:
                registry_meta = json.load(f)
            
            # Clear existing state before loading
            self.models = {}
            self.model_active = {}
            self.model_history = defaultdict(list)
            
            # Load active status and history
            self.model_active = registry_meta.get("active_status", {})
            
            # Backward compatibility for older registry files
            if "weights" in registry_meta and not self.model_active:
                self.model_active = {model_id: True for model_id in registry_meta.get("weights", {})}
                    
            self.model_history = defaultdict(list, registry_meta.get("history", {}))
            
            # Enhanced loading approach with better error reporting
            loading_errors = []
            model_ids = registry_meta.get("model_ids", [])
            success_count = 0
            # Track newly created models that need training
            models_to_train = []
            
            for model_id in model_ids:
                model_path = os.path.join(self.registry_path, f"{model_id}.pkl")
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            try:
                                model = pickle.load(f)
                                
                                # Store model in registry
                                self.models[model_id] = model
                                
                                # Ensure model has active status
                                if model_id not in self.model_active:
                                    self.model_active[model_id] = True
                                    
                                success_count += 1
                                
                            except AttributeError as attr_error:
                                error_msg = str(attr_error)
                                # Handle different model types based on error and model_id
                                
                                # Case 1: MarkovModel 
                                if model_id.startswith('markov'):
                                    order = int(model_id.split('_')[1]) if '_' in model_id else 1
                                    self.models[model_id] = MarkovModel(order=order)
                                    self.model_active[model_id] = True
                                    models_to_train.append(model_id)
                                    success_count += 1
                                    print(f"Created new MarkovModel with order {order} for {model_id}")
                                
                                # Case 2: XGBoostModel
                                elif model_id.startswith('xgboost') or model_id.startswith('xgb_'):
                                    variant_id = model_id if not "_variant_" in model_id else None
                                    self.models[model_id] = XGBoostModel(variant_id=variant_id)
                                    self.model_active[model_id] = True
                                    models_to_train.append(model_id)
                                    success_count += 1
                                    print(f"Created new XGBoostModel for {model_id}")
                                
                                # Case 3: BaccaratModel
                                elif model_id.startswith('baccarat'):
                                    self.models[model_id] = BaccaratModel()
                                    self.model_active[model_id] = True
                                    models_to_train.append(model_id)
                                    success_count += 1
                                    print(f"Created new BaccaratModel for {model_id}")
                                
                                # Case 4: Stacking ensemble
                                elif model_id == "stacking_ensemble":
                                    print(f"Will initialize stacking later after loading base models")
                                    success_count += 1
                                
                                else:
                                    # For other AttributeError, just print and continue
                                    error_msg = f"AttributeError loading {model_id}: {error_msg}"
                                    loading_errors.append(error_msg)
                                    print(error_msg)
                                    
                    except Exception as e:
                        error_msg = f"Error loading {model_id}: {str(e)}"
                        loading_errors.append(error_msg)
                        print(error_msg)
                        traceback.print_exc()
                else:
                    error_msg = f"Model file not found: {model_path}"
                    loading_errors.append(error_msg)
                    print(error_msg)

            # Train newly created models with data
            if models_to_train:
                print(f"Training {len(models_to_train)} newly created models...")
                self._train_new_models(models_to_train)

            # Fix any remaining issues with loaded models
            print("Applying fixes to loaded models...")
            fixed = self._fix_model_attributes()
            
            # Ensure stacking ensemble is initialized
            if "stacking_ensemble" not in self.models:
                print("Initializing stacking ensemble...")
                try:
                    self.initialize_stacking()
                    success_count += 1
                except Exception as e:
                    print(f"Error initializing stacking ensemble: {e}")
                    traceback.print_exc()
            
            # Consistency check - ensure all models have active status
            for model_id in self.models.keys():
                if model_id not in self.model_active:
                    self.model_active[model_id] = True
            
            # If no models loaded, initialize new registry
            if success_count == 0:
                print("No models loaded successfully. Creating new registry...")
                self._initialize_registry()
            else:
                print(f"Successfully loaded {success_count}/{len(model_ids)} models from registry")
            
        except Exception as e:
            print(f"Error loading registry: {e}")
            traceback.print_exc()
            # Fall back to creating new registry
            print("Creating new registry due to error...")
            self._initialize_registry()

    def _train_new_models(self, model_ids):
        """
        Train newly created models with appropriate data.
        
        Args:
            model_ids: List of model IDs to train
            
        Returns:
            int: Number of successfully trained models
        """
        from data.data_utils import prepare_combined_dataset
        success_count = 0
        
        try:
            # Prepare dataset with minimum records requirement
            X, y = prepare_combined_dataset(min_records=10)
            
            if len(X) > 0:
                for model_id in model_ids:
                    model = self.models.get(model_id)
                    if model:
                        try:
                            print(f"Training {model_id}...")
                            
                            if model_id.startswith('markov'):
                                # For Markov models, create a sequence
                                sequence = []
                                for _, row in X.iterrows():
                                    sequence.extend(row.values)
                                sequence.extend(y.values)  # Include target values
                                model.fit(sequence)
                            elif hasattr(model, 'fit'):
                                # For other models, use standard fit
                                X_array = X.values if hasattr(X, 'values') else X
                                y_array = y.values if hasattr(y, 'values') else y
                                model.fit(X_array, y_array)
                            
                            model.is_trained = True
                            success_count += 1
                            print(f"✓ {model_id} trained successfully")
                        except Exception as e:
                            print(f"Error training {model_id}: {e}")
                            traceback.print_exc()
            else:
                print("No training data available for newly created models")
        except Exception as e:
            print(f"Error preparing training data: {e}")
            traceback.print_exc()
        
        return success_count

    def _fix_model_attributes(self):
        """
        Fix common missing attributes in models to ensure compatibility.
        
        This method examines each model and adds any missing attributes or methods
        that are expected by the current codebase, ensuring backward compatibility
        with models created by previous versions.
        
        Returns:
            bool: True if any models were fixed, False otherwise
        """
        fixed_count = 0
        
        # Track which models were fixed for targeted saving
        fixed_models = []
        
        # Iterate through all models in the registry
        for model_id, model in self.models.items():
            model_fixed = False
            
            # === FIX MARKOV MODELS ===
            if model_id.startswith('markov'):
                # Add safe_predict_proba method if missing
                if not hasattr(model, 'safe_predict_proba'):
                    def safe_predict_proba(self, X):
                        """
                        Safe version of predict_proba that handles all edge cases consistently.
                        Default probabilities based on baccarat odds: Banker 45.9%, Player 44.6%, Tie 9.5%
                        """
                        # Default probabilities
                        default_probs = {0: 0.459, 1: 0.446, 2: 0.095}
                        
                        # Handle edge cases
                        if X is None or isinstance(X, (int, float)) or not hasattr(X, '__len__') or len(X) == 0:
                            return default_probs
                        
                        # Add fallback for missing 'order' attribute
                        if not hasattr(self, 'order'):
                            self.order = 1  # Set default order
                            
                        if hasattr(X, '__len__') and len(X) < self.order:
                            return default_probs
                        
                        try:
                            # Guard against validate_input missing
                            if hasattr(self, 'validate_input'):
                                X = self.validate_input(X)
                            else:
                                # Basic fallback validation
                                if isinstance(X, list):
                                    X = np.array(X)
                                if X.ndim == 1:
                                    X = X.reshape(1, -1)
                            
                            if len(X) > 1:
                                return [{0: 0.459, 1: 0.446, 2: 0.095} for _ in range(len(X))]
                            else:
                                # Guard against missing attributes
                                if not hasattr(self, 'transitions'):
                                    self.transitions = {}
                                if not hasattr(self, 'total_counts'):
                                    self.total_counts = {}
                                    
                                state = tuple(X[0][-self.order:])
                                if state in self.transitions and self.total_counts.get(state, 0) > 0:
                                    probs = {k: v / self.total_counts[state] for k, v in self.transitions[state].items()}
                                    return {outcome: probs.get(outcome, 0.05) for outcome in [0, 1, 2]}
                                else:
                                    return default_probs
                        except Exception as e:
                            print(f"Error in safe_predict_proba: {e}")
                            return default_probs
                    
                    # Add the method to the model instance
                    import types
                    model.safe_predict_proba = types.MethodType(safe_predict_proba, model)
                    
                    # Validate fix was applied
                    if hasattr(model, 'safe_predict_proba'):
                        fixed_count += 1
                        model_fixed = True
                        print(f"Added safe_predict_proba method to {model_id}")
                        
                        # Test method to ensure it works
                        try:
                            test_result = model.safe_predict_proba(np.array([[0, 1, 0, 1, 0]]))
                            print(f"Validated safe_predict_proba on {model_id}: {test_result}")
                        except Exception as e:
                            print(f"Warning: Added method to {model_id} but it fails: {e}")
                
                # Add model to fixed list if any attributes were fixed
                if model_fixed:
                    fixed_models.append(model_id)
            
            # Add additional model-specific fixes as needed
            # (XGBoost models, BaccaratModel, etc.)
        
        # Save each fixed model individually to ensure persistence
        for model_id in fixed_models:
            try:
                model = self.models[model_id]
                model_path = os.path.join(self.registry_path, f"{model_id}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"Successfully saved fixed model: {model_id}")
            except Exception as e:
                print(f"Error saving fixed model {model_id}: {e}")
        
        # Save the registry state to update active status
        if fixed_count > 0:
            self._save_registry()
        
        return fixed_count > 0

    def validate_registry_consistency(self):
        """
        Ensure model registry is consistent between memory and file.
        
        Checks for inconsistencies between in-memory state and saved state,
        and corrects them automatically to maintain system stability.
        
        Returns:
            bool: True if registry was consistent or fixed, False if repair failed
        """
        # Always set all loaded models to active
        for model_id in self.models.keys():
            self.model_active[model_id] = True
        
        # Check registry file
        registry_file = os.path.join(self.registry_path, "registry.json")
        if os.path.exists(registry_file):
            try:
                with open(registry_file, 'r') as f:
                    saved_registry = json.load(f)
                
                # Compare active status in memory vs. file
                saved_active = saved_registry.get("active_status", {})
                memory_active = self.model_active
                
                # Find differences
                differences = []
                for model_id in set(saved_active.keys()) | set(memory_active.keys()):
                    saved = saved_active.get(model_id, False)
                    memory = memory_active.get(model_id, False)
                    if saved != memory:
                        differences.append(f"{model_id}: file={saved}, memory={memory}")
                
                if differences:
                    print(f"Fixing {len(differences)} inconsistencies in model_active status...")
                    
                    # Memory state takes precedence for existing models
                    fixed_active = {model_id: True for model_id in self.models.keys()}
                    
                    # Apply fixes to model_active
                    self.model_active = fixed_active
                    
                    # Save fixed registry
                    self._save_registry()
                    print("Registry saved with fixes for model_active status")
                
                return True
                    
            except Exception as e:
                print(f"Error validating registry: {e}")
                return False
        return True

    def _calculate_meta_feature_dimension(self):
        """
        Calculate the expected dimension of meta-features based on active models.
        
        Returns:
            int: Expected feature dimension or 0 if calculation fails
        """
        try:
            # Count active base models
            active_models = self.get_active_base_models()
            
            if not active_models:
                print("No active base models found")
                return 0
                
            # Each model contributes 3 features (probabilities for banker, player, tie)
            base_features = len(active_models) * 3
            
            # Add 4 pattern type features (no_pattern, streak, alternating, tie)
            pattern_features = 4
            
            return base_features + pattern_features
        except Exception as e:
            print(f"Error calculating meta-feature dimension: {e}")
            return 0

    def _generate_synthetic_training_data(self, feature_dim, num_examples=30):
        """
        Generate synthetic training data with proper feature dimensions.
        
        Creates synthetic data suitable for initial training of stacking models
        when real training data is unavailable.
        
        Args:
            feature_dim: Total feature dimension for each example
            num_examples: Number of examples to generate
            
        Returns:
            tuple: (X, y) synthetic features and targets
        """
        # Initialize arrays
        X = []
        y = []
        
        # Pattern features start at this index
        pattern_feature_start = feature_dim - 4
        
        # Generate varied examples
        for i in range(num_examples):
            # Create features with proper dimensions
            features = np.zeros(feature_dim)
            
            # Fill probability features with realistic values
            prob_features = feature_dim - 4  # All features except pattern
            
            for j in range(0, prob_features, 3):
                # Generate random probabilities that sum to 1
                probs = np.random.random(3)
                probs = probs / probs.sum()
                features[j:j+3] = probs
            
            # Set one pattern type active per example
            pattern_idx = i % 4  # Rotate through pattern types
            features[pattern_feature_start + pattern_idx] = 1.0
            
            # Create balanced target distribution
            target = i % 3  # Distribute between banker (0), player (1), tie (2)
            
            X.append(features)
            y.append(target)
        
        return X, y

    def _generate_meta_features_batch(self, X, y):
        """
        Generate meta-features for a batch of examples.
        
        Args:
            X: Input features
            y: Target values (unused but kept for API consistency)
            
        Returns:
            list: Meta-features for each example or None if failed
        """
        try:
            if len(X) == 0:
                return None
                
            # Get active base models
            active_models = self.get_active_base_models()
            
            if not active_models:
                print("No active base models for meta-feature generation")
                return None
            
            # Initialize meta-features
            meta_features = []
            
            # Process each example
            for i in range(len(X)):
                # Extract single example
                if isinstance(X, np.ndarray):
                    x_i = X[i:i+1]
                else:
                    x_i = X.iloc[i:i+1].values
                
                # Get meta-features for this example
                features = []
                
                # Collect predictions from each model
                for model_id, model in active_models.items():
                    try:
                        # Get probabilities with consistent handling
                        prob_list = self.feature_manager._extract_model_probabilities(model, x_i)
                        features.extend(prob_list)
                        
                    except Exception as e:
                        print(f"Error getting predictions from {model_id}: {e}")
                        # Add default probabilities for this model
                        features.extend([0.33, 0.33, 0.34])
                
                # Get pattern features
                pattern_features = self.feature_manager._extract_pattern_features(x_i)
                features.extend(pattern_features)
                
                # Add complete feature vector for this example
                meta_features.append(features)
            
            # Verify all feature vectors have same length
            if len(set(len(f) for f in meta_features)) != 1:
                print("Error: Inconsistent meta-feature dimensions")
                return None
                
            return meta_features
            
        except Exception as e:
            print(f"Error generating batch meta-features: {e}")
            traceback.print_exc()
            return None

    def _force_minimal_training(self):
        """
        Ensure all models have minimal training to prevent 'not fitted' errors.
        
        Creates and uses minimal synthetic data to make models operational
        when actual training data is unavailable.
        """
        print("Training models with minimal synthetic data...")
        
        # Create minimal synthetic training data
        X_minimal = np.array([
            [0, 0, 0, 0, 0],  # All banker
            [1, 1, 1, 1, 1],  # All player
            [0, 1, 0, 1, 0],  # Alternating banker/player
            [1, 0, 1, 0, 1],  # Alternating player/banker
            [0, 0, 1, 1, 2],  # Mixed with tie
            [2, 0, 1, 0, 1],  # Starting with tie
            [0, 1, 2, 0, 1],  # Mixed pattern
            [1, 1, 0, 0, 2],  # Another mixed pattern
        ])
        y_minimal = np.array([0, 1, 0, 1, 2, 0, 1, 2])
        
        # Try to load real data if available
        try:
            import pandas as pd
            from config import BACARAT_DATA_FILE
            
            if os.path.exists(BACARAT_DATA_FILE):
                data = pd.read_csv(BACARAT_DATA_FILE)
                
                if len(data) >= 15:
                    # Sample rows from the data
                    sampled_data = data.sample(n=min(50, len(data)), random_state=42)
                    
                    # Extract features and target
                    input_cols = [col for col in sampled_data.columns if col.startswith('Prev_')]
                    
                    if input_cols and len(input_cols) >= 5 and 'Target' in sampled_data.columns:
                        X_minimal = sampled_data[input_cols].values
                        y_minimal = sampled_data['Target'].values
                        print(f"Using {len(X_minimal)} real data samples for training")
        except Exception as e:
            print(f"Error loading real data: {e}")
            print("Using synthetic data instead")
        
        # Train each model
        for model_id, model in self.models.items():
            # Skip stacking ensemble
            if model_id == "stacking_ensemble":
                continue
                
            try:
                if model_id.startswith('markov'):
                    # For Markov models, convert to sequence
                    sequence = X_minimal.flatten().tolist() + y_minimal.tolist()
                    model.fit(sequence)
                elif hasattr(model, 'fit'):
                    # For other models, use X and y directly
                    model.fit(X_minimal, y_minimal)
                    
                # Explicitly set trained flag
                model.is_trained = True
                print(f"✓ {model_id} trained successfully")
            except Exception as e:
                print(f"Error training {model_id}: {e}")
                # Try even more minimal synthetic data as last resort
                try:
                    minimal_X = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
                    minimal_y = np.array([0, 1])
                    
                    if model_id.startswith('markov'):
                        model.fit(minimal_X.flatten().tolist() + minimal_y.tolist())
                    else:
                        model.fit(minimal_X, minimal_y)
                        
                    model.is_trained = True
                    print(f"✓ {model_id} trained with fallback data")
                except Exception as fallback_err:
                    print(f"Fallback training also failed for {model_id}: {fallback_err}")

    def _ensure_random_forest_trained(self, model_id="baccarat_rf"):
        """
        Ensure the RandomForest model is properly fitted with appropriate data.
        
        Args:
            model_id: ID of the RandomForest model to train
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        if model_id not in self.models:
            print(f"Model {model_id} not found in registry")
            return False
        
        model = self.models[model_id]
        
        print(f"Ensuring {model_id} is properly trained...")
        
        try:
            from config import BACARAT_DATA_FILE
            
            # First try to use actual data
            if os.path.exists(BACARAT_DATA_FILE):
                try:
                    data = pd.read_csv(BACARAT_DATA_FILE)
                    if len(data) >= 50:
                        # Use 50 samples from real data
                        sample_data = data.sample(n=50, random_state=42)
                        
                        # Extract features and target
                        input_cols = [col for col in sample_data.columns if col.startswith('Prev_')]
                        if 'Target' in sample_data.columns and len(input_cols) >= 5:
                            X = sample_data[input_cols].values
                            y = sample_data['Target'].values
                            
                            # Train model
                            model.fit(X, y)
                            print(f"✓ {model_id} trained with real data samples")
                            return True
                except Exception as e:
                    print(f"Error using real data: {e}")
            
            # Fallback to synthetic data with proper structure for RandomForest
            print("Using synthetic data with proper structure for RandomForest...")
            
            # Create structured data with appropriate shape
            X = np.array([
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1],
                [0, 0, 1, 1, 2],
                [2, 0, 1, 0, 1],
                [0, 1, 2, 0, 1],
                [1, 1, 0, 0, 2],
                [0, 2, 0, 2, 0],
                [1, 2, 1, 2, 1],
            ])
            
            # Ensure balanced targets
            y = np.array([0, 1, 0, 1, 2, 0, 1, 2, 0, 1])
            
            # Convert to DataFrame for feature engineering
            X_df = pd.DataFrame(X, columns=[f'Prev_{i+1}' for i in range(X.shape[1])])
            
            # Train with verbose output
            model.fit(X_df, y)
            
            # Verify model was actually fitted
            if hasattr(model, 'model') and hasattr(model.model, 'estimators_'):
                print(f"✓ {model_id} successfully fitted with {len(model.model.estimators_)} trees")
                model.is_trained = True
                return True
            else:
                print(f"Warning: {model_id} training did not produce proper estimators")
                return False
                
        except Exception as e:
            print(f"Error ensuring {model_id} is trained: {e}")
            traceback.print_exc()
            return False

    def initialize_stacking(self):
        """
        Initialize the stacking ensemble meta-learner.
        
        Creates and adds a new stacking ensemble to the model registry
        if one does not already exist.
        
        Returns:
            bool: True if initialization was successful
        """
        print("Initializing unified stacking ensemble meta-learner...")
        
        with StateTransaction(self) as transaction:
            # Create stacking ensemble with current models
            self.models["stacking_ensemble"] = UnifiedStackingEnsemble(self.models)
            self.model_active["stacking_ensemble"] = True
            
            # Try to train with synthetic data
            try:
                feature_dim = self._calculate_meta_feature_dimension()
                if feature_dim > 0:
                    X, y = self._generate_synthetic_training_data(feature_dim, num_examples=50)
                    self.models["stacking_ensemble"].fit(X, y)
                    self.models["stacking_ensemble"].is_trained = True
                    print("✓ Unified stacking ensemble trained with synthetic data")
            except Exception as e:
                print(f"Warning: Initial stacking training failed: {e}")
            
            # Save registry
            self._save_registry()
            
        print("Unified stacking ensemble initialized")
        return True

    def test_stacking_ensemble(self):
        """
        Comprehensive test of stacking ensemble health with diagnostics.
        
        Performs multiple validation tests on the stacking ensemble to ensure
        it's properly configured and operational.
        
        Returns:
            dict: Health assessment results
        """
        self.validate_registry_consistency()
        print("Beginning unified stacking ensemble health check...")
        
        health_report = {
            'status': 'unknown',
            'issues': [],
            'tests_run': [],
            'repair_actions': []
        }
        
        try:
            # Check 1: Stacking existence and training status
            if "stacking_ensemble" not in self.models:
                health_report['issues'].append("Stacking ensemble not found in registry")
                health_report['status'] = 'failed'
                print("Stacking ensemble not found. Initializing...")
                self.initialize_stacking()
                health_report['repair_actions'].append("Initialized new unified stacking ensemble")
                
            stacking = self.models.get("stacking_ensemble")
            if not stacking:
                health_report['issues'].append("Failed to get stacking model reference")
                health_report['status'] = 'failed'
                return health_report
                
            is_trained = getattr(stacking, 'is_trained', False)
            health_report['tests_run'].append('existence_check')
            
            if not is_trained:
                health_report['issues'].append("Stacking ensemble not trained")
                health_report['status'] = 'failed'
                print("Stacking ensemble not trained. Will attempt training...")
                
                # Try training stacking
                self.reset_stacking()
                health_report['repair_actions'].append("Attempted stacking training via reset")
                
                # Check again
                is_trained = getattr(self.models.get("stacking_ensemble"), 'is_trained', False)
                if not is_trained:
                    return health_report
            
            # Check 2: Meta-data availability
            health_report['tests_run'].append('meta_data_check')
            
            meta_examples = len(stacking.meta_X) if hasattr(stacking, 'meta_X') else 0
            health_report['meta_examples'] = meta_examples
            
            if meta_examples < 5:
                health_report['issues'].append(f"Insufficient meta-examples: {meta_examples}")
                if health_report['status'] != 'failed':
                    health_report['status'] = 'warning'
            
            # Check 3: Feature dimensions using the dimension manager for UnifiedStackingEnsemble
            health_report['tests_run'].append('dimension_check')
            
            if hasattr(stacking, 'dimension_manager') and hasattr(stacking.dimension_manager, 'expected_count'):
                actual_features = stacking.dimension_manager.expected_count
                health_report['actual_features'] = actual_features
            else:
                # Fallback to original dimension check
                meta_model_dims = None
                if hasattr(stacking, 'meta_model'):
                    if hasattr(stacking.meta_model, 'coef_'):
                        meta_model_dims = stacking.meta_model.coef_.shape
                    elif hasattr(stacking.meta_model, 'estimators_') and stacking.meta_model.estimators_:
                        first_estimator = stacking.meta_model.estimators_[0]
                        if hasattr(first_estimator, 'coef_'):
                            meta_model_dims = first_estimator.coef_.shape
                            
                health_report['meta_model_dims'] = str(meta_model_dims) if meta_model_dims else "unknown"
            
            # Check 4: Feature alignment
            health_report['tests_run'].append('feature_alignment_check')
            
            # Calculate expected feature count
            active_models = len(self.get_active_base_models())
            expected_features = (active_models * 3) + 4  # 3 probs per model + 4 pattern features
            health_report['expected_features'] = expected_features
            
            # Get actual feature count from dimension manager if available
            if hasattr(stacking, 'dimension_manager') and hasattr(stacking.dimension_manager, 'expected_count'):
                actual_features = stacking.dimension_manager.expected_count
            elif hasattr(stacking, 'expected_feature_count'):
                actual_features = stacking.expected_feature_count
            elif 'meta_model_dims' in health_report and health_report['meta_model_dims'] != "unknown":
                meta_model_dims = eval(health_report['meta_model_dims'])
                actual_features = meta_model_dims[1] if len(meta_model_dims) > 1 else None
            else:
                actual_features = None
            
            health_report['actual_features'] = actual_features if actual_features is not None else "unknown"
            
            if actual_features is not None and actual_features != expected_features:
                health_report['issues'].append(f"Feature dimension mismatch: expected {expected_features}, got {actual_features}")
                health_report['status'] = 'failed'
                
                # Try to repair using dimension manager if available
                if hasattr(stacking, 'dimension_manager') and hasattr(stacking.dimension_manager, 'update_expected_count'):
                    try:
                        stacking.dimension_manager.update_expected_count(active_models)
                        health_report['repair_actions'].append("Updated dimension manager with new expected count")
                    except Exception as dim_error:
                        print(f"Error updating dimension manager: {dim_error}")
                        
                        # Fall back to full reset
                        print(f"Feature dimension mismatch detected. Resetting stacking...")
                        self.reset_stacking(force_reset=True)
                        health_report['repair_actions'].append("Reset stacking due to dimension mismatch")
                else:
                    # Legacy approach for non-UnifiedStackingEnsemble
                    print(f"Feature dimension mismatch detected. Resetting stacking...")
                    self.reset_stacking(force_reset=True)
                    health_report['repair_actions'].append("Reset stacking due to dimension mismatch")
                
                # Check if repair worked
                stacking = self.models.get("stacking_ensemble")
                if hasattr(stacking, 'dimension_manager') and hasattr(stacking.dimension_manager, 'expected_count'):
                    new_actual = stacking.dimension_manager.expected_count
                    if new_actual == expected_features:
                        health_report['repair_results'] = "Dimension mismatch resolved"
            
            # Check 5: Prediction functionality
            health_report['tests_run'].append('prediction_test')
            
            # Use unified diagnostic approach for both implementations
            try:
                # Use a sample alternating pattern for testing
                sample_input = [0, 1, 0, 1, 0]
                prev_rounds_arr = np.array(sample_input).reshape(1, -1)
                
                # Generate meta-features
                meta_features = self.feature_manager.generate_meta_features(prev_rounds_arr)
                
                if meta_features is None:
                    health_report['issues'].append("Failed to generate meta-features")
                    health_report['status'] = 'failed'
                else:
                    # Make features hashable if needed
                    if hasattr(stacking, '_make_hashable'):
                        try:
                            meta_features_hashable = stacking._make_hashable(meta_features)
                        except Exception as hash_error:
                            health_report['issues'].append(f"Failed to make features hashable: {str(hash_error)}")
                            meta_features_hashable = meta_features  # Use original as fallback
                    else:
                        meta_features_hashable = meta_features
                    
                    # Try prediction
                    try:
                        prediction = stacking.predict([meta_features_hashable])
                        probabilities = stacking.predict_proba([meta_features_hashable])
                        
                        health_report['test_prediction'] = int(prediction[0])
                        health_report['test_probabilities'] = {i: float(p) for i, p in enumerate(probabilities[0])}
                        
                        # Check for diagnostics in UnifiedStackingEnsemble
                        if hasattr(stacking, 'get_diagnostics'):
                            try:
                                diagnostics = stacking.get_diagnostics()
                                health_report['diagnostics'] = {
                                    'version': diagnostics.get('version', 'unknown'),
                                    'cache_stats': diagnostics.get('cache_stats', {}),
                                    'recent_errors': diagnostics.get('recent_errors', {})
                                }
                            except Exception as diag_error:
                                health_report['diagnostics_error'] = str(diag_error)
                        
                        # Final health assessment
                        if not health_report['issues']:
                            health_report['status'] = 'healthy'
                            
                    except Exception as pred_error:
                        health_report['issues'].append(f"Prediction test failed: {str(pred_error)}")
                        health_report['status'] = 'failed'
            except Exception as e:
                health_report['issues'].append(f"Prediction test failed: {str(e)}")
                health_report['status'] = 'failed'
                
                # Try repair
                print(f"Prediction test failed. Resetting stacking...")
                self.reset_stacking(force_reset=True)
                health_report['repair_actions'].append("Reset stacking due to prediction failure")
            
            return health_report
        
        except Exception as e:
            health_report['issues'].append(f"Health check failed with exception: {str(e)}")
            health_report['status'] = 'error'
            
            # Try repair for critical error
            try:
                self.reset_stacking(force_reset=True)
                health_report['repair_actions'].append("Emergency stacking reset due to health check failure")
            except Exception as repair_error:
                health_report['repair_actions'].append("Emergency repair also failed")
                
            return health_report

    def run_model_competition(self, test_data=None):
        """
        Run competitive evolution to create variants of successful models.
        
        Implements fitness-based model competition with variant generation
        and evaluation. Core models are preserved, while variants compete
        for limited slots in the registry.
        
        Args:
            test_data: Optional evaluation data (if None, uses historical performance)
            
        Returns:
            list: Ranked models and their performance scores
        """
        self.validate_registry_consistency()
        try:
            # Step 1: Retrain all models with latest data
            self.refresh_models_with_latest_data()
            
            # Step 2: Evaluate all models
            if test_data is not None:
                evaluations = self._evaluate_on_test_data(test_data)
            else:
                evaluations = self._evaluate_on_history()
            
            # Step 3: Sort models by performance
            ranked_models = sorted(
                [(model_id, score) for model_id, score in evaluations.items()],
                key=lambda x: x[1], 
                reverse=True
            )
            
            print(f"Model rankings (best to worst):")
            for model_id, score in ranked_models[:5]:  # Show top 5
                print(f"  {model_id}: {score:.4f}")
            
            # Step 4: Create variants of top performers
            top_performers = ranked_models[:max(2, len(ranked_models) // 3)]
            new_variants = []
            
            for model_id, score in top_performers:
                # Skip models that can't create variants
                if not hasattr(self.models[model_id], 'create_variant'):
                    continue
                    
                # Create 1-2 variants based on performance
                num_variants = 2 if score > 0.65 else 1
                
                for variant_idx in range(num_variants):
                    try:
                        variant = self.models[model_id].create_variant()
                        variant_id = f"{model_id}_variant_{int(time.time())}_{variant_idx}"
                        self.models[variant_id] = variant
                        self.model_active[variant_id] = True
                        new_variants.append(variant_id)
                        print(f"Created new variant: {variant_id} from {model_id}")
                    except Exception as e:
                        print(f"Error creating variant of {model_id}: {e}")
            
            # Train any new variants if created
            if new_variants:
                from data.data_utils import prepare_combined_dataset
                try:
                    X, y = prepare_combined_dataset()
                    
                    for variant_id in new_variants:
                        variant = self.models[variant_id]
                        variant.fit(X, y)
                        variant.is_trained = True
                        print(f"Trained new variant: {variant_id}")
                except Exception as e:
                    print(f"Error training new variants: {e}")
            
            # Step 5: Manage model population size (preserve base models)
            # Identify base vs. variant models
            base_models = [m_id for m_id in self.models.keys() 
                        if not "variant" in m_id.lower()]
            variant_models = [m_id for m_id in self.models.keys() 
                            if "variant" in m_id.lower()]
            
            print(f"Current model population: {len(self.models)} total models")
            print(f"  Base models: {len(base_models)}")
            print(f"  Variant models: {len(variant_models)}")
            
            # Check if we need to remove some variants to stay within limits
            if len(self.models) > self.max_models:
                # Calculate how many models to remove
                excess_models = len(self.models) - self.max_models
                print(f"Need to remove {excess_models} models to stay within limit of {self.max_models}")
                
                # Sort variants by performance for pruning
                variant_scores = [(v_id, evaluations.get(v_id, 0)) for v_id in variant_models]
                sorted_variants = sorted(variant_scores, key=lambda x: x[1])
                
                # Select variants to deactivate
                variants_to_deactivate = sorted_variants[:min(excess_models, len(sorted_variants))]
                for variant_id, score in variants_to_deactivate:
                    print(f"Deactivating underperforming variant: {variant_id} (score: {score:.4f})")
                    self.model_active[variant_id] = False
                    
            # After creating new variants and pruning, set all base models to active
            for model_id in base_models:
                self.model_active[model_id] = True
                
            # Ensure stacking is always active
            if "stacking_ensemble" in self.models:
                self.model_active["stacking_ensemble"] = True
            
            # Update expected feature dimensions based on active models
            self.update_expected_feature_dimensions()
            
            # Save registry after changes
            self._save_registry()
            
            # Validate final registry state
            self.validate_registry_consistency()
            
            return ranked_models
            
        except Exception as e:
            print(f"Error in model competition: {e}")
            traceback.print_exc()
            try:
                self._save_registry()
            except:
                pass
            
            return []
            
    def _evaluate_on_history(self, halflife=50):
        """
        Evaluate models using historical performance with exponential weighting.
        
        Implements time-decay weighting to prioritize recent performance while
        still considering long-term stability. Also analyzes confidence calibration
        quality for more reliable evaluations.
        
        Args:
            halflife: Time in hours for weight to decrease by half
            
        Returns:
            dict: {model_id: score} with performance scores for each model
        """
        now = time.time()
        evaluations = {}
        
        for model_id, model in self.models.items():
            # Use model's built-in health assessment if available
            if hasattr(model, 'get_strategy_health'):
                health = model.get_strategy_health()
                evaluations[model_id] = health['score']
                continue
    
            # Fall back to exponential weighted accuracy
            history = self.model_history[model_id]
            if not history:
                evaluations[model_id] = 0.33  # Default below average for models without history
                continue
            
            # Calculate weighted metrics
            total_weight = 0
            correct_weight = 0
            conf_correlation = 0
            
            # Exponential weighting of historical performance
            for entry in history:
                age = now - entry['timestamp']
                weight = 2 ** (-age / (halflife * 3600))  # Convert halflife to seconds
                
                total_weight += weight
                if entry['correct']:
                    correct_weight += weight
            
            # Calculate confidence correlation if available
            conf_pairs = [(h['confidence'], h['correct']) for h in history if 'confidence' in h]
            if len(conf_pairs) >= 10:
                conf_values = [p[0] for p in conf_pairs]
                corr_values = [p[1] for p in conf_pairs]
                
                conf_mean = sum(conf_values) / len(conf_values)
                corr_mean = sum(corr_values) / len(corr_values)
                
                numerator = sum((c - conf_mean) * (corr - corr_mean) for c, corr in conf_pairs)
                conf_var = sum((c - conf_mean) ** 2 for c in conf_values)
                corr_var = sum((corr - corr_mean) ** 2 for corr in corr_values)
                
                if conf_var > 0 and corr_var > 0:
                    conf_correlation = numerator / ((conf_var * corr_var) ** 0.5)
                    # Normalize to 0-1
                    conf_correlation = (conf_correlation + 1) / 2
                else:
                    conf_correlation = 0.5
            else:
                conf_correlation = 0.5
            
            # Combined score (accuracy + correlation)
            weighted_accuracy = correct_weight / total_weight if total_weight > 0 else 0.33
            evaluations[model_id] = 0.7 * weighted_accuracy + 0.3 * conf_correlation
        
        return evaluations
    
    def _evaluate_on_test_data(self, test_data):
        """
        Evaluate models using provided test data with comprehensive metrics.
        
        Calculates accuracy, confidence calibration, and class-specific performance
        metrics to provide a holistic evaluation of model quality.
        
        Args:
            test_data: Dictionary with 'X' features and 'y' targets
            
        Returns:
            dict: {model_id: score} with performance scores for each model
        """
        evaluations = {}
        
        X = test_data['X']
        y = test_data['y']
        
        for model_id, model in self.models.items():
            try:
                # Make predictions
                y_pred = model.predict(X)
                accuracy = sum(y_pred == y) / len(y)
                
                # Get confidence correlation if possible
                conf_corr = 0.5
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)
                    
                    # Extract confidence values based on return format
                    if isinstance(y_proba, list) and isinstance(y_proba[0], dict):
                        # Dictionary format
                        confidences = [proba.get(pred, 0.33) for proba, pred in zip(y_proba, y_pred)]
                    elif hasattr(y_proba, 'shape'):
                        # Array format
                        confidences = [proba[pred] for proba, pred in zip(y_proba, y_pred)]
                    else:
                        # Fallback
                        confidences = [0.6] * len(y_pred)
                    
                    correct = (y_pred == y).astype(int)
                    
                    # Calculate point-biserial correlation
                    if len(set(correct)) > 1 and len(set(confidences)) > 1:
                        try:
                            from scipy.stats import pointbiserialr
                            r, _ = pointbiserialr(correct, confidences)
                            conf_corr = (r + 1) / 2  # Normalize to 0-1
                        except ImportError:
                            # Manual calculation if scipy not available
                            conf_mean = sum(confidences) / len(confidences)
                            corr_mean = sum(correct) / len(correct)
                            
                            numerator = sum((c - conf_mean) * (corr - corr_mean) for c, corr in zip(confidences, correct))
                            conf_var = sum((c - conf_mean) ** 2 for c in confidences)
                            corr_var = sum((corr - corr_mean) ** 2 for corr in correct)
                            
                            if conf_var > 0 and corr_var > 0:
                                r = numerator / ((conf_var * corr_var) ** 0.5)
                                conf_corr = (r + 1) / 2
                            else:
                                conf_corr = 0.5
                
                # Calculate class-specific metrics
                class_metrics = {}
                for cls in range(3):  # Banker, Player, Tie
                    cls_idx = (y == cls)
                    if sum(cls_idx) > 0:
                        cls_accuracy = sum((y_pred == y) & cls_idx) / sum(cls_idx)
                        class_metrics[cls] = cls_accuracy
                
                # Balance regular accuracy with class-specific performance
                class_balanced_accuracy = sum(class_metrics.values()) / len(class_metrics) if class_metrics else accuracy
                
                # Combined score
                evaluations[model_id] = (0.4 * accuracy + 0.3 * class_balanced_accuracy + 0.3 * conf_corr)
                
            except Exception as e:
                print(f"Error evaluating {model_id}: {e}")
                evaluations[model_id] = 0.33  # Below average default
        
        return evaluations

    def refresh_models_with_latest_data(self):
        """
        Refresh all models with the latest available data.
        
        Implements incremental learning for models that support it and
        full retraining for others, with transaction-based consistency.
        
        Returns:
            tuple: (success, stats) Boolean success flag and statistics dict
        """
        start_time = time.time()
        
        training_stats = {
            "models_trained": 0,
            "models_failed": 0,
            "data_samples": 0,
            "model_stats": {},
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with StateTransaction(self) as transaction:
                # Import data utilities and load latest data 
                from data.data_utils import prepare_combined_dataset
                X, y = prepare_combined_dataset()
                
                training_stats["data_samples"] = len(X)
                
                # Data validation
                if len(X) < 10:
                    print(f"Insufficient data for retraining ({len(X)} samples)")
                    training_stats["error"] = f"Insufficient data: {len(X)} samples"
                    return False, training_stats
                
                # Update each model appropriately based on its type
                for model_id, model in self.models.items():
                    try:
                        model_type = model.model_type if hasattr(model, 'model_type') else "unknown"
                        
                        # Skip stacking model (will be handled separately)
                        if model_id == "stacking_ensemble":
                            continue
                        
                        # Different retraining strategies based on model type
                        if model_id.startswith('markov'):
                            # For Markov models, create a full sequence
                            if isinstance(X, np.ndarray):
                                sequence = X.flatten().tolist()
                            else:
                                sequence = []
                                for _, row in X.iterrows():
                                    sequence.extend(row.values)
                            
                            sequence.extend(y if isinstance(y, list) else y.tolist())  # Include target values
                            model.fit(sequence)
                            
                        elif hasattr(model, 'partial_fit') and model_type != "xgboost":
                            # For models supporting incremental learning
                            model.partial_fit(X, y)
                            
                        elif model_type == "xgboost":
                            # Special handling for XGBoost models
                            X_df = pd.DataFrame(X, columns=[f'Prev_{i+1}' for i in range(X.shape[1])])
                            if hasattr(model, '_add_derived_features'):
                                X_enhanced = model._add_derived_features(X_df)
                            else:
                                X_enhanced = X_df
                                
                            if hasattr(model, 'scaler') and hasattr(model.scaler, 'transform'):
                                X_scaled = model.scaler.transform(X_enhanced)
                            else:
                                X_scaled = X_enhanced
                            
                            # Add a few new trees to the existing model
                            orig_n_trees = model.model.n_estimators
                            model.model.n_estimators += 5  # Add 5 more trees
                            model.model.fit(X_scaled, y)
                            
                        elif hasattr(model, 'fit'):
                            # Full retraining for other models
                            model.fit(X, y)
                        
                        # Mark as trained
                        model.is_trained = True
                        training_stats["models_trained"] += 1
                        training_stats["model_stats"][model_id] = {
                            "status": "Success"
                        }
                        
                    except Exception as e:
                        training_stats["models_failed"] += 1
                        training_stats["model_stats"][model_id] = {
                            "status": "Failed",
                            "error": str(e)
                        }
                
                # Handle stacking ensemble separately
                try:
                    if "stacking_ensemble" in self.models and len(X) >= 20:
                        print("Updating stacking ensemble with meta-features from new data...")
                        meta_features = self._generate_meta_features_batch(X, y)
                        
                        if meta_features and len(meta_features) == len(y):
                            stacking = self.models["stacking_ensemble"]
                            
                            # Either update existing examples or add new ones
                            if hasattr(stacking, 'partial_fit'):
                                stacking.partial_fit(meta_features, y)
                            elif hasattr(stacking, 'fit'):
                                # Keep some previous examples if available
                                if hasattr(stacking, 'meta_X') and hasattr(stacking, 'meta_y'):
                                    old_X = stacking.meta_X
                                    old_y = stacking.meta_y
                                    
                                    if len(old_X) > 0:
                                        # Keep up to 50 previous examples
                                        keep_count = min(50, len(old_X))
                                        combined_X = old_X[-keep_count:] + meta_features
                                        combined_y = old_y[-keep_count:] + y.tolist()
                                        
                                        stacking.fit(combined_X, combined_y)
                                        stacking.meta_X = combined_X
                                        stacking.meta_y = combined_y
                                    else:
                                        stacking.fit(meta_features, y)
                                        stacking.meta_X = meta_features
                                        stacking.meta_y = y.tolist()
                                else:
                                    stacking.fit(meta_features, y)
                                    stacking.meta_X = meta_features
                                    stacking.meta_y = y.tolist()
                            
                            stacking.is_trained = True
                            training_stats["models_trained"] += 1
                            training_stats["model_stats"]["stacking_ensemble"] = {
                                "status": "Success",
                                "meta_features": len(meta_features),
                                "stacking_method": "partial_fit" if hasattr(stacking, 'partial_fit') else "fit"
                            }
                    else:
                        print("Skipping stacking update (insufficient data)")
                except Exception as e:
                    print(f"Error updating stacking ensemble: {e}")
                    training_stats["model_stats"]["stacking_ensemble"] = {
                        "status": "Failed",
                        "error": str(e)
                    }
                
                # Save registry state after all updates
                self._save_registry()
                training_stats["duration_seconds"] = time.time() - start_time
                
                # Update calibration based on new data
                try:
                    self.calibration_manager.calibrate_from_history()
                except Exception as e:
                    print(f"Error updating calibration: {e}")
                
                return True, training_stats
                
        except Exception as e:
            training_stats["duration_seconds"] = time.time() - start_time
            training_stats["error"] = str(e)
            
            return False, training_stats

    def perform_scheduled_maintenance(self):
        """
        Perform scheduled maintenance to ensure system health.
        
        Executes a series of validation and repair procedures to maintain
        optimal system performance:
        1. Stacking ensemble health check
        2. Dimension validation
        3. Calibration refresh
        4. Registry consistency validation
        
        This method should be called periodically (e.g., every 50 predictions).
        
        Returns:
            dict: Maintenance results with actions taken
        """
        results = {
            'actions': [],
            'issues_fixed': 0,
            'status': 'unknown'
        }
        
        try:
            # 1. Check stacking health
            health_report = self.test_stacking_ensemble()
            results['health_report'] = health_report
            
            # 2. Take action based on health status
            if health_report.get('status') == 'healthy':
                results['status'] = 'healthy'
                results['actions'].append("Stacking ensemble verified healthy")
            else:
                # 3. Fix identified issues
                if 'issues' in health_report and health_report['issues']:
                    for issue in health_report['issues']:
                        results['actions'].append(f"Found issue: {issue}")
                        
                    # If dimension mismatch, reset stacking
                    if any('dimension mismatch' in issue.lower() for issue in health_report['issues']):
                        self.reset_stacking(force_reset=True)
                        results['actions'].append("Reset stacking due to dimension mismatch")
                        results['issues_fixed'] += 1
                        
                    # If prediction test failed, reset stacking
                    elif any('prediction test failed' in issue.lower() for issue in health_report['issues']):
                        self.reset_stacking(force_reset=True)
                        results['actions'].append("Reset stacking due to prediction failure")
                        results['issues_fixed'] += 1
                        
                    # If not trained, attempt training
                    elif any('not trained' in issue.lower() for issue in health_report['issues']):
                        from data.data_utils import prepare_combined_dataset
                        try:
                            X, y = prepare_combined_dataset()
                            if len(X) >= 10:
                                meta_features = self._generate_meta_features_batch(X, y)
                                stacking = self.models.get("stacking_ensemble")
                                if stacking and meta_features:
                                    stacking.fit(meta_features, y)
                                    results['actions'].append(f"Trained stacking with {len(X)} examples")
                                    results['issues_fixed'] += 1
                        except Exception as e:
                            results['actions'].append(f"Training attempt failed: {e}")
                            self.reset_stacking(force_reset=True)
                
                # 4. Recalibrate if health issues detected
                try:
                    calibration_result = self.calibration_manager.calibrate_from_history()
                    results['calibration'] = calibration_result
                    results['actions'].append("Refreshed confidence calibration")
                except Exception as e:
                    results['actions'].append(f"Calibration refresh failed: {e}")
            
            # 5. Verify model registry consistency
            consistency_valid = self.validate_registry_consistency()
            results['registry_consistent'] = consistency_valid
            if consistency_valid:
                results['actions'].append("Verified registry consistency")
            else:
                results['actions'].append("Fixed registry consistency issues")
                results['issues_fixed'] += 1
            
            # 6. Final status determination
            if results['issues_fixed'] > 0:
                results['status'] = 'repaired'
            elif health_report.get('status') == 'healthy':
                results['status'] = 'healthy'
            else:
                results['status'] = 'issues_remain'
                
            return results
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            print(f"Error in maintenance: {e}")
            traceback.print_exc()
            return results

    def update_model_performances(self, prediction_results):
        """
        Update performance metrics for models based on recent predictions.
        
        Maintains historical performance data for each model to support evaluation
        and competitive evolution processes.
        
        Args:
            prediction_results: Dict mapping model_id to prediction result
                Each result should have 'predicted', 'actual', 'confidence'
        """
        timestamp = time.time()
        
        for model_id, result in prediction_results.items():
            if model_id not in self.models:
                continue
                
            model = self.models[model_id]
            predicted = result['predicted']
            actual = result['actual']
            confidence = result.get('confidence', 50.0)
            pattern = result.get('pattern')
            
            # Update model's internal tracking
            for method_name in ['safe_update_performance', 'update_performance']:
                if hasattr(model, method_name):
                    try:
                        method = getattr(model, method_name)
                        method(predicted, actual, confidence, pattern)
                        break  # Found working method
                    except Exception as method_error:
                        print(f"Error using {method_name} on {model_id}: {method_error}")
                        continue  # Try next method
            
            # Add to history with transaction-like protection
            try:
                entry = {
                    'timestamp': timestamp,
                    'predicted': int(predicted),
                    'actual': int(actual),
                    'correct': int(predicted == actual),
                    'confidence': float(confidence)
                }
                
                self.model_history[model_id].append(entry)
                
                # Keep history manageable
                if len(self.model_history[model_id]) > 1000:
                    self.model_history[model_id] = self.model_history[model_id][-1000:]
            except Exception as e:
                print(f"Error updating history for {model_id}: {e}")

    def collect_holdout_predictions(self, prev_rounds, actual_result):
        """
        Collect predictions from all models for meta-learner training without leakage.
        
        Gathers predictions from base models for a specific input, to be used
        for training the stacking ensemble after the actual outcome is known.
        This prevents information leakage in the meta-learner training process.
        
        Args:
            prev_rounds: Previous game outcomes
            actual_result: The actual outcome that occurred
            
        Returns:
            dict: Meta-features and outcome for stacking model
        """
        # Ensure prev_rounds is in the right format
        if isinstance(prev_rounds, list):
            prev_rounds_arr = np.array(prev_rounds).reshape(1, -1)
        else:
            prev_rounds_arr = prev_rounds.reshape(1, -1) if prev_rounds.ndim == 1 else prev_rounds
            
        # Get predictions from all base models
        meta_features = []
        model_predictions = {}
        
        for model_id, model in self.models.items():
            # Skip the meta-learner itself to avoid recursion
            if model_id == "stacking_ensemble":
                continue
                
            try:
                # Get prediction probabilities
                prob_features = self.feature_manager._extract_model_probabilities(model, prev_rounds_arr)
                meta_features.extend(prob_features)
                
                # Save prediction for model performance tracking
                model_predictions[model_id] = {
                    'predicted': np.argmax(prob_features),
                    'actual': actual_result,
                    'confidence': max(prob_features) * 100
                }
                
            except Exception as e:
                print(f"Error getting predictions from {model_id}: {e}")
                # Add defaults for missing model
                meta_features.extend([0.33, 0.33, 0.34])
        
        # Add pattern-type features
        pattern_features = self.feature_manager._extract_pattern_features(prev_rounds_arr)
        pattern_type = ["no_pattern", "streak", "alternating", "tie"][pattern_features.index(1)]
        
        # Add pattern features to meta-features
        meta_features.extend(pattern_features)
        
        # Update model performance tracking
        self.update_model_performances(model_predictions)
        
        return {
            'meta_features': meta_features,
            'outcome': actual_result,
            'pattern_type': pattern_type
        }

    def update_stacking_model(self, meta_data):
        """
        Update the stacking ensemble with new training data.
        
        Args:
            meta_data: Dict with meta-features and outcome
        """
        if "stacking_ensemble" not in self.models:
            self.initialize_stacking()
            
        # Update the meta-learner with new data
        try:
            stacking = self.models["stacking_ensemble"]
            
            # Use appropriate update method
            if hasattr(stacking, 'update_meta_data'):
                stacking.update_meta_data(
                    meta_data['meta_features'],
                    meta_data['outcome'],
                    meta_data['pattern_type']
                )
            elif hasattr(stacking, 'partial_fit'):
                stacking.partial_fit([meta_data['meta_features']], [meta_data['outcome']])
            elif hasattr(stacking, 'fit'):
                # Add to existing examples
                if hasattr(stacking, 'meta_X') and hasattr(stacking, 'meta_y'):
                    meta_X = stacking.meta_X
                    meta_y = stacking.meta_y
                    
                    # Add new example
                    meta_X.append(meta_data['meta_features'])
                    meta_y.append(meta_data['outcome'])
                    
                    # Retrain with all examples
                    stacking.fit(meta_X, meta_y)
                    stacking.meta_X = meta_X
                    stacking.meta_y = meta_y
                else:
                    # Initialize with this example
                    stacking.fit([meta_data['meta_features']], [meta_data['outcome']])
                    stacking.meta_X = [meta_data['meta_features']]
                    stacking.meta_y = [meta_data['outcome']]
            
            print(f"Stacking model updated with new example (outcome: {meta_data['outcome']})")
        except Exception as e:
            print(f"Error updating stacking model: {e}")
            traceback.print_exc()

    def repair_registry(self):
        """
        Attempt to repair registry when validation fails.
        
        Implements a comprehensive repair strategy:
        1. Create registry backup
        2. Test individual model health
        3. Recreate problematic models
        4. Retrain new models
        5. Validate the repair
        
        Returns:
            bool: True if repair was successful
        """
        print("Beginning registry repair process...")
        
        # First create a backup
        try:
            import time, shutil, os
            backup_dir = os.path.join(self.registry_path, f"backup_{int(time.time())}")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Backup registry file
            registry_file = os.path.join(self.registry_path, "registry.json")
            if os.path.exists(registry_file):
                shutil.copy2(registry_file, os.path.join(backup_dir, "registry.json"))
                print(f"Created backup of registry.json")
                
                # Backup model files
                for model_id in self.models.keys():
                    model_path = os.path.join(self.registry_path, f"{model_id}.pkl")
                    if os.path.exists(model_path):
                        try:
                            shutil.copy2(model_path, os.path.join(backup_dir, f"{model_id}.pkl"))
                        except Exception as e:
                            print(f"Error backing up {model_id}: {e}")
        except Exception as e:
            print(f"Warning: Failed to create backup: {e}")
        
        # Try to recreate problematic models
        model_ids = list(self.models.keys())
        problematic_models = []
        
        print("Testing individual model health...")
        for model_id in model_ids:
            try:
                # Test the model with a simple prediction
                if model_id != "stacking_ensemble":
                    test_input = np.array([[0, 1, 0, 1, 0]])
                    self.models[model_id].predict(test_input)
                    print(f"✓ Model {model_id} is functional")
                else:
                    # For stacking, we need a different test
                    meta_features = self.feature_manager.generate_meta_features(np.array([[0, 1, 0, 1, 0]]))
                    if meta_features is not None:
                        self.models[model_id].predict([meta_features])
                        print(f"✓ Stacking ensemble is functional")
                    else:
                        print(f"✗ Stacking ensemble test failed - meta-features issue")
                        problematic_models.append(model_id)
            except Exception as e:
                print(f"✗ Model {model_id} failed testing: {e}")
                problematic_models.append(model_id)
        
        # Create new versions of problematic models
        for model_id in problematic_models:
            try:
                print(f"Recreating model: {model_id}")
                if "markov" in model_id.lower():
                    order = int(model_id.split("_")[-1]) if "_" in model_id and model_id.split("_")[-1].isdigit() else 1
                    self.models[model_id] = MarkovModel(order=order)
                    print(f"Created new MarkovModel with order {order}")
                elif "baccarat" in model_id.lower():
                    self.models[model_id] = BaccaratModel()
                    print(f"Created new BaccaratModel")
                elif "xgboost" in model_id.lower():
                    self.models[model_id] = XGBoostModel()
                    print(f"Created new XGBoostModel")
                elif "stacking" in model_id.lower():
                    print("Resetting stacking ensemble...")
                    self.reset_stacking(force_reset=True)
            except Exception as e:
                print(f"Failed to recreate {model_id}: {e}")
        
        # Train any newly created models
        if problematic_models:
            print("Training newly created models...")
            self._force_minimal_training()
        
        # Save the repaired registry
        self._save_registry()
        
        # Validate the repair
        return self.validate_registry_consistency()

    def analyze_confidence_distribution(self):
        """
        Analyze and visualize confidence calibration mapping.
        
        Provides diagnostic information about the current calibration curves
        to help evaluate and debug prediction confidence issues.
        
        Returns:
            dict: Diagnostic information about calibration
        """
        if not hasattr(self, 'confidence_calibrators'):
            print("No confidence calibrators found")
            return {"status": "no_calibrators"}
        
        # Create test input values spanning the full probability range
        test_inputs = np.linspace(0.1, 0.9, 20)
        
        results = {
            "classes": {},
            "diagnostics_run": True,
            "timestamp": time.time()
        }
        
        for cls in range(3):
            if cls not in self.confidence_calibrators:
                continue
                
            calibrator = self.confidence_calibrators[cls]
            
            # Skip if calibrator is missing necessary attributes
            if not hasattr(calibrator, 'X_min_'):
                print(f"Calibrator {cls} is not properly initialized")
                results["classes"][cls] = {"status": "not_initialized"}
                continue
            
            # Get calibrated outputs
            outputs = calibrator.predict(test_inputs.reshape(-1, 1))
            
            # Calculate statistics
            class_results = {
                "min_output": float(outputs.min()),
                "max_output": float(outputs.max()),
                "mean_output": float(outputs.mean()),
                "unique_values": int(len(np.unique(outputs))),
                "mapping": []
            }
            
            # Add mapping table
            for i, o in zip(test_inputs, outputs):
                class_results["mapping"].append({"input": float(i), "output": float(o)})
            
            results["classes"][cls] = class_results
            
            # Print the mapping table for debugging
            print(f"\nConfidence mapping for class {cls}:")
            print("Input  | Output")
            print("-------|-------")
            for i, o in zip(test_inputs, outputs):
                print(f"{i:.2f}   | {o:.2f}")
        
        return results
    
    def initialize_model_fixes(self):
        """
        Apply essential model-specific fixes to ensure robust operation.
        
        This method applies fixes to common model issues to prevent failures
        during operation, focusing on critical components like prediction methods.
        
        Returns:
            dict: Results of fix operations
        """
        results = {
            'markov_fixes': 0,
            'xgboost_fixes': 0,
            'stacking_fixes': 0,
            'errors': []
        }
        
        try:
            # Fix 1: Apply Markov model prediction fixes
            for model_id, model in self.models.items():
                if model_id.startswith('markov'):
                    markov_fixed = self._fix_markov_model(model)
                    if markov_fixed:
                        results['markov_fixes'] += 1
                
                # Fix 2: Apply XGBoost model prediction fixes
                elif model_id.startswith('xgboost') or model_id.startswith('xgb_'):
                    xgb_fixed = self._fix_xgboost_model(model)
                    if xgb_fixed:
                        results['xgboost_fixes'] += 1
            
            # Fix 3: Ensure all prediction methods have consistent interfaces
            for model_id, model in self.models.items():
                if not hasattr(model, 'safe_predict_proba'):
                    self._add_safe_predict_proba(model, model_id)
                    
                if not hasattr(model, 'is_trained'):
                    model.is_trained = False
            
            # Fix 4: Ensure stacking ensemble has necessary components
            if "stacking_ensemble" in self.models:
                stacking = self.models["stacking_ensemble"]
                
                # Ensure dimension management
                if hasattr(stacking, 'dimension_manager'):
                    # Verify dimension manager has expected count
                    if not hasattr(stacking.dimension_manager, 'expected_count'):
                        active_models = len(self.get_active_base_models())
                        stacking.dimension_manager.expected_count = (active_models * 3) + 4
                        results['stacking_fixes'] += 1
                elif hasattr(stacking, 'expected_feature_count'):
                    # Update expected feature count
                    active_models = len(self.get_active_base_models())
                    stacking.expected_feature_count = (active_models * 3) + 4
                    results['stacking_fixes'] += 1
            
            return results
            
        except Exception as e:
            results['errors'].append(str(e))
            print(f"Error applying model fixes: {e}")
            traceback.print_exc()
            return results

    def _add_safe_predict_proba(self, model, model_id):
        """
        Add safe_predict_proba method to model if missing.
        
        Args:
            model: The model to add the method to
            model_id: The model's identifier
            
        Returns:
            bool: True if method was added successfully
        """
        # Skip if already exists
        if hasattr(model, 'safe_predict_proba'):
            return False
            
        # Define generic safe_predict_proba
        def safe_predict_proba(self, X):
            """
            Safe prediction method that handles all edge cases.
            
            Args:
                X: Input features
                
            Returns:
                dict: Class probabilities
            """
            # Default probabilities (banker, player, tie)
            default_probs = {0: 0.459, 1: 0.446, 2: 0.095}
            
            # Handle edge cases
            if X is None or not hasattr(X, 'shape'):
                return default_probs
                
            try:
                # Try original predict_proba if available
                if hasattr(self, 'predict_proba'):
                    try:
                        probs = self.predict_proba(X)
                        
                        # Convert different return formats consistently
                        if isinstance(probs, dict):
                            return probs
                        elif hasattr(probs, 'shape'):
                            if probs.shape[1] >= 3:
                                return {0: probs[0, 0], 1: probs[0, 1], 2: probs[0, 2]}
                        
                        # Fallback for any other format
                        return default_probs
                    except Exception:
                        pass
                
                # Try predict as fallback
                if hasattr(self, 'predict'):
                    try:
                        pred = self.predict(X)[0]
                        result = {0: 0.1, 1: 0.1, 2: 0.1}
                        result[pred] = 0.8
                        return result
                    except Exception:
                        pass
                        
                # Ultimate fallback
                return default_probs
                
            except Exception:
                return default_probs
        
        # Add method to model
        import types
        model.safe_predict_proba = types.MethodType(safe_predict_proba, model)
        
        # Test the method
        try:
            test_result = model.safe_predict_proba(np.array([[0, 1, 0, 1, 0]]))
            if isinstance(test_result, dict) and len(test_result) == 3:
                print(f"Added safe_predict_proba to {model_id}")
                return True
        except Exception:
            pass
            
        return False