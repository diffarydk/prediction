"""
Enhanced XGBoost model implementation for Baccarat Prediction System.
Provides gradient boosting capabilities with adaptive hyperparameters,
improved calibration, and robust error handling.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import pickle
import os
import time
import traceback
from collections import defaultdict, Counter

from models.base_model import BaseModel
from models.baccarat_model import BaccaratModel  # For feature engineering reuse

class XGBoostModel(BaseModel):
    """
    Enhanced XGBoost-based model for Baccarat prediction with:
    - Adaptive hyperparameters
    - Robust error handling
    - Proper pattern tracking
    - Confidence calibration
    - Feature importance analysis
    - Memory-efficient incremental learning
    """
        
    def __init__(self, random_state=42, variant_id=None):
        """Initialize XGBoost model with enhanced parameters"""
        super().__init__()
        # Track model variant for competition purposes
        self.variant_id = variant_id or f"xgb_base_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Check XGBoost version for compatibility
        try:
            import xgboost as xgb
            self.xgb_version = xgb.__version__
            print(f"Initializing XGBoost model with version {self.xgb_version}")
        except (ImportError, AttributeError):
            self.xgb_version = "unknown"
            print("Warning: Could not detect XGBoost version")
        
        # Enhanced default parameters for XGBoost
        self.default_params = {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 4,
            'min_child_weight': 2,
            'gamma': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 1.0,
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': random_state
        }
        
        # Initialize model with default parameters
        try:
            self.model = xgb.XGBClassifier(**self.default_params)
        except Exception as e:
            print(f"Error initializing XGBoost model: {e}")
            traceback.print_exc()
            # Create fallback model with minimal parameters
            self.model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3
            )
        
        # Enhanced preprocessing
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.model_type = "xgboost"
        self.columns = None
        
        # Explicitly initialize all required attributes
        self.feature_mask = None
        self.feature_cols = None
        self.is_calibrated = False
        self.trained_columns = None
        self.important_features = None
        
        # Performance tracking
        self.recent_predictions = []
        self.recent_correct = 0
        self.recent_total = 0
        
        # Pattern tracking with proper initialization
        self.pattern_performance = defaultdict(self._default_pattern_stats)

    def _default_pattern_stats(self):
        """Return default pattern statistics dictionary with enhanced tracking."""
        return {
            "correct": 0, 
            "total": 0,
            "last_seen": time.time(),
            "last_correct": 0,
            "confidence_sum": 0,
            "recent_correct": 0,
            "recent_total": 0
        }
            
    def _add_derived_features(self, X):
        """
        Enhanced feature engineering with consistent feature generation
        between training and prediction phases.
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f'Prev_{i+1}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()
        
        # Apply BaccaratModel's standard feature engineering
        temp_model = BaccaratModel()
        X_enhanced = temp_model._add_derived_features(X_df)
        
        # If we have stored column information from training, use only those columns
        if hasattr(self, 'trained_columns') and self.trained_columns is not None:
            # Keep only columns that were available during training
            X_enhanced = X_enhanced[[col for col in self.trained_columns if col in X_enhanced.columns]]
            
            # If we're missing any columns, add zeros for them
            for col in self.trained_columns:
                if col not in X_enhanced.columns:
                    X_enhanced[col] = 0.0
                    
            # Ensure columns are in the same order as during training
            X_enhanced = X_enhanced[self.trained_columns]
            
            return X_enhanced
        
        # If we're in training mode or don't have stored columns, 
        # just return the enhanced features
        return X_enhanced
    
    def _calculate_entropy(self, row):
        """Calculate information entropy of sequence as a feature."""
        # Count occurrences of each outcome
        counts = Counter(row)
        total = len(row)
        
        # Calculate entropy
        entropy = 0
        for outcome in [0, 1, 2]:
            prob = counts.get(outcome, 0) / total
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _detect_dragon(self, row):
        """Detect 'dragon' pattern (streak of banker or player)."""
        for outcome in [0, 1]:  # Check banker and player only
            streak = 0
            max_streak = 0
            
            for val in row:
                if val == outcome:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
            
            if max_streak >= 3:
                return 1
        
        return 0
    
    def _detect_panda(self, row):
        """Detect 'panda' pattern (alternating banker/player)."""
        alternating_count = 0
        
        for i in range(len(row) - 1):
            if (row.iloc[i] == 0 and row.iloc[i+1] == 1) or (row.iloc[i] == 1 and row.iloc[i+1] == 0):
                alternating_count += 1
        
        return 1 if alternating_count >= 3 else 0
        
    def predict(self, X):
        """
        Make predictions using the trained XGBoost model with enhanced error handling.
        """
        try:
            # Ensure model is trained
            if not self.is_trained:
                print("Warning: Model not trained yet, using fallback prediction")
                return np.array([0] * (X.shape[0] if hasattr(X, 'shape') else 1))
            
            # Add derived features
            X_enhanced = self._add_derived_features(X)
            
            # Apply scaling
            X_scaled = self.scaler.transform(X_enhanced)
            
            # Apply feature mask if available and valid
            if hasattr(self, 'feature_mask') and self.feature_mask is not None:
                if len(self.feature_mask) == X_scaled.shape[1]:
                    X_scaled = X_scaled[:, self.feature_mask]
            
            # Make prediction
            return self.model.predict(X_scaled)
        except Exception as e:
            print(f"Error in XGBoost prediction: {e}")
            traceback.print_exc()
            
            # Return default prediction with banker bias (more common in baccarat)
            return np.array([0] * (X.shape[0] if hasattr(X, 'shape') else 1))
        
    def predict_proba(self, X):
        """
        Get calibrated class probabilities with improved accuracy and conservative estimates.
        """

        if not hasattr(self, 'is_calibrated'):
            self.is_calibrated = False
        if not hasattr(self, 'calibrators'):
            from sklearn.isotonic import IsotonicRegression
            self.calibrators = {
                0: IsotonicRegression(out_of_bounds='clip'),
                1: IsotonicRegression(out_of_bounds='clip'),
                2: IsotonicRegression(out_of_bounds='clip')
            }
            
        try:
            # Ensure model is trained
            if not self.is_trained:
                print("Warning: Model not trained yet, using default probabilities")
                # Default probabilities with slight banker bias
                shape = (X.shape[0] if hasattr(X, 'shape') and len(X) > 0 else 1, 3)
                return np.array([[0.45, 0.45, 0.1]] * shape[0])
            
            # Add derived features
            X_enhanced = self._add_derived_features(X)
            
            # Apply scaling
            X_scaled = self.scaler.transform(X_enhanced)
            
            # Apply feature mask if available and valid
            if hasattr(self, 'feature_mask') and self.feature_mask is not None:
                if len(self.feature_mask) == X_scaled.shape[1]:
                    X_scaled = X_scaled[:, self.feature_mask]
            
            # Get raw probabilities
            raw_probs = self.model.predict_proba(X_scaled)
            
            # Apply calibration if available
            if self.is_calibrated:
                return self._apply_calibration(raw_probs)
            else:
                # Apply conservative scaling for games of chance
                # Shift probabilities toward uniform distribution
                uniform_probs = np.array([[1/3, 1/3, 1/3]] * raw_probs.shape[0])
                conservative_probs = 0.7 * raw_probs + 0.3 * uniform_probs
                
                # Lower tie probabilities (they tend to be overestimated)
                if conservative_probs.shape[1] >= 3:
                    conservative_probs[:, 2] = conservative_probs[:, 2] * 0.8
                    
                    # Renormalize
                    row_sums = conservative_probs.sum(axis=1).reshape(-1, 1)
                    conservative_probs = conservative_probs / row_sums
                
                return conservative_probs
        except Exception as e:
            print(f"Error in XGBoost probability prediction: {e}")
            traceback.print_exc()
            
            # Return default probabilities with slight banker bias
            shape = (X.shape[0] if hasattr(X, 'shape') and len(X) > 0 else 1, 3)
            return np.array([[0.45, 0.45, 0.1]] * shape[0])
    
    def _apply_calibration(self, raw_probs):
        """Apply probability calibration to raw model outputs."""
        calibrated_probs = np.copy(raw_probs)
        
        # Apply each class calibrator
        for cls in range(min(3, raw_probs.shape[1])):
            if hasattr(self.calibrators[cls], 'predict'):
                try:
                    # Extract probabilities for this class
                    cls_probs = raw_probs[:, cls]
                    
                    # Apply calibration
                    calibrated_probs[:, cls] = self.calibrators[cls].predict(cls_probs)
                except Exception as e:
                    print(f"Error in calibration for class {cls}: {e}")
        
        # Renormalize to ensure probabilities sum to 1
        row_sums = calibrated_probs.sum(axis=1).reshape(-1, 1)
        calibrated_probs = calibrated_probs / row_sums
        
        return calibrated_probs
            
    def fit(self, X, y):
        """Train the XGBoost model with enhanced feature engineering and error handling."""
        try:
            start_time = time.time()
            
            # Add derived features
            X_enhanced = self._add_derived_features(X)
            self.feature_cols = X_enhanced.columns.tolist()
            
            # Apply feature scaling
            X_scaled = self.scaler.fit_transform(X_enhanced)
            
            # Apply feature selection if dataset is large enough
            if X_scaled.shape[0] > 50 and X_scaled.shape[1] > 10:
                X_selected, feature_mask = self._select_features(X_scaled, y, X_enhanced.columns)
                self.feature_mask = feature_mask
                print(f"Selected {sum(feature_mask)} features out of {len(feature_mask)}")
            else:
                X_selected = X_scaled
                self.feature_mask = None
            
            # Simple train/validation split without extra parameters
            if len(X) >= 50:
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_selected, y, test_size=0.2, stratify=y, random_state=42
                )
            else:
                X_train, y_train = X_selected, y
            
            # Basic fit without any extra parameters - most compatible approach
            self.model.fit(X_train, y_train)
            
            # Calculate and store feature importance
            if hasattr(self.model, 'feature_importances_'):
                # Get selected feature names
                if self.feature_mask is not None:
                    selected_features = [col for i, col in enumerate(self.feature_cols) if self.feature_mask[i]]
                else:
                    selected_features = self.feature_cols
                
                # Store feature importance
                self.feature_importance = sorted(
                    zip(selected_features, self.model.feature_importances_),
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Print top features
                print("Top 5 important features:")
                for feature, importance in self.feature_importance[:5]:
                    print(f"  {feature}: {importance:.4f}")
            
            # Set training metadata
            self.is_trained = True
            self.last_train_time = time.time()
            self.train_sample_count = len(X)
            
            # Track training duration
            duration = time.time() - start_time
            print(f"XGBoost model trained in {duration:.2f} seconds with {len(X)} samples")
            
            return self
        except Exception as e:
            print(f"Error training XGBoost model: {e}")
            traceback.print_exc()
            return self
    
    def _select_features(self, X, y, column_names):
        """
        Perform feature selection to improve model efficiency.
        Returns selected data and feature mask.
        """
        # For small datasets, skip feature selection
        if X.shape[0] < 30 or X.shape[1] < 10:
            return X, np.ones(X.shape[1], dtype=bool)
        
        try:
            # Use XGBoost's built-in feature importance for selection
            # Train a quick model to get feature importance
            temp_model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                objective='multi:softprob',
                num_class=3,
                random_state=42
            )
            temp_model.fit(X, y)
            
            # Get feature importance
            importances = temp_model.feature_importances_
            
            # Select features above threshold (keep at least 5 features)
            importance_threshold = np.percentile(importances, 70)  # Keep top 30%
            feature_mask = importances >= importance_threshold
            
            # Ensure we keep at least 5 features or 30% of features
            min_features = max(5, int(X.shape[1] * 0.3))
            if sum(feature_mask) < min_features:
                # Take top min_features by importance
                top_indices = np.argsort(importances)[-min_features:]
                feature_mask = np.zeros_like(feature_mask)
                feature_mask[top_indices] = True
            
            # Print selected features
            selected_features = [col for i, col in enumerate(column_names) if feature_mask[i]]
            print(f"Selected features: {', '.join(selected_features[:5])}{'...' if len(selected_features) > 5 else ''}")
            
            return X[:, feature_mask], feature_mask
        except Exception as e:
            print(f"Error in feature selection: {e}")
            # Return all features as fallback
            return X, np.ones(X.shape[1], dtype=bool)
    
    def _train_calibrators(self, X, y):
        """Train probability calibration models for each class."""
        try:
            # Get raw probabilities
            raw_probs = self.model.predict_proba(X)
            
            # For each class, train a calibrator
            for cls in range(min(3, raw_probs.shape[1])):
                # Extract probabilities for this class
                cls_probs = raw_probs[:, cls]
                
                # Create binary targets (1 if this class, 0 otherwise)
                binary_targets = (y == cls).astype(int)
                
                # Only calibrate if we have both positive and negative examples
                if np.sum(binary_targets) >= 3 and np.sum(binary_targets == 0) >= 3:
                    # Train isotonic regression calibrator
                    self.calibrators[cls].fit(cls_probs, binary_targets)
                else:
                    print(f"Insufficient data to calibrate class {cls}")
            
            self.is_calibrated = True
            print("Probability calibration trained successfully")
        except Exception as e:
            print(f"Error training calibrators: {e}")
            self.is_calibrated = False
    
    def update_model(self, prev_rounds, actual_result, confidence=None, pattern=None):
        """
        Enhanced incremental learning with pattern type handling and error recovery.
        """
        try:
            # First update performance tracking
            self.update_performance(predicted=None, actual=actual_result, 
                                   confidence=confidence, pattern=pattern)
            
            # Store data in update buffer
            if not hasattr(self, 'update_buffer'):
                self.update_buffer = {'X': [], 'y': [], 'threshold': 10, 'last_update': time.time()}
            
            self.update_buffer['X'].append(prev_rounds)
            self.update_buffer['y'].append(actual_result)
            
            # Perform batch update when buffer reaches threshold or time threshold exceeded
            current_time = time.time()
            time_threshold = 3600  # 1 hour
            buffer_full = len(self.update_buffer['X']) >= self.update_buffer['threshold']
            time_exceeded = (current_time - self.update_buffer.get('last_update', 0)) > time_threshold
            
            if buffer_full or (time_exceeded and len(self.update_buffer['X']) > 5):
                # Prepare data
                X = np.array(self.update_buffer['X'])
                y = np.array(self.update_buffer['y'])
                
                # Convert to correct shape if needed
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                
                # Add features
                X_df = pd.DataFrame(X, columns=[f'Prev_{i+1}' for i in range(X.shape[1])])
                X_enhanced = self._add_derived_features(X_df)
                X_scaled = self.scaler.transform(X_enhanced)
                
                # Apply feature mask if available and valid
                if hasattr(self, 'feature_mask') and self.feature_mask is not None:
                    if len(self.feature_mask) == X_scaled.shape[1]:
                        X_scaled = X_scaled[:, self.feature_mask]
                
                # Update model with new batch (incremental learning)
                if hasattr(self.model, 'n_estimators'):
                    # Adjust tree count dynamically
                    current = self.model.n_estimators
                    # Add more trees for larger batches, fewer for smaller
                    new_trees = max(5, min(20, len(y) // 2))
                    self.model.n_estimators = current + new_trees
                
                # Try to fit with warm_start if available
                try:
                    if hasattr(self.model, 'warm_start'):
                        # Set warm_start to True temporarily
                        old_warm_start = self.model.warm_start
                        self.model.warm_start = True
                        self.model.fit(X_scaled, y)
                        self.model.warm_start = old_warm_start
                    else:
                        # Just do a regular fit
                        self.model.fit(X_scaled, y)
                    
                    # Update calibrators if enough data
                    if len(y) >= 10:
                        self._update_calibrators(X_scaled, y)
                    
                    print(f"XGBoost model updated with {len(y)} new samples")
                except Exception as fit_error:
                    print(f"Error in incremental update fit: {fit_error}")
                    # Fallback: try full retraining if incremental update fails
                    try:
                        self.model.fit(X_scaled, y)
                        print("Fallback to full retraining successful")
                    except Exception as retrain_error:
                        print(f"Full retraining also failed: {retrain_error}")
                
                # Clear buffer after update
                self.update_buffer = {
                    'X': [], 
                    'y': [], 
                    'threshold': self.update_buffer['threshold'],
                    'last_update': current_time
                }
                
                # Adjust threshold based on update success
                # More frequent updates if successful, less if failed
                if 'fit_error' not in locals():
                    # Successful update: gradually reduce threshold to update more frequently
                    self.update_buffer['threshold'] = max(5, self.update_buffer['threshold'] - 1)
                else:
                    # Failed update: increase threshold to update less frequently
                    self.update_buffer['threshold'] = min(30, self.update_buffer['threshold'] + 2)
                
                return True
            
            return False
        except Exception as e:
            print(f"Error in XGBoost incremental update: {str(e)}")
            traceback.print_exc()
            return False

    def _update_calibrators(self, X, y):
        """Update probability calibration with new data."""
        try:
            # Get raw probabilities
            raw_probs = self.model.predict_proba(X)
            
            # Update each calibrator with new data
            for cls in range(min(3, raw_probs.shape[1])):
                # Extract probabilities for this class
                cls_probs = raw_probs[:, cls]
                
                # Create binary targets
                binary_targets = (y == cls).astype(int)
                
                # Only update if we have both positive and negative examples
                if np.sum(binary_targets) > 0 and np.sum(binary_targets == 0) > 0:
                    try:
                        # For isotonic regression, we can just refit
                        self.calibrators[cls].fit(cls_probs, binary_targets)
                    except Exception as e:
                        print(f"Error updating calibrator for class {cls}: {e}")
            
            self.is_calibrated = True
        except Exception as e:
            print(f"Error updating calibrators: {e}")

    def update_performance(self, predicted, actual, confidence=None, pattern=None):
        """
        Track recent performance with enhanced pattern handling and metrics.
        
        Args:
            predicted: Predicted outcome (can be None if not available)
            actual: Actual outcome
            confidence: Prediction confidence (optional)
            pattern: Pattern type detected (optional)
        """
        # Convert predicted/actual to proper value type
        if predicted is not None:
            predicted = int(predicted) if not isinstance(predicted, bool) else predicted
        if actual is not None:
            actual = int(actual) if not isinstance(actual, bool) else actual
        
        # Check if prediction was correct
        correct = (predicted == actual) if predicted is not None else False
        
        # Prepare tracking data
        tracking_data = {
            'timestamp': time.time(),
            'predicted': predicted,
            'actual': actual,
            'correct': correct,
            'confidence': confidence,
        }
        
        # Only include pattern if it's valid
        if pattern is not None:
            tracking_data['pattern'] = str(pattern)  # Convert to string for consistency
        
        # Add to recent predictions list
        if not hasattr(self, 'recent_predictions'):
            self.recent_predictions = []
        self.recent_predictions.append(tracking_data)
        
        # Limit history length to prevent memory issues
        if len(self.recent_predictions) > 500:
            self.recent_predictions = self.recent_predictions[-500:]
        
        # Update simple counters
        if not hasattr(self, 'recent_total'):
            self.recent_total = 0
        if not hasattr(self, 'recent_correct'):
            self.recent_correct = 0
        
        self.recent_total += 1
        if correct:
            self.recent_correct += 1
        
        # Ensure pattern_performance is initialized
        if not hasattr(self, 'pattern_performance'):
            self.pattern_performance = defaultdict(self._default_pattern_stats)
        
        # Track pattern-specific performance with proper string handling
        if pattern is not None:
            # Convert pattern to string and ensure it's a valid key
            pattern_key = str(pattern) if pattern is not None else "none"
            
            # Track this pattern occurrence
            self.pattern_performance[pattern_key]["total"] += 1
            self.pattern_performance[pattern_key]["recent_total"] += 1
            self.pattern_performance[pattern_key]["last_seen"] = time.time()
            
            # Limit recent pattern tracking to last 20 occurrences
            if self.pattern_performance[pattern_key]["recent_total"] > 20:
                self.pattern_performance[pattern_key]["recent_total"] = 20
            
            # Track correct predictions
            if correct:
                self.pattern_performance[pattern_key]["correct"] += 1
                self.pattern_performance[pattern_key]["recent_correct"] += 1
                self.pattern_performance[pattern_key]["last_correct"] = time.time()
            
            # Track confidence if provided
            if confidence is not None:
                self.pattern_performance[pattern_key]["confidence_sum"] += confidence
        
        return correct

    def get_exponential_weighted_accuracy(self, halflife=20):
        """
        Calculate exponentially weighted accuracy, prioritizing recent results.
        
        Args:
            halflife: Number of predictions after which weight reduces by half
        
        Returns:
            float: Weighted accuracy between 0 and 1
        """
        if not hasattr(self, 'recent_predictions') or not self.recent_predictions:
            return 0.5  # Default to neutral if no history
            
        now = time.time()
        total_weight = 0
        correct_weight = 0
        
        for pred in self.recent_predictions:
            # Calculate time-based weight (more recent = higher weight)
            age = now - pred['timestamp']
            weight = 2 ** (-age / (halflife * 3600))  # Convert halflife to seconds
            
            total_weight += weight
            if pred.get('correct', False):
                correct_weight += weight
                
        if total_weight == 0:
            return 0.5
            
        return correct_weight / total_weight
    
    def get_strategy_health(self):
        """
        Comprehensive health assessment based on multiple metrics.
        
        Returns:
            dict: Health metrics including accuracy, confidence correlation, and pattern health
        """
        # Recent weighted accuracy
        exp_accuracy = self.get_exponential_weighted_accuracy()
        
        # Correlation between confidence and correctness
        conf_corr = self.get_confidence_correlation()
        
        # Pattern dissolution detection
        pattern_health = self.get_pattern_health()
        
        # Calibration quality assessment
        calibration_score = self.assess_calibration_quality()
        
        # Combine metrics into overall health score (0-1)
        health_score = 0.4 * exp_accuracy + 0.2 * conf_corr + 0.2 * pattern_health + 0.2 * calibration_score
        
        return {
            'score': health_score,
            'exp_accuracy': exp_accuracy,
            'conf_correlation': conf_corr,
            'pattern_health': pattern_health,
            'calibration_quality': calibration_score,
            'variant_id': self.variant_id,
            'is_calibrated': self.is_calibrated,
            'samples_trained': self.train_sample_count if hasattr(self, 'train_sample_count') else 0
        }
    
    def assess_calibration_quality(self):
        """Assess how well calibrated the model's probabilities are."""
        if not hasattr(self, 'recent_predictions') or len(self.recent_predictions) < 20:
            return 0.5  # Default with insufficient data
        
        # Filter predictions with confidence scores
        preds_with_conf = [p for p in self.recent_predictions 
                          if p.get('confidence') is not None and p.get('predicted') is not None]
        
        if len(preds_with_conf) < 20:
            return 0.5
        
        # Bin predictions by confidence
        bins = [0, 20, 40, 60, 80, 100]
        bin_correct = [0] * (len(bins)-1)
        bin_total = [0] * (len(bins)-1)
        
        for pred in preds_with_conf:
            conf = pred['confidence']
            correct = pred.get('correct', False)
            
            # Find appropriate bin
            for i in range(len(bins)-1):
                if bins[i] <= conf < bins[i+1]:
                    bin_total[i] += 1
                    if correct:
                        bin_correct[i] += 1
                    break
        
        # Calculate calibration error
        total_error = 0
        total_weight = 0
        
        for i in range(len(bins)-1):
            if bin_total[i] > 0:
                # Expected accuracy based on confidence
                expected_acc = (bins[i] + bins[i+1]) / 2 / 100
                
                # Actual accuracy
                actual_acc = bin_correct[i] / bin_total[i]
                
                # Weighted absolute error
                error = abs(actual_acc - expected_acc)
                weight = bin_total[i]
                
                total_error += error * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        # Convert error to quality score (0-1)
        avg_error = total_error / total_weight
        quality_score = max(0, 1 - 2 * avg_error)  # Scale error to quality
        
        return quality_score
    
    def get_confidence_correlation(self):
        """
        Measure correlation between prediction confidence and accuracy.
        Returns a value between 0 and 1 (higher is better).
        """
        if not hasattr(self, 'recent_predictions') or len(self.recent_predictions) < 10:
            return 0.5  # Default when insufficient data
            
        # Filter predictions that have confidence scores
        preds_with_conf = [p for p in self.recent_predictions if p.get('confidence') is not None]
        
        if len(preds_with_conf) < 10:
            return 0.5
            
        # Calculate point-biserial correlation
        confidences = [p.get('confidence', 50) for p in preds_with_conf]
        correctness = [1 if p.get('correct', False) else 0 for p in preds_with_conf]
        
        # Early return if all predictions are the same (no correlation possible)
        if len(set(correctness)) <= 1 or len(set(confidences)) <= 1:
            return 0.5
        
        # Calculate correlation
        conf_mean = sum(confidences) / len(confidences)
        corr_mean = sum(correctness) / len(correctness)
        
        numerator = sum((c - conf_mean) * (corr - corr_mean) 
                       for c, corr in zip(confidences, correctness))
        conf_var = sum((c - conf_mean) ** 2 for c in confidences)
        corr_var = sum((corr - corr_mean) ** 2 for corr in correctness)
        
        if conf_var == 0 or corr_var == 0:
            return 0.5
            
        correlation = numerator / ((conf_var * corr_var) ** 0.5)
        
        # Normalize to 0-1 range
        return (correlation + 1) / 2
    
    def get_pattern_health(self):
        """
        Enhanced pattern health assessment that detects if patterns are dissolving.
        Returns a score between 0 and 1 (higher means patterns remain predictive).
        """
        if not hasattr(self, 'pattern_performance') or not self.pattern_performance:
            return 0.5
            
        # Calculate weighted average of pattern accuracy changes
        total_weight = 0
        weighted_health = 0
        
        for pattern, stats in self.pattern_performance.items():
            if stats["total"] < 5:  # Skip patterns with too few examples
                continue
                
            # Overall accuracy for this pattern
            overall_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.5
            
            # Recent accuracy (from counters, not recalculating)
            recent_acc = stats["recent_correct"] / stats["recent_total"] if stats["recent_total"] > 0 else 0.5
            
            # Time since last correct prediction (in hours)
            hours_since_correct = (time.time() - stats.get("last_correct", 0)) / 3600
            time_factor = 1.0 if hours_since_correct < 12 else max(0.5, 1.0 - (hours_since_correct - 12) / 24)
            
            # Combined score based on accuracy trend and recency
            pattern_score = (recent_acc / max(0.1, overall_acc)) * time_factor
            
            # Weight by sample size (more weight to common patterns)
            weight = min(stats["total"], 50)  # Cap weight to prevent dominant patterns
            total_weight += weight
            weighted_health += pattern_score * weight
            
            # Debug information
            if stats["total"] >= 10:
                print(f"Pattern {pattern}: overall={overall_acc:.2f}, recent={recent_acc:.2f}, score={pattern_score:.2f}")
        
        if total_weight == 0:
            return 0.5
            
        # Calculate weighted average and normalize to 0-1
        avg_health = weighted_health / total_weight
        norm_health = min(1.0, max(0.0, avg_health))
        
        return norm_health
        
    def _extract_params_safely(self, params):
        """
        Extract parameters from a parameter dictionary with safe defaults.
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            dict: Safe parameters with defaults
        """
        safe_params = self.default_params.copy()
        
        # If params is not a dictionary, return defaults
        if not isinstance(params, dict):
            return safe_params
            
        # Extract each parameter safely with default fallbacks
        for key, default in self.default_params.items():
            value = params.get(key)
            if value is not None:
                try:
                    # Convert to appropriate type based on default
                    if isinstance(default, int):
                        safe_params[key] = int(value)
                    elif isinstance(default, float):
                        safe_params[key] = float(value)
                    else:
                        safe_params[key] = value
                except (ValueError, TypeError):
                    # Keep default if conversion fails
                    pass
                    
        return safe_params
        
    def create_variant(self, mutation_rate=0.2):
        """
        Create improved variant with enhanced parameter optimization.
        
        Args:
            mutation_rate: How much to vary parameters (0.2 = 20% variation)
            
        Returns:
            XGBoostModel: A new variant with mutated parameters
        """
        try:
            # Create new variant with unique ID
            variant_id = f"{self.variant_id}_v{time.strftime('%Y%m%d_%H%M%S')}" if hasattr(self, 'variant_id') else f"xgb_variant_{time.strftime('%Y%m%d_%H%M%S')}"
            print(f"Creating variant: {variant_id}")
            
            variant = XGBoostModel(
                random_state=np.random.randint(1, 10000),
                variant_id=variant_id
            )
            
            # Copy basic structure and data
            variant.is_trained = False  # Will be trained from scratch
            if hasattr(self, 'feature_cols'):
                variant.feature_cols = self.feature_cols.copy() if hasattr(self.feature_cols, 'copy') else self.feature_cols
            if hasattr(self, 'columns'):
                variant.columns = self.columns.copy() if hasattr(self.columns, 'copy') else self.columns
            if hasattr(self, 'scaler'):
                variant.scaler = self.scaler  # Share the scaler
            if hasattr(self, 'feature_mask') and self.feature_mask is not None:
                variant.feature_mask = self.feature_mask.copy() if hasattr(self.feature_mask, 'copy') else self.feature_mask
            
            # Analyze model performance to guide evolution
            performance_factor = 1.0
            if hasattr(self, 'recent_predictions') and len(self.recent_predictions) > 10:
                # Calculate recent accuracy
                recent = self.recent_predictions[-10:]
                recent_correct = sum(1 for p in recent if p.get('correct', False))
                recent_acc = recent_correct / len(recent) if recent else 0.5
                
                # Adjust mutation strategy based on accuracy
                if recent_acc > 0.6:  # Doing well - more conservative mutations
                    performance_factor = 0.8
                    print(f"Good performance ({recent_acc:.2f}), more conservative mutations")
                elif recent_acc < 0.4:  # Doing poorly - more aggressive mutations
                    performance_factor = 1.5
                    print(f"Poor performance ({recent_acc:.2f}), more aggressive mutations")
            
            # Get current parameters with safe defaults
            params = getattr(self.model, 'get_params', lambda: {})()
            safe_params = self._extract_params_safely(params)
            
            # Apply mutations safely with improved strategies
            mutated_params = safe_params.copy()
            
            # Number of trees - increase or decrease by up to 20% (scaled by mutation_rate)
            n_estimators_change = int(safe_params['n_estimators'] * (np.random.random() - 0.5) * mutation_rate * 2 * performance_factor)
            mutated_params['n_estimators'] = max(50, safe_params['n_estimators'] + n_estimators_change)
            
            # Max depth - add or remove levels
            depth_change = np.random.choice([-1, 0, 1]) * (2 if performance_factor > 1 else 1)
            mutated_params['max_depth'] = max(2, min(8, safe_params['max_depth'] + depth_change))
            
            # Learning rate - adjust by up to 20% (scaled by mutation_rate)
            lr_factor = 1 + (np.random.random() - 0.5) * mutation_rate * performance_factor
            mutated_params['learning_rate'] = max(0.005, min(0.3, safe_params['learning_rate'] * lr_factor))
            
            # Subsample and colsample - adjust by up to 10% (scaled by mutation_rate)
            subsample_change = (np.random.random() - 0.5) * mutation_rate
            mutated_params['subsample'] = max(0.5, min(1.0, safe_params['subsample'] + subsample_change))
            
            colsample_change = (np.random.random() - 0.5) * mutation_rate
            mutated_params['colsample_bytree'] = max(0.5, min(1.0, safe_params['colsample_bytree'] + colsample_change))
            
            # Min child weight - add or subtract 1-2
            weight_change = np.random.choice([-2, -1, 0, 1, 2]) * (2 if performance_factor > 1 else 1)
            mutated_params['min_child_weight'] = max(1, safe_params['min_child_weight'] + weight_change)
            
            # Regularization parameters - adjust by up to 30% (scaled by mutation_rate)
            gamma_factor = 1 + (np.random.random() - 0.5) * mutation_rate * 1.5
            mutated_params['gamma'] = max(0, safe_params['gamma'] * gamma_factor)
            
            alpha_factor = 1 + (np.random.random() - 0.5) * mutation_rate * 1.5
            mutated_params['reg_alpha'] = max(0, safe_params['reg_alpha'] * alpha_factor)
            
            lambda_factor = 1 + (np.random.random() - 0.5) * mutation_rate * 1.5
            mutated_params['reg_lambda'] = max(0.01, safe_params['reg_lambda'] * lambda_factor)
            
            # Generate new random state
            mutated_params['random_state'] = np.random.randint(1, 10000)
            
            # Apply advanced parameter testing - adjust for specific issues
            self._smart_parameter_adjustment(mutated_params)
            
            # Print parameter changes
            print(f"Parameter mutations:")
            for param in sorted(set(safe_params.keys()).intersection(mutated_params.keys())):
                if safe_params[param] != mutated_params[param]:
                    print(f"  {param}: {safe_params[param]} → {mutated_params[param]}")
            
            # Create model with mutated parameters - wrapped in try/except
            try:
                variant.model = xgb.XGBClassifier(**mutated_params)
                print(f"Successfully created XGBoost variant with {mutated_params['n_estimators']} trees, depth={mutated_params['max_depth']}")
            except Exception as e:
                print(f"Error creating variant model with custom parameters: {e}")
                print("Falling back to default parameters...")
                variant.model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=3,
                    random_state=np.random.randint(1, 10000)
                )
            
            # Transfer accumulated data if available
            if hasattr(self, 'update_buffer'):
                variant.update_buffer = {
                    'X': self.update_buffer.get('X', []).copy() if hasattr(self.update_buffer.get('X', []), 'copy') else self.update_buffer.get('X', []),
                    'y': self.update_buffer.get('y', []).copy() if hasattr(self.update_buffer.get('y', []), 'copy') else self.update_buffer.get('y', []),
                    'threshold': self.update_buffer.get('threshold', 10),
                    'last_update': time.time()
                }
            
            # Transfer pattern performance data for continuity
            variant.pattern_performance = defaultdict(self._default_pattern_stats)
            for pattern, stats in self.pattern_performance.items():
                variant.pattern_performance[pattern] = stats.copy()
            
            return variant
            
        except Exception as e:
            print(f"Critical error in create_variant: {e}")
            traceback.print_exc()
            
            # Create emergency fallback variant
            emergency_variant = XGBoostModel(
                random_state=42,
                variant_id=f"xgb_emergency_{int(time.time())}"
            )
            emergency_variant.model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3
            )
            
            # Copy minimal information
            if hasattr(self, 'feature_cols'):
                emergency_variant.feature_cols = self.feature_cols
            if hasattr(self, 'scaler'):
                emergency_variant.scaler = self.scaler
                
            print("Created emergency fallback variant with default parameters")
            return emergency_variant
    
    def _smart_parameter_adjustment(self, params):
        """Intelligently adjust parameters based on model history and performance."""
        try:
            # Get pattern performance if available
            if hasattr(self, 'pattern_performance') and self.pattern_performance:
                # Check if patterns are predictive
                pattern_predictive = False
                for pattern, stats in self.pattern_performance.items():
                    if stats['total'] >= 10:
                        accuracy = stats['correct'] / stats['total']
                        if accuracy > 0.55:  # Above random for Baccarat
                            pattern_predictive = True
                            break
                
                # If patterns are predictive, adjust parameters to capture them better
                if pattern_predictive:
                    # Increase max_depth to capture more complex patterns
                    params['max_depth'] = min(8, params['max_depth'] + 1)
                    # Lower min_child_weight to better capture rare patterns
                    params['min_child_weight'] = max(1, params['min_child_weight'] - 1)
                    print("Adjusted parameters to better capture detected patterns")
            
            # Get confidence correlation if available
            if hasattr(self, 'get_confidence_correlation'):
                corr = self.get_confidence_correlation()
                
                # If confidence correlation is low, adjust to improve calibration
                if corr < 0.4:
                    # Reduce complexity to avoid overfitting
                    params['max_depth'] = max(2, params['max_depth'] - 1)
                    # Increase min_samples to improve generalization
                    params['min_child_weight'] = min(10, params['min_child_weight'] + 1)
                    # Add more regularization
                    params['reg_lambda'] = min(5.0, params['reg_lambda'] * 1.5)
                    print("Adjusted parameters to improve confidence calibration")
            
            # Check for overconfidence in recent predictions
            if hasattr(self, 'recent_predictions') and len(self.recent_predictions) >= 20:
                recent = self.recent_predictions[-20:]
                high_conf_preds = [p for p in recent if p.get('confidence', 0) > 70]
                
                if high_conf_preds:
                    high_conf_acc = sum(1 for p in high_conf_preds if p.get('correct', False)) / len(high_conf_preds)
                    
                    # If high confidence predictions are wrong too often
                    if high_conf_acc < 0.6:
                        # Increase regularization to reduce overconfidence
                        params['reg_alpha'] = min(5.0, params['reg_alpha'] * 1.3)
                        params['reg_lambda'] = min(5.0, params['reg_lambda'] * 1.3)
                        print("Adjusted parameters to reduce overconfidence")
            
            return params
        except Exception as e:
            print(f"Error in smart parameter adjustment: {e}")
            return params
    
    def save(self, filename):
        """Save model to file with improved error handling."""
        try:
            dirname = os.path.dirname(filename)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)
                
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            print(f"XGBoostModel saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            traceback.print_exc()
            
            # Try alternative location as fallback
            try:
                fallback_filename = f"xgboost_backup_{int(time.time())}.pkl"
                with open(fallback_filename, 'wb') as f:
                    pickle.dump(self, f)
                print(f"Model saved to fallback location: {fallback_filename}")
                return True
            except:
                return False
            
    @classmethod
    def load(cls, filename):
        """Load model from file with enhanced validation and recovery."""
        if not os.path.exists(filename):
            print(f"Model file {filename} not found. Creating new model.")
            return cls()
        
        try:    
            with open(filename, 'rb') as f:
                model = pickle.load(f)
                
            # Verify it's the correct class
            if not isinstance(model, cls):
                print(f"Warning: Loaded model is not a {cls.__name__}. Converting...")
                new_model = cls()
                
                # Copy core components systematically
                for attr in ['model', 'scaler', 'feature_cols', 'columns', 'feature_mask', 
                            'is_trained', 'feature_names', 'feature_importance',
                            'pattern_performance', 'recent_predictions']:
                    if hasattr(model, attr):
                        setattr(new_model, attr, getattr(model, attr))
                
                # Ensure pattern_performance is properly initialized
                if not hasattr(new_model, 'pattern_performance') or new_model.pattern_performance is None:
                    new_model.pattern_performance = defaultdict(new_model._default_pattern_stats)
                
                # Initialize calibrators if missing
                if not hasattr(new_model, 'calibrators'):
                    new_model.calibrators = {
                        0: IsotonicRegression(out_of_bounds='clip'),
                        1: IsotonicRegression(out_of_bounds='clip'),
                        2: IsotonicRegression(out_of_bounds='clip')
                    }
                    new_model.is_calibrated = False
                
                return new_model
            
            # Validate and fix critical components
            if not hasattr(model, 'pattern_performance') or model.pattern_performance is None:
                model.pattern_performance = defaultdict(model._default_pattern_stats)
                
            if not hasattr(model, 'calibrators'):
                model.calibrators = {
                    0: IsotonicRegression(out_of_bounds='clip'),
                    1: IsotonicRegression(out_of_bounds='clip'),
                    2: IsotonicRegression(out_of_bounds='clip')
                }
                model.is_calibrated = False
                
            if not hasattr(model, 'update_buffer'):
                model.update_buffer = {'X': [], 'y': [], 'threshold': 10, 'last_update': time.time()}
                
            # Model loaded successfully
            print(f"XGBoost model loaded successfully from {filename}")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            
            # Try to recover partial data if possible
            try:
                print("Attempting partial recovery of model data...")
                with open(filename, 'rb') as f:
                    # Create fresh model
                    new_model = cls()
                    
                    # Try to extract some useful data from corrupted file
                    try:
                        corrupt_model = pickle.load(f)
                        if hasattr(corrupt_model, 'recent_predictions'):
                            new_model.recent_predictions = corrupt_model.recent_predictions
                            print("Recovered prediction history")
                        if hasattr(corrupt_model, 'pattern_performance'):
                            new_model.pattern_performance = corrupt_model.pattern_performance
                            print("Recovered pattern performance data")
                    except:
                        pass
                        
                    return new_model
            except:
                # Complete failure, return new model
                return cls()

    def fix_xgboost_models(registry):
        """
        Fix XGBoost models in the registry to ensure they have all required attributes
        and proper feature consistency.
        """
        fixed_count = 0
        for model_id, model in registry.models.items():
            if model_id.startswith('xgboost') or (hasattr(model, 'model_type') and model.model_type == 'xgboost'):
                # Add missing attributes with defaults
                if not hasattr(model, 'is_calibrated'):
                    model.is_calibrated = False
                
                if not hasattr(model, 'calibrators'):
                    from sklearn.isotonic import IsotonicRegression
                    model.calibrators = {
                        0: IsotonicRegression(out_of_bounds='clip'),
                        1: IsotonicRegression(out_of_bounds='clip'),
                        2: IsotonicRegression(out_of_bounds='clip')
                    }
                
                if not hasattr(model, 'feature_mask'):
                    model.feature_mask = None
                    
                if not hasattr(model, 'trained_columns') and hasattr(model, 'feature_cols'):
                    model.trained_columns = model.feature_cols
                
                # Add missing methods with safe implementations
                if not hasattr(model, '_apply_calibration'):
                    def _apply_calibration(self, raw_probs):
                        # Simple pass-through implementation
                        return raw_probs
                    
                    import types
                    model._apply_calibration = types.MethodType(_apply_calibration, model)
                
                # Fix predict_proba method to handle missing attributes safely
                original_predict_proba = model.predict_proba
                
                def safe_predict_proba(self, X):
                    try:
                        return original_predict_proba(X)
                    except AttributeError as e:
                        if "is_calibrated" in str(e):
                            # Handle the specific error by setting a default
                            self.is_calibrated = False
                            return original_predict_proba(X)
                        elif "feature_mask" in str(e):
                            # Handle missing feature mask
                            self.feature_mask = None
                            return original_predict_proba(X)
                        else:
                            # For other attribute errors, use a simpler implementation
                            print(f"Falling back to basic prediction due to: {e}")
                            shape = (X.shape[0] if hasattr(X, 'shape') and len(X) > 0 else 1, 3)
                            return np.array([[0.45, 0.45, 0.1]] * shape[0])
                    except Exception as e:
                        # For any other error, return default probabilities
                        print(f"Error in XGBoost probability prediction: {e}")
                        shape = (X.shape[0] if hasattr(X, 'shape') and len(X) > 0 else 1, 3)
                        return np.array([[0.45, 0.45, 0.1]] * shape[0])
                
                # Replace the method
                model.predict_proba = types.MethodType(safe_predict_proba, model)
                
                fixed_count += 1
                print(f"Fixed XGBoost model: {model_id}")
        
        # Save the registry if models were fixed
        if fixed_count > 0:
            registry._save_registry()
            print(f"Registry saved with {fixed_count} fixed XGBoost models")
        
        return fixed_count > 0