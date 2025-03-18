"""
Base model class for Baccarat prediction system.
This module defines the abstract base class that all prediction models should inherit from,
providing standardized interfaces, error handling, and performance evaluation capabilities.
"""

import numpy as np
import logging
import time
import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all predictive models in the Baccarat prediction system.
    
    This class defines the contract that all model implementations must fulfill,
    ensuring consistent behavior across different modeling approaches. It provides
    standardized interfaces for training, prediction, evaluation, and serialization
    with comprehensive error handling and validation.
    
    Attributes:
        model_type (str): Type identifier for the model
        is_trained (bool): Flag indicating if the model has been trained
        feature_cols (list): List of feature column names
        performance (dict): Dictionary of performance metrics
        created_at (float): Timestamp when model was created
        last_updated (float): Timestamp when model was last updated
    """
    def __init__(self):
        """
        Initialize the base model with common properties and operational metadata.
        """
        self.model_type = "base"
        self.is_trained = False
        self.feature_cols = None
        self.performance = {}
        
        # Add tracking metadata
        self.created_at = time.time()
        self.last_updated = None
        self.prediction_count = 0
        self.error_count = 0
        
    @abstractmethod
    def fit(self, X, y):
        """
        Train the model with data.
        
        Args:
            X: Features/inputs (numpy array or DataFrame)
            y: Target/outputs (numpy array or Series)
            
        Returns:
            self: The trained model instance
        """
        pass
        
    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features/inputs to predict from
            
        Returns:
            numpy array: Predicted outcomes
        """
        pass
        
    @abstractmethod
    def predict_proba(self, X):
        """
        Get prediction probabilities for each class.
        
        Args:
            X: Features/inputs to predict from
            
        Returns:
            numpy array: Probability for each class
        """
        pass
    
    def safe_predict(self, X):
        """
        Make predictions with comprehensive error handling and validation.
        
        This method provides a robust prediction interface that handles
        common edge cases and ensures consistent behavior even when
        prediction fails.
        
        Args:
            X: Features/inputs to predict from
            
        Returns:
            numpy array: Predicted outcomes or fallback values on error
        """
        try:
            self.prediction_count += 1
            
            # Validate input
            X_valid = self.validate_input(X)
            
            # Check if model is trained
            if not self.is_trained:
                logger.warning(f"{self.model_type} model used before training")
                # Return default prediction (banker)
                return np.zeros(len(X_valid), dtype=int)
            
            # Make prediction
            return self.predict(X_valid)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in {self.model_type} prediction: {e}")
            
            # Determine fallback shape
            if isinstance(X, np.ndarray):
                samples = 1 if X.ndim == 1 else len(X)
            elif isinstance(X, list):
                samples = 1 if not X or not isinstance(X[0], list) else len(X)
            else:
                samples = 1
                
            # Return fallback prediction (banker)
            return np.zeros(samples, dtype=int)
    
    def safe_predict_proba(self, X):
        """
        Get prediction probabilities with comprehensive error handling.
        
        Similar to safe_predict but returns probability distributions with
        robust error handling to ensure consistent behavior across all contexts.
        
        Args:
            X: Features/inputs to predict from
            
        Returns:
            dict or numpy array: Probability for each class
        """
        try:
            # Validate input
            X_valid = self.validate_input(X)
            
            # Check if model is trained
            if not self.is_trained:
                logger.warning(f"{self.model_type} model used before training")
                # Return default probabilities (slightly favoring banker)
                return {0: 0.45, 1: 0.45, 2: 0.1}
            
            # Make probability prediction
            probs = self.predict_proba(X_valid)
            
            # Normalize if needed
            if isinstance(probs, dict) and abs(sum(probs.values()) - 1.0) > 0.01:
                total = sum(probs.values())
                return {k: v/total for k, v in probs.items()} if total > 0 else {0: 0.45, 1: 0.45, 2: 0.1}
            
            return probs
            
        except Exception as e:
            logger.error(f"Error in {self.model_type} probability prediction: {e}")
            
            # Return fallback probabilities (based on baccarat odds)
            return {0: 0.45, 1: 0.45, 2: 0.1}
    
    def validate_input(self, X):
        """
        Validate and normalize input data format.
        
        Args:
            X: Input data to validate
            
        Returns:
            numpy array: Validated and properly formatted input
            
        Raises:
            ValueError: If input cannot be properly validated
        """
        # Handle None input
        if X is None:
            raise ValueError("Input cannot be None")
            
        # Convert list to numpy array if needed
        if isinstance(X, list):
            X = np.array(X)
            
        # Ensure 2D shape
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        # Check for NaN values
        if np.isnan(X).any():
            logger.warning("Input contains NaN values, replacing with zeros")
            X = np.nan_to_num(X)
            
        return X
    
    def get_feature_importance(self):
        """
        Get feature importance if available.
        
        Returns:
            list: List of (feature, importance) tuples or None
        """
        if hasattr(self, 'feature_importance'):
            return self.feature_importance
            
        # Try to extract from model attributes
        if hasattr(self, 'model'):
            model = self.model
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                features = self.feature_cols if self.feature_cols else [f"feature_{i}" for i in range(len(importances))]
                return list(zip(features, importances))
            elif hasattr(model, 'coef_'):
                coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                features = self.feature_cols if self.feature_cols else [f"feature_{i}" for i in range(len(coef))]
                return list(zip(features, np.abs(coef)))
                
        return None
    
    def save(self, filename):
        """
        Save model to file with proper error handling and atomicity.
        
        This implementation provides a default serialization approach
        using pickle with atomic file operations to prevent corruption.
        Subclasses can override for custom serialization needs.
        
        Args:
            filename: Path to save the model
            
        Returns:
            bool: True if save was successful
            
        Raises:
            IOError: If saving fails due to file system issues
        """
        try:
            # Update metadata
            self.last_updated = time.time()
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            # Save to temporary file first
            temp_filename = f"{filename}.temp"
            with open(temp_filename, 'wb') as f:
                pickle.dump(self, f)
                
            # Atomic rename to ensure no corruption on failure
            os.replace(temp_filename, filename)
            
            logger.info(f"Model {self.model_type} saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model {self.model_type}: {e}")
            raise IOError(f"Failed to save model: {e}")
    
    @classmethod
    def load(cls, filename):
        """
        Load model from file with proper error handling.
        
        This implementation provides a default deserialization approach
        using pickle. Subclasses can override for custom loading logic.
        
        Args:
            filename: Path to load the model from
            
        Returns:
            BaseModel: The loaded model instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If loaded object is not a valid model
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")
            
        try:
            with open(filename, 'rb') as f:
                model = pickle.load(f)
                
            # Validate model type
            if not isinstance(model, cls):
                raise ValueError(f"Loaded object is not a {cls.__name__}")
                
            logger.info(f"Model {model.model_type} loaded from {filename}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {filename}: {e}")
            raise ValueError(f"Failed to load model: {e}")
    
    def summary(self):
        """
        Get a comprehensive summary of the model with performance metrics.
        
        Returns:
            dict: Model summary information including performance metrics
        """
        summary = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'prediction_count': self.prediction_count,
            'error_rate': self.error_count / max(1, self.prediction_count),
            'performance': self.performance
        }
        
        # Add feature importance if available
        feature_importance = self.get_feature_importance()
        if feature_importance:
            summary['feature_importance'] = feature_importance
            
        # Add model-specific details
        if hasattr(self, 'model'):
            model = self.model
            if hasattr(model, 'n_estimators'):
                summary['n_estimators'] = model.n_estimators
            if hasattr(model, 'max_depth'):
                summary['max_depth'] = model.max_depth
                
        return summary
        
    def evaluate(self, X, y):
        """
        Evaluate model performance on test data with comprehensive metrics.
        
        This method provides detailed performance evaluation including
        accuracy, precision, recall, F1-score, confusion matrix, and
        betting-specific expected value calculation.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            dict: Comprehensive performance metrics
        """
        try:
            # Validate inputs
            X_valid = self.validate_input(X)
            y = np.array(y)
            
            # Generate predictions
            y_pred = self.predict(X_valid)
            
            # Calculate base accuracy
            accuracy = np.mean(y_pred == y)
            
            # Calculate class-specific accuracy
            classes = np.unique(y)
            class_accuracy = {}
            
            for cls in classes:
                # Indices where true label is this class
                indices = (y == cls)
                if np.sum(indices) > 0:
                    # Accuracy for this class
                    class_acc = np.mean(y_pred[indices] == y[indices])
                    class_accuracy[int(cls)] = float(class_acc)
            
            # Initialize metrics dictionary
            metrics = {
                'accuracy': float(accuracy),
                'class_accuracy': class_accuracy,
                'n_samples': len(y),
                'evaluation_time': time.time()
            }
            
            try:
                # Advanced metrics using scikit-learn if available
                from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
                
                # Class-specific metrics
                for cls in range(3):  # Banker (0), Player (1), Tie (2)
                    # Calculate precision, recall, and F1 for each class
                    y_true_binary = (y == cls).astype(int)
                    y_pred_binary = (y_pred == cls).astype(int)
                    
                    cls_precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                    cls_recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                    cls_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                    
                    metrics[f'precision_class_{cls}'] = float(cls_precision)
                    metrics[f'recall_class_{cls}'] = float(cls_recall)
                    metrics[f'f1_class_{cls}'] = float(cls_f1)
                
                # Confusion matrix
                cm = confusion_matrix(y, y_pred)
                metrics['confusion_matrix'] = cm.tolist()
            except ImportError:
                # Skip advanced metrics if scikit-learn not available
                pass
            
            # Calculate baccarat-specific expected value (assuming standard payouts)
            correct_banker = ((y_pred == 0) & (y == 0)).sum()
            correct_player = ((y_pred == 1) & (y == 1)).sum()
            correct_tie = ((y_pred == 2) & (y == 2)).sum()
            
            total_predictions = len(y)
            
            # Banker pays 0.95:1, Player 1:1, Tie 8:1
            expected_value = (0.95 * correct_banker + 1.0 * correct_player + 8.0 * correct_tie) / total_predictions - 1.0
            metrics['expected_value'] = float(expected_value)
            
            # Calculate ROI for other scenarios
            unit_bet = 1.0  # Assuming 1 unit bet for simplicity
            total_roi = 0
            
            for actual_outcome in range(3):
                for predicted_outcome in range(3):
                    # Count matches of this prediction-outcome pair
                    matches = ((y_pred == predicted_outcome) & (y == actual_outcome)).sum()
                    
                    # Skip if no matches
                    if matches == 0:
                        continue
                    
                    # Calculate ROI
                    if predicted_outcome == 0:  # Banker
                        # Banker wins but pays 0.95:1 (5% commission)
                        if actual_outcome == 0:
                            roi = 0.95 * matches
                        else:
                            roi = -1.0 * matches
                    elif predicted_outcome == 1:  # Player
                        # Player pays 1:1
                        if actual_outcome == 1:
                            roi = 1.0 * matches
                        else:
                            roi = -1.0 * matches
                    else:  # Tie
                        # Tie pays 8:1
                        if actual_outcome == 2:
                            roi = 8.0 * matches
                        else:
                            roi = -1.0 * matches
                            
                    total_roi += roi
                    
            # Calculate average ROI per bet
            metrics['roi_per_bet'] = float(total_roi / total_predictions)
            
            # Add calibration metrics if probabilities available
            try:
                y_proba = self.predict_proba(X_valid)
                
                # Convert to standard format
                if isinstance(y_proba, dict):
                    # Convert dict format to array format
                    proba_array = np.zeros((len(y), 3))
                    for i in range(3):
                        if i in y_proba:
                            proba_array[:, i] = y_proba[i]
                    y_proba = proba_array
                
                # Calculate calibration error (Brier score)
                from sklearn.metrics import brier_score_loss
                
                # One-hot encode actual outcomes
                y_one_hot = np.zeros((len(y), 3))
                for i, label in enumerate(y):
                    y_one_hot[i, label] = 1
                
                # Calculate Brier score for each class
                brier_scores = []
                for cls in range(3):
                    brier = brier_score_loss(y_one_hot[:, cls], y_proba[:, cls])
                    metrics[f'brier_score_class_{cls}'] = float(brier)
                    brier_scores.append(brier)
                
                # Average Brier score
                metrics['brier_score_avg'] = float(np.mean(brier_scores))
                
            except Exception as e:
                logger.warning(f"Skipping calibration metrics: {e}")
            
            # Update model performance
            self.performance.update(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            # Return basic metrics on error
            return {
                'accuracy': 0.0,
                'error': str(e),
                'evaluation_time': time.time()
            }
    
    def update_performance(self, predicted, actual, confidence=None, pattern=None):
        """
        Update performance metrics with a single prediction result.
        
        This method provides incremental performance tracking without
        requiring a full evaluation run, useful for online learning
        and continuous monitoring.
        
        Args:
            predicted: Predicted outcome
            actual: Actual outcome
            confidence: Optional prediction confidence
            pattern: Optional pattern type
            
        Returns:
            dict: Updated performance metrics
        """
        # Initialize tracking if needed
        if 'correct_count' not in self.performance:
            self.performance['correct_count'] = 0
        if 'total_count' not in self.performance:
            self.performance['total_count'] = 0
        if 'by_class' not in self.performance:
            self.performance['by_class'] = {0: {'correct': 0, 'total': 0},
                                           1: {'correct': 0, 'total': 0},
                                           2: {'correct': 0, 'total': 0}}
        if 'by_pattern' not in self.performance:
            self.performance['by_pattern'] = {}
        
        # Update counters
        correct = (predicted == actual)
        self.performance['total_count'] += 1
        if correct:
            self.performance['correct_count'] += 1
        
        # Update class-specific stats
        if actual in self.performance['by_class']:
            self.performance['by_class'][actual]['total'] += 1
            if correct:
                self.performance['by_class'][actual]['correct'] += 1
        
        # Update pattern-specific stats if provided
        if pattern:
            if pattern not in self.performance['by_pattern']:
                self.performance['by_pattern'][pattern] = {'correct': 0, 'total': 0}
            self.performance['by_pattern'][pattern]['total'] += 1
            if correct:
                self.performance['by_pattern'][pattern]['correct'] += 1
        
        # Update confidence correlation if provided
        if confidence is not None:
            if 'confidence_sum' not in self.performance:
                self.performance['confidence_sum'] = 0
                self.performance['confidence_correct_sum'] = 0
                self.performance['confidence_count'] = 0
            
            self.performance['confidence_sum'] += confidence
            if correct:
                self.performance['confidence_correct_sum'] += confidence
            self.performance['confidence_count'] += 1
        
        # Calculate accuracy
        if self.performance['total_count'] > 0:
            self.performance['accuracy'] = self.performance['correct_count'] / self.performance['total_count']
        
        # Calculate confidence correlation if possible
        if 'confidence_count' in self.performance and self.performance['confidence_count'] > 0:
            self.performance['avg_confidence'] = self.performance['confidence_sum'] / self.performance['confidence_count']
            if self.performance['correct_count'] > 0:
                self.performance['avg_confidence_when_correct'] = (
                    self.performance['confidence_correct_sum'] / self.performance['correct_count'])
        
        return self.performance
    
    def create_variant(self):
        """
        Create a variant of this model for competitive evolution.
        
        This method supports model competition and evolution by 
        creating modified copies with slightly different parameters.
        Subclasses should override this for model-specific variation.
        
        Returns:
            BaseModel: A variant of this model with modified parameters
        """
        # Default implementation just returns a clone
        import copy
        return copy.deepcopy(self)