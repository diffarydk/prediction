"""
Main Baccarat prediction model using RandomForest and advanced feature engineering.
This module implements the core prediction model that combines machine learning
with domain-specific features to predict Baccarat outcomes.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # For handling imbalanced data
from collections import defaultdict, Counter
import time
from sklearn.ensemble import IsolationForest
from imblearn.combine import SMOTETomek


from models.base_model import BaseModel
from models.markov_model import MarkovModel
from config import MODEL_FILE, MONTE_CARLO_SAMPLES, MARKOV_MEMORY

class BaccaratModel(BaseModel):
    """
    Enhanced Baccarat prediction model using a hybrid approach:
    - Random Forest for pattern recognition
    - Markov models for sequence dependencies
    - Advanced feature engineering
    - Adaptive Monte Carlo simulation
    """
    def __init__(self, random_state=42):
        """
        Initialize the Baccarat model.
        
        Args:
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.feature_cols = None  # Will be set during training
        self.model = RandomForestClassifier(
            n_estimators=300,      # More trees for stability
            max_depth=6,           # Slightly deeper for pattern recognition
            min_samples_leaf=8,    # Less restrictive to capture rare patterns
            min_samples_split=12,  # Better balance against overfitting
            class_weight='balanced_subsample', # Better imbalance handling
            bootstrap=True,
            oob_score=True,        # Enable out-of-bag scoring
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.baseline_accuracy = None
        self.performance = {}
        self.columns = None  # Store column names for prediction
        self.model_type = "baccarat_hybrid"
        
        # Markov models of different orders
        self.markov1 = MarkovModel(order=1)
        self.markov2 = MarkovModel(order=2)
        
        # Recent prediction performance tracking
        self.recent_correct = 0
        self.recent_total = 0
        self.recent_probs = []
    
    def _add_derived_features(self, X):
        """
        Add enhanced derived features that may help the model identify patterns.
        
        Args:
            X: Input features dataframe or numpy array
            
        Returns:
            DataFrame with additional derived features
        """
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f'Prev_{i+1}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()
        
        # Store original column names for later use
        self.columns = X_df.columns.tolist()
        
        # Create a copy of the dataframe for derived features
        X_df_features = X_df.copy()
        
        # Calculate ratio of each outcome in the previous results
        X_df_features["banker_ratio"] = X_df.apply(lambda row: sum(1 for x in row.values if x == 0) / len(row), axis=1)
        X_df_features["player_ratio"] = X_df.apply(lambda row: sum(1 for x in row.values if x == 1) / len(row), axis=1)
        X_df_features["tie_ratio"] = X_df.apply(lambda row: sum(1 for x in row.values if x == 2) / len(row), axis=1)
        
        # Calculate alternation pattern (changes between outcomes)
        # Fixed to use .iloc consistently
        X_df_features["alternation_count"] = X_df.apply(
            lambda row: sum(1 for i in range(len(row)-1) if row.iloc[i] != row.iloc[i+1]), 
            axis=1
        )
        
        # Calculate streaks (consecutive same outcomes)
        def max_streak(row):
            max_s = 1
            current = 1
            for i in range(1, len(row)):
                if row.iloc[i] == row.iloc[i-1]:
                    current += 1
                    max_s = max(max_s, current)
                else:
                    current = 1
            return max_s
        
        X_df_features["max_streak"] = X_df.apply(max_streak, axis=1)
        
        # Calculate current streak (consecutive same most recent outcomes)
        def current_streak(row):
            current = 1
            last = row.iloc[0]  # Most recent outcome
            for i in range(1, len(row)):
                if row.iloc[i] == last:
                    current += 1
                else:
                    break
            return current
        
        X_df_features["current_streak"] = X_df.apply(current_streak, axis=1)
        
        # Calculate time since last occurrence of each outcome
        for outcome in [0, 1, 2]:  # Banker, Player, Tie
            def time_since(row, target):
                for i, val in enumerate(row):
                    if val == target:
                        return i
                return len(row)
            
            X_df_features[f"time_since_{outcome}"] = X_df.apply(lambda row: time_since(row, outcome), axis=1)
        
        # Add parity features (odd/even counts)
        X_df_features["banker_player_diff"] = X_df.apply(
            lambda row: sum(1 for x in row.values if x == 0) - sum(1 for x in row.values if x == 1), 
            axis=1
        )
        
        # Add pattern-based features
        def contains_pattern(row, pattern):
            row_list = row.tolist()
            pattern_str = ''.join(map(str, pattern))
            row_str = ''.join(map(str, row_list))
            return 1 if pattern_str in row_str else 0
        
        # Common patterns to look for
        patterns = [
            [0, 0], [1, 1], [2, 2],  # Repeats
            [0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1],  # Alternations
            [0, 0, 0], [1, 1, 1], [2, 2, 2],  # Triple repeats
            [0, 1, 0], [1, 0, 1], [0, 2, 0], [2, 0, 2], [1, 2, 1], [2, 1, 2]  # Bounce patterns
        ]
        
        for i, pattern in enumerate(patterns):
            X_df_features[f"pattern_{i}"] = X_df.apply(lambda row: contains_pattern(row, pattern), axis=1)
        
        # Add weighted recent history (most recent outcomes have higher weight)
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # Decreasing weights
        for outcome in [0, 1, 2]:
            X_df_features[f"weighted_{outcome}"] = X_df.apply(
                lambda row: sum(weights[i] if row.iloc[i] == outcome else 0 for i in range(len(row))) / sum(weights),
                axis=1
            )
        
        # Return the enhanced features dataframe
        X_df_features["momentum"] = X_df.apply(
            lambda row: sum([1 if row.iloc[i] == row.iloc[0] else -1 for i in range(len(row))]),
            axis=1
        )

        # Add change-point detection (number of transitions between outcomes)
        X_df_features["transitions"] = X_df.apply(
            lambda row: sum([1 for i in range(len(row)-1) if row.iloc[i] != row.iloc[i+1]]),
            axis=1
        )

        # Add last-N detection (specific patterns known in Baccarat)
        if len(X_df.columns) >= 3:
            X_df_features["last3_pattern"] = X_df.apply(
                lambda row: int(str(row.iloc[0]) + str(row.iloc[1]) + str(row.iloc[2])),
                axis=1
    )
        return X_df_features
    
    def fit(self, X, y):
        """
        Train the model with enhanced feature engineering and class balancing.
        
        Args:
            X: Input features
            y: Target outcomes
            
        Returns:
            self: The trained model instance
        """
        if len(X) < 15:  # Need minimum data to train effectively
            print("Insufficient data for training")
            return self
        
        # Calculate baseline accuracy (always predicting most common outcome)
        most_common = np.bincount(y).argmax()
        self.baseline_accuracy = np.mean(y == most_common)
        print(f"Baseline accuracy (always guessing {most_common}): {self.baseline_accuracy:.2f}")
        
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"Class distribution: {class_dist}")
        
        # Add derived features
        X_enhanced = self._add_derived_features(X)
        self.feature_cols = X_enhanced.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_enhanced)
        
        # Use SMOTE to balance the dataset
        print("Applying SMOTE to balance classes...")
        try:
            # For severe imbalance, use combination of SMOTE and Tomek links
            resample = SMOTETomek(random_state=42)
            X_balanced, y_balanced = resample.fit_resample(X_scaled, y)
            print(f"After resampling: {len(X_balanced)} samples")
            
            # Check new class distribution
            unique_balanced, counts_balanced = np.unique(y_balanced, return_counts=True)
            balanced_dist = dict(zip(unique_balanced, counts_balanced))
            print(f"Balanced class distribution: {balanced_dist}")
        except Exception as e:
            print(f"Resampling failed: {e}. Using original data.")
            X_balanced, y_balanced = X_scaled, y

        
        # Split data with stratification to maintain class distribution
        X_train, X_val, y_train, y_val = train_test_split(
            X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
        )
        
        outlier_detector = IsolationForest(contamination=0.05, random_state=42)
        is_outlier = outlier_detector.fit_predict(X_enhanced)
        n_outliers = sum(1 for x in is_outlier if x == -1)
        print(f"Detected {n_outliers} outliers ({n_outliers/len(X_enhanced)*100:.1f}%)")

        # Separate core and outlier instances
        X_core = X_enhanced[is_outlier == 1]
        y_core = y[is_outlier == 1]

        # Train on clean data, then fine-tune with full data
        if len(X_core) >= 50:  # Only use if we have enough data
            print("Training first on clean data, then fine-tuning...")
            self.model.fit(X_core, y_core)
            
            # Adjust weights for fine-tuning
            if hasattr(self.model, 'n_estimators'):
                orig_n_estimators = self.model.n_estimators
                self.model.n_estimators += 50  # Add more trees for fine-tuning
            
            # Fine-tune with all data
            self.model.fit(X_enhanced, y)
        else:
            # Use full dataset if insufficient clean data
            self.model.fit(X_enhanced, y)

        # Train Random Forest model
        self.model.fit(X_train, y_train)
        
        # Train Markov models
        # Extract sequences for Markov model training
        if isinstance(X, pd.DataFrame):
            sequences = X.values.tolist()
            full_sequence = []
            for seq in sequences:
                full_sequence.extend(seq)
            full_sequence.extend(y)  # Include target values in sequence
        else:
            # If X is numpy array, flatten and convert to list
            full_sequence = X.flatten().tolist()
            full_sequence.extend(y)
        
        # Train Markov models with the sequence
        self.markov1.fit(full_sequence)
        self.markov2.fit(full_sequence)
        
        # Evaluate on validation set
        val_pred = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(classification_report(y_val, val_pred))
        
        # Check if model beats baseline
        if val_accuracy <= self.baseline_accuracy + 0.05:
            print("WARNING: Model is not significantly better than baseline")
            print("Baccarat outcomes may be largely random - use predictions with caution")
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            features = self.feature_cols
            self.feature_importance = sorted(zip(features, self.model.feature_importances_), 
                                          key=lambda x: x[1], reverse=True)
            print("Top 5 important features:")
            for feature, importance in self.feature_importance[:5]:
                print(f"  {feature}: {importance:.4f}")
        
        # Store performance metrics
        self.performance = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'val_accuracy': val_accuracy,
            'baseline_accuracy': self.baseline_accuracy
        }
        
        self.is_trained = True

        if hasattr(self.model, 'feature_importances_'):
            features = self.feature_cols
            self.feature_importance = sorted(zip(features, self.model.feature_importances_), 
                                        key=lambda x: x[1], reverse=True)
            
            # Only keep features that contribute at least 1% to the model
            important_features = [f for f, imp in self.feature_importance if imp >= 0.01]
            self.important_feature_indices = [self.feature_cols.index(f) for f in important_features if f in self.feature_cols]
            print(f"Keeping {len(important_features)} important features out of {len(features)}")

        return self
    
    def predict(self, X):
        """
        Make a prediction for new input data.
        
        Args:
            X: Input features (numpy array or dataframe)
            
        Returns:
            numpy array: Predicted outcomes
        """
        try:
            X = self.validate_input(X)
            
            # Create DataFrame with same structure as training data
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X, columns=[f'Prev_{i+1}' for i in range(X.shape[1])])
            else:
                X_df = X.copy()
            
            # Add derived features
            X_enhanced = self._add_derived_features(X_df)
            
            # Apply scaling
            X_scaled = self.scaler.transform(X_enhanced)
            
            # Make prediction with Random Forest
            return self.model.predict(X_scaled)
        except Exception as e:
            print(f"Error in prediction: {e}")
            print(f"Input shape: {X.shape if hasattr(X, 'shape') else 'unknown'}, expected 5 features")
            # Fallback to random outcome
            return np.array([np.random.choice([0, 1, 2])])
    
# In baccarat_model.py, find the predict_proba method
# In models/baccarat_model.py, update the predict_proba method

    def predict_proba(self, X):
        """
        Get class probabilities for predictions using the Random Forest model.
        """
        try:
            X = self.validate_input(X)
            
            # Create DataFrame with same structure as training data
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X, columns=[f'Prev_{i+1}' for i in range(X.shape[1])])
            else:
                X_df = X.copy()
            
            # Add derived features
            X_enhanced = self._add_derived_features(X_df)
            
            # Check if model is fitted
            if not hasattr(self.model, 'classes_'):
                print("Warning: RandomForest model not properly fitted.")
                return np.array([[0.33, 0.33, 0.34]])
            
            # Add this check to ensure scaler is fitted
            if not hasattr(self.scaler, 'n_features_in_'):
                print("Initializing scaler with default values")
                # Fit scaler with the enhanced features as a template
                self.scaler.fit(X_enhanced)
            
            # Apply scaling
            X_scaled = self.scaler.transform(X_enhanced)
            
            # Get probabilities from Random Forest
            return self.model.predict_proba(X_scaled)
        except Exception as e:
            print(f"Error in predict_proba: {e}")
            print(f"Input shape: {X.shape if hasattr(X, 'shape') else 'unknown'}, expected 5 features")
            # Fallback to even probabilities
            return np.array([[0.33, 0.33, 0.34]])
    
    def update_performance(self, predicted, actual, confidence=None, pattern=None):
        """
        Update recent performance metrics with a sliding window approach.
        
        Args:
            predicted: Predicted outcome
            actual: Actual outcome
            confidence: Prediction confidence (optional)
            pattern: Pattern type (optional)
        """
        # Store this prediction's result with additional info
        prediction_info = {
            'predicted': predicted,
            'actual': actual,
            'correct': 1 if predicted == actual else 0,
            'timestamp': time.time()  # Add timestamp for potential time-weighted analysis
        }
        
        # Add optional information if provided
        if confidence is not None:
            prediction_info['confidence'] = confidence
        if pattern is not None:
            prediction_info['pattern'] = pattern
            
        # Add the new prediction to history
        self.recent_probs.append(prediction_info)
        
        # Update running totals
        self.recent_total += 1
        if predicted == actual:
            self.recent_correct += 1
        
        # Instead of discarding data, maintain a separate window for recent accuracy calculation
        # Keep the full history for more sophisticated analysis
        if len(self.recent_probs) > 100:  # Set a reasonable maximum history length
            # For memory considerations, limit total history but keep more than just 20
            oldest = self.recent_probs.pop(0)
            
            # Adjust counters only if we're still tracking this prediction in our window
            if self.recent_total > 0:
                self.recent_total -= 1
                if oldest.get('correct', 0) == 1:
                    self.recent_correct -= 1
        
        # For reporting recent accuracy (last 20 predictions)
        recent_window = min(20, len(self.recent_probs))
        recent_correct = sum(p.get('correct', 0) for p in self.recent_probs[-recent_window:])
        recent_accuracy = recent_correct / recent_window if recent_window > 0 else 0
        
        # You could store this value for quick access later
        self.recent_window_accuracy = recent_accuracy
    
    # Modify get_combined_proba to be stacking-friendly
    def get_combined_proba(self, prev_rounds):
        """
        Get individual model probabilities for use in the stacking ensemble.
        
        This no longer combines probabilities using weights but returns the 
        RandomForest probabilities directly as input to the stacking ensemble.
        """
        try:
            # Ensure prev_rounds is in the right format
            if isinstance(prev_rounds, np.ndarray):
                prev_rounds_array = prev_rounds
            else:
                prev_rounds_array = np.array(prev_rounds)
                
            if prev_rounds_array.ndim == 1:
                prev_rounds_array = prev_rounds_array.reshape(1, -1)
                
            # Get RandomForest probabilities
            rf_probs_array = self.predict_proba(prev_rounds_array)[0]
            
            # Convert to dictionary format for consistency
            return {i: rf_probs_array[i] for i in range(len(rf_probs_array))}
                
        except Exception as e:
            print(f"Error getting probabilities: {e}")
            # Return default probabilities
            return {0: 0.33, 1: 0.33, 2: 0.34}
    
    def save(self, filename=MODEL_FILE):
        """
        Save model to file.
        
        Args:
            filename: Path to save the model
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"BaccaratModel saved to {filename}")
        
    @classmethod
    def load(cls, filename=MODEL_FILE):
        """
        Load model from file.
        
        Args:
            filename: Path to load the model from
            
        Returns:
            BaccaratModel: The loaded model instance
        """
        if not os.path.exists(filename):
            print(f"Model file {filename} not found. Will create a new model.")
            return cls()
            
        try:
            with open(filename, 'rb') as f:
                model = pickle.load(f)
                
            if not isinstance(model, cls):
                print(f"Warning: Loaded model is not a {cls.__name__}. Converting...")
                new_model = cls()
                # Try to copy attributes if possible
                for attr in ['model', 'scaler', 'feature_cols', 'markov1', 'markov2']:
                    if hasattr(model, attr):
                        setattr(new_model, attr, getattr(model, attr))
                return new_model
                        
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return cls()
    
    def get_pattern_insight(self, prev_rounds):
        """
        Analyze the previous rounds for known baccarat patterns.
        
        Args:
            prev_rounds: List of 5 previous outcomes
            
        Returns:
            str: Insight about the pattern, or None if no pattern detected
        """
        # Check for streaks
        if len(set(prev_rounds[:3])) == 1:
            outcome = prev_rounds[0]
            outcome_name = {0: 'Banker', 1: 'Player', 2: 'Tie'}[outcome]
            return f"Detected streak of {outcome_name}. Streaks often continue or break with alternating pattern."
        
        # Check for alternating pattern
        alternating = True
        for i in range(len(prev_rounds) - 2):
            if prev_rounds[i] == prev_rounds[i+2]:
                continue
            else:
                alternating = False
                break
        
        if alternating:
            return "Detected alternating pattern. These often continue or stabilize to one outcome."
        
        # Check for ties
        if 2 in prev_rounds[:2]:
            return "Recent tie detected. After ties, outcomes can become less predictable."
        
        return None
    
    def summary(self):
        """
        Get a summary of the model.
        
        Returns:
            dict: Model summary information
        """
        base_summary = super().summary()
        
        # Add BaccaratModel-specific information
        baccarat_summary = {
            'baseline_accuracy': self.baseline_accuracy,
            'recent_accuracy': self.recent_correct / self.recent_total if self.recent_total > 0 else None,
            'rf_estimators': self.model.n_estimators if hasattr(self.model, 'n_estimators') else None,
            'top_features': self.feature_importance[:5] if self.feature_importance else None,
            'markov1_states': len(self.markov1.transitions) if hasattr(self.markov1, 'transitions') else None,
            'markov2_states': len(self.markov2.transitions) if hasattr(self.markov2, 'transitions') else None
        }
        
        # Combine summaries
        return {**base_summary, **baccarat_summary}