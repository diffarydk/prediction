"""
Implementation of incremental/online learning for Baccarat Prediction.
This module enhances the base models with true online learning capabilities,
updating the model after each prediction instead of batch updates.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time

from config import MODEL_FILE, REALTIME_FILE
from models.baccarat_model import BaccaratModel
from data.data_utils import update_realtime_data

class OnlineBaccaratModel(BaccaratModel):
    """
    Enhanced Baccarat model with online learning capabilities.
    Inherits from the original BaccaratModel but adds incremental learning.
    """
    
    def __init__(self, base_model=None, random_state=42):
        """
        Initialize with option to load from existing model.
        
        Args:
            base_model: Optional existing BaccaratModel to initialize from
            random_state: Random seed for reproducibility
        """
        super().__init__(random_state=random_state)
        
        if base_model:
            # Copy attributes from base model
            self.feature_cols = base_model.feature_cols
            self.model = base_model.model
            self.scaler = base_model.scaler
            self.feature_importance = base_model.feature_importance
            self.baseline_accuracy = base_model.baseline_accuracy
            self.performance = base_model.performance
            self.columns = base_model.columns
            self.markov1 = base_model.markov1
            self.markov2 = base_model.markov2
            self.recent_correct = base_model.recent_correct
            self.recent_total = base_model.recent_total
            self.recent_probs = base_model.recent_probs.copy() if hasattr(base_model, 'recent_probs') else []
            
        # Track all encountered data for incremental updates
        self.all_x = []
        self.all_y = []
        self.update_count = 0
        self.model_type = "online_baccarat"
        self.pattern_performance = {}  # Initialize pattern performance tracking

    def _adjust_learning_parameters(self):
        """Dynamically adjust learning parameters based on performance trends"""
        if not hasattr(self, 'recent_probs') or len(self.recent_probs) < 20:
            return
            
        # Calculate recent accuracy trend
        recent = self.recent_probs[-20:]
        older = self.recent_probs[-40:-20] if len(self.recent_probs) >= 40 else []
        
        if not older:
            return
            
        recent_acc = sum(1 for p in recent if p['correct']) / len(recent)
        older_acc = sum(1 for p in older if p['correct']) / len(older)
        
        # Adjust Random Forest parameters based on trend
        if hasattr(self, 'model') and hasattr(self.model, 'set_params'):
            # If accuracy improving, fine-tune with lower regularization
            if recent_acc > older_acc + 0.05:
                print("Accuracy improving! Adjusting model to exploit patterns")
                current_max_depth = getattr(self.model, 'max_depth', 5)
                new_max_depth = min(8, current_max_depth + 1)
                self.model.set_params(max_depth=new_max_depth)
                
            # If accuracy declining, increase regularization
            elif older_acc > recent_acc + 0.1:
                print("Accuracy declining! Increasing regularization")
                current_max_depth = getattr(self.model, 'max_depth', 5)
                new_max_depth = max(3, current_max_depth - 1)
                current_min_samples = getattr(self.model, 'min_samples_leaf', 15)
                new_min_samples = min(30, current_min_samples + 5)
                self.model.set_params(max_depth=new_max_depth, min_samples_leaf=new_min_samples)

    def _check_pattern_dissolution(self):
        """Check if previously successful patterns are dissolving"""
        if not hasattr(self, 'pattern_performance'):
            return
            
        now = time.time()
        dissolution_threshold = 0.1  # 10% accuracy drop indicates dissolution
        
        for pattern, stats in self.pattern_performance.items():
            if stats['total'] < 10:  # Need minimum data
                continue
                
            # Get overall accuracy for this pattern
            overall_acc = stats['correct'] / stats['total']
            
            # Get recent accuracy (last 10 occurrences)
            recent_entries = [p for p in self.recent_probs[-30:] if p.get('pattern') == pattern]
            if len(recent_entries) >= 5:
                recent_correct = sum(1 for p in recent_entries if p['correct'])
                recent_acc = recent_correct / len(recent_entries)
                
                # Check for dissolution
                if (overall_acc - recent_acc) > dissolution_threshold:
                    time_since_correct = now - stats.get('last_correct', 0)
                    print(f"Pattern dissolution detected: {pattern}")
                    print(f"  Overall accuracy: {overall_acc:.3f}")
                    print(f"  Recent accuracy: {recent_acc:.3f}")
                    print(f"  Time since last correct: {time_since_correct/3600:.1f} hours")

    def _calculate_confidence_correlation(self):
        """Calculate correlation between prediction confidence and correctness"""
        if not hasattr(self, 'recent_probs') or len(self.recent_probs) < 10:
            return 0.5
            
        # Extract confidence and correctness pairs
        pairs = [(p.get('confidence', 50), p.get('correct', False)) 
                for p in self.recent_probs if 'confidence' in p]
        
        if len(pairs) < 10:
            return 0.5
            
        # Calculate correlation
        conf_values = [p[0] for p in pairs]
        corr_values = [1 if p[1] else 0 for p in pairs]
        
        conf_mean = sum(conf_values) / len(conf_values)
        corr_mean = sum(corr_values) / len(corr_values)
        
        numerator = sum((c - conf_mean) * (corr - corr_mean) for c, corr in pairs)
        conf_var = sum((c - conf_mean) ** 2 for c in conf_values)
        corr_var = sum((corr - corr_mean) ** 2 for corr in corr_values)
        
        if conf_var > 0 and corr_var > 0:
            correlation = numerator / ((conf_var * corr_var) ** 0.5)
            # Normalize to 0-1
            return (correlation + 1) / 2
        else:
            return 0.5

    def update_model(self, prev_rounds, actual_result, confidence=None, pattern=None):
        """
        Update the model with a single new observation, with enhanced tracking.
        
        Args:
            prev_rounds: List of 5 previous outcomes
            actual_result: The actual outcome that occurred
            confidence: Prediction confidence (optional)
            pattern: Pattern type identified (optional)
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Original update logic
            self.all_x.append(prev_rounds)
            self.all_y.append(actual_result)
            
            # Update Markov models immediately
            sequence = prev_rounds + [actual_result]
            self.markov1.fit(sequence)
            self.markov2.fit(sequence)
            
            # Track pattern-specific performance
            if not hasattr(self, 'pattern_performance'):
                self.pattern_performance = {}
                
            if pattern:
                if pattern not in self.pattern_performance:
                    self.pattern_performance[pattern] = {'total': 0, 'correct': 0, 'last_correct': 0}
                
                self.pattern_performance[pattern]['total'] += 1
                
                # Check if last prediction for this pattern was correct
                if hasattr(self, 'recent_probs') and len(self.recent_probs) > 0:
                    last_prediction = self.recent_probs[-1]['predicted']
                    if last_prediction == actual_result:
                        self.pattern_performance[pattern]['correct'] += 1
                        self.pattern_performance[pattern]['last_correct'] = time.time()
            
            # Track strategy health
            if not hasattr(self, 'strategy_health_history'):
                self.strategy_health_history = []
                
            # Check for pattern dissolution every 20 updates
            if self.update_count % 20 == 0:
                self._check_pattern_dissolution()
                
            # Update performance tracking with confidence if available
            if hasattr(self, 'recent_probs') and len(self.recent_probs) > 0:
                last_prediction = self.recent_probs[-1]['predicted'] if self.recent_probs else None
                self.update_performance(last_prediction, actual_result, confidence)
            
            # Increment update counter
            self.update_count += 1
            
            # Update RandomForest model periodically with dynamic schedule
            # More frequent updates initially, then stabilize
            update_threshold = min(5, 3 + self.update_count // 100)  # Dynamic update frequency
            if self.update_count >= update_threshold:
                success = self._retrain_forest()
                self.update_count = 0
                if success:
                    print("Random Forest model updated with new data")
                    
                # Check model health and adjust learning parameters if needed
                self._adjust_learning_parameters()
                    
            return True
        except Exception as e:
            print(f"Error in incremental update: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _assess_pattern_health(self):
        """Assess if patterns remain predictive or are dissolving"""
        if not hasattr(self, 'pattern_performance'):
            return 0.5
            
        if not self.pattern_performance:
            return 0.5
            
        # Calculate average pattern effectiveness change
        overall_scores = []
        
        for pattern, stats in self.pattern_performance.items():
            if stats['total'] < 8:
                continue
                
            # Overall accuracy for this pattern
            overall_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.5
            
            # Get recent accuracy for this pattern
            recent_pattern_preds = [p for p in self.recent_probs[-30:] if p.get('pattern') == pattern]
            
            if len(recent_pattern_preds) >= 3:
                recent_correct = sum(1 for p in recent_pattern_preds if p.get('correct', False))
                recent_acc = recent_correct / len(recent_pattern_preds)
                
                # Score based on whether pattern is maintaining effectiveness
                pattern_score = min(1.0, recent_acc / max(0.01, overall_acc))
                overall_scores.append(pattern_score)
        
        if not overall_scores:
            return 0.5
            
        return sum(overall_scores) / len(overall_scores)

    def get_strategy_health(self):
        """Get comprehensive health assessment of this model's prediction strategy"""
        # Accuracy with exponential weighting
        if not hasattr(self, 'recent_probs') or len(self.recent_probs) < 5:
            return {'score': 0.5, 'accuracy': 0.5, 'confidence_corr': 0.5, 'pattern_health': 0.5}
        
        # Exponentially weighted accuracy
        now = time.time()
        total_weight = 0
        correct_weight = 0
        
        for i, pred in enumerate(self.recent_probs):
            # Assign weight based on recency
            age_factor = 0.9 ** (len(self.recent_probs) - i - 1)
            weight = age_factor
            
            total_weight += weight
            if pred.get('correct', False):
                correct_weight += weight
                
        accuracy = correct_weight / total_weight if total_weight > 0 else 0.5
        
        # Confidence correlation
        confidence_corr = self._calculate_confidence_correlation()
        
        # Pattern health - assess if patterns remain predictive
        pattern_health = self._assess_pattern_health()
        
        # Combined health score
        health_score = 0.5 * accuracy + 0.3 * confidence_corr + 0.2 * pattern_health
        
        return {
            'score': health_score,
            'accuracy': accuracy,
            'confidence_corr': confidence_corr,
            'pattern_health': pattern_health
        }

    def create_variant(self):
        """Create a variant of this model with slightly different parameters"""
        variant = OnlineBaccaratModel()
        
        # Copy key attributes
        variant.feature_cols = self.feature_cols
        variant.baseline_accuracy = self.baseline_accuracy
        variant.columns = self.columns
        variant.markov1 = self.markov1  # Share Markov models
        variant.markov2 = self.markov2
        
        # Create new Random Forest with mutated parameters
        from sklearn.ensemble import RandomForestClassifier
        
        # Get current parameters
        n_estimators = getattr(self.model, 'n_estimators', 200)
        max_depth = getattr(self.model, 'max_depth', 5)
        min_samples_leaf = getattr(self.model, 'min_samples_leaf', 15)
        
        # Mutate parameters slightly
        import numpy as np
        n_estimators = max(50, int(n_estimators * (1 + (np.random.random() - 0.5) * 0.2)))
        max_depth = max(3, max_depth + np.random.choice([-1, 0, 1]))
        min_samples_leaf = max(5, min_samples_leaf + np.random.choice([-5, -2, 0, 2, 5]))
        
        # Create model with mutated parameters
        variant.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight='balanced',
            random_state=np.random.randint(1, 1000),
            warm_start=True
        )
        
        # Transfer accumulated data
        variant.all_x = self.all_x.copy() if hasattr(self, 'all_x') else []
        variant.all_y = self.all_y.copy() if hasattr(self, 'all_y') else []
        
        # Initialize training
        if variant.all_x and variant.all_y:
            X = np.array(variant.all_x)
            y = np.array(variant.all_y)
            variant.fit(X, y)
        
        return variant


def load_or_create_online_model():
    """
    Load existing model and convert to online learning model,
    or create a new one if no model exists.
    
    Returns:
        OnlineBaccaratModel: A model ready for incremental updates
    """
    if os.path.exists(MODEL_FILE):
        try:
            # Load existing model
            print(f"Loading model from {MODEL_FILE}...")
            with open(MODEL_FILE, 'rb') as f:
                base_model = pickle.load(f)
            
            # Check if already an online model
            if isinstance(base_model, OnlineBaccaratModel):
                print("Model is already in online learning mode")
                return base_model
                
            # Convert to online model
            online_model = OnlineBaccaratModel(base_model=base_model)
            print("Model converted to online learning mode")
            return online_model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new online learning model...")
            return OnlineBaccaratModel()
    else:
        print("No existing model found. Creating new online learning model...")
        return OnlineBaccaratModel()


def analyze_learning_curve(model, history=None):
    """
    Analyze the learning curve for a model to gauge improvement over time.
    
    Args:
        model: The OnlineBaccaratModel to analyze
        history: Optional prediction history to analyze
        
    Returns:
        dict: Analysis results
    """
    # Basic implementation
    if not hasattr(model, 'recent_probs') or len(model.recent_probs) < 5:
        return {"error": "Insufficient learning data"}
    
    # Get recent performance (last 20 predictions)
    recent = model.recent_probs[-20:] if len(model.recent_probs) >= 20 else model.recent_probs
    recent_acc = sum(1 for p in recent if p.get('correct', False)) / len(recent)
    
    # Get older performance (previous 20)
    older = model.recent_probs[-40:-20] if len(model.recent_probs) >= 40 else []
    older_acc = sum(1 for p in older if p.get('correct', False)) / len(older) if older else 0
    
    # Calculate improvement
    improvement = recent_acc - older_acc if older else 0
    
    return {
        "recent_accuracy": recent_acc,
        "older_accuracy": older_acc,
        "improvement": improvement,
        "total_samples": len(model.recent_probs),
        "trend": "improving" if improvement > 0.05 else "declining" if improvement < -0.05 else "stable"
    }


def save_online_model(model, output_file=MODEL_FILE):
    """Save an online model to file"""
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"Online model saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving online model: {e}")
        return False


def update_after_prediction(model, prev_rounds, actual_result, confidence=None, pattern=None):
    """
    Update an online model after receiving the actual outcome.
    
    Args:
        model: OnlineBaccaratModel to update
        prev_rounds: Previous 5 rounds used for prediction
        actual_result: The actual outcome that occurred
        confidence: Optional confidence value of the prediction
        pattern: Optional pattern type detected
        
    Returns:
        bool: Success status of the update
    """
    if hasattr(model, 'update_model'):
        return model.update_model(prev_rounds, actual_result, confidence, pattern)
    else:
        print("Model does not support online updates")
        return False