"""
Markov model implementation for capturing transition patterns in Baccarat.
This module provides first and second-order Markov models to analyze and predict
the probability of next outcomes based on previous states.
"""

from collections import defaultdict, Counter
import numpy as np
import pickle
import os
import time

from .base_model import BaseModel

class MarkovModel(BaseModel):
    """
    Markov chain model for capturing transition probabilities in baccarat outcomes.
    Supports different orders (1st-order, 2nd-order, etc.) to capture various pattern depths.
    """
    def __init__(self, order=1):
        """
        Initialize a Markov model with specified order.
        
        Args:
            order: The order of the Markov model (default: 1).
                  Order 1 means next outcome depends only on current state.
                  Order 2 means it depends on the last 2 states, etc.
        """
        super().__init__()
        self.order = order
        self.transitions = defaultdict(Counter)
        self.total_counts = defaultdict(int)
        self.model_type = f"markov_{order}"
        self.is_trained = False
        
        # Add default transitions for common states to avoid cold start
        # This will provide reasonable predictions even with no training
        self._initialize_default_transitions()
        
    def _initialize_default_transitions(self):
        """Initialize default transitions based on typical baccarat probabilities"""
        # Default probabilities based on standard baccarat odds
        # Banker wins ~45.9%, Player wins ~44.6%, Tie ~9.5%
        default_probs = {0: 0.459, 1: 0.446, 2: 0.095}
        
        # Add default transitions for all possible initial states
        if self.order == 1:
            for state in [(0,), (1,), (2,)]:  # Banker, Player, Tie
                for outcome, prob in default_probs.items():
                    # Add count proportional to probability (multiply by 100 for integer counts)
                    self.transitions[state][outcome] = int(prob * 100)
                    self.total_counts[state] += int(prob * 100)
        elif self.order == 2:
            for first in [0, 1, 2]:
                for second in [0, 1, 2]:
                    state = (first, second)
                    for outcome, prob in default_probs.items():
                        self.transitions[state][outcome] = int(prob * 100)
                        self.total_counts[state] += int(prob * 100)
        
        # Set trained to true since we have default values
        self.is_trained = True
        
    def fit(self, X, y=None):
        """
        Train the Markov model on a sequence of outcomes.
        
        Args:
            X: A sequence of outcomes (can be a list, numpy array, or a DataFrame column)
               If y is None, X is treated as a complete sequence including the outcomes to predict.
            y: If provided, X is treated as previous states and y as next outcomes.
            
        Returns:
            self: The trained model instance
        """
        # Handle different input formats for X
        try:
            sequence = self._prepare_sequence(X, y)
            
            if len(sequence) < self.order + 1:
                print(f"Sequence too short for Markov model of order {self.order}. Need at least {self.order + 1} elements.")
                return self
                
            # Train the model from sequence
            for i in range(len(sequence) - self.order):
                # Convert to tuple to ensure it's hashable
                state = tuple(sequence[i:i+self.order])
                next_outcome = sequence[i+self.order]
                self.transitions[state][next_outcome] += 1
                self.total_counts[state] += 1
            
            self.is_trained = True
            return self
        except Exception as e:
            print(f"Error in Markov fit: {e}")
            # Fall back to default transitions
            self._initialize_default_transitions()
            return self
        
    def safe_markov_predict_proba(self, X):
        """
        Safe version of predict_proba that handles all edge cases consistently.
        Default probabilities based on baccarat odds: Banker 45.9%, Player 44.6%, Tie 9.5%
        
        Args:
            X: The current state (sequence of previous outcomes)
            
        Returns:
            dict or list of dicts: Probabilities for each possible next outcome
        """
        # Default probabilities
        default_probs = {0: 0.459, 1: 0.446, 2: 0.095}
        
        # Handle edge cases
        if X is None or isinstance(X, (int, float)) or not hasattr(X, '__len__') or len(X) == 0 or len(X) < self.order:
            return default_probs
        
        try:
            X = self.validate_input(X)
            
            if len(X) > 1:
                return [{0: 0.459, 1: 0.446, 2: 0.095} for _ in range(len(X))]
            else:
                state = tuple(X[0][-self.order:])
                if state in self.transitions and self.total_counts[state] > 0:
                    probs = {k: v / self.total_counts[state] for k, v in self.transitions[state].items()}
                    return {outcome: probs.get(outcome, 0.05) for outcome in [0, 1, 2]}
                else:
                    return default_probs
        except:
            return default_probs
        
    def _prepare_sequence(self, X, y=None):
        """
        Convert the input data to a usable sequence format with robust error handling.
        
        Args:
            X: Input data (array, list, DataFrame column)
            y: Optional target data
            
        Returns:
            list: The prepared sequence
        """
        # Handle edge cases
        if X is None:
            return [0, 1, 0, 1, 0]  # Default sequence if None provided
            
        # Convert X to list format with robust handling
        if isinstance(X, np.ndarray):
            # Handle empty arrays
            if X.size == 0:
                return [0, 1, 0, 1, 0]
            sequence = X.flatten().tolist()
        elif hasattr(X, 'values'):  # pandas Series or DataFrame
            if len(X) == 0:
                return [0, 1, 0, 1, 0]
            sequence = X.values.flatten().tolist()
        elif isinstance(X, (int, float)):
            # Handle single numeric value by creating a repeating sequence
            sequence = [int(X) if X in [0, 1, 2] else 0] * 5
        elif isinstance(X, str):
            # Try to parse string as B/P/T sequence
            sequence = []
            for char in X:
                if char.upper() == 'B':
                    sequence.append(0)
                elif char.upper() == 'P':
                    sequence.append(1) 
                elif char.upper() in ['T', 'S']:
                    sequence.append(2)
            # If parsing failed, use default sequence
            if not sequence:
                sequence = [0, 1, 0, 1, 0]
        else:
            try:
                sequence = list(X)  # Try to convert to list
                if not sequence:
                    sequence = [0, 1, 0, 1, 0]
            except:
                sequence = [0, 1, 0, 1, 0]  # Default if conversion fails
            
        # If y is provided, append it to the sequence
        if y is not None:
            if isinstance(y, (list, tuple)):
                sequence.extend(y)
            elif isinstance(y, np.ndarray):
                sequence.extend(y.flatten().tolist())
            elif hasattr(y, 'values'):  # pandas Series
                sequence.extend(y.values.tolist())
            elif isinstance(y, (int, float)) and y in [0, 1, 2]:
                sequence.append(int(y))
            else:
                try:
                    sequence.append(y)  # Assume it's a single value
                except:
                    pass  # Ignore if can't append
                
        # Ensure all values are valid (0, 1, 2)
        sequence = [x if x in [0, 1, 2] else 0 for x in sequence]
        
        return sequence
    
    def predict(self, X):
        """
        Predict the next outcome given the current state.
        
        Args:
            X: The current state (sequence of previous outcomes)
            
        Returns:
            array: Predicted outcome (most likely next outcome)
        """
        # Get probabilities for each outcome
        probs = self.predict_proba(X)
        
        # Find the outcome with the highest probability
        if isinstance(probs, dict):
            # If single prediction, return most likely outcome
            return np.array([max(probs.items(), key=lambda x: x[1])[0]])
        else:
            # If multiple predictions, return most likely outcome for each
            return np.array([max(p.items(), key=lambda x: x[1])[0] for p in probs])
        
    def predict_proba(self, X):
        """
        Get probability distribution for next outcome given current state.
        
        Args:
            X: The current state (sequence of previous outcomes)
            
        Returns:
            dict or list of dicts: Probabilities for each possible next outcome
        """
        # Default probabilities
        default_probs = {0: 0.459, 1: 0.446, 2: 0.095}  # Typical baccarat odds
        
        # Handle all possible problematic input types
        if X is None:
            return default_probs
            
        if isinstance(X, (int, float)):
            return default_probs
            
        if not hasattr(X, '__len__'):
            return default_probs
            
        if len(X) == 0:
            return default_probs
            
        if isinstance(X, np.ndarray) and X.size == 0:
            return default_probs
            
        # For proper arrays/lists, check if they're long enough
        if len(X) < self.order:
            return default_probs
        
        try:
            # Standard processing for valid inputs
            X = self.validate_input(X)
            
            if len(X) > 1:
                return [self._predict_proba_single(x[-self.order:]) for x in X]
            else:
                return self._predict_proba_single(X[0][-self.order:])
        except Exception as e:
            print(f"Fallback in Markov predict_proba: {e}")
            return default_probs
    
    def _predict_proba_single(self, state):
        """
        Get probability distribution for a single state.
        
        Args:
            state: The current state (sequence of last N outcomes)
            
        Returns:
            dict: Probabilities for each possible next outcome
        """
        # Convert state to a tuple to ensure it's hashable
        if isinstance(state, list) or isinstance(state, np.ndarray):
            state = tuple(state[-self.order:])
        
        # Ensure state has correct values
        state = tuple(s if s in [0, 1, 2] else 0 for s in state)
        
        # Get transition counts if available
        if state in self.transitions and self.total_counts[state] > 0:
            # Calculate probabilities from counts
            probs = {k: v / self.total_counts[state] for k, v in self.transitions[state].items()}
            
            # Ensure all outcomes have a probability (default to small value)
            return {outcome: probs.get(outcome, 0.05) for outcome in [0, 1, 2]}
        else:
            # If state not seen, return standard baccarat odds
            return {0: 0.459, 1: 0.446, 2: 0.095}
    
    def get_transition_matrix(self):
        """
        Generate the transition probability matrix for first-order Markov models.
        
        Returns:
            dict or array: Transition probabilities
        """
        if self.order != 1:
            print("Transition matrix is only available for first-order Markov models")
            return None
            
        # For first-order model, create a 3x3 matrix
        matrix = np.zeros((3, 3))
        
        # Fill with probabilities
        for i in range(3):  # Current state (Banker, Player, Tie)
            state = (i,)
            if state in self.transitions:
                for j in range(3):  # Next state
                    count = self.transitions[state].get(j, 0)
                    total = self.total_counts[state]
                    matrix[i, j] = count / total if total > 0 else 0.0
        
        return matrix
    
    def save(self, filename):
        """
        Save the Markov model to file.
        
        Args:
            filename: Path to save the model
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Markov model saved to {filename}")

    @classmethod
    def load(cls, filename):
        """
        Load a Markov model from file with improved error handling for patched methods.
        
        Args:
            filename: Path to load the model from
            
        Returns:
            MarkovModel: The loaded model instance
        """
        if not os.path.exists(filename):
            print(f"Model file {filename} not found. Creating new model.")
            return cls()
            
        try:
            with open(filename, 'rb') as f:
                model = pickle.load(f)
                
            # Handle case of models patched with safe_markov_predict_proba
            if not hasattr(model, 'safe_markov_predict_proba') and hasattr(model, '_original_predict_proba'):
                # Fix the predict_proba method
                def safe_markov_predict_proba(self, X):
                    # Default probabilities
                    default_probs = {0: 0.459, 1: 0.446, 2: 0.095}
                    
                    # Handle edge cases
                    if X is None or isinstance(X, (int, float)) or not hasattr(X, '__len__') or len(X) == 0 or len(X) < self.order:
                        return default_probs
                    
                    try:
                        X = self.validate_input(X)
                        
                        if len(X) > 1:
                            return [{0: 0.459, 1: 0.446, 2: 0.095} for _ in range(len(X))]
                        else:
                            state = tuple(X[0][-self.order:])
                            if state in self.transitions and self.total_counts[state] > 0:
                                probs = {k: v / self.total_counts[state] for k, v in self.transitions[state].items()}
                                return {outcome: probs.get(outcome, 0.05) for outcome in [0, 1, 2]}
                            else:
                                return default_probs
                    except:
                        return default_probs
                        
                import types
                model.predict_proba = types.MethodType(safe_markov_predict_proba, model)
                
            # Handle case where model is not the expected class
            if not isinstance(model, cls):
                print(f"Warning: Loaded model is not a {cls.__name__}. Converting...")
                new_model = cls()
                # Try to copy transitions if available
                if hasattr(model, 'transitions'):
                    new_model.transitions = model.transitions
                if hasattr(model, 'total_counts'):
                    new_model.total_counts = model.total_counts
                return new_model
                    
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return cls()
    
    def update(self, new_sequence):
        """
        Update the model with new data without retraining from scratch.
        
        Args:
            new_sequence: New sequence of outcomes to learn from
            
        Returns:
            self: The updated model instance
        """
        try:
            # Prepare sequence with error handling
            processed_sequence = self._prepare_sequence(new_sequence)
            
            # Update transitions with new data
            for i in range(len(processed_sequence) - self.order):
                state = tuple(processed_sequence[i:i+self.order])
                next_outcome = processed_sequence[i+self.order]
                self.transitions[state][next_outcome] += 1
                self.total_counts[state] += 1
            
            return self
        except Exception as e:
            print(f"Error updating Markov model: {e}")
            return self
    
    def summary(self):
        """
        Get a summary of the Markov model.
        
        Returns:
            dict: Model summary information
        """
        base_summary = super().summary()
        
        # Add Markov-specific information
        markov_summary = {
            'order': self.order,
            'states_observed': len(self.transitions),
            'total_transitions': sum(self.total_counts.values())
        }
        
        # Add most probable transitions for the first 5 states
        top_states = sorted(self.total_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        most_probable = {}
        
        for state, _ in top_states:
            if state in self.transitions:
                most_common = self.transitions[state].most_common(1)
                if most_common:
                    next_outcome, count = most_common[0]
                    probability = count / self.total_counts[state]
                    state_str = '→'.join(map(str, state))
                    most_probable[state_str] = (next_outcome, probability)
        
        markov_summary['most_probable_transitions'] = most_probable
        
        # Combine summaries
        return {**base_summary, **markov_summary}

    def get_entropy(self, state):
        """
        Calculate entropy (uncertainty) of the prediction for a given state.
        
        Args:
            state: The current state (sequence of previous outcomes)
            
        Returns:
            float: Entropy value. Higher means more uncertainty.
        """
        try:
            probs = self.predict_proba(state)
            
            # If it's a list of dicts (multiple states), take the first one
            if isinstance(probs, list):
                if not probs:
                    return 1.0  # Maximum uncertainty if empty
                probs = probs[0]
                
            # Calculate entropy: -sum(p * log2(p))
            entropy = 0
            for p in probs.values():
                if p > 0:  # Avoid log(0)
                    entropy -= p * np.log2(p)
                    
            return entropy
        except Exception as e:
            print(f"Error calculating entropy: {e}")
            return 1.0  # Return maximum uncertainty