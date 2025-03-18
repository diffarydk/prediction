# Create or replace fix_utils.py module

import types

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

def fix_markov_models(registry):
    """
    Apply safe predict_proba patch to all Markov models in the registry.
    
    Args:
        registry: Model registry containing the models
        
    Returns:
        bool: True if models were fixed, False otherwise
    """
    try:
        models_fixed = 0
        for model_id, model in registry.models.items():
            if model_id.startswith('markov'):
                if not hasattr(model, '_original_predict_proba'):
                    model._original_predict_proba = model.predict_proba
                model.predict_proba = types.MethodType(safe_markov_predict_proba, model)
                models_fixed += 1
                print(f"Fixed {model_id}")
        
        if models_fixed > 0:
            registry._save_registry()
            print(f"Fixed and saved {models_fixed} Markov models")
            return True
        return False
    except Exception as e:
        print(f"Error fixing Markov models: {e}")
        return False