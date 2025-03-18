"""
Monte Carlo simulation methods for robust Baccarat prediction.
This module provides enhanced prediction techniques using Monte Carlo simulation
to improve prediction accuracy and estimate confidence levels.
"""

import numpy as np
from config import MONTE_CARLO_SAMPLES

"""
Modify the monte_carlo_prediction function in prediction/monte_carlo.py 
to provide more conservative confidence estimates
"""

def monte_carlo_prediction(model, prev_rounds, samples=MONTE_CARLO_SAMPLES):
    """
    Make robust predictions using enhanced Monte Carlo simulation with improved confidence estimation.
    
    Args:
        model: Trained BaccaratModel instance
        prev_rounds: List of 5 previous outcomes (0=Banker, 1=Player, 2=Tie)
        samples: Number of Monte Carlo simulations to run
        
    Returns:
        dict: Prediction results with calibrated confidence levels
    """
    try:
        # Ensure prev_rounds is in the right format
        if isinstance(prev_rounds, list):
            prev_rounds_list = prev_rounds
            prev_rounds = np.array(prev_rounds)
        else:
            prev_rounds_list = prev_rounds.tolist()
        
        if prev_rounds.ndim == 1:
            prev_rounds = prev_rounds.reshape(1, -1)
        
        # Collection for prediction results
        predictions = []
        probabilities = []
        
        # Define outcome options
        outcomes = [0, 1, 2]  # Banker, Player, Tie
        
        # Get combined probabilities from the hybrid model
        combined_probs = model.get_combined_proba(prev_rounds_list)
        base_probs = np.array([combined_probs[0], combined_probs[1], combined_probs[2]])
        
        print(f"Combined probabilities: Banker={base_probs[0]:.3f}, Player={base_probs[1]:.3f}, Tie={base_probs[2]:.3f}")
        
        # Use probability distribution to generate samples with adaptive sampling
        for _ in range(samples):
            # Apply small random adjustments to probabilities (adds robustness)
            adjusted_probs = base_probs + np.random.normal(0, 0.02, size=3)
            adjusted_probs = np.clip(adjusted_probs, 0.05, 0.95)  # Ensure valid probabilities
            adjusted_probs = adjusted_probs / adjusted_probs.sum()  # Normalize
            
            # Sample prediction based on adjusted probabilities
            pred = np.random.choice(outcomes, p=adjusted_probs)
            predictions.append(pred)
            probabilities.append(adjusted_probs[pred])
        
        # Get the most common prediction and its frequency
        unique, counts = np.unique(predictions, return_counts=True)
        best_prediction = unique[np.argmax(counts)]
        raw_confidence = (counts[np.argmax(counts)] / samples) * 100
        
        # Calculate average probability
        avg_prob = np.mean(probabilities) * 100
        
        # Get prediction counts for each class (with defaults for any missing classes)
        prediction_dist = {}
        for outcome in outcomes:  # Ensure all outcomes are represented
            found = outcome in unique
            count = counts[list(unique).index(outcome)] if found else 0
            prediction_dist[outcome] = (count/samples)*100
        
        # Confidence adjustment based on distribution entropy
        # (lower confidence if predictions are more evenly distributed)
        entropy = -sum((p/100) * np.log2(p/100) if p > 0 else 0 for p in prediction_dist.values())
        max_entropy = np.log2(3)  # Maximum entropy for 3 outcomes
        entropy_ratio = entropy / max_entropy
        
        # More aggressive confidence reduction for high entropy (unpredictable) cases
        # And a baseline scaling to reduce overconfidence
        confidence_scaling = 0.75  # Reduce all confidence values to address general overconfidence
        entropy_penalty = 0.75 * entropy_ratio  # More aggressive penalty based on entropy
        
        # Apply stronger entropy penalty and baseline scaling
        adjusted_confidence = raw_confidence * confidence_scaling * (1 - entropy_penalty)
        
        # For Baccarat, keep confidence within realistic bounds
        # Games of chance shouldn't have extremely high confidence
        max_confidence_cap = 75.0  # Cap maximum confidence at 75%
        adjusted_confidence = min(adjusted_confidence, max_confidence_cap)
        
        # Higher cap for extremely one-sided predictions (over 90% probability)
        if prediction_dist[best_prediction] > 90:
            adjusted_confidence = min(prediction_dist[best_prediction] * 0.8, max_confidence_cap)
        
        # Reduce confidence for ties even further as they're more unpredictable
        if best_prediction == 2:  # Tie outcome
            adjusted_confidence = adjusted_confidence * 0.85
        
        return {
            'prediction': best_prediction,
            'confidence': adjusted_confidence,
            'raw_confidence': raw_confidence,
            'probability': avg_prob,
            'distribution': prediction_dist,
            'entropy': entropy,
            'entropy_ratio': entropy_ratio
        }
    except Exception as e:
        print(f"Error in monte_carlo_prediction: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to random prediction with low confidence if error
        return {
            'prediction': np.random.choice([0, 1, 2]),
            'confidence': 33.3,
            'probability': 33.3,
            'distribution': {0: 33.3, 1: 33.3, 2: 33.3},
            'error': str(e)
        }

def get_pattern_insight(prev_rounds):
    """
    Analyze the previous rounds for known baccarat patterns.
    Handles both NumPy arrays and Python lists.
    
    Args:
        prev_rounds: List or ndarray of 5 previous outcomes (0=Banker, 1=Player, 2=Tie)
        
    Returns:
        str: Insight about the pattern, or None if no pattern detected
    """
    # Convert NumPy array to list if needed
    if hasattr(prev_rounds, 'ndim') or hasattr(prev_rounds, 'dtype'):
        prev_rounds = prev_rounds.tolist()
        
    # Ensure we're working with a flat list
    if isinstance(prev_rounds, list) and prev_rounds and isinstance(prev_rounds[0], list):
        prev_rounds = prev_rounds[0]
    
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


def safe_monte_carlo_prediction(model, prev_rounds, samples=MONTE_CARLO_SAMPLES, fallback_strategy="random"):
    """
    Enhanced Monte Carlo prediction with better error handling and fallback strategies.
    Fixed version to prevent numpy.ndarray callable error.
    
    Args:
        model: Trained prediction model
        prev_rounds: Previous 5 round results as a list
        samples: Number of Monte Carlo simulations
        fallback_strategy: What to do if prediction fails ("random", "banker", "most_common")
        
    Returns:
        dict: Prediction details with confidence levels
    """
    try:
        # Validate input
        if len(prev_rounds) != 5:
            print(f"Warning: Expected 5 previous rounds, got {len(prev_rounds)}. Padding or truncating.")
            # Pad with most common value or truncate
            if prev_rounds:
                # Use Counter instead of direct counting to avoid callable issues
                from collections import Counter
                most_common = Counter(prev_rounds).most_common(1)[0][0]
            else:
                most_common = 0
            prev_rounds = (prev_rounds + [most_common] * 5)[:5]
        
        # Convert prev_rounds to appropriate format
        prev_rounds_array = np.array(prev_rounds)
        
        # Call the enhanced Monte Carlo prediction
        # Use a try/except block to catch any errors in the monte_carlo_prediction function
        try:
            result = monte_carlo_prediction(model, prev_rounds_array, samples)
        except Exception as e:
            print(f"Error in monte_carlo_prediction: {e}")
            # Implement a simpler prediction method as fallback
            predicted = model.predict(prev_rounds_array.reshape(1, -1))[0]
            result = {
                'prediction': int(predicted),
                'confidence': 45.0,  # Lower confidence since we're using fallback
                'distribution': {0: 33.3, 1: 33.3, 2: 33.3},
                'status': 'fallback',
                'error': str(e)
            }
        
        # Add pattern-based insight
        pattern_insight = get_pattern_insight(prev_rounds)
        if pattern_insight:
            result['pattern_insight'] = pattern_insight
        
        return result
        
    except Exception as e:
        print(f"Monte Carlo prediction failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Complete fallback with explanation
        if fallback_strategy == "banker":
            prediction = 0  # Banker
        elif fallback_strategy == "most_common":
            # Use a simple count-based prediction
            if len(prev_rounds) > 0:
                from collections import Counter
                prediction = Counter(prev_rounds).most_common(1)[0][0]
            else:
                prediction = 0  # Default to banker
        else:
            # Random fallback
            prediction = np.random.choice([0, 1, 2])
            
        return {
            'prediction': prediction,
            'confidence': 33.3,
            'probability': 33.3,
            'distribution': {0: 33.3, 1: 33.3, 2: 33.3},
            'status': 'error',
            'error': str(e)
        }

def evaluate_monte_carlo_accuracy(model, test_data, samples=100, runs=10):
    """
    Evaluate the accuracy of Monte Carlo predictions on test data.
    Runs multiple Monte Carlo predictions for each test point to evaluate consistency.
    
    Args:
        model: Trained prediction model
        test_data: DataFrame with features and target
        samples: Number of Monte Carlo samples per prediction
        runs: Number of Monte Carlo runs per test point
        
    Returns:
        dict: Performance metrics
    """
    X = test_data.drop('Target', axis=1).values
    y = test_data['Target'].values
    
    correct = 0
    total = len(y)
    consistency = []
    confidence_scores = []
    
    for i in range(total):
        # Multiple Monte Carlo runs for same input
        run_predictions = []
        
        for _ in range(runs):
            result = monte_carlo_prediction(model, X[i], samples=samples)
            run_predictions.append(result['prediction'])
            
            # For the first run, check if prediction is correct
            if _ == 0:
                if result['prediction'] == y[i]:
                    correct += 1
                confidence_scores.append((result['confidence'], result['prediction'] == y[i]))
        
        # Calculate consistency (how often the same prediction is made)
        unique, counts = np.unique(run_predictions, return_counts=True)
        max_consistency = np.max(counts) / runs
        consistency.append(max_consistency)
    
    # Calculate metrics
    accuracy = correct / total
    avg_consistency = np.mean(consistency)
    
    # Calculate confidence calibration
    confidence_ranges = [(0, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]
    calibration = {}
    
    for low, high in confidence_ranges:
        in_range = [(conf, correct) for conf, correct in confidence_scores if low <= conf < high]
        if in_range:
            range_accuracy = sum(correct for _, correct in in_range) / len(in_range)
            calibration[f"{low}-{high}%"] = {
                'samples': len(in_range),
                'accuracy': range_accuracy,
                'avg_confidence': np.mean([conf for conf, _ in in_range])
            }
    
    return {
        'accuracy': accuracy,
        'consistency': avg_consistency,
        'calibration': calibration,
        'samples': total
    }