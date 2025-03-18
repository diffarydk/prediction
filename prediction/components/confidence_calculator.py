"""
Confidence Calculation Component for Baccarat Prediction System.

This module implements standardized confidence estimation, calibration,
and adjustment mechanisms for baccarat prediction probabilities, ensuring
consistent and well-calibrated confidence scores across the system.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    """
    Manages confidence calculation and calibration for predictions.
    
    This component provides a centralized implementation for converting
    raw probabilities to calibrated confidence scores, applying pattern-specific
    adjustments, and maintaining confidence calibration models based on
    historical performance.
    """
    
    def __init__(self):
        """Initialize confidence calculator with calibration tracking."""
        # Initialize calibration tracking
        self.calibration_data = {
            0: {'bins': defaultdict(lambda: {'correct': 0, 'total': 0})},  # Banker
            1: {'bins': defaultdict(lambda: {'correct': 0, 'total': 0})},  # Player
            2: {'bins': defaultdict(lambda: {'correct': 0, 'total': 0})}   # Tie
        }
        
        # Outcome-specific confidence adjustments
        self.outcome_adjustments = {
            0: 1.0,    # Banker - neutral adjustment
            1: 1.0,    # Player - neutral adjustment
            2: 0.85    # Tie - reduce confidence (harder to predict)
        }
        
        # Confidence caps by outcome
        self.confidence_caps = {
            0: 85.0,   # Banker cap
            1: 85.0,   # Player cap
            2: 70.0    # Tie cap (lower maximum confidence)
        }
        
        # Global entropy penalty factor
        self.entropy_penalty_factor = 0.75
        
        # Calibration curve parameters (default, will be updated with data)
        self.calibration_curves = {
            0: {'slope': 0.9, 'intercept': 5.0},   # Banker
            1: {'slope': 0.9, 'intercept': 5.0},   # Player
            2: {'slope': 0.8, 'intercept': 3.0}    # Tie
        }
        
        # Isotonic regression calibrators (will be initialized when needed)
        self.isotonic_calibrators = {}
        
        # Update timestamp
        self.last_updated = time.time()
    
    def calculate_confidence(
        self, 
        probabilities: Dict[int, float],
        prediction: int,
        entropy: Optional[float] = None
    ) -> float:
        """
        Calculate confidence score from probability distribution.
        
        Args:
            probabilities: Probability distribution for each outcome
            prediction: The predicted outcome
            entropy: Optional entropy of the distribution
            
        Returns:
            float: Calibrated confidence score (0-100)
        """
        # Get raw probability for predicted outcome
        raw_probability = probabilities.get(prediction, 0.33)
        
        # Convert to percentage
        raw_confidence = raw_probability * 100
        
        # Calculate entropy if not provided
        if entropy is None:
            entropy = self._calculate_entropy(probabilities)
        
        # Apply entropy penalty (reduce confidence for high entropy/uncertain predictions)
        entropy_ratio = entropy / 1.58  # Normalize by maximum entropy for 3 outcomes
        entropy_penalty = self.entropy_penalty_factor * entropy_ratio
        
        # Apply entropy penalty and outcome-specific adjustment
        outcome_factor = self.outcome_adjustments.get(prediction, 1.0)
        adjusted_confidence = raw_confidence * outcome_factor * (1 - entropy_penalty)
        
        # Apply calibration curve
        calibrated_confidence = self._apply_calibration(adjusted_confidence, prediction)
        
        # Apply confidence cap based on outcome
        max_confidence = self.confidence_caps.get(prediction, 85.0)
        final_confidence = min(calibrated_confidence, max_confidence)
        
        return final_confidence
    
    def _apply_calibration(self, confidence: float, outcome: int) -> float:
        """
        Apply calibration curve to raw confidence score.
        
        Args:
            confidence: Raw confidence score (0-100)
            outcome: The outcome being predicted
            
        Returns:
            float: Calibrated confidence score
        """
        # Try isotonic calibrator if available
        if outcome in self.isotonic_calibrators:
            try:
                # Convert to probability scale for calibration
                prob = confidence / 100.0
                
                # Apply isotonic regression calibration
                calibrated_prob = self.isotonic_calibrators[outcome].predict([[prob]])[0]
                
                # Convert back to confidence scale
                return calibrated_prob * 100.0
            except Exception as e:
                logger.warning(f"Isotonic calibration failed: {e}")
                # Fall back to linear calibration
        
        # Apply linear calibration curve
        if outcome in self.calibration_curves:
            curve = self.calibration_curves[outcome]
            slope = curve.get('slope', 0.9)
            intercept = curve.get('intercept', 5.0)
            
            return (confidence * slope) + intercept
        
        # Default: return slightly reduced confidence (avoid overconfidence)
        return confidence * 0.9
    
    def adjust_confidence_for_pattern(
        self, 
        confidence: float, 
        pattern_type: str,
        prediction: int
    ) -> float:
        """
        Apply pattern-specific confidence adjustments.
        
        Args:
            confidence: Base confidence score
            pattern_type: Type of pattern identified
            prediction: The predicted outcome
            
        Returns:
            float: Adjusted confidence score
        """
        # Pattern-specific adjustment factors
        pattern_factors = {
            'streak': {
                0: 1.05,  # Slightly boost banker prediction confidence for streaks
                1: 1.05,  # Slightly boost player prediction confidence for streaks
                2: 0.9    # Reduce tie prediction confidence for streaks
            },
            'alternating': {
                0: 1.02,  # Small boost for banker predictions in alternating patterns
                1: 1.02,  # Small boost for player predictions in alternating patterns
                2: 0.85   # Larger reduction for tie predictions in alternating patterns
            },
            'tie_influenced': {
                0: 0.95,  # Reduce banker prediction confidence after ties
                1: 0.95,  # Reduce player prediction confidence after ties
                2: 1.1    # Boost tie prediction confidence in tie-influenced patterns
            },
            'banker_dominated': {
                0: 1.08,  # Boost banker prediction confidence in banker-dominated patterns
                1: 0.92,  # Reduce player prediction confidence in banker-dominated patterns
                2: 0.9    # Reduce tie prediction confidence in banker-dominated patterns
            },
            'player_dominated': {
                0: 0.92,  # Reduce banker prediction confidence in player-dominated patterns
                1: 1.08,  # Boost player prediction confidence in player-dominated patterns
                2: 0.9    # Reduce tie prediction confidence in player-dominated patterns
            },
            'chaotic': {
                0: 0.9,   # Reduce all confidence in chaotic patterns
                1: 0.9,
                2: 0.85
            },
            'no_pattern': {
                0: 1.0,   # Neutral adjustment for no pattern
                1: 1.0,
                2: 0.95
            }
        }
        
        # Get adjustment factor for this pattern and prediction
        if pattern_type in pattern_factors:
            factor = pattern_factors[pattern_type].get(prediction, 1.0)
        else:
            # Default for unknown patterns
            factor = 1.0
        
        # Apply adjustment with capping
        adjusted_confidence = confidence * factor
        
        # Apply confidence cap based on outcome
        max_confidence = self.confidence_caps.get(prediction, 85.0)
        return min(adjusted_confidence, max_confidence)
    
    def update_calibration(
        self, 
        prediction: int, 
        confidence: float, 
        correct: bool
    ) -> None:
        """
        Update calibration data with new prediction results.
        
        Args:
            prediction: The predicted outcome
            confidence: The confidence score assigned
            correct: Whether the prediction was correct
        """
        # Round confidence to nearest 5% for binning
        confidence_bin = round(confidence / 5) * 5
        
        # Update calibration data for this outcome
        if prediction in self.calibration_data:
            self.calibration_data[prediction]['bins'][confidence_bin]['total'] += 1
            if correct:
                self.calibration_data[prediction]['bins'][confidence_bin]['correct'] += 1
        
        # Update calibration curves periodically
        self._update_calibration_curves()
        
        # Track update time
        self.last_updated = time.time()
    
    def _update_calibration_curves(self) -> None:
        """
        Update calibration curves based on accumulated data.
        """
        try:
            # Check if we have sklearn available for isotonic regression
            from sklearn.isotonic import IsotonicRegression
            has_sklearn = True
        except ImportError:
            has_sklearn = False
        
        # Update for each outcome
        for outcome, data in self.calibration_data.items():
            bins = data['bins']
            
            # Need minimum number of bins with data
            if len(bins) < 3:
                continue
            
            # Prepare data for calibration
            confidences = []
            accuracies = []
            
            for conf_bin, stats in bins.items():
                if stats['total'] >= 5:  # Minimum samples for reliable estimates
                    confidences.append(conf_bin / 100.0)  # Convert to probability scale
                    accuracies.append(stats['correct'] / stats['total'])
            
            # Need minimum data points for fitting
            if len(confidences) < 3:
                continue
                
            # Update isotonic calibrators if sklearn is available
            if has_sklearn:
                try:
                    # Create isotonic regression calibrator
                    isotonic = IsotonicRegression(out_of_bounds='clip')
                    isotonic.fit(confidences, accuracies)
                    
                    # Store calibrator
                    self.isotonic_calibrators[outcome] = isotonic
                except Exception as e:
                    logger.warning(f"Error fitting isotonic calibration for outcome {outcome}: {e}")
            
            # Always update linear calibration as fallback
            try:
                # Simple linear regression for calibration curve
                # This is a fallback for when sklearn is not available
                n = len(confidences)
                if n >= 2:
                    x_mean = sum(confidences) / n
                    y_mean = sum(accuracies) / n
                    
                    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(confidences, accuracies))
                    denominator = sum((x - x_mean) ** 2 for x in confidences)
                    
                    if denominator != 0:
                        slope = numerator / denominator
                        intercept = y_mean - slope * x_mean
                        
                        # Update calibration curve
                        self.calibration_curves[outcome] = {
                            'slope': slope,
                            'intercept': intercept * 100  # Convert back to confidence scale
                        }
            except Exception as e:
                logger.warning(f"Error updating linear calibration for outcome {outcome}: {e}")
    
    def _calculate_entropy(self, probabilities: Dict[int, float]) -> float:
        """
        Calculate entropy of a probability distribution.
        
        Args:
            probabilities: Probability distribution
            
        Returns:
            float: Entropy value
        """
        entropy = 0.0
        for outcome, prob in probabilities.items():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive calibration report.
        
        Returns:
            dict: Calibration statistics and curve information
        """
        report = {
            'calibration_curves': {},
            'accuracy_by_confidence': {},
            'sample_counts': {}
        }
        
        for outcome, data in self.calibration_data.items():
            outcome_name = {0: 'Banker', 1: 'Player', 2: 'Tie'}.get(outcome, str(outcome))
            
            # Add calibration curve parameters
            if outcome in self.calibration_curves:
                report['calibration_curves'][outcome_name] = self.calibration_curves[outcome]
            
            # Add accuracy by confidence bin
            accuracy_by_conf = {}
            sample_counts = {}
            
            for conf_bin, stats in data['bins'].items():
                if stats['total'] > 0:
                    accuracy_by_conf[conf_bin] = stats['correct'] / stats['total']
                    sample_counts[conf_bin] = stats['total']
            
            report['accuracy_by_confidence'][outcome_name] = accuracy_by_conf
            report['sample_counts'][outcome_name] = sample_counts
        
        # Add last update timestamp
        report['last_updated'] = self.last_updated
        
        return report


def calibrate_confidence(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone confidence calibration function for backward compatibility.
    
    Args:
        prediction_result: Prediction result to calibrate
        
    Returns:
        dict: Calibrated prediction result
    """
    calculator = ConfidenceCalculator()
    
    # Extract key information
    predicted = prediction_result['prediction']
    distribution = prediction_result['distribution']
    
    # Convert percentage distribution to probabilities
    probabilities = {k: v/100 for k, v in distribution.items()}
    
    # Calculate entropy if available
    entropy = prediction_result.get('entropy')
    
    # Calculate calibrated confidence
    calibrated_confidence = calculator.calculate_confidence(
        probabilities, predicted, entropy
    )
    
    # Update result
    prediction_result['raw_confidence'] = prediction_result['confidence']
    prediction_result['confidence'] = calibrated_confidence
    prediction_result['calibrated'] = True
    
    return prediction_result