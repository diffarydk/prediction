"""
Pattern Analysis Component for Baccarat Prediction System.

This module implements comprehensive pattern recognition and analysis for
baccarat game sequences, detecting meaningful patterns that might influence
prediction accuracy and providing pattern-specific confidence adjustments.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """
    Analyzes baccarat game sequences to identify meaningful patterns.
    
    This component implements sophisticated pattern recognition algorithms
    to detect various baccarat patterns including streaks, alternations,
    tie-influenced patterns, and more complex sequences.
    
    It also provides pattern-specific confidence adjustments based on
    historical pattern effectiveness analysis.
    """
    
    # Pattern type definitions with associated recognition functions
    PATTERN_TYPES = {
        'streak': 'detect_streak',
        'alternating': 'detect_alternating',
        'tie_influenced': 'detect_tie_influence',
        'banker_dominated': 'detect_banker_dominance',
        'player_dominated': 'detect_player_dominance',
        'chaotic': 'detect_chaos'
    }
    
    def __init__(self):
        """Initialize pattern analyzer with performance tracking."""
        # Pattern performance tracking
        self.pattern_performance = {
            'streak': {'correct': 0, 'total': 0, 'last_correct': 0},
            'alternating': {'correct': 0, 'total': 0, 'last_correct': 0},
            'tie_influenced': {'correct': 0, 'total': 0, 'last_correct': 0},
            'banker_dominated': {'correct': 0, 'total': 0, 'last_correct': 0},
            'player_dominated': {'correct': 0, 'total': 0, 'last_correct': 0},
            'chaotic': {'correct': 0, 'total': 0, 'last_correct': 0},
            'no_pattern': {'correct': 0, 'total': 0, 'last_correct': 0}
        }
        
        # Pattern sequence tracking
        self.pattern_transitions = {}
        
    def analyze_pattern(self, sequence: Union[List[int], np.ndarray]) -> Dict[str, Any]:
        """
        Analyze a sequence to identify patterns and generate insights.
        
        Args:
            sequence: Game outcome sequence (0=Banker, 1=Player, 2=Tie)
            
        Returns:
            dict: Pattern analysis results
        """
        # Ensure sequence is in list format
        if isinstance(sequence, np.ndarray):
            if sequence.ndim > 1 and sequence.shape[0] == 1:
                # Handle 2D array with single row
                sequence = sequence[0].tolist()
            else:
                sequence = sequence.tolist()
        
        # Handle nested lists
        if isinstance(sequence, list) and sequence and isinstance(sequence[0], list):
            sequence = sequence[0]
        
        # Ensure we have enough data
        if len(sequence) < 3:
            return {
                'pattern_type': 'insufficient_data',
                'pattern_insight': 'Sequence too short for pattern analysis',
                'confidence_adjustment': 0.0
            }
        
        # Apply all pattern detection methods
        pattern_scores = {}
        pattern_insights = {}
        
        for pattern_type, detector_method in self.PATTERN_TYPES.items():
            # Dynamically call the detection method
            detector = getattr(self, detector_method)
            score, insight = detector(sequence)
            
            pattern_scores[pattern_type] = score
            if insight:
                pattern_insights[pattern_type] = insight
        
        # Determine dominant pattern
        if not pattern_scores:
            pattern_type = 'no_pattern'
            pattern_insight = 'No significant pattern detected in the sequence.'
            confidence_adjustment = 0.0
        else:
            # Get pattern with highest score
            pattern_type = max(pattern_scores, key=pattern_scores.get)
            pattern_score = pattern_scores[pattern_type]
            
            # Only consider it a pattern if score is significant
            if pattern_score < 0.4:
                pattern_type = 'no_pattern'
                pattern_insight = 'No significant pattern detected in the sequence.'
                confidence_adjustment = 0.0
            else:
                pattern_insight = pattern_insights.get(
                    pattern_type, 
                    f"{pattern_type.replace('_', ' ').title()} pattern detected."
                )
                confidence_adjustment = self._calculate_confidence_adjustment(
                    pattern_type, pattern_score
                )
        
        # Combine into result
        result = {
            'pattern_type': pattern_type,
            'pattern_insight': pattern_insight,
            'confidence_adjustment': confidence_adjustment,
            'pattern_scores': pattern_scores
        }
        
        # Add pattern effectiveness if available
        effectiveness = self.get_pattern_effectiveness(pattern_type)
        if effectiveness is not None:
            result['pattern_effectiveness'] = effectiveness
        
        return result
    
    def detect_streak(self, sequence: List[int]) -> Tuple[float, Optional[str]]:
        """
        Detect streaks of the same outcome.
        
        Args:
            sequence: Game outcome sequence
            
        Returns:
            tuple: (pattern_score, pattern_insight)
        """
        # Ignore ties for streak detection
        banker_player_seq = [x for x in sequence if x != 2]
        
        if not banker_player_seq:
            return 0.0, None
            
        # Check for streaks of at least 3
        max_streak = 1
        current_streak = 1
        
        for i in range(1, len(banker_player_seq)):
            if banker_player_seq[i] == banker_player_seq[i-1]:
                current_streak += 1
            else:
                max_streak = max(max_streak, current_streak)
                current_streak = 1
        
        max_streak = max(max_streak, current_streak)
        
        # Calculate score based on streak length
        if max_streak >= 3:
            score = min(1.0, (max_streak / 10) + 0.4)
            
            # Generate insight
            last_outcome = sequence[-1]
            outcome_name = {0: 'Banker', 1: 'Player', 2: 'Tie'}[last_outcome]
            
            # Check if we're currently in a streak
            current_streak_length = 1
            for i in range(len(sequence)-2, -1, -1):
                if sequence[i] == last_outcome and last_outcome != 2:
                    current_streak_length += 1
                else:
                    break
            
            if current_streak_length >= 3:
                insight = (f"Detected active streak of {outcome_name} ({current_streak_length} consecutive). "
                           f"Streaks often continue but may break with alternating pattern.")
            else:
                insight = (f"Detected streak pattern with maximum streak of {max_streak}. "
                           f"Watch for developing streaks in upcoming rounds.")
                           
            return score, insight
        
        return 0.0, None
    
    def detect_alternating(self, sequence: List[int]) -> Tuple[float, Optional[str]]:
        """
        Detect alternating banker/player patterns.
        
        Args:
            sequence: Game outcome sequence
            
        Returns:
            tuple: (pattern_score, pattern_insight)
        """
        # Ignore ties for alternating detection
        banker_player_seq = [x for x in sequence if x != 2]
        
        if len(banker_player_seq) < 4:
            return 0.0, None
            
        # Check for alternating patterns
        alternating_count = 0
        
        for i in range(2, len(banker_player_seq)):
            # Check if current matches pattern from 2 positions back
            if banker_player_seq[i] == banker_player_seq[i-2]:
                alternating_count += 1
        
        # Calculate score based on proportion of alternating pattern
        if len(banker_player_seq) >= 4:
            total_possible = len(banker_player_seq) - 2
            score = alternating_count / total_possible
            
            if score >= 0.7:
                insight = ("Detected strong alternating pattern between Banker and Player. "
                           "These patterns often continue or stabilize to one outcome.")
                return score, insight
            elif score >= 0.4:
                insight = ("Detected partial alternating pattern. May continue alternating "
                           "or transition to a streak.")
                return score, insight
        
        return 0.0, None
    
    def detect_tie_influence(self, sequence: List[int]) -> Tuple[float, Optional[str]]:
        """
        Detect tie-influenced patterns.
        
        Args:
            sequence: Game outcome sequence
            
        Returns:
            tuple: (pattern_score, pattern_insight)
        """
        # Check for ties in the sequence
        tie_positions = [i for i, x in enumerate(sequence) if x == 2]
        
        if not tie_positions:
            return 0.0, None
            
        # Calculate tie frequency
        tie_frequency = len(tie_positions) / len(sequence)
        
        # Check recent ties
        recent_ties = sum(1 for x in sequence[-3:] if x == 2)
        
        if recent_ties >= 1:
            score = min(1.0, tie_frequency + 0.3)
            
            if recent_ties >= 2:
                insight = ("Multiple recent ties detected. After consecutive ties, "
                           "outcomes often become less predictable with potential for streaks.")
            else:
                insight = ("Recent tie detected. After ties, outcomes can shift unpredictably. "
                           "Watch for developing patterns in next rounds.")
            
            return score, insight
        elif tie_frequency >= 0.2:
            score = tie_frequency
            insight = (f"High tie frequency ({tie_frequency:.2f}) in sequence. "
                       "Tie-heavy sequences often show disrupted patterns.")
            return score, insight
        
        return 0.0, None
    
    def detect_banker_dominance(self, sequence: List[int]) -> Tuple[float, Optional[str]]:
        """
        Detect banker-dominated patterns.
        
        Args:
            sequence: Game outcome sequence
            
        Returns:
            tuple: (pattern_score, pattern_insight)
        """
        # Calculate banker frequency
        banker_count = sequence.count(0)
        banker_frequency = banker_count / len(sequence)
        
        # Calculate recent banker frequency
        recent_banker_count = sequence[-3:].count(0)
        recent_banker_frequency = recent_banker_count / len(sequence[-3:])
        
        # Combined score giving more weight to recent outcomes
        score = (0.7 * recent_banker_frequency) + (0.3 * banker_frequency)
        
        if score >= 0.6:
            insight = (f"Banker-dominated pattern ({banker_frequency:.2f} overall, "
                       f"{recent_banker_frequency:.2f} recent). Banker dominance "
                       f"often continues but watch for reversal signals.")
            return score, insight
        
        return 0.0, None
    
    def detect_player_dominance(self, sequence: List[int]) -> Tuple[float, Optional[str]]:
        """
        Detect player-dominated patterns.
        
        Args:
            sequence: Game outcome sequence
            
        Returns:
            tuple: (pattern_score, pattern_insight)
        """
        # Calculate player frequency
        player_count = sequence.count(1)
        player_frequency = player_count / len(sequence)
        
        # Calculate recent player frequency
        recent_player_count = sequence[-3:].count(1)
        recent_player_frequency = recent_player_count / len(sequence[-3:])
        
        # Combined score giving more weight to recent outcomes
        score = (0.7 * recent_player_frequency) + (0.3 * player_frequency)
        
        if score >= 0.6:
            insight = (f"Player-dominated pattern ({player_frequency:.2f} overall, "
                       f"{recent_player_frequency:.2f} recent). Player dominance "
                       f"often continues but watch for reversal signals.")
            return score, insight
        
        return 0.0, None
    
    def detect_chaos(self, sequence: List[int]) -> Tuple[float, Optional[str]]:
        """
        Detect chaotic patterns with no clear structure.
        
        Args:
            sequence: Game outcome sequence
            
        Returns:
            tuple: (pattern_score, pattern_insight)
        """
        # Calculate entropy of the sequence
        counts = Counter(sequence)
        total = len(sequence)
        entropy = 0
        
        for outcome, count in counts.items():
            probability = count / total
            entropy -= probability * np.log2(probability)
        
        # Max entropy for 3 outcomes is log2(3) ≈ 1.58
        normalized_entropy = entropy / 1.58
        
        # Check for repeating subsequences
        has_repeating = self._check_repeating_subsequences(sequence)
        
        # Chaotic sequences have high entropy and lack repeating subsequences
        if normalized_entropy > 0.8 and not has_repeating:
            score = normalized_entropy
            insight = ("Chaotic pattern with high unpredictability. No clear structure "
                       "detected. Consider more conservative betting.")
            return score, insight
        
        return 0.0, None
    
    def _check_repeating_subsequences(self, sequence: List[int], min_length: int = 2) -> bool:
        """
        Check for repeating subsequences in the sequence.
        
        Args:
            sequence: Game outcome sequence
            min_length: Minimum subsequence length to consider
            
        Returns:
            bool: True if repeating subsequences found
        """
        seq_length = len(sequence)
        
        # Check for subsequences of different lengths
        for length in range(min_length, seq_length // 2 + 1):
            for i in range(seq_length - length * 2 + 1):
                subsequence = tuple(sequence[i:i+length])
                
                # Look for this subsequence elsewhere
                for j in range(i + length, seq_length - length + 1):
                    if tuple(sequence[j:j+length]) == subsequence:
                        return True
        
        return False
    
    def _calculate_confidence_adjustment(self, pattern_type: str, pattern_score: float) -> float:
        """
        Calculate confidence adjustment based on pattern type and score.
        
        Args:
            pattern_type: Identified pattern type
            pattern_score: Strength of the pattern (0-1)
            
        Returns:
            float: Confidence adjustment factor (-0.2 to 0.2)
        """
        # Get pattern effectiveness if available
        effectiveness = self.get_pattern_effectiveness(pattern_type)
        
        # Default baseline adjustments by pattern type
        baseline_adjustments = {
            'streak': 0.1,           # Streaks tend to be more predictable
            'alternating': 0.05,     # Alternating patterns moderately predictable
            'tie_influenced': -0.1,  # Ties introduce unpredictability
            'banker_dominated': 0.05,
            'player_dominated': 0.05,
            'chaotic': -0.15,        # Chaotic patterns reduce confidence
            'no_pattern': 0.0
        }
        
        # Start with baseline adjustment
        adjustment = baseline_adjustments.get(pattern_type, 0.0)
        
        # Scale by pattern score
        adjustment *= pattern_score
        
        # Incorporate effectiveness if available (recent performance)
        if effectiveness is not None:
            if effectiveness > 0.5:
                # Pattern is performing well, increase adjustment
                adjustment *= (1.0 + (effectiveness - 0.5) * 2)
            else:
                # Pattern is performing poorly, reduce or reverse adjustment
                adjustment *= (effectiveness / 0.5)
        
        # Limit adjustment range
        return max(-0.2, min(0.2, adjustment))
    
    def update_pattern_performance(
        self, 
        pattern_type: str, 
        prediction: int, 
        actual: int
    ) -> None:
        """
        Update pattern performance tracking.
        
        Args:
            pattern_type: The pattern type identified
            prediction: The predicted outcome
            actual: The actual outcome
        """
        if pattern_type not in self.pattern_performance:
            self.pattern_performance[pattern_type] = {
                'correct': 0, 'total': 0, 'last_correct': 0
            }
        
        # Update statistics
        self.pattern_performance[pattern_type]['total'] += 1
        
        if prediction == actual:
            self.pattern_performance[pattern_type]['correct'] += 1
            self.pattern_performance[pattern_type]['last_correct'] = np.datetime64('now')
    
    def get_pattern_effectiveness(self, pattern_type: str) -> Optional[float]:
        """
        Calculate effectiveness of a pattern type based on historical performance.
        
        Args:
            pattern_type: The pattern type to evaluate
            
        Returns:
            float: Effectiveness score (0-1) or None if insufficient data
        """
        if pattern_type not in self.pattern_performance:
            return None
            
        stats = self.pattern_performance[pattern_type]
        
        # Need minimum samples
        if stats['total'] < 5:
            return None
            
        # Basic accuracy
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        # Apply recency factor - patterns can dissolve over time
        recency_factor = 1.0
        if stats['last_correct'] != 0:
            # Calculate days since last correct prediction
            try:
                last_correct = np.datetime64(stats['last_correct'])
                now = np.datetime64('now')
                days_since = (now - last_correct).astype('timedelta64[D]').astype(int)
                
                # Reduce effectiveness for older patterns
                if days_since > 0:
                    recency_factor = max(0.5, 1.0 - (days_since / 14))  # Decay over two weeks
            except (TypeError, ValueError):
                # Handle case where last_correct isn't a valid timestamp
                pass
        
        return accuracy * recency_factor
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive pattern performance report.
        
        Returns:
            dict: Pattern performance statistics
        """
        report = {
            'pattern_effectiveness': {},
            'sample_counts': {},
            'accuracy': {}
        }
        
        for pattern_type, stats in self.pattern_performance.items():
            if stats['total'] > 0:
                effectiveness = self.get_pattern_effectiveness(pattern_type)
                accuracy = stats['correct'] / stats['total']
                
                report['pattern_effectiveness'][pattern_type] = effectiveness
                report['sample_counts'][pattern_type] = stats['total']
                report['accuracy'][pattern_type] = accuracy
        
        return report


# Legacy functions for backward compatibility

def get_pattern_insight(prev_rounds):
    """
    Legacy wrapper for backward compatibility.
    
    Args:
        prev_rounds: Previous game outcomes
        
    Returns:
        str: Pattern insight text
    """
    analyzer = PatternAnalyzer()
    result = analyzer.analyze_pattern(prev_rounds)
    return result['pattern_insight']

def extract_pattern_type(pattern_insight):
    """
    Legacy wrapper for backward compatibility.
    
    Args:
        pattern_insight: Pattern insight text
        
    Returns:
        str: Pattern type identifier
    """
    if not pattern_insight:
        return "no_pattern"
    
    if "streak" in pattern_insight.lower():
        return "streak"
    elif "alternating" in pattern_insight.lower():
        return "alternating"
    elif "tie" in pattern_insight.lower():
        return "tie_influenced"
    elif "banker" in pattern_insight.lower():
        return "banker_dominated"
    elif "player" in pattern_insight.lower():
        return "player_dominated"
    elif "chaotic" in pattern_insight.lower():
        return "chaotic"
    else:
        return "no_pattern"