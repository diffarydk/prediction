"""
Enhanced Betting System Module for Baccarat Prediction System.

This module implements a robust betting management system with multiple
strategies, transaction-based balance management, and comprehensive
performance tracking. It provides:

1. Strategy-based bet sizing with dynamic adjustment
2. Confidence-calibrated bet recommendations
3. Transaction-based balance management
4. Comprehensive performance analytics
5. Multi-strategy support with consistent interface
"""

import os
import json
import logging
import numpy as np
import time
from colorama import Fore, Style
from typing import Dict, List, Any, Optional, Union, Tuple

from config import (
    INITIAL_BALANCE, MIN_BET, MAX_BET,
    BALANCE_FILE, BET_HISTORY_FILE,
    BANKER_PAYOUT, PLAYER_PAYOUT
)

# Configure logging
logger = logging.getLogger(__name__)


class BettingStrategy:
    """Base class for betting strategies with common interface."""
    
    def __init__(self, system):
        """Initialize with reference to betting system."""
        self.system = system
        self.name = "base"
    
    def calculate_bet(self, confidence: float, certainty: Optional[float] = None) -> float:
        """Calculate bet amount based on strategy-specific algorithm."""
        raise NotImplementedError("Strategy must implement calculate_bet method")
    
    def update_after_win(self) -> None:
        """Update strategy state after a win."""
        pass
    
    def update_after_loss(self) -> None:
        """Update strategy state after a loss."""
        pass
    
    def reset(self) -> None:
        """Reset strategy to initial state."""
        pass


class FibonacciStrategy(BettingStrategy):
    """Fibonacci sequence-based betting strategy."""
    
    def __init__(self, system):
        """Initialize with reference to betting system."""
        super().__init__(system)
        self.name = "fibonacci"
        self.fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
        self.current_index = 0
    
    def calculate_bet(self, confidence: float, certainty: Optional[float] = None) -> float:
        """
        Calculate dynamic bet amount using Fibonacci sequence, confidence, and certainty.
        
        Args:
            confidence: Prediction confidence percentage
            certainty: Optional prediction certainty percentage (higher is better)
            
        Returns:
            float: Recommended bet amount
        """
        multiplier = self.fibonacci_sequence[self.current_index]
        
        # Calculate base unit dynamically - between 1% and 3% based on balance size
        if self.system.balance < 50000:
            base_percentage = 0.01  # 1% for small balances
        elif self.system.balance < 200000:
            base_percentage = 0.015  # 1.5% for medium balances
        else:
            base_percentage = 0.02  # 2% for large balances
        
        base_unit = self.system.balance * base_percentage
        
        # Scale confidence factor - stronger effect from confidence
        confidence_factor = (confidence - 20) / 80
        confidence_factor = max(0.1, min(1.0, confidence_factor))
        
        # Apply certainty adjustment if provided (certainty comes from entropy calculation)
        certainty_multiplier = 1.0
        if certainty is not None and certainty > 0:
            # Certainty under 50% reduces bet, over 50% increases it
            certainty_multiplier = 0.5 + (certainty / 100) 
            certainty_multiplier = max(0.5, min(1.5, certainty_multiplier))
        
        # Combine all factors for final bet calculation
        bet_amount = base_unit * confidence_factor * multiplier * certainty_multiplier
        
        return bet_amount
    
    def update_after_win(self) -> None:
        """Move back two steps in Fibonacci sequence after a win."""
        self.current_index = max(0, self.current_index - 2)
    
    def update_after_loss(self) -> None:
        """Move forward one step in Fibonacci sequence after a loss."""
        self.current_index = min(self.current_index + 1, len(self.fibonacci_sequence) - 1)
    
    def reset(self) -> None:
        """Reset to beginning of Fibonacci sequence."""
        self.current_index = 0


class MartingaleStrategy(BettingStrategy):
    """Martingale (double after loss) betting strategy."""
    
    def __init__(self, system):
        """Initialize with reference to betting system."""
        super().__init__(system)
        self.name = "martingale"
        self.base_bet = None
    
    def calculate_bet(self, confidence: float, certainty: Optional[float] = None) -> float:
        """
        Calculate bet amount using Martingale system (double after loss).
        
        Args:
            confidence: Prediction confidence
            certainty: Optional prediction certainty
            
        Returns:
            float: Recommended bet amount
        """
        # Calculate base bet if not set
        if self.base_bet is None:
            # Base bet is 0.5% of balance, adjusted by confidence
            confidence_factor = (confidence - 20) / 80
            confidence_factor = max(0.1, min(1.0, confidence_factor))
            self.base_bet = self.system.balance * 0.005 * confidence_factor
        
        # In Martingale, we double bet after each loss
        if self.system.consecutive_losses > 0:
            bet_amount = self.base_bet * (2 ** self.system.consecutive_losses)
        else:
            bet_amount = self.base_bet
        
        # Apply certainty adjustment if provided
        if certainty is not None and certainty > 0:
            certainty_factor = 0.8 + (certainty / 500)  # Smaller adjustment for Martingale
            certainty_factor = max(0.8, min(1.2, certainty_factor))
            bet_amount *= certainty_factor
        
        return bet_amount
    
    def update_after_win(self) -> None:
        """Reset to base bet after a win."""
        pass  # Base bet remains unchanged, system will reset consecutive losses
    
    def update_after_loss(self) -> None:
        """Bet is automatically doubled based on consecutive_losses counter."""
        pass
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.base_bet = None


class DAlembert(BettingStrategy):
    """D'Alembert (add unit after loss, subtract after win) betting strategy."""
    
    def __init__(self, system):
        """Initialize with reference to betting system."""
        super().__init__(system)
        self.name = "dalembert"
        self.base_bet = None
        self.unit = None
    
    def calculate_bet(self, confidence: float, certainty: Optional[float] = None) -> float:
        """
        Calculate bet amount using D'Alembert system.
        
        Args:
            confidence: Prediction confidence
            certainty: Optional prediction certainty
            
        Returns:
            float: Recommended bet amount
        """
        # Calculate base values if not set
        if self.base_bet is None:
            # Base bet is 0.5% of balance, adjusted by confidence
            confidence_factor = (confidence - 20) / 80
            confidence_factor = max(0.1, min(1.0, confidence_factor))
            self.base_bet = self.system.balance * 0.005 * confidence_factor
            self.unit = self.base_bet * 0.2  # Unit is 20% of base bet
        
        # D'Alembert: add one unit after loss, subtract after win
        win_loss_diff = self.system.consecutive_losses - self.system.consecutive_wins
        bet_amount = self.base_bet + (win_loss_diff * self.unit)
        bet_amount = max(self.base_bet, bet_amount)  # Minimum bet is base_bet
        
        # Apply certainty adjustment if provided
        if certainty is not None and certainty > 0:
            certainty_factor = 0.9 + (certainty / 1000)  # Smaller adjustment for D'Alembert
            certainty_factor = max(0.9, min(1.1, certainty_factor))
            bet_amount *= certainty_factor
        
        return bet_amount
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.base_bet = None
        self.unit = None


class PercentageStrategy(BettingStrategy):
    """Percentage-based betting strategy."""
    
    def __init__(self, system, percentage=1.0):
        """
        Initialize with reference to betting system.
        
        Args:
            system: The betting system
            percentage: Base percentage for betting (1.0 = 1%)
        """
        super().__init__(system)
        self.name = "percentage"
        self.percentage = percentage
    
    def calculate_bet(self, confidence: float, certainty: Optional[float] = None) -> float:
        """
        Calculate bet amount using percentage of current balance.
        
        Args:
            confidence: Prediction confidence
            certainty: Optional prediction certainty
            
        Returns:
            float: Recommended bet amount
        """
        # Scale confidence factor - stronger effect from confidence
        confidence_factor = (confidence - 20) / 80
        confidence_factor = max(0.1, min(1.0, confidence_factor))
        
        # Calculate bet as percentage of balance, adjusted by confidence
        bet_amount = self.system.balance * (self.percentage / 100) * confidence_factor
        
        # Apply certainty adjustment if provided
        if certainty is not None and certainty > 0:
            certainty_factor = 0.8 + (certainty / 250)
            certainty_factor = max(0.8, min(1.2, certainty_factor))
            bet_amount *= certainty_factor
        
        return bet_amount
    
    def set_percentage(self, percentage: float) -> None:
        """
        Set the betting percentage.
        
        Args:
            percentage: Base percentage for betting (1.0 = 1%)
        """
        self.percentage = max(0.1, min(10.0, percentage))


class BettingSystem:
    """
    Enhanced betting system with multiple strategies and comprehensive tracking.
    
    Provides a robust framework for bet management, strategy execution,
    and performance analysis with transaction-based operations.
    """
    
    def __init__(self):
        """
        Initialize the betting system with balance, history and settings.
        """
        # Initialize core attributes
        self.balance = self._load_balance()
        self.initial_balance = self.balance
        self.bet_history = self._load_bet_history()
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.session_profit = 0
        
        # Initialize available strategies
        self.strategies = {
            "fibonacci": FibonacciStrategy(self),
            "martingale": MartingaleStrategy(self),
            "dalembert": DAlembert(self),
            "percentage": PercentageStrategy(self)
        }
        
        # Set default strategy
        self.active_strategy = "fibonacci"
    
    def _load_balance(self) -> float:
        """
        Load balance from file with robust error handling.
        
        Returns:
            float: Current balance
        """
        if os.path.exists(BALANCE_FILE):
            try:
                with open(BALANCE_FILE, 'r') as f:
                    data = json.load(f)
                    return data.get('balance', INITIAL_BALANCE)
            except Exception as e:
                logger.error(f"Error loading balance: {e}")
                print(f"{Fore.RED}Error loading balance: {e}. Using default.")
                return INITIAL_BALANCE
        else:
            return INITIAL_BALANCE
    
    def _save_balance(self) -> bool:
        """
        Save current balance to file using transaction-based approach.
        
        Returns:
            bool: True if successful
        """
        try:
            # Ensure directory exists
            directory = os.path.dirname(BALANCE_FILE)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Write to temporary file first
            temp_file = f"{BALANCE_FILE}.temp"
            with open(temp_file, 'w') as f:
                json.dump({
                    'balance': self.balance, 
                    'last_updated': time.time()
                }, f)
            
            # Move temporary file to final location (atomic operation)
            os.replace(temp_file, BALANCE_FILE)
            
            return True
        except Exception as e:
            logger.error(f"Error saving balance: {e}")
            print(f"{Fore.RED}Error saving balance: {e}")
            return False
    
    def _load_bet_history(self) -> List[Dict[str, Any]]:
        """
        Load betting history from file with robust error handling.
        
        Returns:
            list: Bet history records
        """
        if os.path.exists(BET_HISTORY_FILE):
            try:
                with open(BET_HISTORY_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading bet history: {e}")
                print(f"{Fore.RED}Error loading bet history: {e}. Starting fresh.")
                return []
        else:
            return []
    
    def _save_bet_history(self) -> bool:
        """
        Save betting history with robust error handling and type conversion.
        
        Returns:
            bool: True if successful
        """
        try:
            # Ensure directory exists
            directory = os.path.dirname(BET_HISTORY_FILE)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            # Process history for JSON serialization
            history_copy = []
            for bet in self.bet_history:
                bet_copy = {}
                for key, value in bet.items():
                    # Convert numpy types to native Python types
                    if hasattr(value, 'item'):
                        bet_copy[key] = value.item()
                    else:
                        bet_copy[key] = value
                history_copy.append(bet_copy)
            
            # Write to temporary file first
            temp_file = f"{BET_HISTORY_FILE}.temp"
            with open(temp_file, 'w') as f:
                json.dump(history_copy, f, indent=2)
            
            # Move temporary file to final location (atomic operation)
            os.replace(temp_file, BET_HISTORY_FILE)
            
            return True
        except Exception as e:
            logger.error(f"Error saving bet history: {e}", exc_info=True)
            print(f"{Fore.RED}Error saving bet history: {e}")
            return False
    
    def set_betting_strategy(self, strategy: str, percentage: float = 1.0) -> bool:
        """
        Set the active betting strategy with parameter configuration.
        
        Args:
            strategy: The strategy name ("fibonacci", "martingale", "dalembert", "percentage")
            percentage: Base percentage for percentage-based betting (1.0 = 1%)
        
        Returns:
            bool: Success status
        """
        strategy = strategy.lower()
        if strategy not in self.strategies:
            print(f"{Fore.RED}Invalid strategy: {strategy}. Valid options are: {', '.join(self.strategies.keys())}")
            return False
        
        # Reset the current strategy
        self.strategies[self.active_strategy].reset()
        
        # Set the new active strategy
        self.active_strategy = strategy
        
        # Configure percentage for percentage strategy
        if strategy == "percentage":
            self.strategies[strategy].set_percentage(percentage)
        
        print(f"{Fore.GREEN}Betting strategy set to: {self.active_strategy}")
        if strategy == "percentage":
            print(f"{Fore.GREEN}Bet percentage set to: {self.strategies[strategy].percentage}%")
        
        return True
    
    def recommend_bet(
        self, 
        prediction: int, 
        confidence: float, 
        certainty: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Recommend a bet amount based on prediction quality and active strategy.
        
        Args:
            prediction: The predicted outcome (0=Banker, 1=Player, 2=Tie)
            confidence: Confidence percentage for the prediction
            certainty: Optional certainty value (from entropy calculation)
            
        Returns:
            dict: Recommended bet details
        """
        # Convert numpy types if needed
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        if hasattr(confidence, 'item'):
            confidence = confidence.item()
        if hasattr(certainty, 'item'):
            certainty = certainty.item()
        
        # Don't bet on Tie (2) outcomes
        if prediction == 2:
            return {
                'should_bet': False,
                'amount': 0,
                'reason': "System doesn't bet on Tie outcomes due to unfavorable odds"
            }
        
        # Determine if confidence is high enough to bet
        if confidence < 20:
            return {
                'should_bet': False,
                'amount': 0,
                'reason': f"Confidence too low ({confidence:.1f}%)"
            }
        
        # Get the active strategy
        strategy = self.strategies[self.active_strategy]
        
        # Calculate bet amount using the active strategy
        bet_amount = strategy.calculate_bet(confidence, certainty)
        
        # Ensure bet is within allowed limits
        bet_amount = max(MIN_BET, min(bet_amount, MAX_BET, self.balance))
        bet_amount = round(bet_amount / 500) * 500  # Round to nearest 500
        
        logger.info(f"Recommending bet of {bet_amount:,} Rp using {self.active_strategy} strategy")
        
        return {
            'should_bet': True,
            'amount': bet_amount,
            'outcome': prediction,
            'confidence': confidence,
            'certainty': certainty,
            'strategy': self.active_strategy
        }
    
    def place_bet(
        self, 
        bet_amount: float, 
        predicted_outcome: int, 
        confidence: float
    ) -> Optional[Dict[str, Any]]:
        """
        Place a bet with transaction-based balance management.
        
        Args:
            bet_amount: Amount to bet
            predicted_outcome: The predicted outcome (0=Banker, 1=Player)
            confidence: Confidence percentage for the prediction
            
        Returns:
            dict: Bet details or None if insufficient balance
        """
        # Check if there's any balance available
        if self.balance <= 0:
            print(f"{Fore.RED}Cannot place bet: No balance available.")
            return None
        
        # Ensure bet amount doesn't exceed balance and is at least MIN_BET
        bet_amount = min(bet_amount, self.balance)
        
        # Check if bet amount meets minimum requirement
        if bet_amount < MIN_BET:
            print(f"{Fore.RED}Cannot place bet: Minimum bet is {MIN_BET:,} Rp but you only have {self.balance:,} Rp.")
            return None
        
        # Convert numpy types to Python native types
        if hasattr(bet_amount, 'item'):
            bet_amount = bet_amount.item()
        if hasattr(predicted_outcome, 'item'):
            predicted_outcome = predicted_outcome.item()
        if hasattr(confidence, 'item'):
            confidence = confidence.item()
        
        # Record the bet
        bet = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'amount': bet_amount,
            'predicted_outcome': predicted_outcome,
            'confidence': confidence,
            'balance_before': self.balance,
            'strategy': self.active_strategy
        }
        
        # Update balance (deducted now, will be returned with profit if won)
        self.balance -= bet_amount
        self._save_balance()
        
        logger.info(f"Placed bet of {bet_amount:,} Rp, Balance before: {self.balance + bet_amount:,} Rp, after: {self.balance:,} Rp")
        
        return bet
    
    def resolve_bet(self, bet: Dict[str, Any], actual_outcome: int) -> Dict[str, Any]:
        """
        Resolve a bet based on the actual outcome with transaction-based balance update.
        
        Args:
            bet: The original bet details
            actual_outcome: The actual outcome that occurred
            
        Returns:
            dict: Updated bet details with results
        """
        # Ensure numpy types are converted
        if hasattr(actual_outcome, 'item'):
            actual_outcome = actual_outcome.item()
        
        # Extract bet details
        predicted = bet['predicted_outcome']
        bet_amount = bet['amount']
        
        logger.info(f"Resolving bet - Amount: {bet_amount:,} Rp, Predicted: {predicted}, Actual: {actual_outcome}")
        
        # Determine if bet won
        won = (predicted == actual_outcome)
        
        # Get active strategy object
        strategy = self.strategies[self.active_strategy]
        
        # Calculate profit/loss
        if won:
            if predicted == 0:  # Banker
                # For banker, payout is 95% (5% commission)
                profit = bet_amount * BANKER_PAYOUT
                winnings = bet_amount + profit
            else:  # Player
                # For player, payout is 100% (1:1)
                profit = bet_amount * PLAYER_PAYOUT
                winnings = bet_amount + profit
            
            # Add winnings to balance
            self.balance += winnings
            self.session_profit += profit
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            
            # Update strategy after win
            strategy.update_after_win()
            
        else:
            # If lost, profit is negative bet amount (already deducted from balance)
            profit = -bet_amount
            self.session_profit += profit
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            # Update strategy after loss
            strategy.update_after_loss()
        
        # Update bet record with results
        bet.update({
            'actual_outcome': actual_outcome,
            'won': won,
            'profit': profit,
            'balance_after': self.balance
        })
        
        # Add to history and save
        self.bet_history.append(bet)
        self._save_bet_history()
        self._save_balance()
        
        return bet
    
    def reset_balance(self, amount: Optional[float] = None) -> float:
        """
        Reset the balance to initial amount or to a specific amount.
        
        Args:
            amount: Optional specific amount to set. If None, uses the initial balance.
            
        Returns:
            float: The new balance
        """
        if amount is not None:
            self.balance = min(amount, self.initial_balance)
        else:
            self.balance = self.initial_balance
        
        # Reset session profit
        self.session_profit = 0
        
        # Save the updated balance
        self._save_balance()
        
        logger.info(f"Balance reset to {self.balance:,} Rp")
        
        return self.balance
    
    def evaluate_strategy_performance(
        self, 
        time_period: str = 'all'
    ) -> Dict[str, Any]:
        """
        Evaluate betting strategy performance with comprehensive metrics.
        
        Args:
            time_period: Time period for evaluation ('all', 'recent', 'medium')
            
        Returns:
            dict: Performance metrics
        """
        if not self.bet_history:
            return {"error": "No betting history available"}
        
        # Filter bets by time period
        if time_period == 'recent':
            # Last 20 bets
            bets = self.bet_history[-20:]
        elif time_period == 'medium':
            # Last 50 bets
            bets = self.bet_history[-50:]
        else:
            # All bets
            bets = self.bet_history
        
        if not bets:
            return {"error": "No bets in the selected time period"}
        
        # Calculate metrics
        total_bets = len(bets)
        total_wagered = sum(bet.get('amount', 0) for bet in bets)
        total_profit = sum(bet.get('profit', 0) for bet in bets)
        win_count = sum(1 for bet in bets if bet.get('won', False))
        
        # Performance metrics
        win_rate = win_count / total_bets
        roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
        
        # Calculate risk-adjusted return
        profits = [bet.get('profit', 0) for bet in bets]
        std_dev = np.std(profits) if len(profits) > 1 else 0
        sharpe = (np.mean(profits) / std_dev) if std_dev > 0 else 0
        
        # Analyze streak data
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for bet in bets:
            if bet.get('won', False):
                current_win_streak += 1
                current_loss_streak = 0
            else:
                current_loss_streak += 1
                current_win_streak = 0
                
            max_win_streak = max(max_win_streak, current_win_streak)
            max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        # Strategy-specific performance
        strategy_performance = {}
        for strategy_name in self.strategies.keys():
            strategy_bets = [bet for bet in bets if bet.get('strategy') == strategy_name]
            if strategy_bets:
                strat_win_count = sum(1 for bet in strategy_bets if bet.get('won', False))
                strat_total = len(strategy_bets)
                strat_total_profit = sum(bet.get('profit', 0) for bet in strategy_bets)
                
                strategy_performance[strategy_name] = {
                    'total_bets': strat_total,
                    'win_rate': strat_win_count / strat_total,
                    'total_profit': strat_total_profit,
                    'profit_per_bet': strat_total_profit / strat_total
                }
        
        return {
            'time_period': time_period,
            'total_bets': total_bets,
            'win_rate': win_rate,
            'roi': roi,
            'total_profit': total_profit,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'sharpe_ratio': sharpe,
            'ev_per_bet': total_profit / total_bets,
            'strategy_performance': strategy_performance
        }
    # Add to BettingSystem class in betting_system.py

    def _initialize_performance_tracking(self) -> None:
        """
        Initialize performance tracking metrics.
        """
        self.performance_metrics = {
            'recommend_time': [],
            'place_bet_time': [],
            'resolve_bet_time': [],
            'total_bet_handling_time': []
        }

    def place_bet_with_profiling(
        self, 
        bet_amount: float, 
        predicted_outcome: int, 
        confidence: float
    ) -> Optional[Dict[str, Any]]:
        """
        Place a bet with transaction-based balance management and performance profiling.
        
        Args:
            bet_amount: Amount to bet
            predicted_outcome: The predicted outcome (0=Banker, 1=Player)
            confidence: Confidence percentage for the prediction
            
        Returns:
            dict: Bet details or None if insufficient balance
        """
        start_time = time.time()
        
        # Check if there's any balance available
        if self.balance <= 0:
            print(f"{Fore.RED}Cannot place bet: No balance available.")
            return None
        
        # Ensure bet amount doesn't exceed balance and is at least MIN_BET
        bet_amount = min(bet_amount, self.balance)
        
        # Check if bet amount meets minimum requirement
        if bet_amount < MIN_BET:
            print(f"{Fore.RED}Cannot place bet: Minimum bet is {MIN_BET:,} Rp but you only have {self.balance:,} Rp.")
            return None
        
        # Convert numpy types to Python native types
        if hasattr(bet_amount, 'item'):
            bet_amount = bet_amount.item()
        if hasattr(predicted_outcome, 'item'):
            predicted_outcome = predicted_outcome.item()
        if hasattr(confidence, 'item'):
            confidence = confidence.item()
        
        # Record the bet
        bet = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'amount': bet_amount,
            'predicted_outcome': predicted_outcome,
            'confidence': confidence,
            'balance_before': self.balance,
            'strategy': self.active_strategy
        }
        
        # Update balance (deducted now, will be returned with profit if won)
        self.balance -= bet_amount
        self._save_balance()
        
        # Record performance metrics
        execution_time = time.time() - start_time
        if hasattr(self, 'performance_metrics'):
            self.performance_metrics['place_bet_time'].append(execution_time)
        
        # Log performance data for optimization
        if execution_time > 0.1:  # Log slow operations (>100ms)
            logger.info(f"Slow bet placement: {execution_time*1000:.2f}ms for {bet_amount:,} Rp")
        
        logger.info(f"Placed bet of {bet_amount:,} Rp, Balance before: {self.balance + bet_amount:,} Rp, after: {self.balance:,} Rp")
        
        return bet

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for betting operations.
        
        Returns:
            dict: Performance metrics
        """
        # Ensure performance metrics are initialized
        if not hasattr(self, 'performance_metrics'):
            self._initialize_performance_tracking()
            return {'error': 'No performance data available yet'}
        
        metrics = {}
        
        # Calculate statistics for each operation type
        for operation, times in self.performance_metrics.items():
            if times:
                metrics[operation] = {
                    'count': len(times),
                    'avg_ms': sum(times) / len(times) * 1000,
                    'min_ms': min(times) * 1000,
                    'max_ms': max(times) * 1000,
                    'total_ms': sum(times) * 1000
                }
        
        # Calculate transaction metrics
        metrics['transaction_success_rate'] = self._calculate_transaction_success_rate()
        
        return metrics

    def _calculate_transaction_success_rate(self) -> float:
        """
        Calculate betting transaction success rate.
        
        Returns:
            float: Success rate percentage
        """
        if not self.bet_history:
            return 100.0
        
        # Count transactions with and without errors
        success_count = len([bet for bet in self.bet_history if 'error' not in bet])
        total_count = len(self.bet_history)
        
        return (success_count / total_count) * 100
    def display_balance(self) -> None:
        """
        Display current balance with enhanced visualization.
        """
        # Determine balance color based on performance
        if self.balance > self.initial_balance:
            balance_color = Fore.GREEN
        elif self.balance < self.initial_balance * 0.8:
            balance_color = Fore.RED
        else:
            balance_color = Fore.YELLOW
            
        # Calculate session profit percentage
        if self.initial_balance > 0:
            profit_percentage = (self.balance - self.initial_balance) / self.initial_balance * 100
        else:
            profit_percentage = 0
            
        profit_color = Fore.GREEN if profit_percentage >= 0 else Fore.RED
        profit_sign = "+" if profit_percentage >= 0 else ""
        
        print(f"\n{Fore.CYAN}=== Balance Information ===")
        print(f"Current Balance: {balance_color}{self.balance:,} Rp")
        print(f"Session Profit: {profit_color}{self.session_profit:,} Rp ({profit_sign}{profit_percentage:.2f}%)")
        
        # Display streak information
        if self.consecutive_wins > 0:
            print(f"Current Streak: {Fore.GREEN}{self.consecutive_wins} wins")
        elif self.consecutive_losses > 0:
            print(f"Current Streak: {Fore.RED}{self.consecutive_losses} losses")
            
        # Display active strategy information
        strategy = self.strategies[self.active_strategy]
        print(f"Active Strategy: {Fore.YELLOW}{self.active_strategy}")
        
        # Display strategy-specific information
        if self.active_strategy == "fibonacci":
            print(f"Fibonacci Level: {strategy.current_index} " +
                 f"(multiplier: {strategy.fibonacci_sequence[strategy.current_index]}x)")
        elif self.active_strategy == "percentage":
            print(f"Bet Percentage: {strategy.percentage}%")
            
        print(f"{Fore.CYAN}===========================\n")
    
    def show_bet_history(self, limit: int = 5) -> None:
        """
        Show recent bet history with enhanced visualization.
        
        Args:
            limit: Number of recent bets to show
        """
        if not self.bet_history:
            print(f"{Fore.YELLOW}No betting history available yet.")
            return
            
        print(f"\n{Fore.CYAN}=== Recent Bet History ===")
        
        # Get most recent bets
        recent_bets = self.bet_history[-limit:]
        outcome_names = {0: 'Banker', 1: 'Player', 2: 'Tie'}
        
        for i, bet in enumerate(reversed(recent_bets)):
            predicted = outcome_names.get(bet.get('predicted_outcome'), 'Unknown')
            actual = outcome_names.get(bet.get('actual_outcome'), 'Unknown')
            
            # Set colors
            result_color = Fore.GREEN if bet.get('won', False) else Fore.RED
            profit = bet.get('profit', 0)
            profit_sign = "+" if profit >= 0 else ""
            
            strategy_name = bet.get('strategy', 'unknown')
            
            print(f"#{i+1}: {result_color}{'WON' if bet.get('won', False) else 'LOST'}{Style.RESET_ALL} - " 
                  f"Bet: {bet.get('amount', 0):,} on {predicted}, "
                  f"Result: {actual}, "
                  f"{result_color}{profit_sign}{profit:,} "
                  f"[{strategy_name}]")
            
        print(f"{Fore.CYAN}===========================\n")
    
    def get_betting_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive betting statistics.
        
        Returns:
            dict: Betting statistics
        """
        if not self.bet_history:
            return {
                'total_bets': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_wagered': 0,
                'total_profit': 0,
                'profit_per_bet': 0,
                'roi': 0
            }
            
        total_bets = len(self.bet_history)
        wins = sum(1 for bet in self.bet_history if bet.get('won', False))
        win_rate = (wins / total_bets) * 100 if total_bets > 0 else 0
        
        total_wagered = sum(bet.get('amount', 0) for bet in self.bet_history)
        total_profit = sum(bet.get('profit', 0) for bet in self.bet_history)
        profit_per_bet = total_profit / total_bets if total_bets > 0 else 0
        roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
        
        # Calculate strategy-specific stats
        strategy_stats = {}
        for strategy_name in self.strategies.keys():
            strategy_bets = [bet for bet in self.bet_history if bet.get('strategy') == strategy_name]
            if strategy_bets:
                strat_win_count = sum(1 for bet in strategy_bets if bet.get('won', False))
                strat_total = len(strategy_bets)
                strat_total_profit = sum(bet.get('profit', 0) for bet in strategy_bets)
                
                strategy_stats[strategy_name] = {
                    'total_bets': strat_total,
                    'win_rate': (strat_win_count / strat_total) * 100,
                    'total_profit': strat_total_profit
                }
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': total_bets - wins,
            'win_rate': win_rate,
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'profit_per_bet': profit_per_bet,
            'roi': roi,
            'strategy_stats': strategy_stats
        }
    
    def display_betting_stats(self) -> None:
        """
        Display comprehensive betting statistics with enhanced visualization.
        """
        stats = self.get_betting_stats()
        
        if stats['total_bets'] == 0:
            print(f"{Fore.YELLOW}No betting statistics available yet.")
            return
            
        print(f"\n{Fore.CYAN}=== Betting Statistics ===")
        
        # Set colors based on performance
        win_rate_color = Fore.RED
        if stats['win_rate'] >= 40:
            win_rate_color = Fore.YELLOW
        if stats['win_rate'] >= 50:
            win_rate_color = Fore.GREEN
            
        roi_color = Fore.RED
        if stats['roi'] >= 0:
            roi_color = Fore.GREEN
            
        profit_color = Fore.RED
        if stats['total_profit'] >= 0:
            profit_color = Fore.GREEN
            
        # Display stats
        print(f"Total Bets: {stats['total_bets']}")
        print(f"Wins/Losses: {Fore.GREEN}{stats['wins']}{Style.RESET_ALL}/{Fore.RED}{stats['losses']}")
        print(f"Win Rate: {win_rate_color}{stats['win_rate']:.2f}%")
        print(f"Total Wagered: {stats['total_wagered']:,} Rp")
        print(f"Total Profit: {profit_color}{stats['total_profit']:,} Rp")
        print(f"Profit per Bet: {profit_color}{stats['profit_per_bet']:,.2f} Rp")
        print(f"Return on Investment: {roi_color}{stats['roi']:.2f}%")
        
        # Display strategy-specific stats if available
        if 'strategy_stats' in stats and stats['strategy_stats']:
            print(f"\n{Fore.CYAN}Strategy Performance:")
            
            for strategy_name, strat_stats in stats['strategy_stats'].items():
                strat_color = Fore.RED
                if strat_stats['total_profit'] >= 0:
                    strat_color = Fore.GREEN
                    
                print(f"  {strategy_name}: {strat_stats['total_bets']} bets, " +
                     f"Win Rate: {strat_stats['win_rate']:.1f}%, " +
                     f"Profit: {strat_color}{strat_stats['total_profit']:,} Rp")
        
        print(f"{Fore.CYAN}===========================\n")
    
    def full_reset(self) -> bool:
        """
        Perform a complete system reset with transaction-based consistency.
        
        Returns:
            bool: True if reset was successful
        """
        try:
            # Reset balance to initial
            self.balance = INITIAL_BALANCE
            self.initial_balance = INITIAL_BALANCE
            self.session_profit = 0
            
            # Reset betting history
            self.bet_history = []
            
            # Reset streak counters
            self.consecutive_wins = 0
            self.consecutive_losses = 0
            
            # Reset all strategies
            for strategy in self.strategies.values():
                strategy.reset()
            
            # Save changes
            self._save_balance()
            self._save_bet_history()
            
            # Optionally delete the files
            if os.path.exists(BALANCE_FILE):
                os.remove(BALANCE_FILE)
            if os.path.exists(BET_HISTORY_FILE):
                os.remove(BET_HISTORY_FILE)
            
            logger.info("System fully reset to default settings")
            
            return True
        except Exception as e:
            logger.error(f"Error during reset: {e}")
            print(f"{Fore.RED}Error during reset: {e}")
            return False

    def recommend_bet_with_pattern(
        self, 
        prediction: int, 
        confidence: float, 
        certainty: Optional[float] = None,
        pattern_type: str = 'unknown'
    ) -> Dict[str, Any]:
        """
        Recommend bet with pattern-specific optimizations.
        
        Args:
            prediction: The predicted outcome (0=Banker, 1=Player, 2=Tie)
            confidence: Confidence percentage for the prediction
            certainty: Optional certainty value (from entropy calculation)
            pattern_type: Pattern type detected in the sequence
            
        Returns:
            dict: Optimized bet recommendation
        """
        # Type safety conversions
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        if hasattr(confidence, 'item'):
            confidence = confidence.item()
        if hasattr(certainty, 'item'):
            certainty = certainty.item()
        
        # Don't bet on Tie (2) outcomes
        if prediction == 2:
            return {
                'should_bet': False,
                'amount': 0,
                'reason': "System doesn't bet on Tie outcomes due to unfavorable odds"
            }
        
        # Determine if confidence is high enough to bet
        if confidence < 20:
            return {
                'should_bet': False,
                'amount': 0,
                'reason': f"Confidence too low ({confidence:.1f}%)"
            }
        
        # Get the active strategy
        strategy = self.strategies[self.active_strategy]
        
        # Calculate base bet amount using the active strategy
        base_bet_amount = strategy.calculate_bet(confidence, certainty)
        
        # Apply pattern-specific modifiers
        pattern_modifier = self._get_pattern_bet_modifier(pattern_type, prediction)
        bet_amount = base_bet_amount * pattern_modifier
        
        # Ensure bet is within allowed limits
        bet_amount = max(MIN_BET, min(bet_amount, MAX_BET, self.balance))
        bet_amount = round(bet_amount / 500) * 500  # Round to nearest 500
        
        logger.info(f"Recommending bet of {bet_amount:,} Rp using {self.active_strategy} strategy " +
                f"with {pattern_type} pattern modifier: {pattern_modifier:.2f}x")
        
        return {
            'should_bet': True,
            'amount': bet_amount,
            'outcome': prediction,
            'confidence': confidence,
            'certainty': certainty,
            'strategy': self.active_strategy,
            'pattern_type': pattern_type,
            'pattern_modifier': pattern_modifier
        }

    def _adjust_strategy_for_pattern(self, pattern_type: str, pattern_effectiveness: Optional[float] = None) -> None:
        """
        Adjust betting strategy parameters based on detected pattern.
        
        Args:
            pattern_type: Type of pattern detected ('streak', 'alternating', etc.)
            pattern_effectiveness: Optional pattern effectiveness score (0-1)
        """
        # Skip if no pattern information available
        if pattern_type == 'unknown' or pattern_effectiveness is None:
            return
        
        # Get active strategy for adjustment
        strategy = self.strategies[self.active_strategy]
        
        # Adjust Fibonacci strategy for streak patterns
        if self.active_strategy == 'fibonacci' and pattern_type == 'streak':
            if pattern_effectiveness > 0.6:
                # Strong streak pattern - more aggressive betting
                if strategy.current_index > 0:
                    strategy.current_index = max(0, strategy.current_index - 1)
        
        # Adjust Martingale for alternating patterns
        elif self.active_strategy == 'martingale' and pattern_type == 'alternating':
            if pattern_effectiveness > 0.6 and hasattr(strategy, 'base_bet') and strategy.base_bet is not None:
                # Increase base bet slightly for high-effectiveness alternating patterns
                strategy.base_bet *= 1.1
                strategy.base_bet = min(strategy.base_bet, self.balance * 0.01)  # Cap at 1% of balance
        
        # Adjust percentage strategy for tie patterns
        elif self.active_strategy == 'percentage' and pattern_type == 'tie':
            # Reduce percentage for tie patterns (which are less predictable)
            if hasattr(strategy, 'percentage'):
                original_percentage = strategy.percentage
                strategy.percentage = max(0.1, original_percentage * 0.8)  # Reduce by 20%
                logger.info(f"Reduced percentage from {original_percentage}% to {strategy.percentage}% for tie pattern")

    def _process_bet_decision(self, bet_rec: Dict[str, Any], predicted: int, confidence: float) -> Optional[Dict[str, Any]]:
        """
        Process user bet decision with transaction integrity and performance instrumentation.
        
        This method implements a structured approach to bet processing with
        comprehensive error handling and performance tracking.
        
        Args:
            bet_rec: Recommendation details generated by the strategy
            predicted: The predicted outcome (0=Banker, 1=Player, 2=Tie)
            confidence: Confidence percentage for the prediction
            
        Returns:
            dict or None: Bet details if placed, None otherwise
        """
        start_time = time.time()
        
        # Extract recommendation details
        should_bet = bet_rec.get('should_bet', False)
        rec_amount = bet_rec.get('amount', 0)
        outcome_names = {0: 'Banker', 1: 'Player', 2: 'Tie'}
        outcome_name = outcome_names.get(predicted, 'Unknown')
        
        # Display recommendation
        print(f"\n{Fore.CYAN}=== Betting Recommendation ===")
        
        if not should_bet:
            print(f"{Fore.YELLOW}Recommendation: DO NOT BET")
            print(f"Reason: {bet_rec.get('reason', 'Insufficient confidence')}")
            
            choice = input("Force bet anyway? (Y/N): ").upper()
            if choice != 'Y':
                return None
            else:
                # Allow betting despite recommendation with minimal amount
                print(f"{Fore.YELLOW}Proceeding with bet despite recommendation...")
                bet_amount = max(MIN_BET, self.balance * 0.02)  # Default to 2% of balance
                bet_amount = round(bet_amount / 500) * 500  # Round to nearest 500
        else:
            # Show recommendation with pattern information if available
            print(f"Recommended bet: {Fore.YELLOW}{rec_amount:,} Rp on {outcome_name}")
            print(f"Strategy: {Fore.YELLOW}{self.active_strategy}")
            
            if 'pattern_type' in bet_rec and bet_rec['pattern_type'] != 'unknown':
                pattern_color = Fore.CYAN
                pattern_mod = bet_rec.get('pattern_modifier', 1.0)
                
                # Color-code pattern modifier
                mod_color = Fore.YELLOW
                if pattern_mod > 1.1:
                    mod_color = Fore.GREEN
                elif pattern_mod < 0.9:
                    mod_color = Fore.RED
                    
                print(f"Pattern detected: {pattern_color}{bet_rec['pattern_type']}")
                print(f"Pattern modifier: {mod_color}x{pattern_mod:.2f}")
            
            # Ask user for bet decision
            choice = input("Place bet (Y=Yes, N=No, C=Custom amount): ").upper()
            
            if choice == 'N':
                print(f"{Fore.YELLOW}Bet not placed.")
                return None
            
            bet_amount = rec_amount
            
            # Handle custom bet amount with validation
            if choice == 'C':
                while True:
                    try:
                        custom_amount = input(f"Enter custom bet amount ({MIN_BET:,}-{self.balance:,} Rp): ")
                        # Remove commas if present
                        custom_amount = custom_amount.replace(',', '')
                        bet_amount = float(custom_amount)
                        
                        # Validate amount
                        if bet_amount < MIN_BET:
                            print(f"{Fore.RED}Minimum bet is {MIN_BET:,} Rp")
                        elif bet_amount > self.balance:
                            print(f"{Fore.RED}Bet cannot exceed your balance of {self.balance:,} Rp")
                        else:
                            break
                    except ValueError:
                        print(f"{Fore.RED}Please enter a valid number")
        
        # Place the bet with performance tracking
        try:
            # Use profiled bet placement for performance tracking
            if hasattr(self, 'place_bet_with_profiling'):
                bet = self.place_bet_with_profiling(bet_amount, predicted, confidence)
            else:
                bet = self.place_bet(bet_amount, predicted, confidence)
            
            # Add pattern information to bet if available
            if bet is not None and 'pattern_type' in bet_rec:
                bet['pattern_type'] = bet_rec['pattern_type']
                if 'pattern_modifier' in bet_rec:
                    bet['pattern_modifier'] = bet_rec['pattern_modifier']
                    
            # Check if bet was successfully placed
            if bet is None:
                print(f"{Fore.RED}Failed to place bet. Please check your balance.")
                return None
                    
            print(f"{Fore.CYAN}Bet placed: {Fore.YELLOW}{bet['amount']:,} Rp on {outcome_name}")
            
            # Display balance after placing bet
            print(f"Remaining balance: {Fore.YELLOW}{self.balance:,} Rp")
            
            # Track execution time
            if hasattr(self, 'performance_metrics'):
                execution_time = time.time() - start_time
                self.performance_metrics['total_bet_handling_time'].append(execution_time)
                
                # Log slow operations
                if execution_time > 0.5:  # Log if processing takes >500ms
                    logger.info(f"Slow bet decision processing: {execution_time*1000:.2f}ms")
            
            return bet
            
        except Exception as e:
            logger.error(f"Error in bet processing: {e}")
            print(f"{Fore.RED}Error processing bet: {e}")
            return None


    def resolve_bet_with_profiling(self, bet: Dict[str, Any], actual_outcome: int) -> Dict[str, Any]:
        """
        Resolve a bet with performance instrumentation and transaction integrity.
        
        Args:
            bet: The original bet details
            actual_outcome: The actual outcome that occurred
            
        Returns:
            dict: Updated bet details with results
        """
        start_time = time.time()
        
        # Ensure numpy types are converted
        if hasattr(actual_outcome, 'item'):
            actual_outcome = actual_outcome.item()
        
        # Extract bet details
        predicted = bet['predicted_outcome']
        bet_amount = bet['amount']
        
        logger.info(f"Resolving bet - Amount: {bet_amount:,} Rp, Predicted: {predicted}, Actual: {actual_outcome}")
        
        # Start transaction tracking
        transaction_state = {
            'balance_before': self.balance,
            'consecutive_wins_before': self.consecutive_wins,
            'consecutive_losses_before': self.consecutive_losses,
            'session_profit_before': self.session_profit,
            'transaction_id': time.time()
        }
        
        try:
            # Determine if bet won
            won = (predicted == actual_outcome)
            
            # Get active strategy object
            strategy = self.strategies[self.active_strategy]
            
            # Calculate profit/loss
            if won:
                if predicted == 0:  # Banker
                    # For banker, payout is 95% (5% commission)
                    profit = bet_amount * BANKER_PAYOUT
                    winnings = bet_amount + profit
                else:  # Player
                    # For player, payout is 100% (1:1)
                    profit = bet_amount * PLAYER_PAYOUT
                    winnings = bet_amount + profit
                
                # Add winnings to balance
                self.balance += winnings
                self.session_profit += profit
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                
                # Update strategy after win
                strategy.update_after_win()
                
            else:
                # If lost, profit is negative bet amount (already deducted from balance)
                profit = -bet_amount
                self.session_profit += profit
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                
                # Update strategy after loss
                strategy.update_after_loss()
            
            # Update bet record with results
            bet.update({
                'actual_outcome': actual_outcome,
                'won': won,
                'profit': profit,
                'balance_after': self.balance,
                'resolution_time': time.time()
            })
            
            # Add to history and save
            self.bet_history.append(bet)
            self._save_bet_history()
            self._save_balance()
            
            # Record performance metrics
            if hasattr(self, 'performance_metrics'):
                execution_time = time.time() - start_time
                self.performance_metrics['resolve_bet_time'].append(execution_time)
                
                # Log slow operations
                if execution_time > 0.2:  # Log if resolution takes >200ms
                    logger.info(f"Slow bet resolution: {execution_time*1000:.2f}ms")
            
            return bet
            
        except Exception as e:
            logger.error(f"Error resolving bet: {e}")
            print(f"{Fore.RED}Error resolving bet: {e}")
            
            # Transaction failure - rollback to previous state
            self.balance = transaction_state['balance_before']
            self.consecutive_wins = transaction_state['consecutive_wins_before']
            self.consecutive_losses = transaction_state['consecutive_losses_before']
            self.session_profit = transaction_state['session_profit_before']
            
            # Save rolled back state
            self._save_balance()
            
            # Return error information
            bet.update({
                'error': str(e),
                'resolution_failed': True
            })
            
            return bet

    def _get_pattern_bet_modifier(self, pattern_type: str, prediction: int) -> float:
        """
        Get pattern-specific betting modifier.
        
        Args:
            pattern_type: Type of pattern detected
            prediction: The predicted outcome
            
        Returns:
            float: Bet amount modifier (1.0 = no change)
        """
        # Default modifier - no adjustment
        modifier = 1.0
        
        # Pattern-specific modifiers based on historical effectiveness
        if pattern_type == 'streak':
            # Check pattern performance in history
            streak_bets = [bet for bet in self.bet_history[-50:] 
                        if bet.get('pattern_type', '') == 'streak']
            
            if streak_bets:
                streak_wins = sum(1 for bet in streak_bets if bet.get('won', False))
                streak_win_rate = streak_wins / len(streak_bets)
                
                # Adjust modifier based on historical performance
                if streak_win_rate > 0.6:
                    modifier = 1.2  # Increase bet for historically successful pattern
                elif streak_win_rate < 0.4:
                    modifier = 0.8  # Decrease bet for historically unsuccessful pattern
        
        elif pattern_type == 'alternating':
            # Similar pattern-specific logic
            alternating_bets = [bet for bet in self.bet_history[-50:]
                            if bet.get('pattern_type', '') == 'alternating']
            
            if alternating_bets:
                alt_wins = sum(1 for bet in alternating_bets if bet.get('won', False))
                alt_win_rate = alt_wins / len(alternating_bets)
                
                if alt_win_rate > 0.6:
                    modifier = 1.2
                elif alt_win_rate < 0.4:
                    modifier = 0.8
        
        # Apply outcome-specific adjustments
        if prediction == 0:  # Banker
            # Slightly decrease banker bets due to the commission
            modifier *= 0.95
        
        return modifier
    def handle_bet_with_transaction(self, prediction_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle the betting process with transaction-based state management.
        
        This method implements a transaction-based approach to betting that ensures
        atomic operations and consistent state management even during failures.
        
        Args:
            prediction_result: Dictionary with prediction details
            
        Returns:
            dict or None: The bet details if a bet was placed, None otherwise
        """
        # Start transaction tracking
        transaction_state = {
            'balance_before': self.balance,
            'consecutive_wins_before': self.consecutive_wins,
            'consecutive_losses_before': self.consecutive_losses,
            'session_profit_before': self.session_profit,
            'transaction_id': time.time()
        }
        
        try:
            # Extract prediction information with type safety
            predicted = prediction_result.get('prediction', 0)
            confidence = prediction_result.get('confidence', 50.0)
            
            # Type conversion for numpy values
            if hasattr(predicted, 'item'):
                predicted = predicted.item()
            if hasattr(confidence, 'item'):
                confidence = confidence.item()
            
            # Extract entropy-based certainty if available
            certainty = None
            if 'entropy' in prediction_result:
                entropy = prediction_result.get('entropy', 0)
                entropy_max = 1.58  # max entropy for 3 outcomes
                entropy_percent = (entropy / entropy_max) * 100
                certainty = 100 - entropy_percent  # Higher number = more certain
                
            # Extract pattern information for strategy adaptation
            pattern_type = prediction_result.get('pattern_type', 'unknown')
            pattern_effectiveness = prediction_result.get('pattern_effectiveness', None)
            
            # Handle insufficient balance with early exit
            if self.balance < MIN_BET:
                print(f"\n{Fore.RED}Insufficient balance: You have {self.balance:,} Rp but minimum bet is {MIN_BET:,} Rp")
                print(f"{Fore.YELLOW}Please add funds to your balance to continue betting.")
                return None
            
            # Apply pattern-specific strategy adjustment
            self._adjust_strategy_for_pattern(pattern_type, pattern_effectiveness)
            
            # Get betting recommendation with enhanced information
            bet_rec = self.recommend_bet_with_pattern(predicted, confidence, certainty, pattern_type)
            
            # Process user decision with transaction safety
            bet = self._process_bet_decision(bet_rec, predicted, confidence)
            
            # If bet was placed, record transaction success
            if bet is not None:
                bet['transaction_id'] = transaction_state['transaction_id']
            
            return bet
            
        except Exception as e:
            # Transaction failure - rollback to previous state
            logger.error(f"Transaction error in betting: {e}")
            print(f"{Fore.RED}Error in betting process: {e}")
            
            # Restore previous state
            self.balance = transaction_state['balance_before']
            self.consecutive_wins = transaction_state['consecutive_wins_before']
            self.consecutive_losses = transaction_state['consecutive_losses_before']
            self.session_profit = transaction_state['session_profit_before']
            
            # Save rolled back state
            self._save_balance()
            
            # Return failure information
            return None
        
    def handle_bet(self, prediction_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle the entire betting process with enhanced user experience.
        
        Args:
            prediction_result: Dictionary with prediction details
            
        Returns:
            dict or None: The bet details if a bet was placed, None otherwise
        """
        # Extract prediction information
        predicted = prediction_result['prediction']
        confidence = prediction_result['confidence']
        
        # Extract certainty if available (from prediction entropy)
        certainty = None
        if 'entropy' in prediction_result:
            entropy = prediction_result['entropy']
            entropy_max = 1.58  # max entropy for 3 outcomes
            entropy_percent = (entropy / entropy_max) * 100
            certainty = 100 - entropy_percent  # Higher number = more certain
        
        # Check if balance is sufficient
        if self.balance < MIN_BET:
            print(f"\n{Fore.RED}Insufficient balance: You have {self.balance:,} Rp but minimum bet is {MIN_BET:,} Rp")
            print(f"{Fore.YELLOW}Please add funds to your balance to continue betting.")
            return None
        
        try:
            # Get betting recommendation with certainty factor
            bet_rec = self.recommend_bet(predicted, confidence, certainty)
        except Exception as e:
            logger.error(f"Error getting bet recommendation: {e}")
            print(f"{Fore.RED}Error getting bet recommendation: {e}")
            bet_rec = {
                'should_bet': False,
                'amount': 0,
                'reason': "Error in betting system"
            }
        
        # If system recommends against betting
        if not bet_rec['should_bet']:
            print(f"\n{Fore.YELLOW}Betting recommendation: DO NOT BET")
            print(f"Reason: {bet_rec['reason']}")
            choice = input("Force bet anyway? (Y/N): ").upper()
            if choice != 'Y':
                return None
            else:
                # Allow betting despite recommendation
                print(f"{Fore.YELLOW}Proceeding with bet despite recommendation...")
                bet_rec = {
                    'should_bet': True,
                    'amount': max(MIN_BET, self.balance * 0.02),  # Default to 2% of balance
                    'outcome': predicted,
                    'confidence': confidence
                }
        
        # Display betting recommendation
        outcome_names = {0: 'Banker', 1: 'Player', 2: 'Tie'}
        outcome_name = outcome_names.get(predicted, 'Unknown')
        
        print(f"\n{Fore.CYAN}=== Betting Recommendation ===")
        print(f"Recommended bet: {Fore.YELLOW}{bet_rec['amount']:,} Rp on {outcome_name}")
        print(f"Strategy: {Fore.YELLOW}{self.active_strategy}")
        
        if 'certainty' in bet_rec and bet_rec['certainty'] is not None:
            certainty_color = Fore.RED
            if bet_rec['certainty'] >= 50:
                certainty_color = Fore.YELLOW
            if bet_rec['certainty'] >= 75:
                certainty_color = Fore.GREEN
                
            print(f"Prediction certainty: {certainty_color}{bet_rec['certainty']:.1f}%")
        
        # Ask user if they want to follow the recommendation
        choice = input("Place bet (Y=Yes, N=No, C=Custom amount): ").upper()
        
        if choice == 'N':
            print(f"{Fore.YELLOW}Bet not placed.")
            return None
        
        bet_amount = bet_rec['amount']
        
        # Handle custom bet amount
        if choice == 'C':
            while True:
                try:
                    custom_amount = input(f"Enter custom bet amount ({MIN_BET:,}-{self.balance:,} Rp): ")
                    # Remove commas if present
                    custom_amount = custom_amount.replace(',', '')
                    bet_amount = float(custom_amount)
                    
                    # Validate amount
                    if bet_amount < MIN_BET:
                        print(f"{Fore.RED}Minimum bet is {MIN_BET:,} Rp")
                    elif bet_amount > self.balance:
                        print(f"{Fore.RED}Bet cannot exceed your balance of {self.balance:,} Rp")
                    else:
                        break
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number")
        
        # Place the bet
        bet = self.place_bet(bet_amount, predicted, confidence)
        
        # Check if bet was successfully placed
        if bet is None:
            print(f"{Fore.RED}Failed to place bet. Please check your balance.")
            return None
            
        print(f"{Fore.CYAN}Bet placed: {Fore.YELLOW}{bet['amount']:,} Rp on {outcome_name}")
        
        # Display balance after placing bet
        print(f"Remaining balance: {Fore.YELLOW}{self.balance:,} Rp")
        
        return bet


# Singleton instance to use throughout the application
betting_system = BettingSystem()