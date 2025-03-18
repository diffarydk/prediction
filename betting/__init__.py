"""
Unified Betting System Package

This package provides a comprehensive implementation of various betting strategies
for the Baccarat Prediction System with integrated risk management and performance analytics.
"""

# Primary class export
from .betting_system import (
    BettingSystem,
    betting_system,  # Pre-initialized singleton instance
    # Strategy classes
    BettingStrategy,
    FibonacciStrategy,
    MartingaleStrategy, 
    DAlembert,
    PercentageStrategy,
)

# Backward compatibility layer for legacy code
from .betting_system import betting_system as system

# Export primary functions for direct access
recommend_bet = betting_system.recommend_bet
place_bet = betting_system.place_bet
resolve_bet = betting_system.resolve_bet
reset_balance = betting_system.reset_balance
evaluate_strategy = betting_system.evaluate_strategy_performance
display_balance = betting_system.display_balance
show_bet_history = betting_system.show_bet_history
get_betting_stats = betting_system.get_betting_stats
set_betting_strategy = betting_system.set_betting_strategy
recommend_bet_with_pattern = betting_system.recommend_bet_with_pattern
adjust_strategy_for_pattern = betting_system._adjust_strategy_for_pattern
get_pattern_bet_modifier = betting_system._get_pattern_bet_modifier
handle_bet_with_transaction = betting_system.handle_bet_with_transaction
process_bet_decision = betting_system._process_bet_decision


__all__ = [
    # Main system classes
    'BettingSystem',
    'betting_system',
    
    # Strategy implementations
    'BettingStrategy',
    'FibonacciStrategy',
    'MartingaleStrategy',
    'DAlembert',
    'PercentageStrategy',
    
    # Primary functions
    'recommend_bet',
    'place_bet',
    'resolve_bet',
    'reset_balance',
    'evaluate_strategy',
    'display_balance',
    'show_bet_history',
    'get_betting_stats',
    'set_betting_strategy',
    'recommend_bet_with_pattern',
    'adjust_strategy_for_pattern',
    'get_pattern_bet_modifier',
    'handle_bet_with_transaction',
    'process_bet_decision',
    
    # Legacy compatibility
    'system'
]