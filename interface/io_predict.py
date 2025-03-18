"""
Enhanced Input/Output Interface for Baccarat Prediction System.

This module provides a robust user interface for the Baccarat prediction system,
implementing consistent error handling, transaction-based model operations,
and seamless integration with the betting system.

Key features:
1. Transaction-based model operations
2. Comprehensive error handling with graceful degradation
3. Enhanced visualization of prediction results
4. Integrated performance monitoring
5. Structured betting workflow
"""

import pandas as pd
import numpy as np
import time
import sys
import traceback
import logging
from models.model_registry import StateTransaction
from typing import Dict, List, Tuple, Any, Optional, Union
from colorama import Fore, Back, Style, init
from config import MIN_BET

# Initialize colorama for cross-platform color support
init(autoreset=True)

# Import configuration
from config import MONTE_CARLO_SAMPLES

# Import betting system
from betting.betting_system import betting_system

# Import unified prediction pipeline
from prediction.prediction_pipeline import PredictionPipeline
from prediction.components.pattern_analyzer import extract_pattern_type
from prediction.monte_carlo import get_pattern_insight

# Import data utilities
from data.data_utils import log_prediction, check_data_balance, update_realtime_data
from models.model_registry import ModelRegistry

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='baccarat_prediction.log'
)

# Global tracking variables
correct_predictions = 0
total_predictions = 0
model_registry = None


def initialize_model_registry() -> ModelRegistry:
    """
    Initialize and validate the model registry with proper error handling.
    
    Ensures that model registry is properly initialized, validated, and ready
    for prediction operations. Implements automatic repair and validation.
    
    Returns:
        ModelRegistry: Initialized and validated model registry
    """
    global model_registry
    
    if model_registry is None:
        try:
            print(f"{Fore.CYAN}Initializing model registry...")
            model_registry = ModelRegistry()
            
            # Validate registry consistency
            print(f"{Fore.CYAN}Validating model registry consistency...")
            validation_result = model_registry.validate_registry_consistency()
            
            if not validation_result:
                print(f"{Fore.YELLOW}Registry validation failed. Attempting repair...")
                repair_success = model_registry.repair_registry()
                
                if repair_success:
                    print(f"{Fore.GREEN}Registry repair successful!")
                else:
                    print(f"{Fore.RED}Registry repair failed. System may be unstable.")
            else:
                print(f"{Fore.GREEN}Registry validation successful.")
                
            # Initialize stacking if not already done
            if "stacking_ensemble" not in model_registry.models:
                print(f"{Fore.CYAN}Initializing stacking ensemble...")
                model_registry.initialize_stacking()
                
            # Perform initial calibration
            try:
                print(f"{Fore.CYAN}Initializing confidence calibration...")
                model_registry.calibration_manager.calibrate_from_history()
            except Exception as e:
                logger.error(f"Error initializing calibration: {e}")
                print(f"{Fore.YELLOW}Warning: Calibration initialization failed: {e}")
            
        except Exception as e:
            logger.error(f"Error initializing model registry: {e}")
            print(f"{Fore.RED}Critical error initializing model registry: {e}")
            raise RuntimeError(f"Failed to initialize model registry: {e}")
    
    return model_registry

# Add to io_predict.py

def make_optimized_prediction(previous_results, system_manager):
    """
    Make prediction using transaction-based system with comprehensive performance profiling.
    
    Args:
        previous_results: Previous game outcomes
        system_manager: SystemManager instance
        
    Returns:
        dict: Prediction results with performance metrics
    """
    try:
        # Use the optimized transaction-based prediction with profiling
        start_time = time.time()
        
        # Get prediction with performance instrumentation
        result = system_manager.predict_with_transaction(previous_results)
        
        # Track execution time
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Enhance result with pattern effectiveness for betting integration
        if 'pattern_type' in result and result['pattern_type'] != 'unknown':
            # Calculate pattern effectiveness based on confidence and entropy
            base_effectiveness = 0.5  # Default middle value
            
            # Higher confidence increases effectiveness
            confidence_factor = (result.get('confidence', 50) - 50) / 50  # -1 to 1 scale
            
            # Lower entropy (higher certainty) increases effectiveness
            certainty = 0.5
            if 'entropy' in result:
                entropy = result['entropy']
                entropy_max = 1.58  # max entropy for 3 outcomes
                entropy_ratio = entropy / entropy_max
                certainty = 1 - entropy_ratio  # 0-1 scale, higher is better
            
            # Calculate overall effectiveness
            pattern_effectiveness = base_effectiveness + (0.2 * confidence_factor) + (0.3 * (certainty - 0.5))
            pattern_effectiveness = max(0.1, min(0.9, pattern_effectiveness))  # Bound between 0.1-0.9
            
            # Add to result
            result['pattern_effectiveness'] = pattern_effectiveness
        
        # Log performance metrics
        if 'performance_metrics' in result:
            metrics = result['performance_metrics']
            logger.info(f"Prediction performance: total={metrics.get('total', 0)*1000:.2f}ms, "
                       f"validation={metrics.get('validation', 0)*1000:.2f}ms, "
                       f"base_prediction={metrics.get('base_prediction', 0)*1000:.2f}ms, "
                       f"meta_prediction={metrics.get('meta_prediction', 0)*1000:.2f}ms")
            
            # Flag slow predictions for further investigation
            if execution_time > 150:  # More than 150ms is considered slow
                logger.warning(f"Slow prediction detected: {execution_time:.2f}ms")
                
                # Identify performance bottleneck
                slowest_stage = max(
                    [(stage, time*1000) for stage, time in metrics.items() 
                     if stage not in ['total', 'error_stage']],
                    key=lambda x: x[1]
                )
                logger.warning(f"Performance bottleneck: {slowest_stage[0]} stage "
                              f"({slowest_stage[1]:.2f}ms, {slowest_stage[1]/execution_time*100:.1f}% of total)")
        
        # Check cache statistics periodically
        if hasattr(system_manager, 'prediction_pipeline') and hasattr(system_manager.prediction_pipeline, 'get_cache_stats'):
            cache_stats = system_manager.prediction_pipeline.get_cache_stats()
            if cache_stats.get('lookups', 0) > 0:
                logger.info(f"Cache performance: hit_rate={cache_stats.get('hit_rate', 0):.1f}%, "
                           f"size={cache_stats.get('size', 0)}/{cache_stats.get('max_size', 0)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in transaction-based prediction: {e}")
        
        # Use FallbackManager directly as final resort
        from prediction.components.fallback_manager import FallbackManager
        fallback_manager = FallbackManager(system_manager.model_registry)
        
        return fallback_manager.generate_fallback(
            previous_results, "transaction_error", error=str(e)
        )
        
    except Exception as e:
        logger.error(f"Error in transaction-based prediction: {e}")
        
        # Use FallbackManager directly as final resort
        from prediction.components.fallback_manager import FallbackManager
        fallback_manager = FallbackManager(system_manager.model_registry)
        
        return fallback_manager.generate_fallback(
            previous_results, "transaction_error", error=str(e)
        )

def track_accuracy(predicted: int, actual_result: int) -> Dict[str, Any]:
    """
    Track and display prediction accuracy with enhanced visualization.
    
    Maintains session-level accuracy tracking and provides detailed
    accuracy metrics including color-coded visualization.
    
    Args:
        predicted: The model's prediction
        actual_result: The actual outcome that occurred
        
    Returns:
        dict: Updated accuracy metrics
    """
    global correct_predictions, total_predictions
    
    total_predictions += 1
    is_correct = predicted == actual_result
    
    if is_correct:
        correct_predictions += 1
        print(f"{Fore.GREEN}✓ Correct prediction!")
    else:
        print(f"{Fore.RED}✗ Incorrect prediction")
    
    accuracy = (correct_predictions / total_predictions) * 100
    
    # Color-code accuracy based on performance
    accuracy_color = Fore.RED
    if accuracy >= 50:
        accuracy_color = Fore.YELLOW
    if accuracy >= 60:
        accuracy_color = Fore.GREEN
    
    print(f"Current session accuracy: {accuracy_color}{correct_predictions}/{total_predictions} ({accuracy:.2f}%)")
    
    # Display class-specific accuracy periodically
    if total_predictions % 3 == 0 or total_predictions <= 3:
        display_class_accuracy()
        
    return {
        'total': total_predictions,
        'correct': correct_predictions,
        'accuracy': accuracy,
        'last_correct': is_correct
    }

def display_class_accuracy() -> None:
    """
    Display enhanced accuracy metrics broken down by class (Banker/Player/Tie).
    
    Retrieves and displays class-specific accuracy metrics from the prediction
    history, providing insights into performance across different outcomes.
    """
    # TODO: Implement detailed class-specific accuracy metrics
    # This would retrieve and visualize accuracy data from the history log
    
    print(f"\n{Fore.CYAN}=== Detailed Accuracy Metrics ===")
    print(f"{Fore.YELLOW}Accuracy by outcome type will display here in future versions")
    print(f"{Fore.CYAN}=================================={Style.RESET_ALL}\n")


def simple_input_with_validation(
    prompt: str, 
    valid_options: set, 
    error_msg: Optional[str] = None,
    case_sensitive: bool = False
) -> str:
    """
    Get validated user input with enhanced error handling.
    
    Ensures user input matches expected options, with consistent
    error handling and validation.
    
    Args:
        prompt: The prompt text to display
        valid_options: Set of valid input options
        error_msg: Custom error message (optional)
        case_sensitive: Whether validation should be case-sensitive
        
    Returns:
        str: The validated input value
    """
    if error_msg is None:
        error_msg = f"Invalid input. Valid options are: {', '.join(valid_options)}"
    
    while True:
        try:
            sys.stdout.write(prompt)
            sys.stdout.flush()
            value = sys.stdin.readline().strip()
            
            # Apply case conversion if not case-sensitive
            check_value = value if case_sensitive else value.upper()
            check_options = valid_options if case_sensitive else {opt.upper() for opt in valid_options}
            
            if check_value in check_options:
                return value
            else:
                print(f"{Fore.RED}{error_msg}")
        except KeyboardInterrupt:
            raise  # Re-raise to allow clean exit
        except Exception as e:
            print(f"{Fore.RED}Error reading input: {e}. Please try again.")


def get_five_results() -> List[int]:
    """
    Get the 5 previous baccarat results with enhanced input validation.
    
    Collects and validates user input for previous game outcomes,
    providing visual feedback and validation.
    
    Returns:
        list: 5 numeric values representing previous outcomes (0=Banker, 1=Player, 2=Tie)
    """
    mapping = {'B': 0, 'P': 1, 'S': 2, 'T': 2}  # Added 'T' as alternative for Tie
    rev_mapping = {0: 'B', 1: 'P', 2: 'S'}
    
    print(f"{Fore.CYAN}\n=== Enter the 5 previous results ===")
    print(f"{Fore.YELLOW}B = Banker, P = Player, S/T = Tie{Style.RESET_ALL}")
    
    results = []
    for i in range(5):
        prompt = f"Result {i+1}/5: "
        result = simple_input_with_validation(
            prompt, 
            valid_options={'B', 'P', 'S', 'T'}, 
            error_msg="Please enter B, P, S, or T only."
        ).upper()
        
        # Add numeric value to results
        results.append(mapping[result])
        
        # Show what was just entered with color
        color = Fore.GREEN if result == 'B' else Fore.BLUE if result == 'P' else Fore.MAGENTA
        print(f"  Added: {color}{result}{Style.RESET_ALL}")
    
    # Show summary of inputs
    print(f"{Fore.CYAN}\nEntered sequence:{Style.RESET_ALL}")
    for i, val in enumerate(results):
        color = Fore.GREEN if val == 0 else Fore.BLUE if val == 1 else Fore.MAGENTA
        print(f"  {i+1}: {color}{rev_mapping[val]}")
    
    return results


# Update in io_predict.py

def display_prediction_result(result: Dict[str, Any], rev_mapping: Dict[int, str]) -> None:
    """
    Display prediction results with enhanced betting integration and performance metrics.
    
    Args:
        result: Prediction result dictionary
        rev_mapping: Mapping from numeric codes to outcome labels
    """
    predicted = result['prediction']
    confidence = result['confidence']
    
    # Choose color based on predicted outcome
    outcome_color = Fore.GREEN if predicted == 0 else Fore.BLUE if predicted == 1 else Fore.MAGENTA
    
    # Choose color for confidence
    conf_color = Fore.RED
    if confidence >= 50:
        conf_color = Fore.YELLOW
    if confidence >= 70:
        conf_color = Fore.GREEN
    
    # Display prediction with colors
    print(f"\n{Fore.CYAN}Next outcome prediction:{Style.RESET_ALL} {outcome_color}{rev_mapping[predicted]} {conf_color}(Confidence: {confidence:.2f}%)")
    
    # Show raw confidence if available (shows effect of calibration)
    if 'raw_confidence' in result and abs(result['raw_confidence'] - confidence) > 2.0:
        raw_conf = result['raw_confidence']
        raw_color = Fore.YELLOW
        
        # Show an indicator of calibration direction
        if raw_conf > confidence:
            direction = "↓"  # Confidence was reduced
            direction_color = Fore.BLUE
        else:
            direction = "↑"  # Confidence was increased
            direction_color = Fore.GREEN
            
        print(f"{Fore.CYAN}Confidence adjustment:{Style.RESET_ALL} {raw_color}{raw_conf:.2f}% {direction_color}{direction} {conf_color}{confidence:.2f}%")
    
    # Indicate if result was calibrated
    if result.get('calibrated', False):
        print(f"{Fore.CYAN}Confidence calibrated:{Style.RESET_ALL} {Fore.GREEN}Yes ✓")
    
    # Display distribution with bar chart visualization
    print(f"{Fore.CYAN}Prediction distribution:{Style.RESET_ALL}")
    for outcome, percentage in result['distribution'].items():
        outcome_name = {0: 'Banker', 1: 'Player', 2: 'Tie'}.get(outcome, str(outcome))
        color = Fore.GREEN if outcome == 0 else Fore.BLUE if outcome == 1 else Fore.MAGENTA
        
        # Create visual bar chart
        bar_length = int(percentage / 2)  # Scale to reasonable length
        bar = '█' * bar_length
        
        print(f"  {color}{outcome_name}: {percentage:.2f}% {bar}")
    
    # Show pattern insight with betting relevance
    if 'pattern_type' in result and result['pattern_type'] != 'unknown':
        pattern_type = result['pattern_type']
        pattern_color = Fore.YELLOW
        
        # More detailed pattern type display
        pattern_label = {
            'streak': 'Streak pattern',
            'alternating': 'Alternating pattern',
            'tie': 'Tie-influenced pattern',
            'no_pattern': 'No distinctive pattern'
        }.get(pattern_type, pattern_type.capitalize())
        
        # Pattern effectiveness display
        if 'pattern_effectiveness' in result:
            effectiveness = result['pattern_effectiveness']
            eff_color = Fore.RED
            if effectiveness >= 0.4:
                eff_color = Fore.YELLOW
            if effectiveness >= 0.6:
                eff_color = Fore.GREEN
                
            # Display with betting integration
            print(f"\n{pattern_color}Pattern detected:{Style.RESET_ALL} {pattern_label}")
            print(f"{pattern_color}Pattern effectiveness:{Style.RESET_ALL} {eff_color}{effectiveness:.2f}")
            
            # Add betting recommendation based on pattern
            if predicted != 2:  # Not Tie
                if effectiveness >= 0.6:
                    print(f"{Fore.GREEN}Betting recommendation: Consider higher bet for this pattern")
                elif effectiveness <= 0.3:
                    print(f"{Fore.RED}Betting recommendation: Consider lower bet for this pattern")
    
    # Show entropy information with betting relevance
    if 'entropy' in result:
        entropy = result['entropy']
        entropy_max = 1.58  # max entropy for 3 outcomes
        entropy_percent = (entropy / entropy_max) * 100
        
        entropy_color = Fore.GREEN
        if entropy_percent > 60:
            entropy_color = Fore.YELLOW
        if entropy_percent > 80:
            entropy_color = Fore.RED
            
        certainty = 100 - entropy_percent
        print(f"{Fore.CYAN}Prediction certainty:{Style.RESET_ALL} {entropy_color}{certainty:.2f}%")
        
        # Add betting interpretation
        if certainty > 70:
            print(f"{Fore.GREEN}High certainty may warrant increased bet size")
        elif certainty < 40:
            print(f"{Fore.YELLOW}Low certainty suggests caution with bet sizing")
    
    # Show model agreement with betting relevance
    if 'base_model_agreement' in result:
        agreement = result['base_model_agreement']
        agreement_color = Fore.RED
        if agreement >= 0.5:
            agreement_color = Fore.YELLOW
        if agreement >= 0.75:
            agreement_color = Fore.GREEN
            
        print(f"{Fore.CYAN}Model agreement:{Style.RESET_ALL} {agreement_color}{agreement:.2f}")
        
        # Add betting interpretation
        if agreement >= 0.75 and predicted != 2:  # Not Tie
            print(f"{Fore.GREEN}Strong model consensus supports confidence in prediction")
    
    # Show performance metrics if available
    if 'performance_metrics' in result:
        metrics = result['performance_metrics']
        total_time = metrics.get('total', 0) * 1000  # Convert to ms
        
        # Only show if performance is notable
        if total_time > 100:  # Only show for slower predictions
            print(f"\n{Fore.CYAN}Performance metrics:{Style.RESET_ALL}")
            print(f"  Total prediction time: {total_time:.1f}ms")
            
            # Show breakdown for significant stages
            for stage, time_ms in sorted(
                [(s, t*1000) for s, t in metrics.items() if s not in ['total', 'error_stage']],
                key=lambda x: x[1],
                reverse=True
            )[:3]:  # Show top 3 stages
                if time_ms > 20:  # Only show significant stages
                    stage_pct = (time_ms / total_time) * 100
                    print(f"  - {stage}: {time_ms:.1f}ms ({stage_pct:.1f}%)")
    
    # Show fallback info if using fallback
    if result.get('fallback', False):
        fallback_method = result.get('fallback_method', 'unknown')
        print(f"{Fore.YELLOW}Note: Using {fallback_method} as fallback method (stacking unavailable)")
        
    # Show emergency info if using emergency fallback
    if result.get('emergency', False):
        print(f"{Fore.RED}Warning: Using emergency fallback prediction (system error occurred)")


def get_actual_result() -> int:
    """
    Get the actual outcome that occurred with enhanced input handling.
    
    Collects and validates the actual game outcome for updating models
    and performance tracking.
    
    Returns:
        int: The actual outcome (0=Banker, 1=Player, 2=Tie)
    """
    mapping = {'B': 0, 'P': 1, 'S': 2, 'T': 2}  # Added 'T' as alternative for Tie
    
    print(f"{Fore.CYAN}\n=== Enter the actual outcome ===")
    
    actual_input = simple_input_with_validation(
        "Actual result (B/P/S/T): ",
        valid_options={'B', 'P', 'S', 'T'},
        error_msg="Please enter B, P, S, or T only."
    ).upper()
    
    # Show confirmation with appropriate color
    color = Fore.GREEN if actual_input == 'B' else Fore.BLUE if actual_input == 'P' else Fore.MAGENTA
    print(f"  Recorded: {color}{actual_input}")
    
    return mapping[actual_input]


# File: interface/io_predict.py
# Replace existing update_models_with_result function, around line 330

def update_models_with_result(
    prev_rounds: List[int], 
    predicted: int, 
    actual_result: int, 
    result: Dict[str, Any]
) -> bool:
    """
    Update all models with the actual outcome using transaction-based approach.
    
    Implements a comprehensive model update process with proper error handling
    and transactional integrity for improved reliability and data consistency.
    
    Args:
        prev_rounds: Previous game outcomes
        predicted: Predicted outcome
        actual_result: Actual outcome that occurred
        result: Full prediction result dictionary
        
    Returns:
        bool: True if update was successful
    """
    # Get model registry
    mr = initialize_model_registry()
    
    # Track which models were successfully updated
    updated_models = []
    
    try:
        # Use transaction to ensure atomic updates
        from models.model_registry import StateTransaction
        
        with StateTransaction(mr) as transaction:
            # Prepare meta data for stacking update
            meta_data = mr.collect_holdout_predictions(prev_rounds, actual_result)
            
            # Track performance metrics for this update
            update_metrics = {
                'base_models_updated': 0,
                'start_time': time.time()
            }
            
            # First update all base models
            for model_id, model in mr.models.items():
                if model_id == "stacking_ensemble":
                    continue
                    
                try:
                    if hasattr(model, 'update_model'):
                        pattern = result.get('pattern_type')
                        confidence = result.get('confidence', 50.0)
                        model.update_model(prev_rounds, actual_result, confidence, pattern)
                        updated_models.append(model_id)
                        update_metrics['base_models_updated'] += 1
                except Exception as e:
                    logger.error(f"Error updating model {model_id}: {e}")
                    print(f"{Fore.YELLOW}Warning: Error updating model {model_id}: {e}")
            
            # Then update the stacking model with new meta-features
            if "stacking_ensemble" in mr.models:
                try:
                    stacking_start = time.time()
                    mr.update_stacking_model(meta_data)
                    updated_models.append("stacking_ensemble")
                    update_metrics['stacking_update_time'] = time.time() - stacking_start
                except Exception as e:
                    logger.error(f"Error updating stacking model: {e}")
                    print(f"{Fore.YELLOW}Warning: Error updating stacking model: {e}")
            
            # Calculate update performance metrics
            update_metrics['total_time'] = time.time() - update_metrics['start_time']
            update_metrics['models_updated'] = len(updated_models)
            
            # Log performance metrics for monitoring
            if update_metrics['total_time'] > 0.1:  # Only log slow updates (>100ms)
                logger.info(f"Model update performance: {update_metrics['total_time']*1000:.2f}ms, "
                           f"{update_metrics['models_updated']} models updated")
            
            # Transaction will be committed automatically if no exceptions occur
        
        # Only save registry after successful transaction
        if updated_models:
            mr._save_registry()
            print(f"{Fore.GREEN}Successfully updated {len(updated_models)} models")
            return True
        else:
            print(f"{Fore.YELLOW}Warning: No models were updated")
            return False
            
    except Exception as e:
        logger.error(f"Error in model update process: {e}")
        print(f"{Fore.RED}Error in model update process: {e}")
        return False


def update_models_with_transaction(
    prev_rounds: List[int], 
    predicted: int, 
    actual_result: int, 
    result: Dict[str, Any],
    betting_result: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Update models and betting system with synchronized transaction support.
    
    This method ensures that model updates and betting transactions are
    synchronized to maintain system consistency.
    
    Args:
        prev_rounds: Previous game outcomes
        predicted: Predicted outcome
        actual_result: Actual outcome that occurred
        result: Full prediction result dictionary
        betting_result: Optional betting result dictionary
        
    Returns:
        bool: True if update was successful
    """
    # Get model registry
    mr = initialize_model_registry()
    
    # Track which models were successfully updated
    updated_models = []
    betting_updated = False
    
    try:
        # Use transaction to ensure atomic updates
        from models.model_registry import StateTransaction
        
        with StateTransaction(mr) as transaction:
            # Prepare meta data for stacking update
            meta_data = mr.collect_holdout_predictions(prev_rounds, actual_result)
            
            # Track performance metrics for this update
            update_metrics = {
                'base_models_updated': 0,
                'start_time': time.time()
            }
            
            # First update all base models
            for model_id, model in mr.models.items():
                if model_id == "stacking_ensemble":
                    continue
                    
                try:
                    if hasattr(model, 'update_model'):
                        pattern = result.get('pattern_type')
                        confidence = result.get('confidence', 50.0)
                        model.update_model(prev_rounds, actual_result, confidence, pattern)
                        updated_models.append(model_id)
                        update_metrics['base_models_updated'] += 1
                except Exception as e:
                    logger.error(f"Error updating model {model_id}: {e}")
                    print(f"{Fore.YELLOW}Warning: Error updating model {model_id}: {e}")
            
            # Then update the stacking model with new meta-features
            if "stacking_ensemble" in mr.models:
                try:
                    stacking_start = time.time()
                    mr.update_stacking_model(meta_data)
                    updated_models.append("stacking_ensemble")
                    update_metrics['stacking_update_time'] = time.time() - stacking_start
                except Exception as e:
                    logger.error(f"Error updating stacking model: {e}")
                    print(f"{Fore.YELLOW}Warning: Error updating stacking model: {e}")
            
            # Update betting system in the same transaction if betting occurred
            if betting_result is not None:
                try:
                    # Ensure betting system has pattern information
                    betting_result['pattern_type'] = result.get('pattern_type', 'unknown')
                    betting_result['pattern_effectiveness'] = result.get('pattern_effectiveness')
                    
                    # Resolve the bet with the actual outcome
                    betting_system.resolve_bet(betting_result, actual_result)
                    betting_updated = True
                    
                    # Add pattern data to betting history for optimization
                    for bet in betting_system.bet_history:
                        if bet.get('timestamp') == betting_result.get('timestamp'):
                            bet['pattern_type'] = result.get('pattern_type', 'unknown')
                            bet['pattern_effectiveness'] = result.get('pattern_effectiveness')
                            break
                except Exception as e:
                    logger.error(f"Error resolving bet: {e}")
                    print(f"{Fore.RED}Error resolving bet: {e}")
                    # Let transaction handle rollback
            
            # Calculate update performance metrics
            update_metrics['total_time'] = time.time() - update_metrics['start_time']
            update_metrics['models_updated'] = len(updated_models)
            
            # Log performance metrics for monitoring
            if update_metrics['total_time'] > 0.1:  # Only log slow updates (>100ms)
                logger.info(f"Model update performance: {update_metrics['total_time']*1000:.2f}ms, "
                          f"{update_metrics['models_updated']} models updated")
            
            # Transaction will be committed automatically if no exceptions occur
        
        # Only save registry after successful transaction
        if updated_models:
            mr._save_registry()
            print(f"{Fore.GREEN}Successfully updated {len(updated_models)} models")
            
            if betting_updated:
                print(f"{Fore.GREEN}Successfully updated betting system")
                betting_system._save_bet_history()
                
            return True
        else:
            print(f"{Fore.YELLOW}Warning: No models were updated")
            return False
            
    except Exception as e:
        logger.error(f"Error in model update process: {e}")
        print(f"{Fore.RED}Error in model update process: {e}")
        return False
# File: interface/io_predict.py
# Replace existing main_prediction_loop function, around line 372

# Update main_prediction_loop in io_predict.py to integrate all components

def main_prediction_loop(system_manager=None) -> None:
    """
    Main prediction loop with comprehensive performance instrumentation and
    integrated transaction-based operations.
    
    Args:
        system_manager: Optional SystemManager instance
    """
    try:
        # Initialize performance tracking
        performance_metrics = {
            'prediction_time': [],
            'betting_time': [],
            'update_time': [],
            'total_cycle_time': []
        }
        
        # Initialize betting system with performance tracking
        if not hasattr(betting_system, 'performance_metrics'):
            betting_system._initialize_performance_tracking()
        
        # Get or initialize system manager
        if system_manager is None:
            try:
                from main import SystemManager
                system_manager = SystemManager()
                system_manager.initialize_system()
            except ImportError:
                # Fallback to direct model registry if SystemManager is unavailable
                mr = initialize_model_registry()
                
                # Create a minimal system manager substitute
                class MinimalSystemManager:
                    def __init__(self, registry):
                        self.model_registry = registry
                        self.prediction_pipeline = PredictionPipeline(registry)
                    
                    def predict_with_transaction(self, prev_rounds):
                        return self.prediction_pipeline.predict_with_profiling(prev_rounds)
                
                system_manager = MinimalSystemManager(mr)
                print(f"{Fore.YELLOW}Note: Using minimal system manager (reduced functionality)")
        
        # Setup outcome mappings
        mapping = {'B': 0, 'P': 1, 'S': 2, 'T': 2}
        rev_mapping = {0: 'B', 1: 'P', 2: 'S'}
        
        # Check data distribution at startup
        print(f"\n{Fore.CYAN}=== Checking data balance ===")
        check_data_balance()
        print(f"{Fore.CYAN}============================={Style.RESET_ALL}\n")
        
        # Operation counters for maintenance tasks
        maintenance_counter = 0
        competition_counter = 0
        
        # Performance monitoring thresholds for adaptive optimization
        slow_predictions_count = 0
        last_performance_check = time.time()
        
        # Main prediction loop
        while True:
            # Track total cycle time
            cycle_start_time = time.time()
            
            try:
                # Get user input
                prev_rounds = get_five_results()
                
                # Track prediction performance
                pred_start_time = time.time()
                
                # Make prediction using enhanced pipeline with performance profiling
                result = system_manager.predict_with_transaction(prev_rounds)
                predicted = result['prediction']
                
                # Record prediction time
                prediction_time = time.time() - pred_start_time
                performance_metrics['prediction_time'].append(prediction_time)
                
                # Display prediction with enhanced visuals
                display_prediction_result(result, rev_mapping)
                
                # Handle betting process with transaction support and performance tracking
                betting_start_time = time.time()

                # Check if pattern-aware betting is available
                if hasattr(betting_system, 'handle_bet_with_transaction'):
                    current_bet = betting_system.handle_bet_with_transaction(result)
                else:
                    # Fallback to standard betting
                    current_bet = betting_system.handle_bet(result)

                # Record betting time
                betting_time = time.time() - betting_start_time
                performance_metrics['betting_time'].append(betting_time)
                
                # Use enhanced transaction-based betting
                current_bet = betting_system.handle_bet_with_transaction(result)
                
                # Record betting time
                betting_time = time.time() - betting_start_time
                performance_metrics['betting_time'].append(betting_time)
                
                # Get actual result
                actual_result = get_actual_result()
                
                # Update models and betting with synchronized transaction support
                update_start_time = time.time()
                
                # Use synchronized update for both models and betting
                update_models_with_transaction(
                    prev_rounds, predicted, actual_result, result, current_bet
                )
                
                # Record update time
                update_time = time.time() - update_start_time
                performance_metrics['update_time'].append(update_time)
                
                # Update the realtime data with the new result
                print(f"{Fore.CYAN}>>> Saving prediction data...")
                update_realtime_data(prev_rounds, actual_result)
                
                # Display betting outcome if a bet was placed
                if current_bet is not None:
                    # Bet resolution happened in the synchronized update
                    for bet in betting_system.bet_history[-1:]:
                        if bet.get('predicted_outcome') == predicted and bet.get('actual_outcome') == actual_result:
                            # Found the matching bet
                            if bet.get('won', False):
                                profit = bet.get('profit', 0)
                                print(f"\n{Fore.GREEN}✓ BET WON! +{profit:,} Rp")
                            else:
                                loss = abs(bet.get('profit', 0))
                                print(f"\n{Fore.RED}✗ BET LOST! -{loss:,} Rp")
                            
                            break
                    
                    betting_system.display_balance()
                    betting_system.show_bet_history(3)
                
                # Track accuracy
                track_accuracy(predicted, actual_result)
                
                # Log full prediction details
                log_prediction(
                    prev_rounds, 
                    predicted, 
                    actual_result, 
                    confidence=result['confidence'], 
                    distribution=result['distribution']
                )
                
                # Calculate and track total cycle time
                cycle_time = time.time() - cycle_start_time
                performance_metrics['total_cycle_time'].append(cycle_time)
                
                # Performance monitoring - track execution time
                if 'performance_metrics' in result:
                    metrics = result['performance_metrics']
                    total_time = metrics.get('total', 0) * 1000  # Convert to ms
                    
                    # Track slow predictions for adaptive optimization
                    if total_time > 150:
                        slow_predictions_count += 1
                        
                        # After multiple slow predictions, consider optimization
                        if slow_predictions_count >= 3 and time.time() - last_performance_check > 300:
                            if hasattr(system_manager, 'prediction_pipeline'):
                                # Adjust cache size based on performance
                                if hasattr(system_manager.prediction_pipeline, '_cache_max_size'):
                                    current_size = system_manager.prediction_pipeline._cache_max_size
                                    system_manager.prediction_pipeline._cache_max_size = min(2000, current_size * 2)
                                    print(f"{Fore.YELLOW}Performance optimization: Increased cache size to "
                                         f"{system_manager.prediction_pipeline._cache_max_size} entries")
                            
                            # Reset tracking after adjustment
                            slow_predictions_count = 0
                            last_performance_check = time.time()
                    else:
                        # Reset counter after a fast prediction
                        slow_predictions_count = max(0, slow_predictions_count - 1)
                
                # Execute transaction-based scheduled maintenance
                maintenance_counter += 1
                if maintenance_counter >= 5:  # Every 5 predictions
                    maintenance_counter = 0
                    
                    # Use transaction for system maintenance
                    from models.model_registry import StateTransaction
                    
                    with StateTransaction(system_manager.model_registry) as transaction:
                        try:
                            print(f"{Fore.CYAN}Performing scheduled maintenance...")
                            
                            # Call perform_system_health_check if available, otherwise use perform_scheduled_maintenance
                            if hasattr(system_manager, 'perform_system_health_check'):
                                maintenance_result = system_manager.perform_system_health_check()
                            else:
                                maintenance_result = system_manager.model_registry.perform_scheduled_maintenance()
                            
                            # Report maintenance results
                            if maintenance_result['status'] == 'healthy':
                                print(f"{Fore.GREEN}✓ System health check passed")
                            elif maintenance_result['status'] == 'repaired':
                                print(f"{Fore.YELLOW}⚠ System issues detected and repaired:")
                                for fix in maintenance_result.get('fixes_applied', []):
                                    print(f"  - {fix}")
                            else:
                                print(f"{Fore.RED}⚠ System issues detected but not fully repaired")
                                for issue in maintenance_result.get('issues', [])[:3]:  # Show top 3 issues
                                    print(f"  - {issue}")
                        except Exception as e:
                            logger.error(f"Error in system maintenance: {e}")
                            print(f"{Fore.RED}Error in system maintenance: {e}")
                            # Transaction will automatically roll back on error
                
                # Run model competition periodically
                competition_counter += 1
                if competition_counter >= 10:  # Every 10 predictions
                    competition_counter = 0
                    try:
                        print(f"{Fore.CYAN}Running model competition...")
                        system_manager.model_registry.run_model_competition()
                        print(f"{Fore.GREEN}✓ Model competition completed")
                    except Exception as e:
                        logger.error(f"Error in model competition: {e}")
                        print(f"{Fore.RED}Error in model competition: {e}")
                
                # Options for next action
                print("\nWhat would you like to do next?")
                cont = simple_input_with_validation(
                    "Options: P=Predict again, B=Betting menu, Q=Quit: ", 
                    valid_options={'P', 'B', 'Q', 'p', 'b', 'q'}, 
                    error_msg="Please enter P, B, or Q."
                ).upper()
                
                if cont == 'B':
                    if not display_betting_menu():
                        break
                    continue
                elif cont == 'Q':
                    betting_system.display_balance()
                    betting_system.display_betting_stats()
                    print(f"{Fore.YELLOW}Exiting prediction system...")
                    break
                    
            except KeyboardInterrupt:
                print(f"{Fore.YELLOW}\n\nOperation canceled. Returning to main menu...")
                break
            except Exception as e:
                logger.error(f"Error in prediction cycle: {e}", exc_info=True)
                print(f"{Fore.RED}\nAn error occurred in prediction loop: {e}")
                print(f"{Fore.YELLOW}Attempting to continue with next prediction...")
                time.sleep(1)  # Brief pause to avoid rapid error loops
                
    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}\nExiting prediction system...")
    except Exception as e:
        logger.critical(f"Critical error in main prediction loop: {e}", exc_info=True)
        print(f"{Fore.RED}Critical error: {e}")
        print(f"{Fore.RED}System cannot continue. Please check logs for details.")

def display_betting_menu() -> bool:
    """
    Display a menu of betting options and settings with enhanced user experience.
    
    Provides an interactive menu for managing betting settings, history,
    and balance operations.
    
    Returns:
        bool: True if the user wants to continue, False if they want to exit
    """
    while True:
        print(f"{Fore.CYAN}\n=== Betting Menu ===")
        print("1. Show balance and statistics")
        print("2. Show betting history")
        print("3. Reset sequence")
        print("4. Change betting strategy")
        print("5. Add funds to balance")
        print("6. Reset balance to initial amount")
        print("7. Full reset (balance, history, and betting sequence)")
        print("8. Return to predictions")
        print("9. Exit program")
        
        try:
            choice = input("\nEnter your choice (1-9): ")
            
            if choice == '1':
                betting_system.display_balance()
                betting_system.display_betting_stats()
                
            elif choice == '2':
                try:
                    num = input("How many bets to show? (default 10): ")
                    num = int(num) if num.strip() else 10
                    betting_system.show_bet_history(num)
                except ValueError:
                    print(f"{Fore.RED}Invalid number. Showing 10 bets.")
                    betting_system.show_bet_history(10)
                    
            elif choice == '3':
                betting_system.current_fibonacci_index = 0
                if hasattr(betting_system, 'current_base_bet'):
                    betting_system.current_base_bet = None
                print(f"{Fore.GREEN}Betting sequence reset to initial level")
                
            elif choice == '4':
                print("Choose betting strategy:")
                print("1. Fibonacci (increase after loss, reset after win)")
                print("2. Martingale (double after loss, reset after win)")
                print("3. D'Alembert (increase after loss, decrease after win)")
                print("4. Percentage (fixed percentage of balance)")
                
                strat_choice = input("Enter choice (1-4): ")
                
                if strat_choice == '1':
                    betting_system.set_betting_strategy("fibonacci")
                elif strat_choice == '2':
                    betting_system.set_betting_strategy("martingale")
                elif strat_choice == '3':
                    betting_system.set_betting_strategy("dalembert")
                elif strat_choice == '4':
                    try:
                        pct = float(input("Enter percentage of balance to bet (0.1-10%): "))
                        betting_system.set_betting_strategy("percentage", pct)
                    except ValueError:
                        print(f"{Fore.RED}Invalid percentage. Using default 1%")
                        betting_system.set_betting_strategy("percentage", 1.0)
                else:
                    print(f"{Fore.RED}Invalid choice. Strategy unchanged.")
                    
            elif choice == '5':
                try:
                    amount = input("Enter amount to add to balance (in thousands): ")
                    amount = int(amount) * 1000
                    betting_system.balance += amount
                    betting_system._save_balance()
                    print(f"{Fore.GREEN}Added {amount:,} Rp to balance")
                    betting_system.display_balance()
                except ValueError:
                    print(f"{Fore.RED}Invalid amount")
                    
            elif choice == '6':
                confirm = input("Are you sure you want to reset balance to initial amount? (y/n): ").lower()
                if confirm == 'y':
                    # Ask if user wants to reset to a specific amount
                    custom_amount = input("Reset to a specific amount? (Leave blank for initial amount): ")
                    
                    if custom_amount.strip():
                        try:
                            amount = int(custom_amount.replace(',', ''))
                            betting_system.reset_balance(amount)
                        except ValueError:
                            print(f"{Fore.RED}Invalid amount. Using initial balance instead.")
                            betting_system.reset_balance()
                    else:
                        # Reset to initial balance
                        betting_system.reset_balance()
                    
                    betting_system.display_balance()
                else:
                    print(f"{Fore.YELLOW}Balance reset cancelled")
                    
            elif choice == '7':
                confirm = input("Are you sure you want to perform a FULL RESET? This will reset your balance, betting history, and sequences. (y/n): ").lower()
                if confirm == 'y':
                    betting_system.full_reset()
                    print(f"{Fore.GREEN}Full reset completed successfully!")
                    betting_system.display_balance()
                else:
                    print(f"{Fore.YELLOW}Full reset cancelled")
                    
            elif choice == '8':
                return True  # Continue to predictions
                
            elif choice == '9':
                print(f"{Fore.YELLOW}\nExiting program...")
                return False  # Exit the program
                
        except KeyboardInterrupt:
            print(f"{Fore.YELLOW}\nReturning to main menu...")
            return True
        except Exception as e:
            logger.error(f"Error in betting menu: {e}")
            print(f"{Fore.RED}Error: {e}")

def test_prediction_betting_integration(system_manager=None, test_bet=False):
    """
    Run comprehensive integration test between prediction pipeline and betting system.
    
    This function validates the full prediction-to-betting workflow with various
    pattern types, ensuring proper transaction handling, error recovery, and
    performance measurement under different scenarios.
    
    Args:
        system_manager: Optional SystemManager instance
        test_bet: Whether to test actual betting functionality
    
    Returns:
        dict: Detailed test results with performance metrics
    """
    print(f"\n{Fore.CYAN}=== Running Prediction-Betting Integration Test ===")
    
    # Create or get system manager
    if system_manager is None:
        try:
            from main import SystemManager
            system_manager = SystemManager()
            system_manager.initialize_system()
        except ImportError:
            # Fallback to model registry if SystemManager is unavailable
            from models.model_registry import ModelRegistry
            mr = ModelRegistry()
            
            # Create a minimal system manager substitute
            from prediction.prediction_pipeline import PredictionPipeline
            
            class MinimalSystemManager:
                def __init__(self, registry):
                    self.model_registry = registry
                    self.prediction_pipeline = PredictionPipeline(registry)
                
                def predict_with_transaction(self, prev_rounds):
                    result = self.prediction_pipeline.predict_with_profiling(prev_rounds)
                    
                    # Add pattern effectiveness for betting integration
                    if 'pattern_type' in result and result['pattern_type'] != 'unknown':
                        # Simple pattern effectiveness calculation
                        confidence = result.get('confidence', 50.0)
                        effectiveness = 0.5 + ((confidence - 50) / 100)
                        result['pattern_effectiveness'] = max(0.1, min(0.9, effectiveness))
                    
                    return result
            
            system_manager = MinimalSystemManager(mr)
            print(f"{Fore.YELLOW}Note: Using minimal system manager (reduced functionality)")
    
    # Import betting system
    from betting.betting_system import betting_system
    
    # Initialize betting system performance tracking if needed
    if not hasattr(betting_system, 'performance_metrics'):
        betting_system._initialize_performance_tracking()
    
    # Test data for different pattern types
    test_cases = [
        {'pattern': 'streak', 'data': [0, 0, 0, 0, 0], 'desc': 'Banker streak pattern'},
        {'pattern': 'streak', 'data': [1, 1, 1, 1, 1], 'desc': 'Player streak pattern'},
        {'pattern': 'alternating', 'data': [0, 1, 0, 1, 0], 'desc': 'Alternating banker/player pattern'},
        {'pattern': 'alternating', 'data': [1, 0, 1, 0, 1], 'desc': 'Alternating player/banker pattern'},
        {'pattern': 'tie_influenced', 'data': [0, 2, 0, 1, 0], 'desc': 'Pattern with tie influence'},
        {'pattern': 'mixed', 'data': [0, 1, 2, 0, 1], 'desc': 'Mixed pattern with banker/player/tie'}
    ]
    
    results = []
    performance_metrics = {
        'prediction_time': [],
        'betting_recommendation_time': [],
        'bet_placement_time': [],
        'bet_resolution_time': [],
        'total_workflow_time': []
    }
    
    # Run test cases
    for i, test_case in enumerate(test_cases):
        print(f"\n{Fore.YELLOW}Test case {i+1}: {test_case['desc']}")
        print(f"Input: {test_case['data']}")
        
        try:
            # Track total workflow time
            workflow_start_time = time.time()
            
            # 1. Make prediction with performance tracking
            pred_start = time.time()
            prediction = system_manager.predict_with_transaction(test_case['data'])
            pred_time = time.time() - pred_start
            performance_metrics['prediction_time'].append(pred_time * 1000)  # ms
            
            print(f"Prediction: {prediction.get('prediction')} with {prediction.get('confidence'):.1f}% confidence")
            print(f"Pattern: {prediction.get('pattern_type', 'unknown')}")
            
            # 2. Test bet recommendation with performance tracking
            bet_start = time.time()
            
            # Extract certainty from entropy if available
            certainty = None
            if 'entropy' in prediction:
                entropy = prediction['entropy']
                entropy_max = 1.58  # max entropy for 3 outcomes
                entropy_percent = (entropy / entropy_max) * 100
                certainty = 100 - entropy_percent
            
            # Use pattern-aware betting if available
            if hasattr(betting_system, 'recommend_bet_with_pattern'):
                bet_rec = betting_system.recommend_bet_with_pattern(
                    prediction['prediction'],
                    prediction['confidence'],
                    certainty,
                    prediction.get('pattern_type', 'unknown')
                )
            else:
                # Fall back to standard recommendation
                bet_rec = betting_system.recommend_bet(
                    prediction['prediction'],
                    prediction['confidence']
                )
            
            bet_rec_time = time.time() - bet_start
            performance_metrics['betting_recommendation_time'].append(bet_rec_time * 1000)  # ms
            
            print(f"Bet recommendation: {bet_rec.get('should_bet')}, Amount: {bet_rec.get('amount', 0):,} Rp")
            
            # 3. Test actual betting if requested and recommendation is positive
            bet_result = None
            if test_bet and bet_rec.get('should_bet', False):
                # Save original balance
                original_balance = betting_system.balance
                
                # Place a minimal test bet
                from config import MIN_BET
                test_amount = MIN_BET
                print(f"Placing test bet of {test_amount:,} Rp...")
                
                bet_place_start = time.time()
                if hasattr(betting_system, 'place_bet_with_profiling'):
                    bet = betting_system.place_bet_with_profiling(test_amount, prediction['prediction'], prediction['confidence'])
                else:
                    bet = betting_system.place_bet(test_amount, prediction['prediction'], prediction['confidence'])
                bet_place_time = time.time() - bet_place_start
                performance_metrics['bet_placement_time'].append(bet_place_time * 1000)  # ms
                
                if bet:
                    # Resolve with alternating outcomes to test both win/loss
                    actual_outcome = 0 if i % 2 == 0 else 1
                    
                    bet_resolve_start = time.time()
                    if hasattr(betting_system, 'resolve_bet_with_profiling'):
                        bet_result = betting_system.resolve_bet_with_profiling(bet, actual_outcome)
                    else:
                        bet_result = betting_system.resolve_bet(bet, actual_outcome)
                    bet_resolve_time = time.time() - bet_resolve_start
                    performance_metrics['bet_resolution_time'].append(bet_resolve_time * 1000)  # ms
                    
                    # Restore original balance
                    betting_system.balance = original_balance
                    betting_system._save_balance()
                    
                    # Remove test bet from history
                    if betting_system.bet_history:
                        betting_system.bet_history.pop()
                        betting_system._save_bet_history()
                    
                    print(f"Test bet resolved: {'Won' if bet_result.get('won', False) else 'Lost'}")
                    print(f"Resolve time: {bet_resolve_time*1000:.2f}ms")
            
            # Track total workflow time
            workflow_time = time.time() - workflow_start_time
            performance_metrics['total_workflow_time'].append(workflow_time * 1000)  # ms
            
            # Record comprehensive test results
            test_result = {
                'test_case': test_case['desc'],
                'input_data': test_case['data'],
                'prediction': prediction.get('prediction'),
                'confidence': prediction.get('confidence'),
                'pattern_type': prediction.get('pattern_type', 'unknown'),
                'pattern_effectiveness': prediction.get('pattern_effectiveness'),
                'should_bet': bet_rec.get('should_bet', False),
                'bet_amount': bet_rec.get('amount', 0),
                'performance': {
                    'prediction_time_ms': pred_time * 1000,
                    'betting_recommendation_time_ms': bet_rec_time * 1000,
                    'total_workflow_time_ms': workflow_time * 1000
                }
            }
            
            # Add bet results if applicable
            if bet_result:
                test_result['bet_result'] = {
                    'won': bet_result.get('won', False),
                    'profit': bet_result.get('profit', 0),
                    'bet_placement_time_ms': bet_place_time * 1000,
                    'bet_resolution_time_ms': bet_resolve_time * 1000
                }
            
            results.append(test_result)
            
        except Exception as e:
            print(f"{Fore.RED}Error in test case {i+1}: {e}")
            traceback.print_exc()
            results.append({
                'test_case': test_case['desc'],
                'input_data': test_case['data'],
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    # Calculate aggregated performance metrics
    aggregated_metrics = {}
    for metric, times in performance_metrics.items():
        if times:
            aggregated_metrics[metric] = {
                'avg_ms': sum(times) / len(times),
                'min_ms': min(times),
                'max_ms': max(times),
                'total_ms': sum(times),
                'samples': len(times)
            }
    
    # Summarize results
    success_count = sum(1 for r in results if 'error' not in r)
    
    print(f"\n{Fore.CYAN}=== Integration Test Summary ===")
    print(f"Test cases: {len(test_cases)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(test_cases) - success_count}")
    
    # Performance metrics
    if success_count > 0:
        print(f"\n{Fore.CYAN}=== Performance Metrics ===")
        print(f"Average prediction time: {aggregated_metrics['prediction_time']['avg_ms']:.2f}ms")
        print(f"Average bet recommendation time: {aggregated_metrics['betting_recommendation_time']['avg_ms']:.2f}ms")
        print(f"Average total workflow time: {aggregated_metrics['total_workflow_time']['avg_ms']:.2f}ms")
        
        if 'bet_placement_time' in aggregated_metrics:
            print(f"Average bet placement time: {aggregated_metrics['bet_placement_time']['avg_ms']:.2f}ms")
        if 'bet_resolution_time' in aggregated_metrics:
            print(f"Average bet resolution time: {aggregated_metrics['bet_resolution_time']['avg_ms']:.2f}ms")
    
    # Return comprehensive test results
    return {
        'success_count': success_count,
        'failure_count': len(test_cases) - success_count,
        'success_rate': (success_count / len(test_cases)) * 100 if test_cases else 0,
        'test_results': results,
        'performance_metrics': aggregated_metrics,
        'timestamp': time.time(),
        'is_betting_integrated': hasattr(betting_system, 'recommend_bet_with_pattern')
    }

if __name__ == "__main__":
    try:
        print(f"{Fore.CYAN}\n=== Enhanced Baccarat Prediction System ===")
        print(f"{Fore.CYAN}=== With Online Learning & Betting System ===\n")
        
        # Display betting system info first
        betting_system.display_balance()
        betting_system.display_betting_stats()
        
        # Start with betting menu first, then go to prediction loop if user chooses
        if display_betting_menu():
            main_prediction_loop()
        
    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}\nExiting...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        print(f"{Fore.RED}Fatal error: {e}")