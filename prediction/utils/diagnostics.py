"""
prediction/utils/diagnostics.py

Diagnostics utilities for Baccarat Prediction System.
This module provides comprehensive error tracking, logging, and diagnostic functionality
to assist with debugging and system monitoring.
"""

import logging
import time
import traceback
import json
import os
from typing import Dict, Any, Optional, List, Union
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)


def log_prediction_error(error_context: Dict[str, Any]) -> None:
    """
    Log detailed prediction error information.
    
    Args:
        error_context: Error context information
    """
    # Format error message
    error_msg = f"Prediction error in stage '{error_context.get('stage', 'unknown')}': {error_context.get('error', 'Unknown error')}"
    
    # Add traceback if available
    if 'traceback' in error_context:
        error_msg += f"\nTraceback: {error_context['traceback']}"
    
    # Log to both file and console
    logger.error(error_msg)
    print(f"Error: {error_msg}")
    
    # Add to error history if tracking is enabled
    if hasattr(log_prediction_error, 'error_history'):
        log_prediction_error.error_history.append(error_context)
        
        # Trim history if needed
        while len(log_prediction_error.error_history) > log_prediction_error.history_limit:
            log_prediction_error.error_history.popleft()


# Initialize error history
log_prediction_error.error_history = deque(maxlen=100)
log_prediction_error.history_limit = 100


def get_error_context(
    stage: str,
    input_data: Any,
    error: Exception,
    partial_result: Optional[Dict[str, Any]] = None,
    performance_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive error context for diagnostics.
    
    Args:
        stage: Pipeline stage where error occurred
        input_data: Input data that caused the error
        error: The exception that was raised
        partial_result: Partial result if available
        performance_metrics: Performance metrics if available
        
    Returns:
        dict: Error context information
    """
    context = {
        'timestamp': time.time(),
        'stage': stage,
        'error': str(error),
        'error_type': type(error).__name__,
        'traceback': traceback.format_exc()
    }
    
    # Include input data shape/type information
    if input_data is not None:
        if hasattr(input_data, 'shape'):
            context['input_shape'] = str(input_data.shape)
        elif isinstance(input_data, list):
            context['input_length'] = len(input_data)
        context['input_type'] = str(type(input_data))
    
    # Include partial result if available (with large field filtering)
    if partial_result:
        context['partial_result'] = {
            k: v for k, v in partial_result.items()
            if k not in ['distribution', 'base_predictions', 'performance_metrics']
        }
    
    # Include performance metrics if available
    if performance_metrics:
        context['performance_metrics'] = performance_metrics
    
    return context


def get_recent_errors(limit: int = 10, error_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get recent errors from the error history.
    
    Args:
        limit: Maximum number of errors to return
        error_type: Filter by error type (optional)
        
    Returns:
        list: Recent errors
    """
    if not hasattr(log_prediction_error, 'error_history'):
        return []
        
    errors = list(log_prediction_error.error_history)
    
    # Apply filtering if specified
    if error_type:
        errors = [e for e in errors if e.get('error_type') == error_type]
    
    # Return most recent errors first, up to limit
    return list(reversed(errors[-limit:]))


def log_system_diagnostics(diagnostics: Dict[str, Any], log_file: Optional[str] = None) -> None:
    """
    Log system diagnostic information.
    
    Args:
        diagnostics: Diagnostic information to log
        log_file: Optional file path for diagnostic logging
    """
    # Format diagnostics message
    diag_msg = "System Diagnostics:\n"
    for key, value in diagnostics.items():
        diag_msg += f"  {key}: {value}\n"
    
    # Log to main logger
    logger.info(diag_msg)
    
    # Log to separate file if specified
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'a') as f:
                f.write(f"{time.time()}: {json.dumps(diagnostics)}\n")
        except Exception as e:
            logger.error(f"Failed to write diagnostics to {log_file}: {e}")


class DiagnosticsCollector:
    """
    Collects and manages diagnostic information for system monitoring.
    
    This class provides methods for tracking various system metrics,
    execution times, error rates, and other diagnostic data to assist
    with debugging and performance optimization.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize diagnostics collector.
        
        Args:
            max_history: Maximum number of history entries to maintain
        """
        self.max_history = max_history
        self.execution_times = {}
        self.error_counts = {}
        self.prediction_counts = {}
        self.model_performance = {}
        self.system_events = deque(maxlen=max_history)
        self.start_time = time.time()
    
    def track_execution_time(self, stage: str, execution_time: float) -> None:
        """
        Track execution time for a specific pipeline stage.
        
        Args:
            stage: Pipeline stage name
            execution_time: Execution time in seconds
        """
        if stage not in self.execution_times:
            self.execution_times[stage] = deque(maxlen=self.max_history)
        
        self.execution_times[stage].append(execution_time)
    
    def track_error(self, error_type: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Track occurrence of a specific error type.
        
        Args:
            error_type: Type of error
            context: Error context information (optional)
        """
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to system events
        event = {
            'type': 'error',
            'error_type': error_type,
            'timestamp': time.time()
        }
        
        if context:
            event['context'] = {
                k: v for k, v in context.items()
                if k not in ['traceback']  # Skip large fields
            }
        
        self.system_events.append(event)
    
    def track_prediction(self, prediction_type: str) -> None:
        """
        Track occurrence of a prediction.
        
        Args:
            prediction_type: Type of prediction
        """
        self.prediction_counts[prediction_type] = self.prediction_counts.get(prediction_type, 0) + 1
    
    def track_model_performance(self, model_id: str, accuracy: float) -> None:
        """
        Track model performance.
        
        Args:
            model_id: Model identifier
            accuracy: Model accuracy
        """
        if model_id not in self.model_performance:
            self.model_performance[model_id] = deque(maxlen=self.max_history)
        
        self.model_performance[model_id].append(accuracy)
    
    def get_execution_time_stats(self, stage: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Get execution time statistics.
        
        Args:
            stage: Pipeline stage (optional, if None returns all stages)
            
        Returns:
            dict: Execution time statistics
        """
        import numpy as np
        
        stats = {}
        
        # Determine which stages to process
        stages = [stage] if stage else self.execution_times.keys()
        
        for s in stages:
            if s in self.execution_times and self.execution_times[s]:
                times = list(self.execution_times[s])
                stats[s] = {
                    'min': np.min(times),
                    'max': np.max(times),
                    'mean': np.mean(times),
                    'median': np.median(times),
                    'std': np.std(times),
                    'count': len(times)
                }
            
        return stats
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Returns:
            dict: Error statistics
        """
        total_errors = sum(self.error_counts.values())
        
        return {
            'total': total_errors,
            'by_type': self.error_counts,
            'error_rate': total_errors / max(1, sum(self.prediction_counts.values()))
        }
    
    def get_model_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get model performance statistics.
        
        Returns:
            dict: Model performance statistics
        """
        import numpy as np
        
        stats = {}
        
        for model_id, accuracies in self.model_performance.items():
            if accuracies:
                acc_list = list(accuracies)
                stats[model_id] = {
                    'min': np.min(acc_list),
                    'max': np.max(acc_list),
                    'mean': np.mean(acc_list),
                    'median': np.median(acc_list),
                    'std': np.std(acc_list),
                    'recent': acc_list[-1],
                    'count': len(acc_list)
                }
        
        return stats
    
    def get_system_events(self, limit: int = 50, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent system events.
        
        Args:
            limit: Maximum number of events to return
            event_type: Filter by event type (optional)
            
        Returns:
            list: Recent system events
        """
        events = list(self.system_events)
        
        # Apply filtering if specified
        if event_type:
            events = [e for e in events if e.get('type') == event_type]
        
        # Return most recent events first, up to limit
        return list(reversed(events[-limit:]))
    
    def get_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive system diagnostics.
        
        Returns:
            dict: Comprehensive diagnostic information
        """
        uptime = time.time() - self.start_time
        
        return {
            'uptime': uptime,
            'uptime_formatted': f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
            'prediction_counts': self.prediction_counts,
            'total_predictions': sum(self.prediction_counts.values()),
            'execution_time_summary': {
                stage: {
                    'mean': round(sum(times) / len(times) * 1000, 2) if times else 0,
                    'count': len(times)
                }
                for stage, times in self.execution_times.items()
            },
            'error_summary': {
                'total': sum(self.error_counts.values()),
                'by_type': self.error_counts
            },
            'model_performance_summary': {
                model_id: round(sum(accs) / len(accs), 4) if accs else 0
                for model_id, accs in self.model_performance.items()
            },
            'recent_events': self.get_system_events(limit=10)
        }
    
    def save_diagnostics(self, file_path: str) -> bool:
        """
        Save diagnostic information to file.
        
        Args:
            file_path: File path for saving diagnostics
            
        Returns:
            bool: Success flag
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            diagnostics = self.get_comprehensive_diagnostics()
            
            with open(file_path, 'w') as f:
                json.dump(diagnostics, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save diagnostics to {file_path}: {e}")
            return False