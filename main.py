"""
Main entry point for Baccarat Prediction System.

This module serves as the central hub that connects all components of the system,
implementing a structured initialization sequence, menu-driven user interface,
and comprehensive error handling mechanisms.

The system architecture follows a modular design with transaction-based state
management to ensure consistent operation even under failure conditions.
"""

import os
import sys
import time
import traceback
import logging
import numpy as np
from colorama import Fore, init, Style
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Initialize colorama for cross-platform color support
init(autoreset=True)

# Import configuration
from config import (
    MODEL_FILE, REALTIME_FILE, LOG_FILE, 
    REGISTRY_PATH, MODEL_REGISTRY_PATH
)

# Import core components with proper error handling
try:
    # First, ensure required directories exist
    from data.data_utils import ensure_all_directories_exist
    ensure_all_directories_exist()
    
    # Import components - each with individual error handling
    from models.model_registry import ModelRegistry, StateTransaction
    from interface.io_predict import main_prediction_loop, display_betting_menu
    from betting.betting_system import betting_system
    from analytics.analytics import run_analytics_menu
except ImportError as e:
    print(f"{Fore.RED}Critical import error: {e}")
    print(f"{Fore.YELLOW}Attempting to fix path issues...")
    
    # Add parent directory to path as fallback
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    # Try imports again
    from data.data_utils import ensure_all_directories_exist
    ensure_all_directories_exist()
    
    from models.model_registry import ModelRegistry, StateTransaction
    from interface.io_predict import main_prediction_loop, display_betting_menu
    from betting.betting_system import betting_system
    from analytics.analytics import run_analytics_menu


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='baccarat_system.log'
)
logger = logging.getLogger(__name__)

class InitializationError(Exception):
    """
    Custom exception for initialization failures.
    
    This exception is raised when a critical component fails to initialize
    and the system cannot continue with normal operation.
    """
    pass
class DependencyResolver:
    """
    Manages component dependencies and initialization order.
    
    This utility class helps track dependencies between system components
    and ensures they are initialized in the correct order with proper validation.
    """
    
    def __init__(self):
        """Initialize dependency tracker."""
        self.dependencies = {}
        self.initialized = set()
        self.failed = set()
        
    def register_dependency(self, component, depends_on=None):
        """
        Register component dependency.
        
        Args:
            component: Component identifier
            depends_on: List of dependencies or None
        """
        if depends_on is None:
            depends_on = []
        self.dependencies[component] = depends_on
        
    def calculate_initialization_order(self):
        """
        Calculate correct initialization order based on dependencies.
        
        Returns:
            list: Components in initialization order
        """
        # Implementation of topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected with {node}")
            if node in visited:
                return
                
            temp_visited.add(node)
            
            for dependency in self.dependencies.get(node, []):
                visit(dependency)
                
            temp_visited.remove(node)
            visited.add(node)
            order.append(node)
            
        # Visit all nodes
        for component in self.dependencies:
            if component not in visited:
                visit(component)
                
        return order
        
    def mark_initialized(self, component):
        """Mark component as successfully initialized."""
        self.initialized.add(component)
        
    def mark_failed(self, component):
        """Mark component initialization as failed."""
        self.failed.add(component)
        
    def can_initialize(self, component):
        """
        Check if component can be initialized.
        
        Args:
            component: Component identifier
            
        Returns:
            bool: True if all dependencies are satisfied
        """
        dependencies = self.dependencies.get(component, [])
        return all(dep in self.initialized for dep in dependencies)
        
    def get_initialization_status(self):
        """
        Get overall initialization status.
        
        Returns:
            dict: Initialization status report
        """
        all_components = set(self.dependencies.keys())
        pending = all_components - self.initialized - self.failed
        
        return {
            "initialized": list(self.initialized),
            "failed": list(self.failed),
            "pending": list(pending),
            "complete": len(self.initialized) == len(all_components),
            "failed_count": len(self.failed),
            "success_ratio": len(self.initialized) / max(1, len(all_components))
        }    

class SystemManager:
    def __init__(self):
        """Initialize the system manager with proper error handling."""
        self.model_registry = None
        self.initialization_complete = False
        self.registry_healthy = False
        
        # Track system state
        self.system_state = {
            'initialized': False,
            'stacking_healthy': False,
            'registry_loaded': False,
            'calibration_ready': False
        }
    def validate_system_components(self):
        """
        Validate critical system components and apply recovery measures.
        
        Returns:
            dict: Validation results with recovery actions
        """
        results = {
            'registry_initialized': False,
            'registry_methods_valid': False,
            'fixes_applied': [],
            'issues_detected': []
        }
        
        # Check model registry initialization
        if not hasattr(self, 'model_registry') or self.model_registry is None:
            results['issues_detected'].append("Model registry not initialized")
            
            # Try to initialize
            try:
                from models.model_registry import ModelRegistry
                self.model_registry = ModelRegistry()
                results['fixes_applied'].append("Initialized missing model registry")
                results['registry_initialized'] = True
            except Exception as e:
                results['issues_detected'].append(f"Failed to initialize model registry: {e}")
                return results
        else:
            results['registry_initialized'] = True
        
        # Check registry methods
        if not hasattr(self.model_registry, 'get_active_base_models'):
            results['issues_detected'].append("Registry missing get_active_base_models method")
            
            # Add method dynamically
            import types
            
            def get_active_base_models(registry_self):
                """Dynamic implementation of missing method"""
                if hasattr(registry_self, 'models') and hasattr(registry_self, 'model_active'):
                    return {model_id: model for model_id, model in registry_self.models.items() 
                        if registry_self.model_active.get(model_id, False) and model_id != "stacking_ensemble"}
                return {}
            
            # Add method to registry
            self.model_registry.get_active_base_models = types.MethodType(
                get_active_base_models, self.model_registry
            )
            results['fixes_applied'].append("Added missing get_active_base_models method")
            results['registry_methods_valid'] = True
        else:
            results['registry_methods_valid'] = True
        
        return results
    def test_pattern_bridge():
        """Test harness for pattern analysis bridge"""
        from models.model_registry import PatternAnalysisBridge
        bridge = PatternAnalysisBridge()
        
        # Test with sample pattern
        test_input = [0, 1, 0, 1, 0]  # Alternating pattern
        
        print("=== Pattern Analysis Bridge Test ===")
        result = bridge.analyze_pattern(test_input)
        print(f"Pattern type: {result.get('pattern_type')}")
        print(f"Pattern insight: {result.get('pattern_insight')}")
        
        # Test pattern features extraction
        features = bridge.extract_pattern_features(test_input)
        print(f"Pattern features: {features}")
        print("==================================")



    def perform_system_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.
        
        This method conducts a thorough health assessment of all system components,
        identifying and attempting to fix any issues to maintain optimal performance.
        
        Returns:
            dict: Health assessment results
        """
        health_report = {
            'status': 'unknown',
            'components': {},
            'issues': [],
            'fixes_applied': [],
            'timestamp': time.time()
        }
        
        try:
            # 1. Check registry consistency
            print(f"{Fore.CYAN}Checking registry consistency...")
            registry_consistent = self.model_registry.validate_registry_consistency()
            health_report['components']['registry'] = {
                'status': 'healthy' if registry_consistent else 'inconsistent'
            }
            
            if not registry_consistent:
                health_report['issues'].append("Registry inconsistency detected")
                # Apply fix
                fixed = self.model_registry.repair_registry()
                if fixed:
                    health_report['fixes_applied'].append("Registry inconsistency fixed")
                else:
                    health_report['issues'].append("Failed to fix registry inconsistency")
            
            # 2. Check stacking ensemble health
            print(f"{Fore.CYAN}Checking stacking ensemble health...")
            stacking_health = self.model_registry.test_stacking_ensemble()
            health_report['components']['stacking'] = stacking_health
            
            if stacking_health.get('status') != 'healthy':
                health_report['issues'].append(f"Stacking health issues: {stacking_health.get('issues', [])}")
                
                # Apply fix if needed
                if stacking_health.get('status') == 'failed':
                    print(f"{Fore.YELLOW}Attempting to reset stacking ensemble...")
                    stacking_reset = self.model_registry.reset_stacking(force_reset=True)
                    if stacking_reset:
                        health_report['fixes_applied'].append("Stacking ensemble reset")
                    else:
                        health_report['issues'].append("Failed to reset stacking ensemble")
            
            # 3. Check model training status
            print(f"{Fore.CYAN}Checking model training status...")
            for model_id, model in self.model_registry.models.items():
                if model_id == "stacking_ensemble":
                    continue  # Already checked above
                    
                is_trained = getattr(model, 'is_trained', False)
                if not is_trained:
                    health_report['issues'].append(f"Model {model_id} not trained")
                    # Will be addressed in training verification step
            
            # 4. Check feature dimensions
            print(f"{Fore.CYAN}Checking feature dimensions...")
            dimension_check = self.model_registry.feature_manager.ensure_consistent_dimensions()
            health_report['components']['dimensions'] = dimension_check
            
            if not dimension_check.get('success', False):
                health_report['issues'].append(f"Dimension inconsistencies: {dimension_check.get('issues', [])}")
                
                # Apply fix if fixed_models were reported
                if dimension_check.get('fixed_models', []):
                    health_report['fixes_applied'].append(f"Fixed dimensions for {len(dimension_check['fixed_models'])} models")
            
            # 5. Check calibration health
            print(f"{Fore.CYAN}Checking calibration health...")
            if hasattr(self.model_registry, 'confidence_calibrators'):
                calibration_status = 'healthy'
                for cls, calibrator in self.model_registry.confidence_calibrators.items():
                    if not hasattr(calibrator, 'X_min_'):
                        calibration_status = 'unhealthy'
                        health_report['issues'].append(f"Calibrator for class {cls} not properly initialized")
                
                health_report['components']['calibration'] = {'status': calibration_status}
                
                if calibration_status == 'unhealthy':
                    # Reinitialize calibration
                    calibration_result = self.model_registry.calibration_manager.calibrate_from_history()
                    health_report['fixes_applied'].append("Reinitialized calibration")
            
            # 6. Determine overall system health
            if not health_report['issues']:
                health_report['status'] = 'healthy'
            elif len(health_report['issues']) <= len(health_report['fixes_applied']):
                health_report['status'] = 'repaired'
            else:
                health_report['status'] = 'degraded'
            
            return health_report
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
            health_report['status'] = 'error'
            health_report['error'] = str(e)
            
            return health_report

    def integrated_prediction_betting(self, prev_rounds):
        """
        Execute a fully integrated prediction and betting transaction.
        
        Args:
            prev_rounds: Previous game outcomes
            
        Returns:
            tuple: (prediction_result, betting_recommendation)
        """
        # Transaction-based prediction
        prediction_result = self.predict_with_transaction(prev_rounds)
        
        # Use prediction_result for betting recommendation
        try:
            from betting.betting_system import betting_system
            if hasattr(betting_system, 'recommend_bet_with_pattern'):
                # Extract certainty from entropy if available
                certainty = None
                if 'entropy' in prediction_result:
                    entropy = prediction_result['entropy']
                    entropy_max = 1.58  # max entropy for 3 outcomes
                    entropy_percent = (entropy / entropy_max) * 100
                    certainty = 100 - entropy_percent
                
                # Pattern-aware betting recommendation
                bet_rec = betting_system.recommend_bet_with_pattern(
                    prediction_result['prediction'],
                    prediction_result['confidence'],
                    certainty,
                    prediction_result.get('pattern_type', 'unknown')
                )
            else:
                # Fallback to standard recommendation
                bet_rec = betting_system.recommend_bet(
                    prediction_result['prediction'],
                    prediction_result['confidence']
                )
            
            return prediction_result, bet_rec
        except Exception as e:
            logger.error(f"Error in integrated betting: {e}")
            # Return prediction only if betting fails
            return prediction_result, None
        
    def _async_log(self, message, level='info'):
        """Log asynchronously to avoid blocking."""
        # Use a thread or task queue to handle logging
        import threading
        def _log_task():
            if level == 'info':
                logger.info(message)
            elif level == 'error':
                logger.error(message)
            # etc.
        
        threading.Thread(target=_log_task).start()

    def _initialize_directories(self) -> dict:
        """Initialize required directory structure with comprehensive validation."""
        result = {"success": False, "critical": True}
        
        try:
            from data.data_utils import ensure_all_directories_exist
            directories_created = ensure_all_directories_exist()
            
            # Verify critical directories
            critical_dirs = ["models", "models/registry", "data", "logs"]
            missing_dirs = [d for d in critical_dirs if not os.path.exists(d)]
            
            if missing_dirs:
                result["error"] = f"Critical directories missing: {missing_dirs}"
                return result
                
            result["success"] = True
            result["directories_created"] = directories_created
            return result
        except Exception as e:
            result["error"] = str(e)
            return result

    def _initialize_registry(self) -> dict:
        """Initialize model registry with transaction-safe operations."""
        result = {"success": False, "critical": True}
        
        try:
            # Define temporary registry path for initialization
            import tempfile
            temp_registry_path = os.path.join(tempfile.gettempdir(), f"temp_registry_{int(time.time())}")
            os.makedirs(temp_registry_path, exist_ok=True)
            
            # Initialize registry in temporary location first
            print(f"{Fore.CYAN}Initializing model registry in temporary location...")
            self.model_registry = ModelRegistry(registry_path=temp_registry_path)
            
            # Validate basic functionality
            basic_functionality = self._validate_registry_functionality()
            if not basic_functionality["success"]:
                result["error"] = f"Registry functionality validation failed: {basic_functionality.get('error')}"
                return result
                
            # Copy to permanent location using atomic operations
            try:
                final_registry_path = MODEL_REGISTRY_PATH
                os.makedirs(final_registry_path, exist_ok=True)
                
                # Update registry path
                self.model_registry.registry_path = final_registry_path
                self.model_registry._save_registry()
                
                # Clean up temporary files
                import shutil
                shutil.rmtree(temp_registry_path, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Registry location update failed: {e}")
                # Continue with temporary location - non-critical
                
            result["success"] = True
            result["registry_path"] = self.model_registry.registry_path
            return result
        except Exception as e:
            result["error"] = str(e)
            return result
        
    def _validate_registry_functionality(self) -> dict:
        """Verify critical registry functionality."""
        result = {"success": False}
        
        try:
            # Check basic attributes
            if not hasattr(self.model_registry, 'models'):
                result["error"] = "Registry missing 'models' attribute"
                return result
                
            if not hasattr(self.model_registry, 'model_active'):
                result["error"] = "Registry missing 'model_active' attribute"
                return result
                
            # Verify critical methods with systematic reflection
            critical_methods = [
                ("get_active_base_models", 0),  # Method name and minimum arg count (excluding self)
                ("get_prediction", 1),
                ("validate_registry_consistency", 0)
            ]
            
            for method_name, min_args in critical_methods:
                if not hasattr(self.model_registry, method_name):
                    result["error"] = f"Registry missing '{method_name}' method"
                    return result
                    
                # Validate method signature
                import inspect
                method = getattr(self.model_registry, method_name)
                sig = inspect.signature(method)
                # Count parameters excluding 'self'
                param_count = len(sig.parameters)
                if param_count < min_args:
                    result["error"] = f"Method '{method_name}' has insufficient parameters: {param_count} (need {min_args})"
                    return result
                    
            # Test get_active_base_models with exception safety
            try:
                active_models = self.model_registry.get_active_base_models()
                # Should return a dictionary, even if empty
                if not isinstance(active_models, dict):
                    result["error"] = f"get_active_base_models returned {type(active_models)}, expected dict"
                    return result
            except Exception as e:
                result["error"] = f"get_active_base_models raised exception: {e}"
                return result
                
            result["success"] = True
            return result
        except Exception as e:
            result["error"] = str(e)
            return result
    def initialize_system(self) -> bool:
        """
        Comprehensive system initialization with dependency validation and recovery.
        
        Returns:
            bool: True if initialization was fully successful
        """
        print(f"{Fore.CYAN}Starting systematic initialization sequence...")
        
        # Track each component's initialization status
        completion_status = {
            "directories": False,
            "registry": False,
            "base_models": False,
            "stacking": False,
            "pipeline": False,
            "calibration": False,
            "betting": False
        }
        
        try:
            # Stage 1: Ensure directory structure is present
            stage_result = self._initialize_directories()
            completion_status["directories"] = stage_result['success']
            if not stage_result['success'] and stage_result.get('critical', True):
                raise InitializationError(f"Directory initialization failed: {stage_result.get('error')}")
            
            # Stage 2: Initialize model registry with robust error recovery
            stage_result = self._initialize_registry()
            completion_status["registry"] = stage_result['success']
            if not stage_result['success'] and stage_result.get('critical', True):
                raise InitializationError(f"Registry initialization failed: {stage_result.get('error')}")
            
            # Stage 3: Validate and initialize base models
            stage_result = self._initialize_base_models()
            completion_status["base_models"] = stage_result['success']
            if not stage_result['success'] and stage_result.get('critical', True):
                raise InitializationError(f"Base models initialization failed: {stage_result.get('error')}")
            
            # Stage 4: Initialize stacking ensemble
            stage_result = self._initialize_stacking()
            completion_status["stacking"] = stage_result['success']
            # Stacking failure is non-critical, continue with fallback
            
            # Stage 5: Initialize prediction pipeline
            stage_result = self._initialize_prediction_pipeline()
            completion_status["pipeline"] = stage_result['success']
            if not stage_result['success'] and stage_result.get('critical', True):
                raise InitializationError(f"Pipeline initialization failed: {stage_result.get('error')}")
            
            # Stage 6: Initialize calibration system and verify calibrators
            stage_result = self._initialize_calibration()
            completion_status["calibration"] = stage_result['success']
            
            # Add calibrator verification step
            if hasattr(self.model_registry, 'verify_calibrators'):
                self.model_registry.verify_calibrators()
            
            # Stage 7: Initialize betting system
            stage_result = self._initialize_betting()
            completion_status["betting"] = stage_result['success']
            # Betting failure is non-critical
            
            # Determine overall initialization status
            critical_components = ["directories", "registry", "base_models", "pipeline"]
            critical_success = all(completion_status[component] for component in critical_components)
            partial_success = any(completion_status.values())
            
            self.initialization_complete = critical_success
            self.system_state['initialized'] = critical_success
            self.system_state['partially_initialized'] = partial_success and not critical_success
            
            # Create detailed initialization report
            self.initialization_report = {
                "timestamp": time.time(),
                "status": "complete" if critical_success else "partial" if partial_success else "failed",
                "component_status": completion_status,
                "successful_components": [k for k, v in completion_status.items() if v],
                "failed_components": [k for k, v in completion_status.items() if not v]
            }
            
            # Report initialization status
            if critical_success:
                print(f"{Fore.GREEN}System initialization completed successfully.")
                return True
            elif partial_success:
                print(f"{Fore.YELLOW}System initialization partially completed. Some features may be limited.")
                return True  # Return success for partial initialization to allow operation with limited functionality
            else:
                print(f"{Fore.RED}System initialization failed. System may be unstable.")
                return False
                
        except InitializationError as e:
            logger.error(f"Initialization error: {e}")
            print(f"{Fore.RED}Critical initialization error: {e}")
            print(f"{Fore.YELLOW}Attempting emergency initialization...")
            
            # Execute emergency initialization
            emergency_result = self._emergency_initialization()
            
            # Update initialization report with emergency status
            self.initialization_report = {
                "timestamp": time.time(),
                "status": "emergency",
                "component_status": completion_status,
                "emergency_result": emergency_result,
                "error": str(e)
            }
            
            return emergency_result
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            traceback.print_exc()
            
            # Update initialization report with failure
            self.initialization_report = {
                "timestamp": time.time(),
                "status": "failed",
                "component_status": completion_status,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
            # Attempt emergency initialization as last resort
            print(f"{Fore.RED}Critical system error during initialization: {e}")
            print(f"{Fore.YELLOW}Attempting emergency initialization...")
            return self._emergency_initialization()
        
    def _initialize_performance_monitoring(self):
        """
        Initialize system-wide performance monitoring for all components.
        
        This method sets up performance tracking for both prediction and
        betting components to enable system-wide optimization.
        """
        # Initialize prediction pipeline cache
        if hasattr(self, 'prediction_pipeline'):
            # Optimize cache size based on system specs
            import psutil
            try:
                # Use 0.5% of system memory for cache, with bounds
                mem = psutil.virtual_memory()
                mem_mb = mem.total / (1024 * 1024)  # Convert to MB
                optimal_cache_size = int(mem_mb * 0.005)  # 0.5% of memory
                
                # Keep within reasonable bounds (100-2000 entries)
                optimal_cache_size = max(100, min(2000, optimal_cache_size))
                
                self.prediction_pipeline._cache_max_size = optimal_cache_size
                print(f"{Fore.GREEN}✓ Prediction cache optimized for system memory ({optimal_cache_size} entries)")
            except:
                # Default if memory detection fails
                self.prediction_pipeline._cache_max_size = 500
                print(f"{Fore.YELLOW}Using default prediction cache size (500 entries)")
        
        # Initialize betting system performance tracking
        try:
            from betting.betting_system import betting_system
            if not hasattr(betting_system, 'performance_metrics'):
                betting_system._initialize_performance_tracking()
                print(f"{Fore.GREEN}✓ Betting system performance tracking initialized")
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not initialize betting performance tracking: {e}")
    
    def _verify_stacking_health(self) -> Dict[str, Any]:
        """
        Verify stacking ensemble health.
        
        This method uses the enhanced stacking health check from ModelRegistry
        to ensure that the stacking ensemble is properly initialized, trained,
        and has consistent feature dimensions.
        
        Returns:
            dict: Health status report
        """
        if not hasattr(self.model_registry, 'test_stacking_ensemble'):
            return {'status': 'unknown', 'reason': 'Test method not available'}
        
        try:
            # Use ModelRegistry's comprehensive health check
            health_report = self.model_registry.test_stacking_ensemble()
            
            # Update system state based on health report
            self.system_state['stacking_healthy'] = (health_report.get('status') == 'healthy')
            
            return health_report
            
        except Exception as e:
            logger.error(f"Error verifying stacking health: {e}")
            print(f"{Fore.RED}Error verifying stacking health: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _apply_system_fixes(self) -> Dict[str, Any]:
        """
        Apply critical system fixes with transaction-based operations.
        
        This method ensures the system is in a consistent and operational state
        by applying targeted fixes to common issues.
        
        Returns:
            dict: Results of fix operations
        """
        results = {
            'fixes_applied': 0,
            'errors': []
        }
        
        # Apply fixes in transaction to ensure consistency
        with StateTransaction(self.model_registry) as transaction:
            try:
                # Fix 1: Apply Markov model prediction fixes
                if hasattr(self.model_registry, 'models'):
                    for model_id, model in self.model_registry.models.items():
                        if model_id.startswith('markov'):
                            self._fix_markov_model(model)
                            results['fixes_applied'] += 1
                
                # Fix 2: Ensure consistent model activation
                if hasattr(self.model_registry, 'model_active') and hasattr(self.model_registry, 'models'):
                    for model_id in self.model_registry.models:
                        if model_id not in self.model_registry.model_active:
                            self.model_registry.model_active[model_id] = True
                            results['fixes_applied'] += 1
                
                # Fix 3: Validate registry consistency
                if hasattr(self.model_registry, 'validate_registry_consistency'):
                    consistency_valid = self.model_registry.validate_registry_consistency()
                    if not consistency_valid:
                        results['errors'].append("Registry consistency validation failed")
                
                # Fix 4: Update expected feature dimensions
                if hasattr(self.model_registry, 'update_expected_feature_dimensions'):
                    self.model_registry.update_expected_feature_dimensions()
                    results['fixes_applied'] += 1
                
                # Fix 5: Apply model-specific fixes if methods exist
                if hasattr(self.model_registry, 'initialize_model_fixes'):
                    self.model_registry.initialize_model_fixes()
                    results['fixes_applied'] += 1
                
                return results
                
            except Exception as e:
                logger.error(f"Error applying system fixes: {e}")
                results['errors'].append(str(e))
                return results
    # Add to SystemManager class in main.py (after existing methods)
    def validate_calibrators(self):
        """
        Verify calibrators are properly initialized and create fallbacks as needed.
        """
        results = {
            'verified': 0,
            'created_fallbacks': 0,
            'issues': []
        }
        
        # Return early if required attributes are missing
        if not hasattr(self, 'model_registry'):
            results['issues'].append('Model registry not available')
            return results
            
        if not hasattr(self.model_registry, 'confidence_calibrators'):
            results['issues'].append('No calibrators found in registry')
            return results
        
        # Verify each calibrator
        for cls in [0, 1, 2]:  # Banker, Player, Tie
            if cls not in self.model_registry.confidence_calibrators:
                # Create missing calibrator
                self._create_fallback_calibrator(cls)
                results['created_fallbacks'] += 1
                results['issues'].append(f'Created missing calibrator for class {cls}')
                continue
                
            calibrator = self.model_registry.confidence_calibrators[cls]
            
            # Check for required attributes
            required_attrs = ['X_min_', 'y_min_']
            missing_attrs = [attr for attr in required_attrs if not hasattr(calibrator, attr)]
            
            if missing_attrs:
                # Replace invalid calibrator
                self._create_fallback_calibrator(cls)
                results['created_fallbacks'] += 1
                results['issues'].append(f'Replaced calibrator for class {cls}, missing: {missing_attrs}')
            else:
                results['verified'] += 1
        
        return results

    def _create_fallback_calibrator(self, cls):
        """
        Create a fallback calibrator for a specific class when proper initialization fails.
        
        Args:
            cls: The class index (0=Banker, 1=Player, 2=Tie)
        """
        import numpy as np
        from sklearn.isotonic import IsotonicRegression
        
        # Create synthetic calibration data with statistically informed defaults
        if cls == 2:  # Tie-specific calibration
            # Ties are rare (~9.5%), use conservative calibration curve
            X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            y = np.array([0.05, 0.1, 0.15, 0.25, 0.35, 0.45, 0.6, 0.75, 0.85])
        else:  # Banker/Player calibration
            # More common outcomes (~45.5%/44.9%), use balanced calibration curve
            X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        # Create and fit fallback calibrator
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(X.reshape(-1, 1), y)
        
        # Replace the problematic calibrator in the registry
        self.model_registry.confidence_calibrators[cls] = calibrator
        logger.info(f"Created fallback calibrator for class {cls}")
    
    def _initialize_stacking(self) -> Dict[str, Any]:
        """
        Initialize and validate the stacking ensemble.
        
        This method ensures that the stacking ensemble is properly initialized,
        trained, and validated for consistent predictions.
        
        Returns:
            dict: Stacking initialization results
        """
        if not hasattr(self.model_registry, 'reset_stacking'):
            return {'status': 'failed', 'reason': 'Registry missing reset_stacking method'}
        
        try:
            # First, check if stacking ensemble exists and is healthy
            if "stacking_ensemble" in self.model_registry.models:
                # Test stacking health
                health_report = self.model_registry.test_stacking_ensemble()
                
                if health_report.get('status') == 'healthy':
                    print(f"{Fore.GREEN}✓ Stacking ensemble health check passed")
                    return health_report
                
                print(f"{Fore.YELLOW}⚠ Stacking requires reset. Status: {health_report.get('status', 'unknown')}")
            
            # Reset stacking ensemble
            stacking = self.model_registry.reset_stacking(force_reset=True)
            
            if stacking is None:
                print(f"{Fore.RED}⚠ Stacking reset failed. Using fallback mechanisms.")
                return {'status': 'reset_failed'}
            
            # Verify stacking was properly initialized
            health_report = self.model_registry.test_stacking_ensemble()
            
            if health_report.get('status') == 'healthy':
                print(f"{Fore.GREEN}✓ Stacking ensemble initialized and healthy")
                return health_report
            else:
                print(f"{Fore.YELLOW}⚠ Stacking health check after reset: {health_report.get('status', 'unknown')}")
                return health_report
                
        except Exception as e:
            logger.error(f"Error initializing stacking ensemble: {e}")
            print(f"{Fore.RED}Error initializing stacking ensemble: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _initialize_calibration(self) -> Dict[str, Any]:
        """Initialize confidence calibration system with robust validation."""
        # First try using the dedicated initializer if available
        try:
            from prediction.utils.calibration_initializer import initialize_calibration
            result = initialize_calibration()
            if result.get('status') == 'success':
                print(f"{Fore.GREEN}✓ Confidence calibration initialized via dedicated initializer")
                
                # Update system state if calibrators were properly initialized
                self.system_state['calibration_ready'] = True
                return result
        except ImportError:
            pass  # Fall through to original implementation
        
        # Original implementation as fallback
        if not hasattr(self.model_registry, 'calibration_manager'):
            return {'status': 'failed', 'reason': 'Registry missing calibration_manager'}
        
        try:
            # Initialize calibration from historical data
            calibration_result = self.model_registry.calibration_manager.calibrate_from_history()
            print(f"{Fore.GREEN}✓ Confidence calibration initialized")
            
            # Critical addition: Verify calibrators are fully initialized
            validation_results = self.validate_calibrators()
            
            # Update system state
            self.system_state['calibration_ready'] = validation_results['verified'] == 3  # All three classes verified
            
            if validation_results['issues']:
                print(f"{Fore.YELLOW}Calibration initialized with {validation_results['created_fallbacks']} fallback mechanisms")
                for issue in validation_results['issues']:
                    logger.warning(f"Calibration issue: {issue}")
                calibration_result['validation_issues'] = validation_results['issues']
                calibration_result['status'] = 'partial'
            else:
                print(f"{Fore.GREEN}✓ All calibrators successfully verified")
                    
            return calibration_result
                    
        except Exception as e:
            logger.error(f"Error initializing calibration: {e}")
            print(f"{Fore.YELLOW}Warning: Error initializing calibration: {e}")
            
            # Create fallback calibrators for all classes
            for cls in [0, 1, 2]:
                self._create_fallback_calibrator(cls)
            
            print(f"{Fore.YELLOW}Created fallback calibrators for all classes")
            self.system_state['calibration_ready'] = True  # Using fallbacks
            
            return {'status': 'fallback', 'error': str(e)}
                
        except Exception as e:
            logger.error(f"Error initializing calibration: {e}")
            print(f"{Fore.YELLOW}Warning: Error initializing calibration: {e}")
            
            # Create fallback calibrators for all classes
            for cls in [0, 1, 2]:
                self._create_fallback_calibrator(cls)
            
            print(f"{Fore.YELLOW}Created fallback calibrators for all classes")
            self.system_state['calibration_ready'] = True  # Using fallbacks
            
            return {'status': 'fallback', 'error': str(e)}
    
    def _verify_model_training(self) -> Dict[str, Any]:
        """
        Verify that all models are properly trained.
        
        This method checks training status of all models and initiates
        training for any untrained models to ensure system readiness.
        
        Returns:
            dict: Model training verification results
        """
        results = {
            'trained_models': [],
            'untrained_models': [],
            'training_errors': []
        }
        
        try:
            # Check each model's training status
            for model_id, model in self.model_registry.models.items():
                if not hasattr(model, 'is_trained') or not model.is_trained:
                    results['untrained_models'].append(model_id)
                else:
                    results['trained_models'].append(model_id)
            
            # If untrained models found, offer to train them
            if results['untrained_models']:
                print(f"{Fore.YELLOW}Found {len(results['untrained_models'])} untrained models.")
                
                # Auto-train models with available data
                self._train_untrained_models(results['untrained_models'])
                
                # Update results after training
                results['untrained_models'] = []
                for model_id, model in self.model_registry.models.items():
                    if not hasattr(model, 'is_trained') or not model.is_trained:
                        results['untrained_models'].append(model_id)
                    else:
                        if model_id not in results['trained_models']:
                            results['trained_models'].append(model_id)
            
            # Report final status
            if not results['untrained_models']:
                print(f"{Fore.GREEN}All models ({len(results['trained_models'])}) are properly trained.")
            else:
                print(f"{Fore.YELLOW}Warning: {len(results['untrained_models'])} models remain untrained.")
            
            return results
                
        except Exception as e:
            logger.error(f"Error verifying model training: {e}")
            print(f"{Fore.RED}Error verifying model training: {e}")
            results['error'] = str(e)
            return results
    
    def _train_untrained_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """
        Train models that aren't properly initialized.
        
        Args:
            model_ids: List of model IDs that need training
            
        Returns:
            dict: Training results
        """
        from data.data_utils import prepare_combined_dataset
        import numpy as np
        
        results = {
            'success': [],
            'failed': []
        }
        
        try:
            # Get data for training
            try:
                X, y = prepare_combined_dataset()
                data_available = len(X) >= 10
                
                print(f"{Fore.CYAN}Training with {len(X)} data samples...")
                
            except Exception as e:
                logger.error(f"Error preparing training data: {e}")
                print(f"{Fore.YELLOW}Error loading training data: {e}")
                data_available = False
            
            # Train with available data or use minimal synthetic data
            for model_id in model_ids:
                if model_id not in self.model_registry.models:
                    results['failed'].append(model_id)
                    continue
                
                model = self.model_registry.models[model_id]
                
                try:
                    if data_available:
                        # Train with real data
                        if model_id.startswith('markov'):
                            # For Markov models, convert to sequence
                            if hasattr(X, 'values'):
                                # Handle pandas DataFrame
                                sequence = []
                                for _, row in X.iterrows():
                                    sequence.extend(row.values)
                                sequence.extend(y.values)
                            else:
                                # Handle numpy array
                                sequence = X.flatten().tolist() + y.tolist()
                                
                            model.fit(sequence)
                        else:
                            # For other models, use standard interface
                            model.fit(X, y)
                            
                        model.is_trained = True
                        results['success'].append(model_id)
                        print(f"{Fore.GREEN}✓ Successfully trained {model_id}")
                    else:
                        # Use synthetic data for minimal training
                        self._train_with_synthetic_data(model, model_id)
                        results['success'].append(model_id)
                
                except Exception as e:
                    logger.error(f"Error training {model_id}: {e}")
                    print(f"{Fore.RED}Error training {model_id}: {e}")
                    results['failed'].append(model_id)
                    
                    # Try with synthetic data as fallback
                    try:
                        if not (hasattr(model, 'is_trained') and model.is_trained):
                            self._train_with_synthetic_data(model, model_id)
                            results['success'].append(model_id)
                            results['failed'].remove(model_id)
                    except Exception:
                        pass
            
            # Save registry after training
            if results['success']:
                self.model_registry._save_registry()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in model training process: {e}")
            print(f"{Fore.RED}Error in model training process: {e}")
            return {'error': str(e)}
    

    def _initialize_prediction_pipeline(self):
        """
        Initialize the optimized prediction pipeline with comprehensive validation.
        """
        from prediction.prediction_pipeline import PredictionPipeline
        
        try:
            # Validate system components first
            validation_results = self.validate_system_components()
            if validation_results['issues_detected']:
                print(f"{Fore.YELLOW}Addressed system initialization issues:")
                for fix in validation_results['fixes_applied']:
                    print(f"  + {fix}")
            
            # Create pipeline with registry (now guaranteed to be initialized)
            if not hasattr(self, 'prediction_pipeline'):
                print(f"{Fore.CYAN}Initializing optimized prediction pipeline...")
                
                # Registry will be wrapped with adapter inside PredictionPipeline
                self.prediction_pipeline = PredictionPipeline(self.model_registry)
                
                # Configure pipeline
                self.prediction_pipeline._cache_max_size = 500
                
                # Verify pipeline with test prediction
                try:
                    test_input = np.array([[0, 1, 0, 1, 0]])
                    self.prediction_pipeline.predict(test_input)
                    print(f"{Fore.GREEN}✓ Prediction pipeline verified with test prediction")
                except Exception as test_error:
                    print(f"{Fore.YELLOW}Pipeline test failed but continuing: {test_error}")
                
                print(f"{Fore.GREEN}✓ Prediction pipeline initialized")
            
            return self.prediction_pipeline
        except Exception as e:
            logger.error(f"Error initializing prediction pipeline: {e}")
            print(f"{Fore.RED}Error initializing prediction pipeline: {e}")
            traceback.print_exc()
            
            # Try emergency initialization with adapter only
            try:
                from prediction.registry_adapter import ModelRegistryAdapter
                from prediction.prediction_pipeline import PredictionPipeline
                
                # Create empty adapter and pipeline
                print(f"{Fore.YELLOW}Attempting emergency pipeline initialization...")
                adapter = ModelRegistryAdapter(None)  # No registry reference
                self.prediction_pipeline = PredictionPipeline(adapter)
                print(f"{Fore.GREEN}✓ Emergency prediction pipeline initialized")
                return self.prediction_pipeline
            except Exception as emergency_error:
                print(f"{Fore.RED}Emergency initialization also failed: {emergency_error}")
                return None

    def predict_with_transaction(self, prev_rounds):
        """
        Make prediction with transaction-based error handling and recovery.
        
        This method implements a comprehensive prediction workflow with transaction
        semantics, ensuring system stability through progressive fallback mechanisms,
        detailed performance monitoring, and pattern-specific optimizations.
        
        Args:
            prev_rounds: Previous game outcomes (list or ndarray)
            
        Returns:
            dict: Prediction results with comprehensive metadata
        """
        # Initialize performance monitoring
        prediction_start_time = time.time()
        diagnostics = {
            'validation_time_ms': 0,
            'pipeline_time_ms': 0,
            'fallbacks_triggered': [],
            'optimization_paths': []
        }
        
        try:
            # PHASE 1: System State Validation
            validation_start = time.time()
            
            # Verify model registry initialization
            if not hasattr(self, 'model_registry') or self.model_registry is None:
                print(f"{Fore.YELLOW}Warning: Model registry not initialized. Attempting recovery...")
                validation_results = self.validate_system_components()
                diagnostics['fallbacks_triggered'].append('model_registry_initialization')
                
                if not validation_results['registry_initialized']:
                    print(f"{Fore.RED}Critical error: Unable to initialize model registry")
                    # Return emergency prediction via adapter
                    from prediction.registry_adapter import ModelRegistryAdapter
                    adapter = ModelRegistryAdapter(None)
                    fallback = adapter._generate_fallback_prediction()
                    fallback['diagnostics'] = diagnostics
                    return fallback
            
            # Ensure prediction pipeline is initialized
            if not hasattr(self, 'prediction_pipeline') or self.prediction_pipeline is None:
                print(f"{Fore.CYAN}Initializing prediction pipeline on-demand...")
                self._initialize_prediction_pipeline()
                diagnostics['fallbacks_triggered'].append('pipeline_initialization')
                
                # If initialization failed, use adapter directly
                if not hasattr(self, 'prediction_pipeline') or self.prediction_pipeline is None:
                    print(f"{Fore.RED}Critical error: Unable to initialize prediction pipeline")
                    from prediction.registry_adapter import ModelRegistryAdapter
                    adapter = ModelRegistryAdapter(self.model_registry)
                    fallback = adapter._generate_fallback_prediction()
                    fallback['diagnostics'] = diagnostics
                    return fallback
                    
            # Record validation time
            diagnostics['validation_time_ms'] = (time.time() - validation_start) * 1000
            
            # PHASE 2: Input Processing
            # Format input consistently for processing
            try:
                if isinstance(prev_rounds, list):
                    # Convert to numpy array for consistent handling
                    prev_rounds_arr = np.array(prev_rounds).reshape(1, -1)
                elif isinstance(prev_rounds, np.ndarray):
                    # Ensure proper shape for prediction
                    prev_rounds_arr = prev_rounds.reshape(1, -1) if prev_rounds.ndim == 1 else prev_rounds
                else:
                    # Try to handle other input types
                    prev_rounds_arr = np.array(prev_rounds).reshape(1, -1)
                    
                # Validate input values (ensure all values are 0, 1, or 2)
                if np.any((prev_rounds_arr < 0) | (prev_rounds_arr > 2)):
                    logger.warning("Input contains invalid values, clipping to valid range")
                    prev_rounds_arr = np.clip(prev_rounds_arr, 0, 2)
                    diagnostics['fallbacks_triggered'].append('input_value_correction')
                    
            except Exception as input_error:
                logger.error(f"Error processing input: {input_error}")
                diagnostics['fallbacks_triggered'].append('input_processing_error')
                
                # Create fallback input (simple alternating pattern)
                prev_rounds_arr = np.array([[0, 1, 0, 1, 0]])
                
            # PHASE 3: Transaction-protected Prediction Execution
            pipeline_start = time.time()
            
            # Check cache before prediction if enabled
            cache_key = None
            if hasattr(self.prediction_pipeline, '_get_cache_key'):
                try:
                    cache_key = self.prediction_pipeline._get_cache_key(prev_rounds_arr)
                    cached_result = self.prediction_pipeline._get_from_cache(cache_key)
                    if cached_result:
                        logger.debug("Using cached prediction result")
                        cached_result['cached'] = True
                        cached_result['execution_time_ms'] = (time.time() - prediction_start_time) * 1000
                        return cached_result
                except Exception as cache_error:
                    logger.warning(f"Cache lookup failed: {cache_error}")
                    diagnostics['fallbacks_triggered'].append('cache_lookup_error')
            
            try:
                # Execute prediction with full profiling
                result = self.prediction_pipeline.predict_with_profiling(prev_rounds_arr)
                
                # Record pipeline execution time
                diagnostics['pipeline_time_ms'] = (time.time() - pipeline_start) * 1000
                
                # PHASE 4: Result Enhancement
                
                # Extract performance metrics for logging and optimization
                if 'performance_metrics' in result:
                    performance = result['performance_metrics']
                    
                    # Log slow predictions for optimization
                    total_time = performance.get('total', 0) * 1000  # Convert to ms
                    if total_time > 100:  # Log if prediction takes over 100ms
                        logger.warning(f"Slow prediction detected: {total_time:.2f}ms")
                        
                        # Identify performance bottleneck
                        slowest_stage = max(
                            [(stage, time) for stage, time in performance.items() 
                            if stage not in ['total', 'error_stage']],
                            key=lambda x: x[1]
                        )
                        logger.warning(f"Slowest stage: {slowest_stage[0]} ({slowest_stage[1]*1000:.2f}ms)")
                        
                        # Track repeated slow patterns for adaptive optimization
                        if not hasattr(self, '_slow_prediction_patterns'):
                            self._slow_prediction_patterns = {}
                        
                        # Log pattern type if available for targeted optimization
                        pattern_type = result.get('pattern_type', 'unknown')
                        if pattern_type not in self._slow_prediction_patterns:
                            self._slow_prediction_patterns[pattern_type] = 0
                        self._slow_prediction_patterns[pattern_type] += 1
                
                # Add pattern effectiveness for betting integration
                if 'pattern_type' in result and result['pattern_type'] != 'unknown':
                    # Calculate pattern effectiveness based on confidence and entropy
                    base_effectiveness = 0.5  # Default middle value
                    
                    # Higher confidence increases effectiveness
                    confidence_factor = (result.get('confidence', 50) - 50) / 50  # -1 to 1 scale
                    confidence_factor = max(-0.5, min(0.5, confidence_factor))  # Limit influence
                    
                    # Lower entropy (higher certainty) increases effectiveness
                    certainty = 0.5
                    if 'entropy' in result:
                        entropy = result['entropy']
                        entropy_max = 1.58  # max entropy for 3 outcomes
                        entropy_ratio = entropy / entropy_max
                        certainty = 1 - entropy_ratio  # 0-1 scale, higher is better
                    
                    # Base model agreement increases effectiveness
                    agreement_factor = 0
                    if 'base_model_agreement' in result:
                        agreement = result['base_model_agreement']
                        agreement_factor = (agreement - 0.5) * 0.4  # Scale to appropriate range
                    
                    # Calculate overall effectiveness
                    pattern_effectiveness = base_effectiveness + (0.2 * confidence_factor) + (0.3 * (certainty - 0.5)) + agreement_factor
                    pattern_effectiveness = max(0.1, min(0.9, pattern_effectiveness))  # Bound between 0.1-0.9
                    
                    # Add to result
                    result['pattern_effectiveness'] = pattern_effectiveness
                
                # PHASE 5: Adaptive Optimization
                
                # Apply pattern-specific optimizations based on historical performance
                if hasattr(self, '_pattern_optimizations') and 'pattern_type' in result:
                    pattern_type = result['pattern_type']
                    
                    if pattern_type in self._pattern_optimizations:
                        # Apply historical optimization rules based on pattern type
                        optimization = self._pattern_optimizations[pattern_type]
                        
                        # Example: Adjust confidence based on historical calibration
                        if 'confidence_adjustment' in optimization and 'confidence' in result:
                            original_confidence = result['confidence']
                            adjustment = optimization['confidence_adjustment']
                            result['confidence'] = min(95, max(5, original_confidence + adjustment))
                            diagnostics['optimization_paths'].append(f"confidence_adjusted_{pattern_type}")
                
                # PHASE 6: Cache Result
                if cache_key is not None and hasattr(self.prediction_pipeline, '_add_to_cache'):
                    try:
                        self.prediction_pipeline._add_to_cache(cache_key, result)
                    except Exception as cache_error:
                        logger.warning(f"Cache storage failed: {cache_error}")
                
                # PHASE 7: Periodically update stacking model (if needed)
                self._maybe_update_stacking(prev_rounds_arr, result)
                
                # Add complete execution time
                result['execution_time_ms'] = (time.time() - prediction_start_time) * 1000
                
                # Add diagnostics
                result['execution_diagnostics'] = diagnostics
                
                return result
                
            except Exception as prediction_error:
                # Record failure details
                logger.error(f"Error in prediction execution: {prediction_error}")
                diagnostics['fallbacks_triggered'].append('prediction_execution_error')
                diagnostics['error'] = str(prediction_error)
                
                # Attempt to retrieve partial result from pipeline if available
                partial_result = None
                if hasattr(self.prediction_pipeline, 'partial_result'):
                    partial_result = self.prediction_pipeline.partial_result
                    
                # PHASE 8: Fallback and Recovery
                
                # Fallback Strategy 1: Use pattern-specific fallback
                if 'pattern_type' in locals() and hasattr(self.prediction_pipeline, 'fallback_manager'):
                    try:
                        fallback_result = self.prediction_pipeline.fallback_manager.generate_fallback(
                            prev_rounds_arr, "transaction_error", error=str(prediction_error)
                        )
                        
                        # Add execution diagnostics
                        fallback_result['execution_time_ms'] = (time.time() - prediction_start_time) * 1000
                        fallback_result['execution_diagnostics'] = diagnostics
                        fallback_result['pattern_effectiveness'] = 0.3  # Conservative value for fallback
                        
                        return fallback_result
                    except Exception as fallback_error:
                        logger.error(f"Fallback strategy 1 failed: {fallback_error}")
                        diagnostics['fallbacks_triggered'].append('pattern_fallback_failed')
                
                # Fallback Strategy 2: Direct adapter fallback
                try:
                    from prediction.registry_adapter import ModelRegistryAdapter
                    adapter = ModelRegistryAdapter(self.model_registry)
                    fallback_result = adapter._generate_fallback_prediction()
                    
                    # Add error context
                    fallback_result['error'] = str(prediction_error)
                    fallback_result['execution_time_ms'] = (time.time() - prediction_start_time) * 1000
                    fallback_result['execution_diagnostics'] = diagnostics
                    
                    return fallback_result
                except Exception as adapter_error:
                    logger.error(f"Fallback strategy 2 failed: {adapter_error}")
                    diagnostics['fallbacks_triggered'].append('adapter_fallback_failed')
                
                # Fallback Strategy 3: Ultimate emergency fallback
                emergency_result = {
                    'prediction': 0,  # Default to banker (slight edge in baccarat)
                    'confidence': 33.3,
                    'distribution': {0: 45.0, 1: 45.0, 2: 10.0},
                    'fallback': True,
                    'error': str(prediction_error),
                    'emergency': True,
                    'execution_time_ms': (time.time() - prediction_start_time) * 1000,
                    'execution_diagnostics': diagnostics
                }
                
                return emergency_result
                
        except Exception as e:
            # Complete system failure - handle with ultra-robust fallback
            logger.critical(f"Critical error in prediction transaction: {e}")
            
            try:
                # Try traceback capture for diagnostics
                import traceback
                error_trace = traceback.format_exc()
                logger.critical(f"Error trace: {error_trace}")
            except:
                error_trace = "Traceback capture failed"
            
            # Ultra-fallback result with complete insulation from further errors
            return {
                'prediction': 0,  # Default to banker (slight edge in baccarat)
                'confidence': 33.3,
                'distribution': {0: 45.0, 1: 45.0, 2: 10.0},
                'fallback': True,
                'ultra_emergency': True,
                'error': str(e),
                'traceback': error_trace,
                'execution_time_ms': (time.time() - prediction_start_time) * 1000
            }

    def _maybe_update_stacking(self, prev_rounds, prediction_result):
        """
        Conditionally update stacking model to improve future predictions.
        
        This method intelligently decides when to update the stacking ensemble
        based on prediction patterns and system load.
        
        Args:
            prev_rounds: Previous game outcomes
            prediction_result: Prediction results
        """
        # Only update occasionally to avoid overwhelming the system
        if not hasattr(self, '_stacking_update_counter'):
            self._stacking_update_counter = 0
        
        self._stacking_update_counter += 1
        
        # Update every 5 predictions to balance learning vs performance
        if self._stacking_update_counter % 5 == 0:
            try:
                # Check if we need to execute system maintenance
                if self._stacking_update_counter % 50 == 0:
                    # Perform periodic system maintenance
                    self.model_registry.perform_scheduled_maintenance()
                
                # No actual outcome yet, so we can't update the model
                # This would happen after the game result is known
                pass
                
            except Exception as e:
                logger.error(f"Error in stacking update: {e}")
    def _train_with_synthetic_data(self, model, model_id: str) -> bool:
        """
        Train a model with synthetic data when real data isn't available.
        
        Args:
            model: The model to train
            model_id: The model's identifier
            
        Returns:
            bool: True if training was successful
        """
        import numpy as np
        
        # Create synthetic training data
        X_synthetic = np.array([
            [0, 0, 0, 0, 0],  # All banker
            [1, 1, 1, 1, 1],  # All player
            [0, 1, 0, 1, 0],  # Alternating B/P
            [1, 0, 1, 0, 1],  # Alternating P/B
            [0, 0, 1, 1, 2],  # Mixed with tie
            [2, 0, 1, 0, 1],  # Starting with tie
            [0, 1, 2, 0, 1],  # Mixed
            [1, 1, 0, 0, 2],  # Mixed
        ])
        y_synthetic = np.array([0, 1, 0, 1, 2, 0, 1, 2])
        
        # Train based on model type
        if model_id.startswith('markov'):
            # For Markov models, convert to sequence
            sequence = X_synthetic.flatten().tolist() + y_synthetic.tolist()
            model.fit(sequence)
        else:
            # For other models
            model.fit(X_synthetic, y_synthetic)
        
        # Mark as trained
        model.is_trained = True
        print(f"{Fore.GREEN}✓ {model_id} trained with synthetic data")
        
        return True
    
    def _run_initial_competition(self) -> Dict[str, Any]:
        """
        Run initial model competition to optimize model weights.
        
        This method executes the model competition process to identify
        and prioritize the best-performing models based on historical data.
        
        Returns:
            dict: Competition results
        """
        if not hasattr(self.model_registry, 'run_model_competition'):
            return {'status': 'skipped', 'reason': 'Registry missing competition method'}
        
        try:
            # Run competition with proper error handling
            results = self.model_registry.run_model_competition()
            
            if results:
                print(f"{Fore.GREEN}✓ Model competition completed successfully")
                return {'status': 'success', 'results': results}
            else:
                print(f"{Fore.YELLOW}Model competition completed with no results")
                return {'status': 'no_results'}
                
        except Exception as e:
            logger.error(f"Error running model competition: {e}")
            print(f"{Fore.YELLOW}Warning: Error running model competition: {e}")
            return {'status': 'error', 'error': str(e)}
    

    def run_integrated_prediction_system(self) -> None:
        """
        Run the integrated prediction system with enhanced performance optimization.
        
        This method provides a direct entry point to the prediction system with
        comprehensive performance monitoring, adaptive optimization, and
        transaction-based operations.
        """
        # Ensure system is initialized
        if not self.initialization_complete:
            print(f"{Fore.YELLOW}System initialization not complete. Attempting initialization...")
            success = self.initialize_system()
            if not success:
                print(f"{Fore.RED}System initialization failed. Cannot proceed.")
                return
        
        # Initialize prediction pipeline if not already done
        if not hasattr(self, 'prediction_pipeline'):
            self._initialize_prediction_pipeline()
        
        # Import prediction interface
        try:
            from interface.io_predict import main_prediction_loop
            
            # Display welcome message
            self.show_welcome_message()
            
            # Start prediction loop with system manager
            main_prediction_loop(system_manager=self)
            
        except ImportError as e:
            print(f"{Fore.RED}Error loading prediction interface: {e}")
            print(f"{Fore.RED}Please ensure the interface package is installed correctly.")
            return
        except Exception as e:
            logger.error(f"Error in integrated prediction system: {e}")
            print(f"{Fore.RED}Error in integrated prediction system: {e}")
            return

    def _initialize_betting_integration(self):
        """
        Initialize betting system integration with prediction pipeline.
        
        This method ensures the betting system is properly integrated with
        the prediction pipeline and transaction management systems.
        
        Returns:
            bool: True if initialization successful
        """
        from betting.betting_system import betting_system
        
        try:
            # Ensure betting system has performance tracking
            if not hasattr(betting_system, 'performance_metrics'):
                betting_system._initialize_performance_tracking()
            
            # Check if betting system needs to initialize pattern-specific methods
            if not hasattr(betting_system, 'recommend_bet_with_pattern'):
                print(f"{Fore.YELLOW}Adding pattern-specific betting optimizations...")
                
                # Add method references dynamically
                import types
                
                # Use the implementation code from earlier
                betting_system.recommend_bet_with_pattern = types.MethodType(
                    betting_system.recommend_bet_with_pattern, betting_system
                )
                
                betting_system._adjust_strategy_for_pattern = types.MethodType(
                    betting_system._adjust_strategy_for_pattern, betting_system
                )
                
                betting_system._get_pattern_bet_modifier = types.MethodType(
                    betting_system._get_pattern_bet_modifier, betting_system
                )
                
                betting_system.handle_bet_with_transaction = types.MethodType(
                    betting_system.handle_bet_with_transaction, betting_system
                )
                
                betting_system._process_bet_decision = types.MethodType(
                    betting_system._process_bet_decision, betting_system
                )
                
                print(f"{Fore.GREEN}✓ Betting system integration complete")
            
            # Save any configuration changes
            betting_system._save_balance()
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing betting integration: {e}")
            print(f"{Fore.RED}Error initializing betting integration: {e}")
            return False

    # Update the initialize_system method to include betting integration
    def initialize_system(self) -> bool:
        # Existing code...
        
        # After initializing prediction pipeline (Stage 8)
        print(f"{Fore.CYAN}Initializing prediction pipeline...")
        self._initialize_prediction_pipeline()
        
        # Add Stage 9: Initialize betting integration
        print(f"{Fore.CYAN}Initializing betting system integration...")
        self._initialize_betting_integration()

    def _emergency_initialization(self) -> bool:
        """
        Perform emergency initialization when normal initialization fails.
        
        This method implements a more aggressive initialization approach
        that focuses on establishing minimal functionality when the
        standard initialization process fails.
        
        Returns:
            bool: True if emergency initialization succeeded
        """
        print(f"{Fore.YELLOW}Performing emergency system initialization...")
        
        try:
            # Create fresh registry instance
            self.model_registry = ModelRegistry()
            
            # Apply essential fixes only
            try:
                # Fix XGBoost and Markov models specifically
                for model_id, model in self.model_registry.models.items():
                    if model_id.startswith('markov'):
                        self._fix_markov_model(model)
                    
                # Ensure all models are active
                for model_id in self.model_registry.models:
                    self.model_registry.model_active[model_id] = True
                    
                # Initialize stacking with minimal functionality
                if "stacking_ensemble" not in self.model_registry.models:
                    self.model_registry.reset_stacking(force_reset=True)
                
            except Exception as inner_e:
                logger.error(f"Error in emergency fixes: {inner_e}")
            
            # Mark initialization as partially complete
            self.initialization_complete = True
            self.registry_healthy = False
            
            print(f"{Fore.YELLOW}Emergency initialization completed with limited functionality.")
            return True
            
        except Exception as e:
            logger.critical(f"Emergency initialization failed: {e}")
            print(f"{Fore.RED}Emergency initialization failed: {e}")
            print(f"{Fore.RED}System may be unstable or non-functional.")
            
            # Mark initialization as failed
            self.initialization_complete = False
            self.registry_healthy = False
            
            return False
    
    def show_welcome_message(self) -> None:
        """
        Display welcome message and system status information.
        
        This method provides a user-friendly introduction with information
        about the current system state, available data, and model status.
        """
        print(Fore.CYAN + "\n" + "="*60)
        print(Fore.CYAN + "=== Enhanced Baccarat Prediction System ===".center(60))
        print(Fore.CYAN + "=== With Online Learning & Betting System ===".center(60))
        print(Fore.CYAN + "="*60 + "\n")
        
        # Check for existing model and data files
        model_exists = os.path.exists(MODEL_FILE)
        data_exists = os.path.exists(REALTIME_FILE)
        log_exists = os.path.exists(LOG_FILE)
        
        # Display model information
        if model_exists:
            model_size = os.path.getsize(MODEL_FILE) / (1024*1024)  # Convert to MB
            print(f"{Fore.GREEN}✓ Model found: {MODEL_FILE} ({model_size:.2f} MB)")
        else:
            print(f"{Fore.YELLOW}⚠ No model file found. A new model will be created during first prediction.")
        
        # Display data information
        if data_exists:
            try:
                import pandas as pd
                df = pd.read_csv(REALTIME_FILE)
                print(f"{Fore.GREEN}✓ Data found: {REALTIME_FILE} ({len(df)} records)")
            except Exception:
                print(f"{Fore.GREEN}✓ Data found: {REALTIME_FILE}")
        else:
            print(f"{Fore.YELLOW}⚠ No realtime data found. Starting fresh.")
        
        # Display prediction log information
        if log_exists:
            try:
                import pandas as pd
                log_df = pd.read_csv(LOG_FILE)
                total = len(log_df)
                if 'Correct' in log_df.columns:
                    correct = log_df['Correct'].sum()
                    accuracy = (correct / total) * 100 if total > 0 else 0
                    print(f"{Fore.GREEN}✓ Prediction log found: {total} predictions, {accuracy:.2f}% accuracy")
                else:
                    print(f"{Fore.GREEN}✓ Prediction log found: {total} predictions")
            except Exception:
                print(f"{Fore.GREEN}✓ Prediction log found: {LOG_FILE}")
        else:
            print(f"{Fore.YELLOW}⚠ No prediction log found. Starting fresh.")
        
        # Display registry status if initialization was attempted
        if hasattr(self, 'registry_healthy'):
            if self.registry_healthy:
                if hasattr(self.model_registry, 'models'):
                    model_count = len(self.model_registry.models)
                    active_count = sum(1 for active in self.model_registry.model_active.values() if active)
                    print(f"{Fore.GREEN}✓ Model registry initialized with {model_count} models ({active_count} active)")
                else:
                    print(f"{Fore.GREEN}✓ Model registry initialized")
            else:
                print(f"{Fore.YELLOW}⚠ Model registry initialization incomplete or in limited functionality mode")
        
        print("\n")
    
    def run_prediction_system(self) -> None:
        """
        Run the main prediction system with comprehensive error handling.
        
        This method provides a menu-driven interface for interacting with
        the prediction system, implementing proper error handling and
        ensuring system stability throughout the session.
        """
        # Display initialization status
        if not self.initialization_complete:
            print(f"{Fore.YELLOW}Warning: System initialization was not fully completed.")
            print(f"{Fore.YELLOW}Some functionality may be limited or unavailable.")
        
        # Display betting system info
        betting_system.display_balance()
        betting_system.display_betting_stats()
            
        # Main menu loop
        while True:
            try:
                choice = self.get_user_choice()
                
                if choice == 1:
                    # Make predictions
                    main_prediction_loop()
                
                elif choice == 2:
                    # Betting system
                    display_betting_menu()
                
                elif choice == 3:
                    # Analytics
                    run_analytics_menu()
                
                elif choice == 4:
                    # System settings
                    self.system_settings()
                
                elif choice == 5:
                    # Help and info
                    self.show_help_info()
                
                elif choice == 0:
                    # Exit
                    print(Fore.YELLOW + "\nExiting Baccarat Prediction System...")
                    print("Thank you for using the system!")
                    break
                
                else:
                    print(Fore.RED + "Invalid choice. Please enter a number between 0 and 5.")
            
            except KeyboardInterrupt:
                print(Fore.YELLOW + "\n\nOperation cancelled. Returning to main menu...")
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                print(Fore.RED + f"\nAn error occurred: {e}")
                print(Fore.YELLOW + "Continuing...")
    
    def get_user_choice(self) -> int:
        """
        Display main menu and get user choice with input validation.
        
        Returns:
            int: The user's choice (0-5)
        """
        print(Fore.CYAN + "\nMain Menu:")
        print("1. Make Predictions")
        print("2. Betting System")
        print("3. Analytics & Visualization")
        print("4. System Settings")
        print("5. Help & Information")
        print("0. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (0-5): ")
                return int(choice)
            except ValueError:
                print(Fore.RED + "Please enter a valid number between 0 and 5.")
    
    def system_settings(self) -> None:
        """
        Display and manage system settings with comprehensive options.
        
        This method provides a menu for system maintenance, model management,
        data operations, and diagnostic tools with proper error handling.
        """
        while True:
            print(Fore.CYAN + "\n=== System Settings ===")
            print("1. Reset prediction model")
            print("2. Reset betting system")
            print("3. Check data balance")
            print("4. Backup current model")
            print("5. Analyze model contributions")
            print("6. Force retrain all models")
            print("7. Debug model registry")
            print("8. Reinitialize model registry")
            print("9. Reset stacking ensemble")
            print("10. Return to main menu")
            
            try:
                choice = input("\nEnter your choice (1-10): ")
                
                if choice == '1':
                    self._reset_prediction_model()
                
                elif choice == '2':
                    self._reset_betting_system()
                
                elif choice == '3':
                    self._check_data_balance()
                
                elif choice == '4':
                    self._backup_current_model()
                
                elif choice == '5':
                    self._analyze_model_contributions()
                
                elif choice == '6':
                    self._force_retrain_all_models()
                
                elif choice == '7':
                    self._debug_model_registry()
                
                elif choice == '8':
                    self._reinitialize_model_registry()
                
                elif choice == '9':
                    self._reset_stacking_ensemble()
                
                elif choice == '10':
                    return
                
                else:
                    print(Fore.RED + "Invalid choice. Please enter a number between 1 and 10.")
                    
            except Exception as e:
                logger.error(f"Error in system settings: {e}")
                print(f"{Fore.RED}Error: {e}")
    
    def _reset_prediction_model(self) -> bool:
        """
        Reset the prediction model with proper backup and validation.
        
        Returns:
            bool: True if reset was successful
        """
        confirm = input("Are you sure you want to reset the prediction model? This cannot be undone. (y/n): ").lower()
        if confirm != 'y':
            print(f"{Fore.YELLOW}Reset cancelled.")
            return False
            
        try:
            if os.path.exists(MODEL_FILE):
                # Create backup before deletion
                backup_file = f"{MODEL_FILE}.backup.{int(time.time())}"
                import shutil
                shutil.copy2(MODEL_FILE, backup_file)
                os.remove(MODEL_FILE)
                print(f"{Fore.GREEN}Model reset successfully. Backup saved to {backup_file}")
                return True
            else:
                print(f"{Fore.YELLOW}No model file found to reset.")
                return False
        except Exception as e:
            logger.error(f"Error resetting model: {e}")
            print(f"{Fore.RED}Error resetting model: {e}")
            return False
    
    def _reset_betting_system(self) -> bool:
        """
        Reset the betting system with proper confirmation.
        
        Returns:
            bool: True if reset was successful
        """
        confirm = input("Are you sure you want to reset the betting system? This cannot be undone. (y/n): ").lower()
        if confirm != 'y':
            print(f"{Fore.YELLOW}Reset cancelled.")
            return False
            
        try:
            result = betting_system.full_reset()
            return result
        except Exception as e:
            logger.error(f"Error resetting betting system: {e}")
            print(f"{Fore.RED}Error resetting betting system: {e}")
            return False
    
    def _check_data_balance(self) -> None:
        """
        Check and display data balance information.
        """
        try:
            from data.data_utils import check_data_balance
            print("\n" + Fore.CYAN + "=== Checking data balance ===")
            check_data_balance()
            print(Fore.CYAN + "=================================\n")
        except Exception as e:
            logger.error(f"Error checking data balance: {e}")
            print(f"{Fore.RED}Error checking data balance: {e}")
    
    def _backup_current_model(self) -> bool:
        """
        Create backup of current model with timestamp.
        
        Returns:
            bool: True if backup was successful
        """
        try:
            if os.path.exists(MODEL_FILE):
                backup_folder = "models/pkl/backup"
                if not os.path.exists(backup_folder):
                    os.makedirs(backup_folder)
                    
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_file = os.path.join(backup_folder, f"backup_{timestamp}_{os.path.basename(MODEL_FILE)}")
                import shutil
                shutil.copy2(MODEL_FILE, backup_file)
                print(f"{Fore.GREEN}Model backup created: {backup_file}")
                return True
            else:
                print(f"{Fore.YELLOW}No model file found to backup.")
                return False
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            print(f"{Fore.RED}Error creating backup: {e}")
            
            # Try alternative backup approach
            try:
                if os.path.exists(MODEL_FILE):
                    backup_file = f"{MODEL_FILE}.backup.{int(time.time())}"
                    import shutil
                    shutil.copy2(MODEL_FILE, backup_file)
                    print(f"{Fore.GREEN}Model backup created (alternative method): {backup_file}")
                    return True
                else:
                    print(f"{Fore.YELLOW}No model file found to backup.")
                    return False
            except Exception as alt_e:
                logger.error(f"Error creating backup (alternative method): {alt_e}")
                print(f"{Fore.RED}Error creating backup (alternative method): {alt_e}")
                return False
    
    def _analyze_model_contributions(self) -> None:
        """
        Analyze and display contribution of each model to ensemble predictions.
        
        This method provides detailed analysis of model performance, importance
        in the stacking ensemble, and pattern-specific effectiveness.
        """
        import numpy as np
        from collections import Counter
        
        if not self.model_registry:
            print(f"{Fore.RED}Model registry not initialized.")
            return
            
        if not self.model_registry.models:
            print(f"{Fore.RED}No models found in registry.")
            return
        
        print(f"\n{Fore.CYAN}=== Model Contribution Analysis (Stacking) ===")
        
        # Get stacking model if available
        stacking_model = self.model_registry.models.get("stacking_ensemble")
        has_stacking = stacking_model is not None and getattr(stacking_model, 'is_trained', False)
        
        if not has_stacking:
            print(f"{Fore.YELLOW}Stacking ensemble not trained yet. Base model analysis only.")
        
        # Calculate model performance statistics
        models_data = []
        for model_id, model in self.model_registry.models.items():
            # Skip analysis of the stacking model itself
            if model_id == "stacking_ensemble":
                continue
                
            # Check if model is active
            is_active = self.model_registry.model_active.get(model_id, False)
                
            # Check training status
            is_trained = getattr(model, 'is_trained', False)
            
            # Get performance metrics
            history = self.model_registry.model_history.get(model_id, [])
            
            if history:
                recent = history[-min(20, len(history)):]  # Last 20 predictions
                correct = sum(1 for entry in recent if entry.get('correct', False))
                accuracy = (correct / len(recent)) * 100 if recent else 0
            else:
                accuracy = 0
            
            # Get confidence correlation if available
            conf_correlation = 0
            if history and len(history) >= 10:
                pairs = [(h.get('confidence', 50), h.get('correct', False)) for h in history if 'confidence' in h]
                if pairs:
                    # Calculate correlation between confidence and correctness
                    conf_values = [p[0] for p in pairs]
                    corr_values = [1 if p[1] else 0 for p in pairs]
                    
                    # Calculate mean values
                    conf_mean = sum(conf_values) / len(conf_values)
                    corr_mean = sum(corr_values) / len(corr_values)
                    
                    # Calculate correlation numerator and denominators
                    numerator = sum((c - conf_mean) * (corr - corr_mean) for c, corr in zip(conf_values, corr_values))
                    conf_var = sum((c - conf_mean) ** 2 for c in conf_values)
                    corr_var = sum((corr - corr_mean) ** 2 for corr in corr_values)
                    
                    # Calculate correlation coefficient
                    if conf_var > 0 and corr_var > 0:
                        conf_correlation = numerator / ((conf_var * corr_var) ** 0.5)
                        conf_correlation = (conf_correlation + 1) / 2  # Normalize to 0-1
            
            # Get model complexity/size info
            complexity = "Unknown"
            if hasattr(model, 'model'):
                if hasattr(model.model, 'n_estimators'):
                    complexity = f"{model.model.n_estimators} trees"
                elif hasattr(model.model, 'estimators_'):
                    complexity = f"{len(model.model.estimators_)} trees"
            elif hasattr(model, 'transitions'):
                complexity = f"{len(model.transitions)} states"
            
            # Calculate stacking importance if available
            stacking_importance = 0
            if has_stacking:
                if hasattr(stacking_model, 'meta_model') and hasattr(stacking_model.meta_model, 'coef_'):
                    # For linear meta-models
                    base_models = [m for m in self.model_registry.models if m != "stacking_ensemble"]
                    if model_id in base_models:
                        idx = base_models.index(model_id)
                        # Calculate importance as sum of absolute coefficients for this model's features
                        start_idx = idx * 3
                        end_idx = start_idx + 3
                        if start_idx < stacking_model.meta_model.coef_.shape[1]:
                            feature_importance = np.abs(stacking_model.meta_model.coef_[:, start_idx:end_idx]).sum()
                            # Normalize against total importance
                            total_importance = np.abs(stacking_model.meta_model.coef_).sum()
                            if total_importance > 0:
                                stacking_importance = feature_importance / total_importance
                            
                elif hasattr(stacking_model, 'meta_model') and hasattr(stacking_model.meta_model, 'feature_importances_'):
                    # For tree-based meta-models
                    base_models = [m for m in self.model_registry.models if m != "stacking_ensemble"]
                    if model_id in base_models:
                        idx = base_models.index(model_id)
                        start_idx = idx * 3
                        end_idx = start_idx + 3
                        if start_idx < len(stacking_model.meta_model.feature_importances_):
                            importances = stacking_model.meta_model.feature_importances_[start_idx:end_idx]
                            feature_importance = np.sum(importances)
                            stacking_importance = feature_importance / np.sum(stacking_model.meta_model.feature_importances_)
            
            # Collect all data for this model
            models_data.append({
                'model_id': model_id,
                'trained': is_trained,
                'active': is_active,
                'accuracy': accuracy,
                'correlation': conf_correlation,
                'history_len': len(history),
                'complexity': complexity,
                'stacking_importance': stacking_importance
            })
        
        # Sort by stacking importance if available, otherwise by accuracy
        if has_stacking:
            models_data.sort(key=lambda x: x['stacking_importance'], reverse=True)
        else:
            models_data.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Display table header
        if has_stacking:
            print(f"\n{Fore.YELLOW}{'Model ID':<15} {'Trained':<8} {'Active':<8} {'Accuracy':<10} {'Conf.Corr':<10} {'Stacking Imp.':<15} {'History':<10} {'Complexity':<15}")
        else:
            print(f"\n{Fore.YELLOW}{'Model ID':<15} {'Trained':<8} {'Active':<8} {'Accuracy':<10} {'Conf.Corr':<10} {'History':<10} {'Complexity':<15}")
        print("-" * (75 + (15 if has_stacking else 0)))
        
        # Display each model's data
        for data in models_data:
            # Color code based on status
            model_color = Fore.GREEN if data['trained'] and data['active'] else Fore.RED
            acc_color = Fore.RED
            if data['accuracy'] >= 40:
                acc_color = Fore.YELLOW
            if data['accuracy'] >= 50:
                acc_color = Fore.GREEN
                
            active_str = "✓" if data['active'] else "✗"
            trained_str = "✓" if data['trained'] else "✗"
            acc_str = f"{data['accuracy']:.1f}%" if data['accuracy'] > 0 else "N/A"
            corr_str = f"{data['correlation']:.2f}" if data['correlation'] > 0 else "N/A"
            
            if has_stacking:
                imp_str = f"{data['stacking_importance']*100:.2f}%"
                print(f"{model_color}{data['model_id']:<15}{Style.RESET_ALL} "
                      f"{trained_str:<8} "
                      f"{active_str:<8} "
                      f"{acc_color}{acc_str:<10}{Style.RESET_ALL} "
                      f"{corr_str:<10} "
                      f"{imp_str:<15} "
                      f"{data['history_len']:<10} "
                      f"{data['complexity']:<15}")
            else:
                print(f"{model_color}{data['model_id']:<15}{Style.RESET_ALL} "
                      f"{trained_str:<8} "
                      f"{active_str:<8} "
                      f"{acc_color}{acc_str:<10}{Style.RESET_ALL} "
                      f"{corr_str:<10} "
                      f"{data['history_len']:<10} "
                      f"{data['complexity']:<15}")
        
        # Print summary and recommendations
        print(f"\n{Fore.CYAN}Summary:")
        untrained = [d for d in models_data if not d['trained']]
        inactive = [d for d in models_data if not d['active']]
        low_acc = [d for d in models_data if d['accuracy'] < 40 and d['history_len'] >= 10]
        
        if untrained:
            print(f"{Fore.RED}⚠ {len(untrained)} untrained models found.")
        
        if inactive:
            print(f"{Fore.YELLOW}⚠ {len(inactive)} inactive models found.")
        
        if low_acc:
            print(f"{Fore.YELLOW}⚠ {len(low_acc)} models with low accuracy (<40%).")
        
        if has_stacking:
            top_model = max(models_data, key=lambda x: x['stacking_importance']) if models_data else None
            if top_model:
                print(f"Most important model in stacking: {top_model['model_id']} "
                      f"(importance: {top_model['stacking_importance']*100:.2f}%, "
                      f"accuracy: {top_model['accuracy']:.1f}%)")
        
        print(f"{Fore.CYAN}===================================")
    
    def _force_retrain_all_models(self) -> bool:
        """
        Force retraining of all models with available data.
        
        This method performs a comprehensive retraining of all models
        using all available data with detailed progress reporting.
        
        Returns:
            bool: True if retraining was successful
        """
        from data.data_utils import prepare_combined_dataset
        import time
        import os
        import numpy as np
        
        print(f"\n{Fore.CYAN}=== Force Training All Models ===")
        start_time = time.time()
        
        # Ensure required directories exist
        os.makedirs("models/registry", exist_ok=True)
        
        try:
            # First prepare the dataset
            print(f"{Fore.YELLOW}Preparing dataset from all available sources...")
            try:
                X, y = prepare_combined_dataset()
                print(f"{Fore.GREEN}✓ Dataset prepared: {len(X)} samples")
                
                # Check if dataset is too small
                if len(X) < 20:
                    print(f"{Fore.RED}⚠ Warning: Dataset is very small ({len(X)} samples).")
                    print(f"{Fore.YELLOW}Training may not be effective with limited data.")
                    
                    proceed = input("Proceed with training anyway? (y/n): ").lower()
                    if proceed != 'y':
                        print(f"{Fore.YELLOW}Training cancelled.")
                        return False
            except Exception as e:
                logger.error(f"Error preparing dataset: {e}")
                print(f"{Fore.RED}Error preparing dataset: {e}")
                print(f"{Fore.YELLOW}Will attempt training with synthetic data.")
                X, y = None, None
            
            # Train with real data if available
            if X is not None and len(X) > 0:
                success_count = 0
                
                # Train each model
                print(f"\n{Fore.CYAN}Training models sequentially:")
                for model_id, model in self.model_registry.models.items():
                    print(f"{Fore.YELLOW}Training {model_id}...")
                    
                    # Skip stacking (will be trained last)
                    if model_id == "stacking_ensemble":
                        continue
                        
                    try:
                        # Train based on model type
                        if model_id.startswith('markov'):
                            # For Markov models, prepare a full sequence
                            if isinstance(X, np.ndarray):
                                sequence = X.flatten().tolist()
                            else:
                                sequence = []
                                for _, row in X.iterrows():
                                    sequence.extend(row.values)
                            
                            sequence.extend(y.tolist() if isinstance(y, np.ndarray) else y.values)
                            model.fit(sequence)
                        elif hasattr(model, 'fit'):
                            # For machine learning models
                            model.fit(X, y)
                        
                        # Mark as trained
                        model.is_trained = True
                        success_count += 1
                        print(f"{Fore.GREEN}✓ {model_id} trained successfully")
                    except Exception as e:
                        logger.error(f"Error training {model_id}: {e}")
                        print(f"{Fore.RED}✗ Error training {model_id}: {e}")
                
                # Handle stacking ensemble separately
                if "stacking_ensemble" in self.model_registry.models:
                    print(f"{Fore.YELLOW}Training stacking ensemble...")
                    try:
                        # Generate meta-features for stacking
                        meta_features = self.model_registry._generate_meta_features_batch(X, y)
                        
                        if meta_features and len(meta_features) == len(y):
                            stacking = self.model_registry.models["stacking_ensemble"]
                            stacking.fit(meta_features, y)
                            stacking.meta_X = meta_features
                            stacking.meta_y = y
                            stacking.is_trained = True
                            success_count += 1
                            print(f"{Fore.GREEN}✓ stacking_ensemble trained successfully")
                        else:
                            print(f"{Fore.RED}✗ Failed to generate valid meta-features for stacking")
                    except Exception as e:
                        logger.error(f"Error training stacking ensemble: {e}")
                        print(f"{Fore.RED}✗ Error training stacking ensemble: {e}")
                
                # Save registry after training
                try:
                    self.model_registry._save_registry()
                    print(f"{Fore.GREEN}Registry saved successfully.")
                except Exception as e:
                    logger.error(f"Error saving registry: {e}")
                    print(f"{Fore.RED}Error saving registry: {e}")
                
                # Synchronize dimensions after training
                try:
                    self.model_registry.update_expected_feature_dimensions()
                    print(f"{Fore.GREEN}Feature dimensions updated.")
                except Exception as e:
                    logger.error(f"Error updating dimensions: {e}")
                    print(f"{Fore.RED}Error updating dimensions: {e}")
                
                # Validate registry consistency
                try:
                    self.model_registry.validate_registry_consistency()
                    print(f"{Fore.GREEN}Registry consistency validated.")
                except Exception as e:
                    logger.error(f"Error validating registry: {e}")
                    print(f"{Fore.RED}Error validating registry: {e}")
                
                # Print training summary
                total_time = time.time() - start_time
                print(f"\n{Fore.CYAN}Training Summary:")
                print(f"Total models: {len(self.model_registry.models)}")
                print(f"Successfully trained: {success_count}")
                print(f"Failed: {len(self.model_registry.models) - success_count}")
                print(f"Training time: {total_time:.2f} seconds")
                print(f"{Fore.CYAN}===================================")
                
                return success_count > 0
            else:
                print(f"{Fore.YELLOW}No training data available. Using synthetic data.")
                
                # Use synthetic data as fallback
                try:
                    # Create synthetic training data
                    X_synthetic = np.array([
                        [0, 0, 0, 0, 0],  # All banker
                        [1, 1, 1, 1, 1],  # All player
                        [0, 1, 0, 1, 0],  # Alternating B/P
                        [1, 0, 1, 0, 1],  # Alternating P/B
                        [0, 0, 1, 1, 2],  # Mixed with tie
                        [2, 0, 1, 0, 1],  # Starting with tie
                        [0, 1, 2, 0, 1],  # Mixed
                        [1, 1, 0, 0, 2],  # Mixed
                    ])
                    y_synthetic = np.array([0, 1, 0, 1, 2, 0, 1, 2])
                    
                    # Train with synthetic data
                    success_count = 0
                    for model_id, model in self.model_registry.models.items():
                        if model_id == "stacking_ensemble":
                            continue
                            
                        try:
                            if model_id.startswith('markov'):
                                sequence = X_synthetic.flatten().tolist() + y_synthetic.tolist()
                                model.fit(sequence)
                            else:
                                model.fit(X_synthetic, y_synthetic)
                                
                            model.is_trained = True
                            success_count += 1
                            print(f"{Fore.GREEN}✓ {model_id} trained with synthetic data")
                        except Exception as e:
                            logger.error(f"Error training {model_id} with synthetic data: {e}")
                            print(f"{Fore.RED}✗ Error training {model_id}: {e}")
                    
                    # Save registry after training
                    self.model_registry._save_registry()
                    print(f"{Fore.GREEN}Registry saved with {success_count} trained models.")
                    
                    return success_count > 0
                except Exception as e:
                    logger.error(f"Error in synthetic training: {e}")
                    print(f"{Fore.RED}Error in synthetic training: {e}")
                    return False
        except Exception as e:
            logger.error(f"Error in model training process: {e}")
            print(f"{Fore.RED}Error in model training process: {e}")
            print(f"{Fore.CYAN}===================================")
            return False
    
    def _debug_model_registry(self) -> Dict[str, Any]:
        """
        Debug model registry with comprehensive diagnostics.
        
        This method performs in-depth diagnostics on the model registry,
        file system checks, and consistency validation with detailed reporting.
        
        Returns:
            dict: Diagnostic results
        """
        import os
        
        print(f"\n{Fore.CYAN}=== Model Registry Debug ===")
        results = {
            'registry_dir_exists': False,
            'registry_file_exists': False,
            'model_files_count': 0,
            'models_in_memory': 0,
            'issues': []
        }
        
        # Check registry directory
        registry_dir = "models/registry"
        if not os.path.exists(registry_dir):
            print(f"{Fore.RED}Registry directory missing: {registry_dir}")
            results['issues'].append("Registry directory missing")
            os.makedirs(registry_dir, exist_ok=True)
            print(f"{Fore.GREEN}Created registry directory")
        else:
            print(f"{Fore.GREEN}Registry directory exists: {registry_dir}")
            results['registry_dir_exists'] = True
        
        # Check registry file
        registry_file = os.path.join(registry_dir, "registry.json")
        if os.path.exists(registry_file):
            print(f"{Fore.GREEN}Registry file exists: {registry_file}")
            results['registry_file_exists'] = True
            
            try:
                import json
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    model_ids = data.get('model_ids', [])
                    print(f"{Fore.YELLOW}Models in registry file: {len(model_ids)}")
                    print(f"{Fore.YELLOW}Model IDs: {model_ids}")
                    results['models_in_file'] = len(model_ids)
                    results['model_ids_in_file'] = model_ids
            except Exception as e:
                logger.error(f"Error reading registry file: {e}")
                print(f"{Fore.RED}Error reading registry file: {e}")
                results['issues'].append(f"Registry file error: {e}")
        else:
            print(f"{Fore.RED}Registry file missing: {registry_file}")
            results['issues'].append("Registry file missing")
        
        # Check model files
        print(f"\n{Fore.CYAN}Checking model files in registry:")
        if os.path.exists(registry_dir):
            model_files = [f for f in os.listdir(registry_dir) if f.endswith('.pkl')]
            print(f"{Fore.YELLOW}Found {len(model_files)} model files:")
            for file in model_files:
                print(f"  - {file}")
            results['model_files_count'] = len(model_files)
            results['model_files'] = model_files
        
        # Check registry object
        print(f"\n{Fore.CYAN}Analyzing registry object:")
        if self.model_registry is not None:
            model_count = len(self.model_registry.models) if hasattr(self.model_registry, 'models') else 0
            print(f"{Fore.GREEN}Models in registry object: {model_count}")
            
            if model_count > 0:
                for model_id in self.model_registry.models:
                    model = self.model_registry.models[model_id]
                    is_trained = getattr(model, 'is_trained', False)
                    status = "✓ Trained" if is_trained else "✗ Untrained"
                    print(f"  - {model_id}: {status}")
            
            results['models_in_memory'] = model_count
            
            # Check model activation status
            if hasattr(self.model_registry, 'model_active'):
                print(f"\n{Fore.CYAN}Model Active Status:")
                for model_id, active in self.model_registry.model_active.items():
                    status = "✓ Active" if active else "✗ Inactive"
                    print(f"  - {model_id}: {status}")
                
                # Check for inconsistencies
                missing_active = [model_id for model_id in self.model_registry.models 
                                if model_id not in self.model_registry.model_active]
                if missing_active:
                    print(f"{Fore.RED}Models missing active status: {missing_active}")
                    results['issues'].append(f"Models missing active status: {missing_active}")
                
                missing_models = [model_id for model_id in self.model_registry.model_active 
                                if model_id not in self.model_registry.models]
                if missing_models:
                    print(f"{Fore.RED}Active status for non-existent models: {missing_models}")
                    results['issues'].append(f"Active status for non-existent models: {missing_models}")
            else:
                print(f"{Fore.RED}model_active attribute missing from registry")
                results['issues'].append("model_active attribute missing")
            
            # Check stacking ensemble
            print(f"\n{Fore.CYAN}Stacking Ensemble Status:")
            if "stacking_ensemble" in self.model_registry.models:
                stacking = self.model_registry.models["stacking_ensemble"]
                is_trained = getattr(stacking, 'is_trained', False)
                meta_examples = len(getattr(stacking, 'meta_X', [])) if hasattr(stacking, 'meta_X') else 0
                
                print(f"Stacking is {'trained' if is_trained else 'untrained'}")
                print(f"Meta-examples: {meta_examples}")
                
                # Check expected feature dimensions
                if hasattr(stacking, 'expected_feature_count'):
                    print(f"Expected feature count: {stacking.expected_feature_count}")
                    
                    # Calculate actual feature count
                    active_base_models = len([m for m in self.model_registry.model_active 
                                           if self.model_registry.model_active[m] and m != "stacking_ensemble"])
                    expected_dim = (active_base_models * 3) + 4  # 3 probs per model + 4 pattern features
                    
                    if stacking.expected_feature_count != expected_dim:
                        print(f"{Fore.RED}Dimension mismatch: {stacking.expected_feature_count} vs {expected_dim}")
                        results['issues'].append(f"Stacking dimension mismatch: {stacking.expected_feature_count} vs {expected_dim}")
                
                results['stacking_trained'] = is_trained
                results['stacking_meta_examples'] = meta_examples
            else:
                print(f"{Fore.RED}Stacking ensemble not found in registry")
                results['issues'].append("Stacking ensemble missing")
        else:
            print(f"{Fore.RED}Model registry not initialized")
            results['issues'].append("Model registry not initialized")
    
    
if __name__ == "__main__":
    # Configure diagnostic logging for initialization debugging
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler("initialization_debug.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("main")
    
    # Display execution environment information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Define the test_pattern_bridge function first as a standalone function
    def test_pattern_bridge():
        """Test harness for pattern analysis bridge"""
        from models.model_registry import PatternAnalysisBridge
        bridge = PatternAnalysisBridge()
        
        # Test with sample pattern
        test_input = [0, 1, 0, 1, 0]  # Alternating pattern
        
        print("=== Pattern Analysis Bridge Test ===")
        result = bridge.analyze_pattern(test_input)
        print(f"Pattern type: {result.get('pattern_type')}")
        print(f"Pattern insight: {result.get('pattern_insight')}")
        
        # Test pattern features extraction
        features = bridge.extract_pattern_features(test_input)
        print(f"Pattern features: {features}")
        print("==================================")
    
    # Optional pattern bridge test if requested
    if "--test-bridge" in sys.argv:
        test_pattern_bridge()
    
    print("\nInitializing Baccarat Prediction System...")
    
    # Initialize dependency resolver
    dependency_resolver = DependencyResolver()
    dependency_resolver.register_dependency("directories")
    dependency_resolver.register_dependency("registry", ["directories"])
    dependency_resolver.register_dependency("base_models", ["registry"])
    dependency_resolver.register_dependency("stacking", ["base_models"])
    dependency_resolver.register_dependency("pipeline", ["registry", "base_models"])
    dependency_resolver.register_dependency("calibration", ["registry", "base_models"])
    dependency_resolver.register_dependency("betting", ["pipeline"])
    
    # Calculate initialization order
    try:
        init_order = dependency_resolver.calculate_initialization_order()
        logger.info(f"Component initialization order: {init_order}")
    except ValueError as e:
        logger.error(f"Dependency resolution error: {e}")
        init_order = ["directories", "registry", "base_models", "stacking", "pipeline", "calibration", "betting"]
        logger.info(f"Using fallback initialization order: {init_order}")
    
    # Create system manager with validation
    system_manager = SystemManager()
    
    # Perform initialization with comprehensive reporting
    start_time = time.time()
    try:
        init_success = system_manager.initialize_system()
        initialization_time = time.time() - start_time
        
        logger.info(f"System initialization completed in {initialization_time:.2f} seconds")
        logger.info(f"Initialization status: {'Success' if init_success else 'Failed'}")
        
        if hasattr(system_manager, 'initialization_report'):
            logger.info(f"Initialization report: {system_manager.initialization_report}")
            
        # Run system with appropriate mode based on initialization status
        if init_success:
            if system_manager.initialization_complete:
                # Full functionality
                system_manager.run_prediction_system()
            else:
                # Limited functionality
                print(f"{Fore.YELLOW}Running with limited functionality due to partial initialization")
                system_manager.run_prediction_system(safe_mode=True)
        else:
            print(f"{Fore.RED}System initialization failed. Cannot continue execution.")
            sys.exit(1)
    except Exception as e:
        # Catastrophic initialization failure
        logger.critical(f"Catastrophic initialization failure: {e}")
        traceback.print_exc()
        print(f"{Fore.RED}Critical system error: {e}")
        print(f"{Fore.RED}System cannot start due to fatal initialization error.")
        sys.exit(1)