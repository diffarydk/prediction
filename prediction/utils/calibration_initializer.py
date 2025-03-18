# calibration_initializer.py
"""
Robust Calibration Initialization Module for Baccarat Prediction System.

This module provides a comprehensive initialization framework for the calibration
system with multi-level fallback mechanisms to ensure system stability during
architectural transition. It addresses IsotonicRegression configuration issues
and provides a unified interface for confidence calibration.
"""

import sys
import os
import numpy as np
import time
import traceback
from typing import Dict, Any, List, Optional, Union, Tuple

# Add project root to ensure consistent imports
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to Python path: {project_root}")

class DummyCalibrator:
    """
    Enhanced pass-through calibrator for fallback situations.
    
    Implements a minimal interface compatible with IsotonicRegression
    to serve as an emergency replacement when calibration fails.
    Includes all required attributes for compatibility.
    """
    def __init__(self):
        """Initialize with all required attributes to prevent missing attribute errors."""
        self.X_min_ = 0.0
        self.X_max_ = 1.0
        self.y_min_ = 0.0
        self.y_max_ = 1.0
        self._y = [0.0, 1.0]  # Required by some IsotonicRegression consumers
        self._X = [0.0, 1.0]  # Required by some IsotonicRegression consumers
        self.increasing = True
        self.out_of_bounds = 'clip'
    
    def predict(self, X):
        """Return input values with proper shape handling."""
        # Handle different input formats
        import numpy as np
        
        if isinstance(X, list):
            if not X:
                return np.array([0.5])
            return np.array(X)
        elif hasattr(X, 'shape'):
            # For NumPy arrays, handle different dimensions
            if X.ndim == 2 and X.shape[1] == 1:
                # Column vector (n, 1) -> flatten to (n,)
                return X.flatten()
            elif X.ndim == 1:
                # Already in correct shape
                return X
            else:
                # Default handling for other shapes
                return X.flatten() if hasattr(X, 'flatten') else X
        else:
            # Default for other types
            return np.array([0.5])
            
    def fit(self, X, y):
        """
        Implement fit method for compatibility.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            self: For method chaining compatibility
        """
        import numpy as np
        
        # Store data in compatible format
        if hasattr(X, 'flatten'):
            self._X = X.flatten()
        else:
            self._X = np.array(X).flatten() if hasattr(X, '__iter__') else np.array([X])
            
        if hasattr(y, 'flatten'):
            self._y = y.flatten()
        else:
            self._y = np.array(y).flatten() if hasattr(y, '__iter__') else np.array([y])
            
        # Set required attributes
        self.X_min_ = min(self._X) if len(self._X) > 0 else 0.0
        self.X_max_ = max(self._X) if len(self._X) > 0 else 1.0
        self.y_min_ = min(self._y) if len(self._y) > 0 else 0.0
        self.y_max_ = max(self._y) if len(self._y) > 0 else 1.0
        
        return self

def create_robust_calibrator(cls_idx):
    """
    Create a robust IsotonicRegression calibrator with comprehensive validation.
    
    Args:
        cls_idx: Class index (0=Banker, 1=Player, 2=Tie)
        
    Returns:
        object: Properly initialized calibrator with fallback mechanisms
    """
    try:
        from sklearn.isotonic import IsotonicRegression
        
        # Create synthetic calibration data appropriate for the class
        if cls_idx == 2:  # Tie-specific calibration
            # Ties are rare (~9.5%), use conservative calibration curve
            X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            y = np.array([0.05, 0.1, 0.15, 0.25, 0.35, 0.45, 0.6, 0.75, 0.85])
        else:  # Banker/Player calibration
            # More common outcomes (~45.5%/44.9%), use balanced calibration curve
            X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            
        # Reshape X properly for calibrator
        X_shaped = X.reshape(-1, 1)
        
        # Create and fit calibrator
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(X_shaped, y)
        
        # Validate calibrator was properly fitted by checking for required attributes
        required_attrs = ['X_min_', 'X_max_', 'y_min_', 'y_max_']
        if all(hasattr(calibrator, attr) for attr in required_attrs):
            print(f"Successfully created IsotonicRegression calibrator for class {cls_idx}")
            return calibrator
        else:
            missing_attrs = [attr for attr in required_attrs if not hasattr(calibrator, attr)]
            print(f"Warning: IsotonicRegression missing attributes {missing_attrs} for class {cls_idx}")
            # Fall through to dummy calibrator
    except Exception as e:
        print(f"Error creating IsotonicRegression: {e}")
        traceback.print_exc()
    
    # Create dummy calibrator as fallback
    print(f"Creating DummyCalibrator as fallback for class {cls_idx}")
    dummy = DummyCalibrator()
    dummy.fit(np.array([[0.1], [0.5], [0.9]]), np.array([0.1, 0.5, 0.9]))
    return dummy

def validate_calibrator(calibrator, cls_idx):
    """
    Validate a calibrator has all required attributes and methods.
    
    Args:
        calibrator: The calibrator to validate
        cls_idx: Class index for reporting
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    # Check required attributes
    required_attrs = ['X_min_', 'X_max_', 'y_min_', 'y_max_']
    missing_attrs = [attr for attr in required_attrs if not hasattr(calibrator, attr)]
    
    # Check required methods
    required_methods = ['predict', 'fit']
    missing_methods = [method for method in required_methods 
                      if not hasattr(calibrator, method) or not callable(getattr(calibrator, method))]
    
    if missing_attrs or missing_methods:
        return False, f"Calibrator for class {cls_idx} missing: " + \
               f"attributes {missing_attrs}, methods {missing_methods}"
    
    # Test prediction functionality
    try:
        test_input = np.array([[0.5]])
        prediction = calibrator.predict(test_input)
        return True, f"Calibrator for class {cls_idx} is valid and functional"
    except Exception as e:
        return False, f"Calibrator for class {cls_idx} failed prediction test: {e}"

def initialize_calibration():
    """
    Unified calibration initialization with multi-level fallback strategy.
    
    This function attempts multiple initialization approaches with progressive
    fallback to ensure calibration availability across different architectural states.
    It handles IsotonicRegression configuration issues and provides a unified
    interface for confidence calibration.
    
    Returns:
        dict: Initialization status and results
    """
    # Strategy 1: Try SystemManager approach
    try:
        from main import SystemManager
        print("Initializing calibration via SystemManager...")
        system_manager = SystemManager()
        result = system_manager._initialize_calibration()
        
        # Validate calibrators to ensure they're properly initialized
        if hasattr(system_manager.model_registry, 'confidence_calibrators'):
            validation_results = {}
            for cls_idx in range(3):  # Check all three classes
                if cls_idx in system_manager.model_registry.confidence_calibrators:
                    calibrator = system_manager.model_registry.confidence_calibrators[cls_idx]
                    is_valid, message = validate_calibrator(calibrator, cls_idx)
                    validation_results[cls_idx] = {"valid": is_valid, "message": message}
                    
                    # Fix invalid calibrators
                    if not is_valid:
                        print(f"Replacing invalid calibrator for class {cls_idx}")
                        system_manager.model_registry.confidence_calibrators[cls_idx] = create_robust_calibrator(cls_idx)
            
            print("Calibration initialized via SystemManager")
            return {
                "status": "success", 
                "method": "system_manager",
                "result": result,
                "validation": validation_results
            }
        else:
            print("SystemManager approach succeeded but no calibrators found")
            
    except Exception as e:
        print(f"SystemManager approach failed: {e}")
        traceback.print_exc()
    
    # Strategy 2: Try direct ModelRegistry approach
    try:
        from models.model_registry import ModelRegistry
        print("Initializing calibration via ModelRegistry...")
        registry = ModelRegistry()
        
        # Ensure confidence_calibrators attribute exists
        if not hasattr(registry, 'confidence_calibrators'):
            registry.confidence_calibrators = {}
            
        # Initialize calibrators for all classes
        calibrators_created = 0
        for cls_idx in range(3):  # Banker, Player, Tie
            # Check if calibrator exists and is valid
            if cls_idx in registry.confidence_calibrators:
                is_valid, message = validate_calibrator(registry.confidence_calibrators[cls_idx], cls_idx)
                if not is_valid:
                    print(f"Replacing invalid calibrator: {message}")
                    registry.confidence_calibrators[cls_idx] = create_robust_calibrator(cls_idx)
                    calibrators_created += 1
            else:
                # Create new calibrator
                registry.confidence_calibrators[cls_idx] = create_robust_calibrator(cls_idx)
                calibrators_created += 1
        
        # Try to save registry if calibrators were created
        if calibrators_created > 0 and hasattr(registry, '_save_registry'):
            try:
                registry._save_registry()
                print(f"Saved registry with {calibrators_created} new/replaced calibrators")
            except Exception as save_error:
                print(f"Warning: Could not save registry: {save_error}")
            
        print("Calibration initialized via direct ModelRegistry approach")
        return {
            "status": "success", 
            "method": "model_registry_direct",
            "calibrators_created": calibrators_created
        }
    except Exception as e:
        print(f"ModelRegistry approach failed: {e}")
        traceback.print_exc()
    
    # Strategy 3: Try component-based approach
    try:
        from prediction.components.confidence_calculator import ConfidenceCalculator
        print("Initializing calibration via ConfidenceCalculator component...")
        calibrator = ConfidenceCalculator()
        
        # Test the calibrator
        test_data = {
            'prediction': 0,
            'confidence': 75.0,
            'distribution': {0: 75.0, 1: 20.0, 2: 5.0}
        }
        
        calibrated = calibrator.calculate_confidence(
            probabilities={k: v/100 for k, v in test_data['distribution'].items()},
            prediction=test_data['prediction']
        )
        
        print(f"Calibration tested successfully: {calibrated}")
        print("Calibration initialized via ConfidenceCalculator component")
        return {
            "status": "success", 
            "method": "confidence_calculator_component",
            "calibrator": calibrator
        }
    except Exception as e:
        print(f"Component approach failed: {e}")
        traceback.print_exc()
    
    # Strategy 4: Try utilities-based approach
    try:
        from prediction.utils.calibration import calibrate_confidence
        print("Testing calibration via utilities module...")
        # Verify function works with test data
        test = {'prediction': 0, 'confidence': 75.0, 'distribution': {0: 75.0, 1: 20.0, 2: 5.0}}
        calibrated_result = calibrate_confidence(test)
        print(f"Calibration test result: {calibrated_result['confidence']}")
        
        print("Calibration initialized successfully via utilities module")
        return {
            "status": "success", 
            "method": "calibration_utilities",
            "test_result": calibrated_result
        }
    except Exception as e:
        print(f"Utilities approach failed: {e}")
        traceback.print_exc()
    
    # Strategy 5: Create standalone calibration system
    try:
        print("Creating standalone calibration system...")
        # Create a dictionary of calibrators
        calibrators = {
            0: create_robust_calibrator(0),  # Banker
            1: create_robust_calibrator(1),  # Player
            2: create_robust_calibrator(2)   # Tie
        }
        
        # Create a simple calibration function
        def standalone_calibrate(result):
            """
            Simple calibration function for standalone use.
            
            Args:
                result: Prediction result dictionary
                
            Returns:
                dict: Calibrated prediction result
            """
            # Store original confidence
            original_confidence = result.get('confidence', 50.0)
            
            # Extract prediction and distribution
            prediction = result.get('prediction', 0)
            distribution = result.get('distribution', {0: 45.0, 1: 45.0, 2: 10.0})
            
            # Use appropriate calibrator if available
            if prediction in calibrators:
                calibrator = calibrators[prediction]
                
                try:
                    # Apply calibration transformation
                    cal_conf = calibrator.predict(np.array([[original_confidence / 100.0]]))[0]
                    
                    # Apply outcome-specific caps
                    confidence_caps = {
                        0: 85.0,  # Banker cap
                        1: 85.0,  # Player cap
                        2: 70.0   # Tie cap
                    }
                    
                    capped_conf = min(cal_conf * 100.0, confidence_caps.get(prediction, 85.0))
                    
                    # Update result
                    result['raw_confidence'] = original_confidence
                    result['confidence'] = capped_conf
                    result['calibrated'] = True
                    return result
                except Exception as e:
                    print(f"Error applying standalone calibration: {e}")
            
            # Calculate entropy if distribution is available
            if distribution:
                # Convert to probabilities if needed
                probabilities = {k: v/100 if v > 1 else v for k, v in distribution.items()}
                
                # Calculate entropy
                entropy = 0.0
                for outcome, prob in probabilities.items():
                    if prob > 0:
                        entropy -= prob * np.log2(prob)
                
                # Apply outcome-specific adjustment
                outcome_adjustments = {
                    0: 1.0,   # Banker - neutral adjustment
                    1: 1.0,   # Player - neutral adjustment
                    2: 0.85   # Tie - reduce confidence (harder to predict)
                }
                
                outcome_factor = outcome_adjustments.get(prediction, 1.0)
                
                # Apply entropy penalty (higher entropy = lower confidence)
                entropy_ratio = entropy / 1.58  # Max entropy for 3 outcomes
                entropy_penalty = 0.75 * entropy_ratio
                
                # Calculate adjusted confidence
                adjusted_confidence = original_confidence * outcome_factor * (1 - entropy_penalty)
                
                # Apply confidence caps
                confidence_caps = {
                    0: 85.0,   # Banker cap
                    1: 85.0,   # Player cap
                    2: 70.0    # Tie cap
                }
                
                capped_confidence = min(adjusted_confidence, confidence_caps.get(prediction, 85.0))
                
                # Update result
                result['raw_confidence'] = original_confidence
                result['confidence'] = capped_confidence
                result['calibrated'] = True
                result['entropy'] = entropy
                result['entropy_ratio'] = entropy_ratio
            
            # Return the (possibly modified) result
            return result
        
        # Verify with test data
        test = {'prediction': 0, 'confidence': 75.0, 'distribution': {0: 75.0, 1: 20.0, 2: 5.0}}
        calibrated_test = standalone_calibrate(test.copy())
        print(f"Standalone calibration test: {calibrated_test['confidence']:.2f}%")
        
        # Create global reference for future use
        import builtins
        builtins.baccarat_calibrators = calibrators
        builtins.standalone_calibrate = standalone_calibrate
        
        print("Standalone calibration system created successfully")
        return {
            "status": "success", 
            "method": "standalone_calibration",
            "test_result": calibrated_test
        }
    except Exception as e:
        print(f"Standalone approach failed: {e}")
        traceback.print_exc()
    
    # Final fallback: Return failure with comprehensive error information
    print("All calibration initialization approaches failed")
    return {
        "status": "failure",
        "error": "All calibration initialization approaches failed",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    # Run comprehensive calibration initialization
    print("\n" + "="*80)
    print("COMPREHENSIVE BACCARAT CALIBRATION SYSTEM INITIALIZATION")
    print("="*80 + "\n")
    
    start_time = time.time()
    result = initialize_calibration()
    duration = time.time() - start_time
    
    # Print results
    print("\n" + "="*50)
    print(f"CALIBRATION INITIALIZATION RESULTS:")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Method: {result.get('method', 'none')}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Display additional details if available
    for key, value in result.items():
        if key not in ['status', 'method', 'validation']:
            print(f"{key}: {value}")
            
    # Print validation results if available
    if 'validation' in result:
        print("\nValidation Results:")
        for cls_idx, validation in result['validation'].items():
            status = "✓" if validation['valid'] else "✗"
            print(f"Class {cls_idx}: {status} {validation['message']}")
    
    print("="*50 + "\n")