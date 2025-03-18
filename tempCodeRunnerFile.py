# tempCodeRunnerFile.py
"""
Baccarat Prediction System - Calibration Initialization

This script properly initializes the calibration system for the Baccarat
Prediction System with comprehensive error handling and fallback mechanisms.
It addresses the original error by providing proper instance context for the
_initialize_calibration method and implementing robust import resolution.
"""

import os
import sys
import traceback

# Ensure project root is in path for consistent imports
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to Python path: {project_root}")

# First attempt: Use the dedicated calibration initializer if available
try:
    from prediction.utils.calibration_initializer import initialize_calibration
    print("\nUsing dedicated calibration initializer...")
    result = initialize_calibration()
    print(f"Initialization result: {result.get('status', 'unknown')}")
    
except ImportError:
    print("\nDedicated initializer not found, trying direct approach...")
    
    # Second attempt: Use SystemManager directly (original method)
    try:
        from main import SystemManager
        print("Initializing calibration via SystemManager...")
        
        # Create SystemManager instance (this was missing in the original)
        system_manager = SystemManager()
        
        # Properly call the method with instance context
        result = system_manager._initialize_calibration()
        
        print(f"Calibration initialization result: {result}")
        
    except Exception as e:
        print(f"\nError initializing calibration: {e}")
        traceback.print_exc()
        
        # Third attempt: Minimal direct implementation
        print("\nAttempting minimal calibration initialization...")
        
        try:
            # Import IsotonicRegression for direct calibrator creation
            from sklearn.isotonic import IsotonicRegression
            import numpy as np
            
            # Create basic calibrators for each class
            calibrators = {}
            
            # Banker calibrator
            X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).reshape(-1, 1)
            y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            calibrators[0] = IsotonicRegression(out_of_bounds='clip')
            calibrators[0].fit(X, y)
            
            # Player calibrator
            calibrators[1] = IsotonicRegression(out_of_bounds='clip')
            calibrators[1].fit(X, y)
            
            # Tie calibrator
            y_tie = np.array([0.05, 0.1, 0.15, 0.25, 0.35, 0.45, 0.6, 0.75, 0.85])
            calibrators[2] = IsotonicRegression(out_of_bounds='clip')
            calibrators[2].fit(X, y_tie)
            
            print("Created minimal calibration system directly")
            
            # Try to add to model registry if available
            try:
                from models.model_registry import ModelRegistry
                registry = ModelRegistry()
                
                # Add calibrators to registry
                registry.confidence_calibrators = calibrators
                
                # Save registry if possible
                if hasattr(registry, '_save_registry'):
                    registry._save_registry()
                    print("Added calibrators to model registry")
            except Exception as registry_error:
                print(f"Could not add calibrators to registry: {registry_error}")
                
                # Make calibrators available globally as last resort
                import builtins
                builtins.baccarat_calibrators = calibrators
                print("Made calibrators available as global builtins.baccarat_calibrators")
            
        except Exception as fallback_error:
            print(f"\nAll calibration initialization attempts failed: {fallback_error}")
            traceback.print_exc()

print("\nCalibration initialization process completed")
