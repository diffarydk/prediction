
# Import path fixer for main.py
# Add this at the top of main.py:
# import sys, os
# sys.path.insert(0, os.path.abspath('.'))
# from path_fix import fix_paths
# fix_paths()

def fix_paths():
    '''Add project root to Python path to ensure consistent imports.'''
    import sys
    import os
    
    # Get project root (current directory when run from main.py)
    project_root = os.path.abspath('.')
    
    # Add to path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        return True
    
    return False

# Add compatibility import helpers
def ensure_imports():
    '''Ensure critical imports are available, create placeholders if needed.'''
    try:
        from prediction.utils.calibration import calibrate_confidence
    except ImportError:
        # Create temporary module with necessary function
        import sys
        import types
        
        # Create module
        calibration_module = types.ModuleType('calibration')
        calibration_module.calibrate_confidence = lambda x: x
        
        # Add to sys.modules
        sys.modules['prediction.utils.calibration'] = calibration_module
        
    # More imports could be handled here
    
    return True
