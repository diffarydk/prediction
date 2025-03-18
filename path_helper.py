
# path_helper.py
# Helper module to ensure consistent import paths

import os
import sys
import importlib
import inspect

def fix_import_paths():
    '''Add project root to Python path to ensure consistent imports.'''
    # Get the calling module's filename
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename
    
    # Get the project root (assumes that models and prediction are top-level packages)
    project_root = None
    current_dir = os.path.dirname(os.path.abspath(caller_file))
    
    # Try to find project root by looking for models and prediction directories
    while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
        if (os.path.exists(os.path.join(current_dir, 'models')) and 
            os.path.exists(os.path.join(current_dir, 'prediction'))):
            project_root = current_dir
            break
        current_dir = os.path.dirname(current_dir)
    
    if project_root:
        # Add to path if not already there
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            return True
    
    return False

def ensure_imports(modules):
    '''Try to import modules and return success status.'''
    success = {}
    
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            success[module_name] = True
        except ImportError:
            success[module_name] = False
    
    return success
