"""
Baccarat Prediction System - Import Path Fixer

This utility script diagnoses and repairs import path issues in the system,
creates missing files for backward compatibility, and enables graceful degradation
when components are not fully implemented yet.

Usage:
    python fix_imports.py

This script should be run from the project root directory.
"""

import os
import sys
import importlib
import inspect
import logging
import traceback
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("import_fix.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Directory structure that should exist
REQUIRED_DIRS = [
    "models",
    "prediction",
    "prediction/utils",
    "prediction/components",
]

# Essential files that should exist with minimal content
ESSENTIAL_FILES = {
    "prediction/__init__.py": True,  # Already fixed
    "prediction/utils/__init__.py": False,
    "prediction/utils/calibration.py": True,  # Already fixed
    "prediction/utils/diagnostics.py": True,  # Already fixed
    "prediction/utils/validation.py": True,  # Already exists
    "prediction/components/__init__.py": False,
    "models/__init__.py": False,
}

# Default content for missing files
DEFAULT_CONTENT = {
    "prediction/utils/__init__.py": """
# prediction/utils/__init__.py
# Utility functions for the prediction system

from .calibration import calibrate_confidence
from .diagnostics import log_prediction_error, get_error_context
from .validation import validate_input, format_input

# Export all public symbols
__all__ = [
    'calibrate_confidence',
    'log_prediction_error',
    'get_error_context',
    'validate_input',
    'format_input'
]
""",
    "prediction/components/__init__.py": """
# prediction/components/__init__.py
# This module will contain component implementations in future updates.

# Placeholder to ensure proper imports
__all__ = []

# This package will eventually contain:
# - PatternAnalyzer
# - ConfidenceCalculator
# - FallbackManager
""",
    "models/__init__.py": """
# models/__init__.py
# Base models for Baccarat prediction system

# Import model registry when module is available
try:
    from .model_registry import ModelRegistry
    __all__ = ['ModelRegistry']
except ImportError:
    # Create placeholder for backward compatibility
    class ModelRegistry:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ModelRegistry not fully implemented yet")
            
    __all__ = ['ModelRegistry']
"""
}


def check_and_create_dirs():
    """Check and create required directories."""
    logger.info("Checking and creating required directories...")
    
    for dir_path in REQUIRED_DIRS:
        if not os.path.exists(dir_path):
            logger.info(f"Creating directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        else:
            logger.info(f"Directory exists: {dir_path}")


def check_and_create_files():
    """Check and create essential files with minimal content."""
    logger.info("Checking and creating essential files...")
    
    for file_path, already_fixed in ESSENTIAL_FILES.items():
        if not os.path.exists(file_path):
            logger.info(f"Creating file: {file_path}")
            
            # Make sure parent directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write default content if available
            if file_path in DEFAULT_CONTENT:
                with open(file_path, "w") as f:
                    f.write(DEFAULT_CONTENT[file_path])
            else:
                # Create empty file
                with open(file_path, "w") as f:
                    f.write(f"# {file_path}\n# Created by import fixer\n")
        else:
            if already_fixed:
                logger.info(f"File exists and already fixed: {file_path}")
            else:
                logger.info(f"File exists: {file_path}")
                
                # Check if file is empty or minimal
                if os.path.getsize(file_path) < 10:
                    logger.info(f"File is very small, adding default content: {file_path}")
                    with open(file_path, "w") as f:
                        if file_path in DEFAULT_CONTENT:
                            f.write(DEFAULT_CONTENT[file_path])


def check_python_path():
    """Check and modify Python path if needed."""
    logger.info("Checking Python path...")
    
    # Get the project root directory
    project_root = os.path.abspath(".")
    
    # Check if project root is in sys.path
    if project_root not in sys.path:
        logger.info(f"Adding project root to Python path: {project_root}")
        sys.path.insert(0, project_root)
        
    # Check if we can import key modules
    try:
        import models
        logger.info("Successfully imported models package")
    except ImportError as e:
        logger.error(f"Failed to import models package: {e}")
        
    try:
        import prediction
        logger.info("Successfully imported prediction package")
    except ImportError as e:
        logger.error(f"Failed to import prediction package: {e}")


def create_path_helper():
    """Create a path helper module for consistent imports."""
    logger.info("Creating path helper...")
    
    path_helper_content = """
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
"""
    
    with open("path_helper.py", "w") as f:
        f.write(path_helper_content)
        
    # Create a simple file that can be used in main.py
    main_path_fix_content = """
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
"""
    
    with open("path_fix.py", "w") as f:
        f.write(main_path_fix_content)


def main():
    """Main function to fix import issues."""
    logger.info("Starting import path fixer...")
    
    try:
        # Step 1: Create required directories
        check_and_create_dirs()
        
        # Step 2: Create essential files
        check_and_create_files()
        
        # Step 3: Check Python path
        check_python_path()
        
        # Step 4: Create path helper
        create_path_helper()
        
        logger.info("Import path fixing completed successfully")
        
    except Exception as e:
        logger.error(f"Error fixing import paths: {e}")
        logger.error(traceback.format_exc())
        
    logger.info("Finished")


if __name__ == "__main__":
    main()