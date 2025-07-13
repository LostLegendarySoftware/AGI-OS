#!/usr/bin/env python3
"""
Four-Team AGI Framework Launcher
Simple launcher script for the framework
"""

import sys
import os

# Add the framework directory to Python path
framework_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, framework_dir)

try:
    from framework_orchestrator import main

    if __name__ == "__main__":
        success = main()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"Failed to import framework: {e}")
    print("Please ensure all dependencies are installed and the framework is properly configured.")
    sys.exit(1)
except Exception as e:
    print(f"Framework execution failed: {e}")
    sys.exit(1)
