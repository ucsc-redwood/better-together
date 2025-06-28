#!/usr/bin/env python3
"""
Wrapper script for schedule parsing and analysis.

This script provides a simple entry point to the modular schedule parsing system.
It imports and runs the main orchestrator from the results package.
"""

import sys
import os

# Add the results directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'results'))

# Import and run the main function
from results.parse_schedules_main import main

if __name__ == "__main__":
    sys.exit(main()) 