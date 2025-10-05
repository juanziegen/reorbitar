#!/usr/bin/env python3
"""
Genetic Algorithm Satellite Route Optimizer

Standalone command-line interface for genetic algorithm-based satellite route optimization.
This script provides both batch and interactive modes for finding optimal satellite hopping routes.

Usage:
    # Batch mode with specific parameters
    python genetic_optimizer.py --deltav-budget 5.0 --timeframe 63072000 --verbose
    
    # Interactive mode for parameter tuning
    python genetic_optimizer.py --interactive
    
    # Custom GA parameters
    python genetic_optimizer.py --deltav-budget 3.0 --timeframe 31536000 \
        --population-size 200 --generations 1000 --output results.json
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.genetic_cli import main

if __name__ == "__main__":
    sys.exit(main())