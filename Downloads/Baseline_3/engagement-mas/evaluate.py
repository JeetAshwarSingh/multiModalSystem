#!/usr/bin/env python3
"""
Convenience entry point for evaluation.
Delegates directly to evaluation.evaluate.main().
"""
import sys
import os
import pathlib

# Ensure engagement-mas directory is on sys.path
BASE_DIR = pathlib.Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from evaluation.evaluate import main

if __name__ == "__main__":
    main()
