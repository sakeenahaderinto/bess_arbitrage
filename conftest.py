"""Pytest configuration ensuring the project root is on the Python path."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))