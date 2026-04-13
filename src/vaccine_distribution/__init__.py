"""Vaccine Distribution Optimization Package.

A comprehensive toolkit for optimizing vaccine distribution in public health scenarios,
including demand prediction, logistics optimization, and resource allocation.
"""

__version__ = "1.0.0"
__author__ = "kryptologyst"
__email__ = "kryptologyst@example.com"

from .data import VaccineDataGenerator, VaccineDataProcessor
from .models import (
    BaselineRegressor,
    GradientBoostingRegressor,
    NeuralNetworkRegressor,
    VaccineAllocationOptimizer,
)
from .evaluation import VaccineEvaluator, MetricsCalculator
from .visualization import VaccineVisualizer, MapVisualizer

__all__ = [
    "VaccineDataGenerator",
    "VaccineDataProcessor", 
    "BaselineRegressor",
    "GradientBoostingRegressor",
    "NeuralNetworkRegressor",
    "VaccineAllocationOptimizer",
    "VaccineEvaluator",
    "MetricsCalculator",
    "VaccineVisualizer",
    "MapVisualizer",
]
