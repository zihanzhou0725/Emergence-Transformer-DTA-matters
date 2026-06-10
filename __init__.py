"""
DTA Platform: Numerical Experiment Framework for Synchronization/Desynchronization Control
machine-learning-based oscillatorsynchronization/Desynchronization controllernumerical experiment framework

Based on paper:
- "Synchronization Transformer: Dynamical Temporal Attention Matters"

Main components:
- models: Synchronization Transformer model
- utils: network generation、Visualize、metrics
- configs: configuration files
"""

__version__ = "1.0.0"
__author__ = "DTA Platform"

from .models import SynchronizationTransformer, SyncController, DesyncController
from .utils.networks import generate_watts_strogatz, network_summary

__all__ = [
    'SynchronizationTransformer',
    'SyncController',
    'DesyncController',
    'generate_watts_strogatz',
    'network_summary',
]
