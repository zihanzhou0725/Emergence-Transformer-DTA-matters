"""
Utility functions for DTA platform
"""

from .networks import (
    generate_watts_strogatz,
    generate_fully_connected,
    generate_ring_network,
    compute_average_shortest_path_length
)

from .visualization import (
    plot_phase_evolution,
    plot_order_parameter,
    plot_phase_distribution,
    plot_training_curves
)

from .metrics import (
    compute_order_parameter,
    compute_synchronizability,
    compute_phase_coherence
)

__all__ = [
    'generate_watts_strogatz',
    'generate_fully_connected',
    'generate_ring_network',
    'compute_average_shortest_path_length',
    'plot_phase_evolution',
    'plot_order_parameter',
    'plot_phase_distribution',
    'plot_training_curves',
    'compute_order_parameter',
    'compute_synchronizability',
    'compute_phase_coherence',
]
