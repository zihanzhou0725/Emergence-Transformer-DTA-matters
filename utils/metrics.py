"""
Metric utilities
"""

import torch
import numpy as np


def compute_order_parameter(phases):
    """
    Compute order parameter R (Order Parameter)
    R = |<e^{iθ}>| ∈ [0, 1]
    R = 1: fully synchronized
    R = 0: fully desynchronized (uniform distribution)
    
    Args:
        phases: (N,) or (batch, N) phases (radians)
    
    Returns:
        R: scalaror (batch,) order parameter
    """
    if isinstance(phases, np.ndarray):
        phases = torch.from_numpy(phases)
    
    # e^{iθ}
    complex_phases = torch.exp(1j * phases)
    
    # Compute mean value
    if phases.dim() == 1:
        mean_complex = complex_phases.mean()
        R = torch.abs(mean_complex)
    else:
        mean_complex = complex_phases.mean(dim=-1)
        R = torch.abs(mean_complex)
    
    return R


def compute_phase_coherence(phases):
    """
    Compute phase coherence (same as the order parameter)
    
    Args:
        phases: (N,) phases
    
    Returns:
        coherence: scalar coherence
    """
    return compute_order_parameter(phases)


def compute_synchronizability(adjacency, coupling_strength=None):
    """
    Estimate network synchronizability
    Based on Laplacian eigenvalues
    
    Args:
        adjacency: (N, N) adjacency matrix
        coupling_strength: Coupling strength λ (optional)
    
    Returns:
        synchronizability: synchronizability metric
        lambda_c: estimated critical coupling strength
    """
    if isinstance(adjacency, torch.Tensor):
        adjacency = adjacency.numpy()
    
    N = adjacency.shape[0]
    
    # Compute degree matrix
    degrees = adjacency.sum(axis=1)
    D = np.diag(degrees)
    
    # Laplacian matrix L = D - A
    L = D - adjacency
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(L)
    
    # synchronizability metric: ratio of smallest nonzero eigenvalue to largest eigenvalue
    # λ_2 / λ_N (ratio of Fiedler value to largest eigenvalue)
    nonzero_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(nonzero_eigenvalues) > 0:
        lambda_2 = nonzero_eigenvalues.min()
        lambda_N = eigenvalues.max()
        sync_metric = lambda_2 / lambda_N if lambda_N > 0 else 0
    else:
        sync_metric = 0
        lambda_2 = 0
    
    # Estimate critical coupling strength (based on the Kuramoto model)
    # λ_c ≈ 2D / (π * g(0)) for fully connected networks
    # use a simplified estimate here
    if coupling_strength is not None:
        lambda_c = 2 * coupling_strength / (lambda_2 + 1e-10)
    else:
        lambda_c = 2.0 / (lambda_2 + 1e-10)
    
    return sync_metric, lambda_c


def compute_phase_difference(phases):
    """
    Compute phase-difference matrix
    
    Args:
        phases: (N,) phases
    
    Returns:
        diff_matrix: (N, N) phase-difference matrix
    """
    if isinstance(phases, torch.Tensor):
        phases = phases.cpu().numpy()
    
    N = len(phases)
    diff_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            diff = np.abs(phases[i] - phases[j])
            # Account for periodicity
            diff = min(diff, 2 * np.pi - diff)
            diff_matrix[i, j] = diff
    
    return diff_matrix


def compute_synchronization_error(phases, target_phases=None):
    """
    Compute synchronization error
    
    Args:
        phases: (N,) current phases
        target_phases: (N,) target phases (None means complete synchronization)
    
    Returns:
        error: synchronization error
    """
    if target_phases is None:
        # Target is complete synchronization，use the complement of the order parameter
        R = compute_order_parameter(phases)
        return 1 - R
    else:
        # Compute differences from target phases
        if isinstance(phases, torch.Tensor):
            phases = phases.cpu().numpy()
        if isinstance(target_phases, torch.Tensor):
            target_phases = target_phases.cpu().numpy()
        
        diff = np.abs(phases - target_phases)
        diff = np.minimum(diff, 2 * np.pi - diff)
        return diff.mean()


def compute_entrainment(phases, target_phase=0.0):
    """
    Compute phase-locking degree (deviation from a target phase)
    
    Args:
        phases: (N,) phases
        target_phase: target phases
    
    Returns:
        entrainment: entrainment degree [0, 1]
    """
    if isinstance(phases, torch.Tensor):
        phases = phases.cpu().numpy()
    
    # Compute each phase difference from the target phases
    diff = np.abs(phases - target_phase)
    diff = np.minimum(diff, 2 * np.pi - diff)
    
    # Convert to cosine similarity
    similarity = np.cos(diff)
    
    return similarity.mean()


def evaluate_control_performance(order_params_history, task='sync', threshold=0.9):
    """
    Evaluate control performance
    
    Args:
        order_params_history: (T,) order-parameter history
        task: 'sync' or 'desync'
        threshold: success threshold
    
    Returns:
        metrics: dictionary of performance metrics
    """
    if isinstance(order_params_history, torch.Tensor):
        order_params_history = order_params_history.detach().cpu().numpy()
    
    metrics = {}
    
    # Final order parameter
    final_R = order_params_history[-1]
    metrics['final_order_param'] = float(final_R)
    
    # mean order parameter
    mean_R = order_params_history.mean()
    metrics['mean_order_param'] = float(mean_R)
    
    # convergence time (steps until the threshold is reached)
    if task == 'sync':
        converged = order_params_history >= threshold
        if converged.any():
            convergence_time = np.where(converged)[0][0]
        else:
            convergence_time = len(order_params_history)
        metrics['convergence_time'] = int(convergence_time)
        metrics['success'] = bool(final_R >= threshold)
    else:  # desync
        converged = order_params_history <= (1 - threshold)
        if converged.any():
            convergence_time = np.where(converged)[0][0]
        else:
            convergence_time = len(order_params_history)
        metrics['convergence_time'] = int(convergence_time)
        metrics['success'] = bool(final_R <= (1 - threshold))
    
    # standard deviation of the order parameter (stability)
    metrics['stability'] = float(order_params_history[-100:].std()) if len(order_params_history) >= 100 else float(order_params_history.std())
    
    return metrics
