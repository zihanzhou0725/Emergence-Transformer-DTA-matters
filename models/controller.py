"""
Synchronization/desynchronization controller implementation
Learn attention parameters for the control objective
"""

import torch
import torch.nn as nn
import numpy as np


class SyncController(nn.Module):
    """
    Synchronization controller
    Objective: align all oscillator phases (order parameter R -> 1)
    """
    
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        
    def compute_loss(self, order_params, target_R=1.0, mode='final', 
                     convergence_weight=0.5, stability_weight=0.1):
        """
        Compute synchronization loss（improved version）
        
        Args:
            order_params: (T,) order-parameter sequence
            target_R: target order parameter (default: 1.0)
            mode: 'final', 'mean', 'traj', 'convergence'
            convergence_weight: convergence-speed weight
            stability_weight: stability weight
        Returns:
            loss: scalar
        """
        if mode == 'final':
            loss = (target_R - order_params[-1]) ** 2
        elif mode == 'mean':
            loss = (target_R - order_params.mean()) ** 2
        elif mode == 'traj':
            weights = torch.linspace(0.5, 1.0, len(order_params), device=order_params.device)
            loss = (weights * (target_R - order_params) ** 2).mean()
        elif mode == 'convergence':
            # combined objective：final state + convergence speed + stability
            final_loss = (target_R - order_params[-1]) ** 2
            
            # Convergence-speed loss encourages a high order parameter early.
            T = len(order_params)
            time_weights = torch.exp(-torch.linspace(0, 3, T, device=order_params.device))
            convergence_loss = (time_weights * (target_R - order_params)).sum()
            
            # Stability loss penalizes decreases in the order parameter.
            diff = torch.diff(order_params)
            stability_loss = torch.relu(-diff).mean()
            
            loss = final_loss + convergence_weight * convergence_loss + stability_weight * stability_loss
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return loss
    
    def forward(self, initial_phases, n_steps, **kwargs):
        """Forward pass"""
        final_phases, order_params = self.model(initial_phases, n_steps)
        loss = self.compute_loss(order_params, **kwargs)
        return loss, order_params, final_phases


class DesyncController(nn.Module):
    """
    Desynchronization controller
    target: spread oscillator phases apart (order parameter R -> 0)
    """
    
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        
    def compute_loss(self, order_params, target_R=0.0, mode='final', 
                    diversity_weight=0.1, final_phases=None):
        """
        Compute desynchronization loss
        
        Args:
            order_params: (T,) order-parameter sequence
            target_R: target order parameter (default: 0.0)
            mode: 'final' - only optimize the final state, 'mean' - optimize the mean state
            diversity_weight: phase-diversity penalty weight
            final_phases: final phases (used to compute diversity)
        Returns:
            loss: scalar
        """
        if mode == 'final':
            loss = (order_params[-1] - target_R) ** 2
        elif mode == 'mean':
            loss = (order_params.mean() - target_R) ** 2
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Add phase-diversity penalty
        if diversity_weight > 0 and final_phases is not None:
            # compute phase differences，encourage uniform distribution
            N = len(final_phases)
            # sort phases and compute spacings
            sorted_phases = torch.sort(final_phases)[0]
            phase_diffs = torch.diff(sorted_phases, append=sorted_phases[:1] + 2*np.pi)
            # ideally，phases should be uniformly distributed，with spacing 2π/N
            ideal_diff = 2 * np.pi / N
            diversity_penalty = ((phase_diffs - ideal_diff) ** 2).mean()
            loss = loss + diversity_weight * diversity_penalty
        
        return loss
    
    def forward(self, initial_phases, n_steps, **kwargs):
        """Forward pass"""
        final_phases, order_params = self.model(initial_phases, n_steps)
        loss = self.compute_loss(order_params, final_phases=final_phases, **kwargs)
        return loss, order_params, final_phases


class HybridController(nn.Module):
    """
    Hybrid controller - can switch between synchronization and desynchronization
    """
    
    def __init__(self, base_model, task='sync'):
        super().__init__()
        self.model = base_model
        self.task = task
        
    def set_task(self, task):
        """Set task type"""
        assert task in ['sync', 'desync']
        self.task = task
        
    def compute_loss(self, order_params, target_R=None, final_phases=None):
        """
        Compute loss
        
        Args:
            order_params: (T,) order-parameter sequence
            target_R: target order parameter (None uses the default value)
            final_phases: final phases
        """
        if self.task == 'sync':
            target = 1.0 if target_R is None else target_R
            loss = (target - order_params[-1]) ** 2
        else:  # desync
            target = 0.0 if target_R is None else target_R
            loss = (order_params[-1] - target) ** 2
            
            # For desynchronization，add diversity penalty
            if final_phases is not None:
                N = len(final_phases)
                sorted_phases = torch.sort(final_phases % (2*np.pi))[0]
                phase_diffs = torch.diff(sorted_phases, append=sorted_phases[:1] + 2*np.pi)
                ideal_diff = 2 * np.pi / N
                diversity_penalty = ((phase_diffs - ideal_diff) ** 2).mean()
                loss = loss + 0.1 * diversity_penalty
        
        return loss
    
    def forward(self, initial_phases, n_steps, **kwargs):
        """Forward pass"""
        final_phases, order_params = self.model(initial_phases, n_steps)
        loss = self.compute_loss(order_params, final_phases=final_phases, **kwargs)
        return loss, order_params, final_phases
