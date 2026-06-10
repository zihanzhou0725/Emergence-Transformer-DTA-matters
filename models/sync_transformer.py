"""
Core Synchronization Transformer model implementation
Based on paper equations (S30)-(S36) for discrete-time dynamics
history window is time-varying：accumulates all historical information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SynchronizationTransformer(nn.Module):
    """
    Synchronization Transformer for networked oscillators with Dynamical Temporal Attention (DTA)
    
    Core equations:
    - Phase update (S30): θ_T+1 = θ_T + ω + λ*Im[I_T * exp(-iθ_T)] + ξ_T
    - Total information (S31): I_T = (1-α) * spatial_coupling + α * attention_coupling
    - attention mechanism (S33-S36): uses Transformer-style attention
    
    History window: time-varying，accumulates all historical information (t=0,1,...,T)
    """
    
    def __init__(self, 
                 n_oscillators: int,
                 d_model: int = 64,
                 spatial_network: torch.Tensor = None,
                 attention_network: torch.Tensor = None,
                 natural_frequencies: torch.Tensor = None,
                 coupling_strength: float = 1.5,
                 noise_strength: float = 0.1,
                 learnable_alpha: bool = True,
                 alpha_init: float = 0.5,
                 learnable_w_qk: bool = True,
                 w_v_identity: bool = True):
        """
        Args:
            n_oscillators: Number of oscillators N
            d_model: feature dimension d
            spatial_network: spatial coupling network A (N x N adjacency matrix)
            attention_network: attention network Â (N x N adjacency matrix)
            natural_frequencies: natural frequencies ω (N,)
            coupling_strength: Coupling strength λ
            noise_strength: noise strength D
            learnable_alpha: whether to learn the mixture coefficient α
            alpha_init: α initial value
            learnable_w_qk: whether to learn W^Q and W^K
            w_v_identity: W^V whether to use identity matrix
        """
        super().__init__()
        
        self.N = n_oscillators
        self.d = d_model
        self.lambda_coupling = coupling_strength
        self.D = noise_strength
        
        # Register network topology (fixed parameter)
        if spatial_network is None:
            spatial_network = torch.eye(n_oscillators)
        self.register_buffer('A', spatial_network.float())
        
        if attention_network is None:
            attention_network = torch.eye(n_oscillators)
        self.register_buffer('A_hat', attention_network.float())
        
        # Compute degrees
        self.d_i = self.A.sum(dim=1, keepdim=True)  # (N, 1)
        self.d_hat_i = self.A_hat.sum(dim=1, keepdim=True)  # (N, 1)
        
        # Register natural frequencies (fixed parameter)
        if natural_frequencies is None:
            natural_frequencies = torch.zeros(n_oscillators)
        self.register_buffer('omega', natural_frequencies.float())
        
        # Learnable parameters: mixture coefficient α
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        else:
            self.register_buffer('alpha', torch.tensor(alpha_init))
        
        # Learnable parameters: attention weight matrices W^Q, W^K
        if learnable_w_qk:
            # W^Q, W^K: (N, d)
            self.W_Q = nn.Parameter(torch.randn(n_oscillators, d_model) * 0.01)
            self.W_K = nn.Parameter(torch.randn(n_oscillators, d_model) * 0.01)
        else:
            self.register_buffer('W_Q', torch.randn(n_oscillators, d_model) * 0.01)
            self.register_buffer('W_K', torch.randn(n_oscillators, d_model) * 0.01)
        
        # W^V: usually set to identity (N, N)
        if w_v_identity:
            self.register_buffer('W_V', torch.eye(n_oscillators))
        else:
            self.W_V = nn.Parameter(torch.eye(n_oscillators))
        
        # Used to store phase history - uses a list for the time-varying window（accumulates all history）
        self.phase_history_list = []
        self.current_time = 0
        
    def reset_history(self):
        """Reset phase-history cache"""
        self.phase_history_list = []
        self.current_time = 0
        
    def update_history(self, phases_complex):
        """
        Update phase-history cache - accumulates all historical information（core idea from the paper）
        Args:
            phases_complex: (N,) complex phase e^{iθ}
        """
        # Use detach() to avoid retaining old computation graphs
        self.phase_history_list.append(phases_complex.detach().clone())
        self.current_time += 1
        
    def compute_attention(self):
        """
        Compute attention output M_T (equations S33-S36)
        Use all historical information（time-varying window）
        
        Returns:
            M_T: (N,) attention output (complex)
        """
        T = len(self.phase_history_list)
        
        if T == 0:
            # If there is no history，return zeros
            return torch.zeros(self.N, dtype=torch.cfloat, device=self.A.device)
        
        if T == 1:
            # With only one history item，return it directly
            return self.phase_history_list[0]
        
        # Stack all history into a matrix Theta_T: (T, N) complex
        Theta_T = torch.stack(self.phase_history_list, dim=0)  # (T, N)
        
        # Linear projection (S33)
        # Q, K: (T, d) complex; V: (T, N) complex
        W_Q_complex = self.W_Q.to(Theta_T.dtype)
        W_K_complex = self.W_K.to(Theta_T.dtype)
        W_V_complex = self.W_V.to(Theta_T.dtype)
        
        Q = Theta_T @ W_Q_complex  # (T, d)
        K = Theta_T @ W_K_complex  # (T, d)
        V = Theta_T @ W_V_complex  # (T, N)
        
        # Compute attention matrix (S34)
        # |Q K^T|: (T, d) @ (d, T) = (T, T)
        scores = torch.abs(Q @ K.T) / np.sqrt(self.d)  # (T, T)
        
        # Softmax row-wise (S34)
        C = F.softmax(scores, dim=-1)  # (T, T)
        
        # Use the last row as the current attention weights (S36)
        C_T = C[-1]  # (T,)
        
        # Compute attention output (S35-S36)
        # M_T = sum_k C_Tk * v_k
        M_T = torch.sum(C_T.unsqueeze(-1) * V, dim=0)  # (N,) complex
        
        return M_T
    
    def compute_spatial_coupling(self, phases_complex):
        """
        Compute the traditional spatial coupling term
        Args:
            phases_complex: (N,) complex phase
        Returns:
            spatial_term: (N,) spatial coupling information
        """
        # Move A and d_i to the same device and dtype as phases_complex
        A_complex = self.A.to(phases_complex.dtype).to(phases_complex.device)
        d_i = self.d_i.to(phases_complex.device)
        spatial_term = (A_complex @ phases_complex) / d_i.squeeze(-1)
        return spatial_term
    
    def compute_attention_coupling(self, M_T):
        """
        Compute the attention coupling term.
        Args:
            M_T: (N,) attention output
        Returns:
            attention_term: (N,) attention coupling information
        """
        # Move A_hat and d_hat_i to the same device and dtype as M_T
        A_hat_complex = self.A_hat.to(M_T.dtype).to(M_T.device)
        d_hat_i = self.d_hat_i.to(M_T.device)
        attention_term = (A_hat_complex @ M_T) / d_hat_i.squeeze(-1)
        return attention_term
    
    def forward_step(self, phases, noise=None):
        """
        Single-step forward pass (equations S30-S31)
        
        Args:
            phases: (N,) current phases θ_T (radians)
            noise: (N,) noise term (optional)
        Returns:
            new_phases: (N,) updated phases θ_{T+1}
            order_param: scalar order parameter R
        """
        # Set simulation time step dt
        dt = 0.05

        # Convert to complex representation
        phases_complex = torch.exp(1j * phases)  # e^{iθ}
        
        # Update history
        self.update_history(phases_complex)
        
        # Compute spatial coupling
        spatial_term = self.compute_spatial_coupling(phases_complex)
        
        # Compute attention coupling
        M_T = self.compute_attention()
        attention_term = self.compute_attention_coupling(M_T)
        
        # Total information I_T (S31)
        # Use sigmoid to constrain alpha to [0, 1]
        alpha = torch.sigmoid(self.alpha)
        I_T = (1 - alpha) * spatial_term + alpha * attention_term  # (N,) complex
        
        # Phase update (S30)
        # θ_T+1 = θ_T + ω + λ * Im[I_T * exp(-iθ_T)] + noise
        coupling_term = self.lambda_coupling * (I_T * phases_complex.conj()).imag
        
        # Add noise
        if noise is None:
            noise = torch.randn_like(phases) * np.sqrt(2 * self.D * dt)
        
        new_phases = phases + (self.omega + coupling_term) * dt + noise
        
        # Compute order parameter R
        order_param = torch.abs(phases_complex.mean())
        
        return new_phases, order_param
    
    def forward(self, initial_phases, n_steps, return_trajectory=False):
        """
        Multi-step forward simulation
        
        Args:
            initial_phases: (N,) initial phases
            n_steps: Simulation steps
            return_trajectory: whether to return the full trajectory
        Returns:
            final_phases: (N,) final phases
            order_params: (n_steps,) order-parameter sequence
            phases_trajectory: (n_steps, N) phase trajectory (if return_trajectory=True)
        """
        self.reset_history()
        
        phases = initial_phases.clone()
        order_params = []
        
        if return_trajectory:
            phases_trajectory = []
        
        for t in range(n_steps):
            phases, R = self.forward_step(phases)
            order_params.append(R)
            
            if return_trajectory:
                phases_trajectory.append(phases.clone())
        
        order_params = torch.stack(order_params)
        
        if return_trajectory:
            phases_trajectory = torch.stack(phases_trajectory)
            return phases, order_params, phases_trajectory
        
        return phases, order_params


class SimplifiedDTA(nn.Module):
    """
    Simplified DTA model，uses a direct decaying-memory mechanism
    corresponds to the exponential-decay self-interaction in the paper (S24-S29)
    """
    
    def __init__(self, 
                 n_oscillators: int,
                 decay_rate: float = 0.01,
                 learnable_decay: bool = False):
        super().__init__()
        
        self.N = n_oscillators
        
        if learnable_decay:
            self.beta = nn.Parameter(torch.tensor(decay_rate))
        else:
            self.register_buffer('beta', torch.tensor(decay_rate))
        
        # memory state M_t (S29)
        self.register_buffer('M_t', torch.zeros(n_oscillators, dtype=torch.cfloat))
        
    def reset_memory(self):
        """Reset memory state"""
        self.M_t.zero_()
    
    def forward(self, phases_complex):
        """
        Update memory and return attention information
        Args:
            phases_complex: (N,) complex phase
        Returns:
            M_t: (N,) memory state
        """
        # dM_t = β * (e^{iθ_t} - M_t) * dt
        beta = torch.abs(self.beta)
        self.M_t = self.M_t + beta * (phases_complex - self.M_t)
        return self.M_t
