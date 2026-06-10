"""
Comparison experiment: Natural Memory model vs pure traditional coupling (α=0)
Compare the exponential-decay memory model with traditional coupling
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.sync_transformer import SynchronizationTransformer
from utils.networks import generate_watts_strogatz, generate_fully_connected


class NaturalMemoryModel(nn.Module):
    """
    Natural Memory Model (Natural Memory Model)
    Based on an exponential-decay memory kernel: dM/dt = beta * (e^(i*theta) - M)
    Total information: I = (1-alpha) * spatial_coupling + alpha * memory_coupling
    
    Parameters:
        alpha: memory fraction (0-1)
        beta: exponential decay rate
    """
    
    def __init__(self, n_oscillators, spatial_network, attention_network, natural_frequencies,
                 coupling_strength=1.5, noise_strength=0.05,
                 alpha=0.5, beta=0.1, device='cpu'):
        super().__init__()
        
        self.N = n_oscillators
        self.device = device
        self.lambda_coupling = coupling_strength
        self.D = noise_strength
        self.alpha = alpha  # memory fraction
        self.beta = beta    # decay rate
        self.dt = 0.05      # time step
        
        # Register network parameters
        self.register_buffer('A', spatial_network.float())
        self.register_buffer('A_hat', attention_network.float())  # memory coupling network
        self.register_buffer('omega', natural_frequencies.float())
        
        # Compute degrees
        self.d_i = self.A.sum(dim=1, keepdim=True)
        self.d_hat_i = self.A_hat.sum(dim=1, keepdim=True)
        
        # Initialize memory state
        self.reset_memory()
        
    def reset_memory(self):
        """Reset memory state M_t"""
        self.M_t = torch.zeros(self.N, dtype=torch.cfloat, device=self.device)
        
    def compute_spatial_coupling(self, phases_complex):
        """Compute the traditional spatial coupling term"""
        A_complex = self.A.to(phases_complex.dtype).to(phases_complex.device)
        d_i = self.d_i.to(phases_complex.device)
        spatial_term = (A_complex @ phases_complex) / d_i.squeeze(-1)
        return spatial_term
    
    def compute_memory_coupling(self, M_t):
        """
        Compute the memory coupling term
        memory_coupling = A_hat @ M_t / d_hat_i
        """
        A_hat_complex = self.A_hat.to(M_t.dtype).to(M_t.device)
        d_hat_i = self.d_hat_i.to(M_t.device)
        memory_term = (A_hat_complex @ M_t) / d_hat_i.squeeze(-1)
        return memory_term
    
    def update_memory(self, phases_complex):
        """
        Update memory state: dM/dt = beta * (e^(i*theta) - M)
        Discretization: M_{t+1} = M_t + beta * dt * (e^(i*theta_t) - M_t)
        """
        self.M_t = self.M_t + self.beta * self.dt * (phases_complex - self.M_t)
        
    def forward_step(self, phases, noise=None):
        """Single-step forward pass"""
        phases_complex = torch.exp(1j * phases)
        
        # Update memory state
        self.update_memory(phases_complex)
        
        # Compute spatial coupling
        spatial_term = self.compute_spatial_coupling(phases_complex)
        
        # Compute memory coupling
        memory_term = self.compute_memory_coupling(self.M_t)
        
        # Total information I = (1-alpha) * spatial + alpha * memory
        I_t = (1 - self.alpha) * spatial_term + self.alpha * memory_term
        
        # Phase update
        coupling_term = self.lambda_coupling * (I_t * phases_complex.conj()).imag
        
        if noise is None:
            noise = torch.randn_like(phases) * np.sqrt(2 * self.D * self.dt)
        
        new_phases = phases + (self.omega + coupling_term) * self.dt + noise
        order_param = torch.abs(phases_complex.mean())
        
        return new_phases, order_param
    
    def forward(self, initial_phases, n_steps, return_trajectory=False):
        """Multi-step forward simulation"""
        self.reset_memory()
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


def load_model_for_comparison(checkpoint_path, device='cpu'):
    """
    Load model metadata from checkpoint
    """
    print(f"Load checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model architecture parameters
    if 'model_state_dict' in checkpoint and 'W_Q' in checkpoint['model_state_dict']:
        d_model = checkpoint['model_state_dict']['W_Q'].shape[1]
        print(f"[OK] D_MODEL: {d_model}")
    else:
        d_model = 64
        print(f"[WARNING] Using default D_MODEL: {d_model}")
    
    # Get basic parameters
    N = checkpoint.get('spatial_network', torch.eye(100)).shape[0]
    task = checkpoint.get('task', 'sync')
    coupling_strength = checkpoint.get('coupling_strength', 1.5)
    noise_strength = checkpoint.get('noise_strength', 0.05)
    attention_type = checkpoint.get('attention_type', 'neighbor')
    
    # Load network from checkpoint
    if 'spatial_network' in checkpoint and checkpoint['spatial_network'] is not None:
        spatial_network = checkpoint['spatial_network'].to(device)
        natural_frequencies = checkpoint['natural_frequencies'].to(device)
        
        # Set attention network
        if attention_type == 'neighbor':
            attention_network = spatial_network.clone()
            print(f"[OK] Network type: neighbor (Â=A)")
        else:
            attention_network = torch.eye(N, device=device)
            print(f"[OK] Network type: self (Â=I)")
    else:
        print("[WARNING] Using default network topology")
        spatial_network = generate_watts_strogatz(N, 4, 0.1, seed=42)
        attention_network = spatial_network.clone()
        natural_frequencies = torch.randn(N) * 0.1
    
    print(f"[INFO] Number of oscillators N: {N}")
    print(f"[INFO] Task type: {task}")
    print(f"[INFO] Coupling strength λ: {coupling_strength}")
    print(f"[INFO] noise strength D: {noise_strength}")
    
    return {
        'N': N,
        'd_model': d_model,
        'spatial_network': spatial_network,
        'attention_network': attention_network,
        'natural_frequencies': natural_frequencies,
        'coupling_strength': coupling_strength,
        'noise_strength': noise_strength,
        'task': task,
        'device': device
    }


def create_memory_model(model_info, memory_alpha, memory_beta, device='cpu',
                        coupling_strength=None, noise_strength=None):
    """
    Create natural memory model（alpha=0 means traditional coupling）
    """
    coupling_strength = model_info['coupling_strength'] if coupling_strength is None else coupling_strength
    noise_strength = model_info['noise_strength'] if noise_strength is None else noise_strength

    model = NaturalMemoryModel(
        n_oscillators=model_info['N'],
        spatial_network=model_info['spatial_network'],
        attention_network=model_info['attention_network'],
        natural_frequencies=model_info['natural_frequencies'],
        coupling_strength=coupling_strength,
        noise_strength=noise_strength,
        alpha=memory_alpha,
        beta=memory_beta,
        device=device
    ).to(device)
    model.eval()
    
    if memory_alpha == 0:
        print(f"\n[Create traditional coupling baseline]")
        print(f"  Alpha: {memory_alpha} (pure spatial coupling)")
    else:
        print(f"\n[Create natural memory model]")
        print(f"  Alpha: {memory_alpha}, Beta: {memory_beta}")
    
    print(f"  Network type: {'neighbor (Â=A)' if torch.allclose(model_info['attention_network'], model_info['spatial_network']) else 'self (Â=I)'}")
    print(f"  Coupling strength λ: {coupling_strength}")
    print(f"  noise strength D: {noise_strength}")
    
    return model


def generate_trajectory(model, initial_phases, n_steps, device='cpu', return_trajectory=False):
    """Generate a trajectory for the memory model"""
    with torch.no_grad():
        model.reset_memory()
        initial_phases = initial_phases.to(device)
        result = model(initial_phases, n_steps, return_trajectory=return_trajectory)

    if return_trajectory:
        final_phases, order_params, trajectory = result
    else:
        final_phases, order_params = result
        trajectory = None

    return final_phases, order_params.cpu().numpy(), trajectory


def plot_comparison(order_params_traditional, order_params_memory,
                   initial_phases, task, save_path, case_id,
                   memory_alpha, memory_beta):
    """Plot comparison: traditional coupling vs Natural Memory Model（plot one point every 100 steps）"""
    plt.figure(figsize=(12, 7))
    
    initial_R = order_params_traditional[0]
    n_points = len(order_params_traditional)
    
    # sample one point every 100 steps
    step = 100
    indices = list(range(0, n_points, step))
    if indices[-1] != n_points - 1:
        indices.append(n_points - 1)  # ensure the last point is included
    
    traditional_sampled = order_params_traditional[indices]
    memory_sampled = order_params_memory[indices]
    
    # Plot both curves（plot one point every 100 steps）
    plt.plot(indices, traditional_sampled, linewidth=2.5, color='red',
            label='Traditional Coupling (α=0)', linestyle='-')
    plt.plot(indices, memory_sampled, linewidth=2.5, color='green',
            label=f'Natural Memory (α={memory_alpha}, β={memory_beta})', linestyle='-')
    
    # Reference lines
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='Perfect Sync (R=1)')
    plt.axhline(y=0.0, color='gray', linestyle='--', alpha=0.3, linewidth=1.5)
    
    # Fill the advantage region（using full data）
    plt.fill_between(range(n_points),
                    order_params_memory, order_params_traditional,
                    alpha=0.2, color='purple', 
                    label=f'Memory Advantage (ΔR={order_params_memory[-1] - order_params_traditional[-1]:.3f})')
    
    # Set labels and title
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Order Parameter R', fontsize=14)
    plt.title(f'Case {case_id}: Natural Memory vs Traditional Coupling\n'
             f'Task: {task.upper()}, Initial R={initial_R:.3f}\n'
             f'Final R: Traditional={order_params_traditional[-1]:.3f} | Memory={order_params_memory[-1]:.3f}',
             fontsize=13)
    
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.05, 1.05])
    plt.xlim([0, n_points-1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved: {save_path}")


def plot_average_comparison(all_traditional, all_memory, task, save_path,
                           memory_alpha, memory_beta):
    """Plot average comparison（plot one point every 100 steps）"""
    mean_traditional = np.mean(all_traditional, axis=0)
    std_traditional = np.std(all_traditional, axis=0)
    mean_memory = np.mean(all_memory, axis=0)
    std_memory = np.std(all_memory, axis=0)
    
    plt.figure(figsize=(13, 7))
    
    n_points = len(mean_traditional)
    
    # sample one point every 100 steps
    step = 100
    indices = list(range(0, n_points, step))
    if indices[-1] != n_points - 1:
        indices.append(n_points - 1)
    
    mean_traditional_sampled = mean_traditional[indices]
    mean_memory_sampled = mean_memory[indices]
    std_traditional_sampled = std_traditional[indices]
    std_memory_sampled = std_memory[indices]
    
    # Plot average curves（plot one point every 100 steps）
    plt.plot(indices, mean_traditional_sampled, linewidth=3, color='red',
            label=f'Traditional Coupling (α=0) - Mean')
    plt.plot(indices, mean_memory_sampled, linewidth=3, color='green',
            label=f'Natural Memory (α={memory_alpha}, β={memory_beta}) - Mean')
    
    # Plot standard-deviation bands（every 100 points）
    plt.fill_between(indices,
                    mean_traditional_sampled - std_traditional_sampled, 
                    mean_traditional_sampled + std_traditional_sampled,
                    alpha=0.15, color='red')
    plt.fill_between(indices,
                    mean_memory_sampled - std_memory_sampled, 
                    mean_memory_sampled + std_memory_sampled,
                    alpha=0.15, color='green')
    
    # Reference lines
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='Perfect Sync (R=1)')
    plt.axhline(y=0.0, color='gray', linestyle='--', alpha=0.3, linewidth=1.5)
    
    # Fill the advantage region（using full data）
    plt.fill_between(range(n_points),
                    mean_memory, mean_traditional,
                    alpha=0.2, color='purple', 
                    label=f'Memory Advantage (ΔR={mean_memory[-1] - mean_traditional[-1]:.3f})')
    
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Order Parameter R', fontsize=14)
    plt.title(f'Average Comparison: Natural Memory vs Traditional Coupling\n'
             f'Task: {task.upper()}, {len(all_traditional)} Random Initial Conditions\n'
             f'Final R: Traditional={mean_traditional[-1]:.3f} | Memory={mean_memory[-1]:.3f}',
             fontsize=13)
    
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.05, 1.05])
    plt.xlim([0, n_points-1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Average comparison plot saved: {save_path}")


def save_matlab_data_sampled(all_traditional, all_memory, task, save_dir, memory_alpha, memory_beta):
    """
    Save sampled data as a MATLAB .mat file by sampling every 100 points to save space.
    
    Variable descriptions:
    - timestep: sampled time-step sequence
    - mean_traditional: mean order parameter of the traditional model（after sampling）
    - std_traditional: standard deviation of the traditional model（after sampling）
    - mean_memory: mean order parameter of the memory model（after sampling）
    - std_memory: standard deviation of the memory model（after sampling）
    - traditional_upper/lower: traditional modelbounds
    - memory_upper/lower: memory modelbounds
    """
    mean_traditional = np.mean(all_traditional, axis=0)
    std_traditional = np.std(all_traditional, axis=0)
    mean_memory = np.mean(all_memory, axis=0)
    std_memory = np.std(all_memory, axis=0)
    
    # Sample every 100 points
    n_points = len(mean_traditional)
    step = 100
    indices = list(range(0, n_points, step))
    if indices[-1] != n_points - 1:
        indices.append(n_points - 1)
    
    timestep = np.array(indices)
    mean_traditional_sampled = mean_traditional[indices]
    std_traditional_sampled = std_traditional[indices]
    mean_memory_sampled = mean_memory[indices]
    std_memory_sampled = std_memory[indices]
    
    # Compute upper and lower bounds
    traditional_upper = mean_traditional_sampled + std_traditional_sampled
    traditional_lower = mean_traditional_sampled - std_traditional_sampled
    memory_upper = mean_memory_sampled + std_memory_sampled
    memory_lower = mean_memory_sampled - std_memory_sampled
    
    # Build the data dictionary with sampled data only
    mat_data = {
        'timestep': timestep,
        'mean_traditional': mean_traditional_sampled,
        'std_traditional': std_traditional_sampled,
        'mean_memory': mean_memory_sampled,
        'std_memory': std_memory_sampled,
        'traditional_upper': traditional_upper,
        'traditional_lower': traditional_lower,
        'memory_upper': memory_upper,
        'memory_lower': memory_lower,
        'memory_alpha': memory_alpha,
        'memory_beta': memory_beta,
        'n_trials': len(all_traditional),
        'task': task,
        'sampling_info': 'Every 100 points sampled to reduce file size'
    }
    
    # Build the file name
    filename = f"memory_compare_{task}_alpha{memory_alpha}_beta{memory_beta}.mat"
    filepath = os.path.join(save_dir, filename)
    
    # Save as a .mat file
    savemat(filepath, mat_data)
    print(f"\n[OK] MATLAB data saved: {filepath}")
    print(f"    number of sampled points: {len(timestep)} / {n_points} (sampled every 100 points)")
    print(f"    Variables included: timestep, mean_traditional, std_traditional, mean_memory, std_memory")
    print(f"    and upper/lower bounds: traditional_upper/lower, memory_upper/lower")
    
    return filepath


def parse_args():
    parser = argparse.ArgumentParser(description='Natural Memory vs traditional coupling comparison experiment')
    parser.add_argument('--checkpoint', type=str, default='results/sync_self_ws/final_model.pt',
                       help='checkpoint used to read network, frequency, and task metadata')
    parser.add_argument('--memory_alpha', type=float, default=0.5,
                       help='memory fraction alpha')
    parser.add_argument('--memory_beta', type=float, default=0.01,
                       help='exponential decay rate beta')
    parser.add_argument('--n_steps', type=int, default=100000,
                       help='Simulation steps')
    parser.add_argument('--n_cases', type=int, default=100,
                       help='number of test cases')
    parser.add_argument('--coupling_strength', type=float, default=None,
                       help='override coupling strength lambda from the checkpoint')
    parser.add_argument('--noise_strength', type=float, default=None,
                       help='override noise strength D from the checkpoint')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'auto'],
                       help='runtime device; CPU by default to avoid limited GPU memory')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='results output directory; generated from alpha/beta by default')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA is unavailable; using CPU")
        device = 'cpu'

    print(f"[INFO] Using {device.upper()} run")

    CHECKPOINT_PATH = args.checkpoint
    MEMORY_ALPHA = args.memory_alpha
    MEMORY_BETA = args.memory_beta
    N_STEPS = args.n_steps
    N_CASES = args.n_cases
    
    # Check checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: model file not found {CHECKPOINT_PATH}")
        return
    
    # Load modelinformation
    print("\n" + "="*70)
    print("Load modelinformation")
    print("="*70)
    model_info = load_model_for_comparison(CHECKPOINT_PATH, device)
    
    # Create traditional coupling baseline (alpha=0, beta arbitrary)
    traditional_model = create_memory_model(
        model_info,
        memory_alpha=0,
        memory_beta=0,
        device=device,
        coupling_strength=args.coupling_strength,
        noise_strength=args.noise_strength
    )
    
    # Create natural memory model (alpha>0)
    memory_model = create_memory_model(
        model_info,
        MEMORY_ALPHA,
        MEMORY_BETA,
        device,
        coupling_strength=args.coupling_strength,
        noise_strength=args.noise_strength
    )
    
    # Create output directory
    save_dir = args.save_dir or f'results/memory_comparison_alpha{MEMORY_ALPHA}_beta{MEMORY_BETA}'
    os.makedirs(save_dir, exist_ok=True)
    
    N = model_info['N']
    
    print("\n" + "="*70)
    print("Start comparison experiment")
    print("="*70)
    print(f"Simulation steps: {N_STEPS}")
    print(f"number of test cases: {N_CASES}")
    print(f"Number of oscillators: {N}")
    
    all_traditional = []
    all_memory = []
    
    for i in range(N_CASES):
        # Use the same random initial condition
        initial_phases = torch.rand(N) * 2 * np.pi
        
        # Run traditional coupling model
        final_traditional, order_params_traditional, traj_traditional = generate_trajectory(
            traditional_model, initial_phases, N_STEPS, device
        )
        
        # Run the Natural Memory Model with the same initial condition
        final_memory, order_params_memory, traj_memory = generate_trajectory(
            memory_model, initial_phases, N_STEPS, device
        )
        
        # Save results for averaging
        all_traditional.append(order_params_traditional)
        all_memory.append(order_params_memory)
        
        # Print results
        print(f"\nCase {i+1}:")
        print(f"  Initial R: {order_params_traditional[0]:.4f}")
        print(f"  Traditional coupling final R: {order_params_traditional[-1]:.4f}")
        print(f"  Natural memory final R: {order_params_memory[-1]:.4f}")
        print(f"  Memory advantage (ΔR): {order_params_memory[-1] - order_params_traditional[-1]:+.4f}")
        
        # Plot comparison
        save_path = os.path.join(save_dir, f'comparison_case_{i+1}.png')
        plot_comparison(order_params_traditional, order_params_memory,
                       initial_phases, model_info['task'],
                       save_path, i+1, MEMORY_ALPHA, MEMORY_BETA)
    
    # Plot average comparison
    print("\nGenerate average comparison plot...")
    plot_average_comparison(all_traditional, all_memory, model_info['task'],
                           os.path.join(save_dir, 'average_comparison.png'),
                           MEMORY_ALPHA, MEMORY_BETA)
    
    # Summary statistics
    print("\n" + "="*70)
    print("Comparison experiment summary")
    print("="*70)
    
    final_traditional_all = [traj[-1] for traj in all_traditional]
    final_memory_all = [traj[-1] for traj in all_memory]
    improvements = [m - t for m, t in zip(final_memory_all, final_traditional_all)]
    
    print(f"Memory model parameters: α={MEMORY_ALPHA}, β={MEMORY_BETA}")
    print("-"*70)
    print(f"Mean final R of traditional coupling: {np.mean(final_traditional_all):.4f} ± {np.std(final_traditional_all):.4f}")
    print(f"Mean final R of natural memory: {np.mean(final_memory_all):.4f} ± {np.std(final_memory_all):.4f}")
    print(f"Mean improvement (ΔR): {np.mean(improvements):+.4f} ± {np.std(improvements):.4f}")
    if np.mean(final_traditional_all) > 0.01:
        print(f"Relative improvement: {(np.mean(improvements) / np.mean(final_traditional_all) * 100):+.1f}%")
    print("="*70)
    
    print(f"\nAll results saved in: {save_dir}/")
    print("\nTip：")
    print(f"  - Change MEMORY_ALPHA and MEMORY_BETA to tune the memory model")
    print(f"  - Current parameters: α={MEMORY_ALPHA}, β={MEMORY_BETA}")
    
    # Save MATLAB data file（after sampling，save space）
    print("\n" + "="*70)
    print("Save sampled MATLAB data file...")
    print("="*70)
    mat_filepath = save_matlab_data_sampled(
        all_traditional, all_memory, 
        model_info['task'], save_dir,
        MEMORY_ALPHA, MEMORY_BETA
    )
    print("="*70)


if __name__ == '__main__':
    main()
