"""
Demo script
Quickly demonstrate Synchronization Transformer functionality
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Avoid OpenMP duplicate runtime warnings

import torch
import numpy as np
import os

from models.sync_transformer import SynchronizationTransformer
from utils.networks import generate_watts_strogatz, network_summary
from utils.metrics import compute_order_parameter
from utils.visualization import plot_order_parameter, plot_phase_distribution


def demo_sync_task():
    """Demo synchronization task"""
    print("=" * 60)
    print("Demo: synchronization task (Sync Task)")
    print("=" * 60)
    
    # Parameter settings
    N = 50
    n_steps = 100
    
    # Generate network
    print("\n1. Generate network topology...")
    spatial_network = generate_watts_strogatz(N, k_neighbors=4, rewiring_prob=0.1, seed=42)
    network_summary(spatial_network)
    
    # Synchronization task: use neighbor-DTA
    attention_network = spatial_network.clone()
    
    # Create model
    print("\n2. Create model...")
    model = SynchronizationTransformer(
        n_oscillators=N,
        d_model=64,
        spatial_network=spatial_network,
        attention_network=attention_network,
        natural_frequencies=torch.randn(N) * 0.1,
        coupling_strength=1.5,
        noise_strength=0.05,
        learnable_alpha=True,
        alpha_init=0.3
    )
    
    # Simulation
    print("\n3. Run simulation...")
    initial_phases = torch.rand(N) * 2 * np.pi
    
    with torch.no_grad():
        model.reset_history()
        final_phases, order_params, trajectory = model(
            initial_phases, n_steps, return_trajectory=True
        )
    
    # Results
    print(f"\n4. Results:")
    print(f"   Initial order parameter R(0): {order_params[0].item():.4f}")
    print(f"   Final order parameter R(T): {order_params[-1].item():.4f}")
    print(f"   Current alpha value: {torch.sigmoid(model.alpha).item():.4f}")
    
    # Visualize
    os.makedirs('./results/demo', exist_ok=True)
    plot_order_parameter(
        order_params.numpy(),
        save_path='./results/demo/sync_order_param.png',
        title='Sync Task: Order Parameter Evolution'
    )
    print(f"   Plot saved to: ./results/demo/sync_order_param.png")


def demo_desync_task():
    """Demo desynchronization task"""
    print("\n" + "=" * 60)
    print("Demo: desynchronization task (Desync Task)")
    print("=" * 60)
    
    # Parameter settings
    N = 50
    n_steps = 100
    
    # Generate network
    print("\n1. Generate network topology...")
    spatial_network = generate_watts_strogatz(N, k_neighbors=4, rewiring_prob=0.1, seed=42)
    
    # Desynchronization task: use self-DTA
    attention_network = torch.eye(N)
    
    # Create model
    print("\n2. Create model...")
    model = SynchronizationTransformer(
        n_oscillators=N,
        d_model=64,
        spatial_network=spatial_network,
        attention_network=attention_network,
        natural_frequencies=torch.randn(N) * 0.1,
        coupling_strength=2.0,
        noise_strength=0.05,
        learnable_alpha=True,
        alpha_init=0.7
    )
    
    # Simulation
    print("\n3. Run simulation...")
    initial_phases = torch.ones(N) * np.pi / 2  # initial state is nearly synchronized
    
    with torch.no_grad():
        model.reset_history()
        final_phases, order_params, trajectory = model(
            initial_phases, n_steps, return_trajectory=True
        )
    
    # Results
    print(f"\n4. Results:")
    print(f"   Initial order parameter R(0): {order_params[0].item():.4f}")
    print(f"   Final order parameter R(T): {order_params[-1].item():.4f}")
    print(f"   Current alpha value: {torch.sigmoid(model.alpha).item():.4f}")
    
    # Visualize
    plot_order_parameter(
        order_params.numpy(),
        save_path='./results/demo/desync_order_param.png',
        title='Desync Task: Order Parameter Evolution'
    )
    print(f"   Plot saved to: ./results/demo/desync_order_param.png")


def demo_attention_mechanism():
    """Demo attention mechanism"""
    print("\n" + "=" * 60)
    print("Demo: attention mechanism (Attention Mechanism)")
    print("=" * 60)
    
    N = 20
    
    # Create a simple network
    spatial_network = torch.eye(N)
    attention_network = torch.eye(N)
    
    model = SynchronizationTransformer(
        n_oscillators=N,
        d_model=32,
        spatial_network=spatial_network,
        attention_network=attention_network,
        learnable_alpha=True,
        learnable_w_qk=True
    )
    
    print("\n1. Learnable parameters:")
    print(f"   W_Q shape: {model.W_Q.shape}")
    print(f"   W_K shape: {model.W_K.shape}")
    print(f"   W_V shape: {model.W_V.shape}")
    print(f"   α (mixture coefficient): {torch.sigmoid(model.alpha).item():.4f}")
    
    print("\n2. Simulate phase evolution...")
    initial_phases = torch.rand(N) * 2 * np.pi
    
    with torch.no_grad():
        model.reset_history()
        final_phases, order_params, trajectory = model(
            initial_phases, n_steps=50, return_trajectory=True
        )
    
    print(f"\n3. Order parameter change:")
    print(f"   Initial: {order_params[0].item():.4f}")
    print(f"   Middle: {order_params[25].item():.4f}")
    print(f"   Final: {order_params[-1].item():.4f}")
    
    print("\n4. History cache state:")
    print(f"   History length: {len(model.phase_history_list)}")
    print(f"   Current time step: {model.current_time}")


def main():
    print("\n")
    print("#" * 60)
    print("# Synchronization Transformer demo")
    print("# Based on paper: 'Synchronization Transformer: Dynamical")
    print("#          Temporal Attention Matters'")
    print("#" * 60)
    
    # Run demos
    demo_sync_task()
    demo_desync_task()
    demo_attention_mechanism()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nUsage:")
    print("  1. Train synchronization controller: python train.py --config sync")
    print("  2. Train desynchronization controller: python train.py --config desync")
    print("  3. Quick test: python train.py --config test")
    print("  4. Evaluate model: python evaluate.py --checkpoint <path>")
    print("=" * 60)


if __name__ == '__main__':
    main()
