"""
Load a trained model and generate R-vs-timestep plots
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Avoid OpenMP duplicate runtime warnings

import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.sync_transformer import SynchronizationTransformer
from utils.networks import generate_watts_strogatz
from utils.visualization import plot_order_parameter
from utils.metrics import compute_order_parameter

def load_trained_model(checkpoint_path, device='cpu'):
    """Load a trained model"""
    print(f"Load model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Rebuild the model with default configuration
    N = 20  # based on the test configuration
    T_context = 10
    
    spatial_network = generate_watts_strogatz(N, k_neighbors=4, rewiring_prob=0.1, seed=42)
    
    # Infer task type
    task = checkpoint.get('task', 'sync')
    if task == 'sync':
        attention_network = spatial_network.clone()
        print("Task type: synchronization (Sync)")
    else:
        attention_network = torch.eye(N)
        print("Task type: desynchronization (Desync)")
    
    natural_frequencies = torch.randn(N) * 0.1
    
    model = SynchronizationTransformer(
        n_oscillators=N,
        d_model=64,
        spatial_network=spatial_network,
        attention_network=attention_network,
        natural_frequencies=natural_frequencies,
        coupling_strength=1.5,
        noise_strength=0.05,
        learnable_alpha=True,
        learnable_w_qk=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Show learned parameters
    alpha_learned = torch.sigmoid(model.alpha).item()
    print(f"Learned α: {alpha_learned:.4f}")
    
    return model, task


def generate_trajectory(model, initial_phases, n_steps, device='cpu'):
    """Generate trajectory"""
    with torch.no_grad():
        model.reset_history()
        initial_phases = initial_phases.to(device)
        
        final_phases, order_params, trajectory = model(
            initial_phases, n_steps, return_trajectory=True
        )
    
    return final_phases, order_params, trajectory


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model path
    checkpoint_path = 'results/test_experiment/final_model.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: model file not found {checkpoint_path}")
        print("Please train first: python train.py --config test")
        return
    
    # Load model
    model, task = load_trained_model(checkpoint_path, device)
    
    # Create output directory
    save_dir = 'results/trained_trajectories'
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate multiple test cases
    n_test_cases = 5
    n_steps = 100
    N = 20
    
    print(f"\nGenerating trajectories for {n_test_cases} test cases...")
    
    for i in range(n_test_cases):
        # Random initial condition
        initial_phases = torch.rand(N) * 2 * np.pi
        
        # Generate trajectory
        final_phases, order_params, trajectory = generate_trajectory(
            model, initial_phases, n_steps, device
        )
        
        # Print results
        initial_R = order_params[0].item()
        final_R = order_params[-1].item()
        print(f"\nCase {i+1}:")
        print(f"  Initial R: {initial_R:.4f}")
        print(f"  Final R: {final_R:.4f}")
        print(f"  Change: {final_R - initial_R:+.4f}")
        
        # Plot and save
        save_path = os.path.join(save_dir, f'trajectory_case_{i+1}.png')
        plot_order_parameter(
            order_params.cpu().numpy(),
            save_path=save_path,
            title=f'Trained Model - Case {i+1}: R(0)={initial_R:.3f} → R(T)={final_R:.3f}'
        )
        print(f"  Plot saved: {save_path}")
    
    # Plot the average trajectory
    print("\nGenerate average trajectory...")
    all_order_params = []
    for i in range(20):  # more samples for averaging
        initial_phases = torch.rand(N) * 2 * np.pi
        _, order_params, _ = generate_trajectory(model, initial_phases, n_steps, device)
        all_order_params.append(order_params.cpu().numpy())
    
    mean_order_params = np.mean(all_order_params, axis=0)
    std_order_params = np.std(all_order_params, axis=0)
    
    # Save average trajectory plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(mean_order_params, linewidth=2, color='blue', label='Mean R')
    plt.fill_between(
        range(len(mean_order_params)),
        mean_order_params - std_order_params,
        mean_order_params + std_order_params,
        alpha=0.3, color='blue', label='±1 Std Dev'
    )
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Sync (R=1)')
    plt.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Desync (R=0)')
    plt.xlabel('Time Step')
    plt.ylabel('Order Parameter R')
    plt.title(f'Trained Model - Average Trajectory (20 samples)\nTask: {task}, Final R={mean_order_params[-1]:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.05, 1.05])
    
    save_path = os.path.join(save_dir, 'average_trajectory.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Average trajectory plot saved: {save_path}")
    
    print(f"\nAll results saved in: {save_dir}/")


if __name__ == '__main__':
    main()
