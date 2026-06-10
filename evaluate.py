"""
Model evaluation script
Evaluate a trained controller
"""

import argparse
import torch
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.sync_transformer import SynchronizationTransformer
from utils.networks import generate_watts_strogatz
from utils.metrics import evaluate_control_performance, compute_order_parameter
from utils.visualization import (
    plot_phase_evolution, plot_order_parameter, 
    plot_phase_distribution, plot_network_topology
)


def load_model(checkpoint_path, device='cpu'):
    """Load a trained model"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    from configs.default_config import Config

    state_dict = checkpoint['model_state_dict']
    task = checkpoint.get('task', 'sync')
    attention_type = checkpoint.get(
        'attention_type',
        'neighbor' if task == 'sync' else 'self'
    )

    if checkpoint.get('spatial_network') is not None:
        spatial_network = checkpoint['spatial_network'].to(device)
    else:
        n_oscillators = state_dict['W_Q'].shape[0] if 'W_Q' in state_dict else Config.N_OSCILLATORS
        spatial_network = generate_watts_strogatz(
            n_oscillators,
            Config.K_NEIGHBORS,
            Config.REWIRING_PROB,
            seed=42
        ).to(device)

    n_oscillators = spatial_network.shape[0]

    if checkpoint.get('attention_network') is not None:
        attention_network = checkpoint['attention_network'].to(device)
    elif attention_type == 'self':
        attention_network = torch.eye(n_oscillators, device=device)
    else:
        attention_network = spatial_network.clone()

    if checkpoint.get('natural_frequencies') is not None:
        natural_frequencies = checkpoint['natural_frequencies'].to(device)
    else:
        natural_freq_std = checkpoint.get('natural_freq_std', Config.NATURAL_FREQ_STD)
        natural_frequencies = torch.randn(n_oscillators, device=device) * natural_freq_std

    d_model = state_dict['W_Q'].shape[1] if 'W_Q' in state_dict else Config.D_MODEL
    coupling_strength = checkpoint.get('coupling_strength', Config.COUPLING_STRENGTH)
    noise_strength = checkpoint.get('noise_strength', Config.NOISE_STRENGTH)

    model = SynchronizationTransformer(
        n_oscillators=n_oscillators,
        d_model=d_model,
        spatial_network=spatial_network,
        attention_network=attention_network,
        natural_frequencies=natural_frequencies,
        coupling_strength=coupling_strength,
        noise_strength=noise_strength,
        learnable_alpha=True,
        learnable_w_qk=True
    )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, task


def evaluate_model(model, n_trials, n_steps, n_oscillators, task='sync', device='cpu', save_dir=None):
    """
    Evaluate model performance
    
    Args:
        model: trained model
        n_trials: number of trials
        n_steps: Simulation steps
        n_oscillators: Number of oscillators
        device: device
        save_dir: output directory
    """
    results = []
    
    print(f"\nRunning {n_trials} trials...")
    
    for trial in range(n_trials):
        # Random initial condition
        initial_phases = torch.rand(n_oscillators, device=device) * 2 * np.pi
        
        with torch.no_grad():
            model.reset_history()
            final_phases, order_params, trajectory = model(
                initial_phases, n_steps, return_trajectory=True
            )
        
        # Evaluate performance
        metrics = evaluate_control_performance(
            order_params.cpu().numpy(),
            task=task
        )
        
        results.append(metrics)
        
        # Visualize the first trial
        if trial == 0 and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Order-parameter evolution
            plot_order_parameter(
                order_params.cpu().numpy(),
                save_path=os.path.join(save_dir, 'order_param_evolution.png'),
                title='Order Parameter Evolution'
            )
            
            # Phase evolution
            plot_phase_evolution(
                trajectory.cpu().numpy(),
                save_path=os.path.join(save_dir, 'phase_evolution.png'),
                title='Phase Evolution'
            )
            
            # Final phase distribution
            plot_phase_distribution(
                final_phases.cpu().numpy(),
                save_path=os.path.join(save_dir, 'phase_distribution.png'),
                title=f'Final Phase Distribution (R={metrics["final_order_param"]:.3f})'
            )
    
    # Summarize results
    print("\nEvaluation results:")
    print("=" * 50)
    
    success_rate = np.mean([r['success'] for r in results])
    avg_final_R = np.mean([r['final_order_param'] for r in results])
    avg_convergence_time = np.mean([r['convergence_time'] for r in results])
    
    print(f"Success rate: {success_rate:.2%}")
    print(f"Mean final order parameter: {avg_final_R:.4f}")
    print(f"Mean convergence time: {avg_convergence_time:.1f} steps")
    print("=" * 50)
    
    # Save results
    if save_dir:
        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
            json.dump({
                'success_rate': float(success_rate),
                'avg_final_R': float(avg_final_R),
                'avg_convergence_time': float(avg_convergence_time),
                'all_results': results
            }, f, indent=2)
    
    return results


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Load model: {args.checkpoint}")
    model, task = load_model(args.checkpoint, device)
    
    print(f"Task type: {task}")
    print(f"Learned α: {torch.sigmoid(model.alpha).item():.4f}")
    
    # Evaluate
    results = evaluate_model(
        model,
        n_trials=args.n_trials,
        n_steps=args.n_steps,
        n_oscillators=model.N,
        task=task,
        device=device,
        save_dir=args.save_dir
    )
    
    print(f"\nEvaluation complete. Results saved in: {args.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained controller')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='model checkpoint path')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='number of trials')
    parser.add_argument('--n_steps', type=int, default=100,
                       help='Simulation steps')
    parser.add_argument('--save_dir', type=str, default='./results/evaluation',
                       help='output directory')
    
    args = parser.parse_args()
    main(args)
