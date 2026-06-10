"""
Comparison experiment: trainedDTA model vs pure traditional coupling (α=0)
Compare both curves in one figure to show the DTA advantage
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Avoid OpenMP duplicate runtime warnings

import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.io import savemat  # for saving MATLAB format files

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.sync_transformer import SynchronizationTransformer
from utils.networks import generate_watts_strogatz, generate_fully_connected


def load_trained_model(checkpoint_path, device='cpu'):
    """Load a trained model"""
    print(f"Load a trained model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load the network topology and natural frequencies from the checkpoint to match training
    if 'spatial_network' in checkpoint and checkpoint['spatial_network'] is not None:
        spatial_network = checkpoint['spatial_network'].to(device)
        natural_frequencies = checkpoint['natural_frequencies'].to(device)
        print("[OK] Loaded training network topology and natural frequencies from checkpoint")
        
        # Load ATTENTION_TYPE if it is stored
        if 'attention_type' in checkpoint:
            attention_type = checkpoint['attention_type']
            print(f"[OK] ATTENTION_TYPE: {attention_type}")
        else:
            # Backward compatibility: infer it from the task
            attention_type = 'neighbor' if checkpoint.get('task') == 'sync' else 'self'
            print(f"[WARNING] Using default ATTENTION_TYPE: {attention_type}")
        
        # Load NETWORK_TYPE if it is stored
        if 'network_type' in checkpoint:
            network_type = checkpoint['network_type']
            print(f"[OK] NETWORK_TYPE: {network_type}")
        else:
            network_type = 'ws'  # Backward-compatible default WS network
            print(f"[WARNING] Using default NETWORK_TYPE: {network_type}")
        
        # Load NATURAL_FREQ_STD if it is stored
        if 'natural_freq_std' in checkpoint:
            natural_freq_std = checkpoint['natural_freq_std']
            print(f"[OK] NATURAL_FREQ_STD: {natural_freq_std}")
        else:
            natural_freq_std = 0.1  # Backward-compatible default value
            print(f"[WARNING] Using default NATURAL_FREQ_STD: {natural_freq_std}")
        
        # Load COUPLING_STRENGTH if it is stored
        if 'coupling_strength' in checkpoint:
            coupling_strength = checkpoint['coupling_strength']
            print(f"[OK] COUPLING_STRENGTH: {coupling_strength}")
        else:
            coupling_strength = 1.5  # Backward-compatible default value
            print(f"[WARNING] Using default COUPLING_STRENGTH: {coupling_strength}")

        noise_strength = checkpoint.get('noise_strength', 0.05)
        print(f"[OK] NOISE_STRENGTH: {noise_strength}")
        
        # Set attention_network according to ATTENTION_TYPE
        if attention_type == 'neighbor':
            attention_network = spatial_network.clone()
        else:  # 'self'
            attention_network = torch.eye(spatial_network.shape[0], device=device)
    else:
        # Backward compatibility
        print("[WARNING] Using default network topology; it may differ from training")
        N = 20
        spatial_network = generate_watts_strogatz(N, k_neighbors=2, rewiring_prob=0.1, seed=42)
        attention_network = spatial_network.clone()
        natural_frequencies = torch.randn(N) * 0.1
        coupling_strength = 1.5  # default value
        noise_strength = 0.05
        attention_type = 'neighbor'  # default value
        network_type = 'ws'  # default value
        natural_freq_std = 0.1  # default value
    
    N = spatial_network.shape[0]
    task = checkpoint.get('task', 'sync')
    
    # Infer d_model from weights
    if 'model_state_dict' in checkpoint and 'W_Q' in checkpoint['model_state_dict']:
        d_model = checkpoint['model_state_dict']['W_Q'].shape[1]
        print(f"[OK] D_MODEL: {d_model} (inferred from checkpoint weights)")
    else:
        d_model = 64  # default value
        print(f"[WARNING] Using default D_MODEL: {d_model}")
    
    # Create the model with full accumulated history
    model = SynchronizationTransformer(
        n_oscillators=N,
        d_model=d_model,
        spatial_network=spatial_network,
        attention_network=attention_network,
        natural_frequencies=natural_frequencies,
        coupling_strength=coupling_strength,
        noise_strength=noise_strength,
        learnable_alpha=True,
        learnable_w_qk=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    alpha_learned = torch.sigmoid(model.alpha).item()
    print(f"Learned α: {alpha_learned:.4f}")
    print(f"Coupling strength λ: {model.lambda_coupling}")
    
    # Collect configuration information
    config_info = {
        'task': task,
        'attention_type': attention_type if 'attention_type' in locals() else 'neighbor',
        'network_type': network_type if 'network_type' in locals() else 'ws',
        'natural_freq_std': natural_freq_std if 'natural_freq_std' in locals() else 0.1
    }
    
    return model, task, alpha_learned, config_info


def create_baseline_model(trained_model, device='cpu'):
    """
    Create a baseline model with the same parameters as the trained model, but with alpha=0 for pure traditional coupling
    """
    N = trained_model.N
    
    # Copy trained model parameters but set alpha=0
    baseline_model = SynchronizationTransformer(
        n_oscillators=N,
        d_model=trained_model.d,
        spatial_network=trained_model.A.clone(),
        attention_network=trained_model.A_hat.clone(),
        natural_frequencies=trained_model.omega.clone(),
        coupling_strength=trained_model.lambda_coupling,
        noise_strength=trained_model.D,
        learnable_alpha=False,  # not learnable
        alpha_init=-10.0,       # sigmoid gives approximately 0, so this is pure traditional coupling
        learnable_w_qk=False,   # attention matrices are not learnable
        w_v_identity=True
    )
    
    # Copy W_Q and W_K for consistency, although they are unused when alpha=0
    baseline_model.W_Q.data = trained_model.W_Q.data.clone()
    baseline_model.W_K.data = trained_model.W_K.data.clone()
    
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    
    return baseline_model


def generate_trajectory(model, initial_phases, n_steps, device='cpu'):
    """Generate trajectory"""
    with torch.no_grad():
        model.reset_history()
        initial_phases = initial_phases.to(device)
        final_phases, order_params, trajectory = model(
            initial_phases, n_steps, return_trajectory=True
        )
    
    # Clear CUDA cache
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    print(f"[Debug] generate_trajectory: n_steps={n_steps}, returned order_params length={len(order_params)}")
    return final_phases, order_params.cpu().numpy(), trajectory


def plot_comparison(order_params_dta, order_params_baseline, 
                   initial_phases, alpha_value, task,
                   save_path, case_id):
    """
    Plot comparison: DTA vs traditional coupling
    """
    print(f"[Debug] plot_comparison Case {case_id}: DTA length={len(order_params_dta)}, Baselinelength={len(order_params_baseline)}")
    
    plt.figure(figsize=(12, 7))
    
    # Compute initial R
    initial_R = order_params_dta[0]
    
    # Plot both curves
    plt.plot(order_params_dta, linewidth=2.5, color='blue', 
            label=f'DTA Model (α={alpha_value:.3f})', marker='o', markersize=3, markevery=10)
    plt.plot(order_params_baseline, linewidth=2.5, color='red', 
            label='Traditional Coupling Only (α=0)', marker='s', markersize=3, markevery=10)
    
    # Reference lines
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Perfect Sync (R=1)')
    plt.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Desync (R=0)')
    
    # Fill the gap between curves
    plt.fill_between(range(len(order_params_dta)), 
                    order_params_dta, order_params_baseline,
                    alpha=0.2, color='purple', 
                    label=f'DTA Advantage (ΔR={order_params_dta[-1] - order_params_baseline[-1]:.3f})')
    
    # Set labels and title
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Order Parameter R', fontsize=14)
    plt.title(f'Case {case_id}: DTA vs Traditional Coupling\n'
             f'Task: {task.upper()}, Initial R={initial_R:.3f}, '
             f'Final: DTA={order_params_dta[-1]:.3f} vs Baseline={order_params_baseline[-1]:.3f}',
             fontsize=14)
    
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.05, 1.05])
    plt.xlim([0, len(order_params_dta)-1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved: {save_path}")


def plot_average_comparison(all_dta, all_baseline, alpha_value, task, save_path):
    """Plot average comparison"""
    mean_dta = np.mean(all_dta, axis=0)
    std_dta = np.std(all_dta, axis=0)
    mean_baseline = np.mean(all_baseline, axis=0)
    std_baseline = np.std(all_baseline, axis=0)
    
    plt.figure(figsize=(13, 7))
    
    # Plot average curves
    plt.plot(mean_dta, linewidth=3, color='blue', 
            label=f'DTA Model (α={alpha_value:.3f}) - Mean of {len(all_dta)} trials')
    plt.plot(mean_baseline, linewidth=3, color='red', 
            label='Traditional Coupling (α=0) - Mean')
    
    # Plot standard-deviation bands
    plt.fill_between(range(len(mean_dta)), 
                    mean_dta - std_dta, mean_dta + std_dta,
                    alpha=0.2, color='blue')
    plt.fill_between(range(len(mean_baseline)), 
                    mean_baseline - std_baseline, mean_baseline + std_baseline,
                    alpha=0.2, color='red')
    
    # Reference lines
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Perfect Sync (R=1)')
    plt.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Fill the advantage region
    plt.fill_between(range(len(mean_dta)), 
                    mean_dta, mean_baseline,
                    alpha=0.25, color='purple', 
                    label=f'DTA Advantage (ΔR={mean_dta[-1] - mean_baseline[-1]:.3f})')
    
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Order Parameter R', fontsize=14)
    plt.title(f'Average Comparison: DTA vs Traditional Coupling\n'
             f'Task: {task.upper()}, {len(all_dta)} Random Initial Conditions',
             fontsize=14)
    
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Average comparison plot saved: {save_path}")
    
    return mean_dta, std_dta, mean_baseline, std_baseline


def save_matlab_data(all_dta, all_baseline, task, network_type, attention_type, alpha_value, save_dir):
    """
    Save data as a MATLAB .mat file
    
    Variable descriptions:
    - timestep: time-step sequence (0, 1, 2, ..., T-1)
    - mean_dta: mean order parameter of the DTA model
    - std_dta: standard deviation of the DTA model
    - mean_baseline: mean order parameter of the baseline model  
    - std_baseline: standard deviation of the baseline model
    - dta_upper: DTA upper bound (mean + std)
    - dta_lower: DTA lower bound (mean - std)
    - baseline_upper: baseline upper bound
    - baseline_lower: baseline lower bound
    """
    mean_dta = np.mean(all_dta, axis=0)
    std_dta = np.std(all_dta, axis=0)
    mean_baseline = np.mean(all_baseline, axis=0)
    std_baseline = np.std(all_baseline, axis=0)
    
    # Compute upper and lower bounds
    dta_upper = mean_dta + std_dta
    dta_lower = mean_dta - std_dta
    baseline_upper = mean_baseline + std_baseline
    baseline_lower = mean_baseline - std_baseline
    
    # Time steps
    timestep = np.arange(len(mean_dta))
    
    # Build the data dictionary
    mat_data = {
        'timestep': timestep,
        'mean_dta': mean_dta,
        'std_dta': std_dta,
        'mean_baseline': mean_baseline,
        'std_baseline': std_baseline,
        'dta_upper': dta_upper,
        'dta_lower': dta_lower,
        'baseline_upper': baseline_upper,
        'baseline_lower': baseline_lower,
        'alpha_learned': alpha_value,
        'n_trials': len(all_dta),
        'task': task,
        'network_type': network_type,
        'attention_type': attention_type
    }
    
    # Build the file name: results/compare_<task>_<attention_type>_<network_type>.mat
    filename = f"compare_{task}_{attention_type}_{network_type}.mat"
    filepath = os.path.join(save_dir, filename)
    
    # Save as a .mat file
    savemat(filepath, mat_data)
    print(f"\n[OK] MATLAB data saved: {filepath}")
    print(f"    Variables included: timestep, mean_dta, std_dta, mean_baseline, std_baseline")
    print(f"    and upper/lower bounds: dta_upper/lower, baseline_upper/lower")
    
    return filepath


def main():
    # Use CPU if GPU memory is limited:
    # device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"[INFO] If GPU memory is insufficient, set: device = 'cpu'")
    else:
        print("[INFO] Using CPU")
    
    checkpoint_path = 'results/sync_self_ws/final_model.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: model file not found {checkpoint_path}")
        return
    
    # Load a trained model
    trained_model, task, alpha_learned, config_info = load_trained_model(checkpoint_path, device)
    attention_type = config_info['attention_type']
    network_type = config_info['network_type']
    
    # Create the alpha=0 baseline model
    print("\nCreate baseline model (alpha=0, pure traditional coupling)...")
    baseline_model = create_baseline_model(trained_model, device)
    
    # Create output directory
    save_dir = 'results/comparison_dta_vs_baseline'
    os.makedirs(save_dir, exist_ok=True)
    
    # Parameters
    n_steps = 2000
    N = trained_model.N  # Get the node count from the model instead of hard-coding it
    
    print(f"\n[Debug] Simulation steps n_steps = {n_steps}")
    print(f"[Debug] Number of oscillators N = {N}")
    
    # Run multiple test cases
    print("\nGenerate comparison cases...")
    n_cases = 20
    
    all_dta = []
    all_baseline = []
    
    for i in range(n_cases):
        # Use the same random initial condition
        initial_phases = torch.rand(N) * 2 * np.pi
        
        # Run DTA model
        final_dta, order_params_dta, traj_dta = generate_trajectory(
            trained_model, initial_phases, n_steps, device
        )
        
        # Run baseline model with the same initial condition
        final_baseline, order_params_baseline, traj_baseline = generate_trajectory(
            baseline_model, initial_phases, n_steps, device
        )
        
        # Save results for averaging
        all_dta.append(order_params_dta)
        all_baseline.append(order_params_baseline)
        
        # Clear GPU cache every 5 cases
        if device == 'cuda' and (i + 1) % 5 == 0:
            torch.cuda.empty_cache()
            print(f"  [Clear GPU memory] completed {i+1}/{n_cases} cases")
        
        # Print results
        print(f"\nCase {i+1}:")
        print(f"  Initial R: {order_params_dta[0]:.4f}")
        print(f"  DTA final R: {order_params_dta[-1]:.4f}")
        print(f"  Traditional coupling final R: {order_params_baseline[-1]:.4f}")
        print(f"  DTA advantage (ΔR): {order_params_dta[-1] - order_params_baseline[-1]:+.4f}")
        
        # Plot comparison
        save_path = os.path.join(save_dir, f'comparison_case_{i+1}.png')
        plot_comparison(order_params_dta, order_params_baseline,
                       initial_phases, alpha_learned, task,
                       save_path, i+1)
    
    # Plot average comparison
    print("\nGenerate average comparison plot...")
    mean_dta, std_dta, mean_baseline, std_baseline = plot_average_comparison(
        all_dta, all_baseline, alpha_learned, task,
        os.path.join(save_dir, 'average_comparison.png')
    )
    
    # Summary statistics
    print("\n" + "="*60)
    print("Comparison experiment summary")
    print("="*60)
    
    final_dta_all = [traj[-1] for traj in all_dta]
    final_baseline_all = [traj[-1] for traj in all_baseline]
    improvements = [d - b for d, b in zip(final_dta_all, final_baseline_all)]
    
    print(f"Mean final R of DTA model: {np.mean(final_dta_all):.4f} ± {np.std(final_dta_all):.4f}")
    print(f"Mean final R of traditional coupling: {np.mean(final_baseline_all):.4f} ± {np.std(final_baseline_all):.4f}")
    print(f"Mean improvement (ΔR): {np.mean(improvements):+.4f} ± {np.std(improvements):.4f}")
    print(f"Relative improvement: {(np.mean(improvements) / np.mean(final_baseline_all) * 100):+.1f}%")
    print("="*60)
    
    print(f"\nAll comparison plots saved in: {save_dir}/")
    print("File descriptions:")
    print("  - comparison_case_X.png: single-case comparison")
    print("  - average_comparison.png: average trajectory comparison")
    
    # Save MATLAB data file
    print("\n" + "="*60)
    print("Save MATLAB data file...")
    print("="*60)
    mat_filepath = save_matlab_data(
        all_dta, all_baseline, 
        task, network_type, attention_type, 
        alpha_learned, save_dir
    )
    print("="*60)


if __name__ == '__main__':
    main()
