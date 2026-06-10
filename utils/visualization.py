"""
Visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend


def plot_phase_evolution(phases_trajectory, save_path=None, title="Phase Evolution"):
    """
    Plot phase evolution
    
    Args:
        phases_trajectory: (T, N) phase trajectory
        save_path: save path
        title: plot title
    """
    if isinstance(phases_trajectory, np.ndarray):
        phases = phases_trajectory
    else:
        phases = phases_trajectory.detach().cpu().numpy()
    
    T, N = phases.shape
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each oscillator phase
    for i in range(N):
        ax.plot(range(T), phases[:, i], alpha=0.6, linewidth=0.5)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Phase (rad)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_order_parameter(order_params, save_path=None, title="Order Parameter Evolution"):
    """
    Plot order-parameter evolution
    
    Args:
        order_params: (T,) order-parameter sequence
        save_path: save path
        title: plot title
    """
    if isinstance(order_params, np.ndarray):
        R_values = order_params
    else:
        R_values = order_params.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(len(R_values)), R_values, linewidth=2, color='blue')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Sync (R=1)')
    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Desync (R=0)')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Order Parameter R')
    ax.set_title(title)
    ax.set_ylim([-0.05, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_phase_distribution(phases, save_path=None, title="Phase Distribution"):
    """
    Plot phase-distribution histogram
    
    Args:
        phases: (N,) phases
        save_path: save path
        title: plot title
    """
    if isinstance(phases, np.ndarray):
        phase_values = phases
    else:
        phase_values = phases.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Histogram
    ax.hist(phase_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    
    # Compute and plot mean phase
    mean_phase = np.arctan2(np.sin(phase_values).mean(), np.cos(phase_values).mean())
    R = np.sqrt(np.cos(phase_values).mean()**2 + np.sin(phase_values).mean()**2)
    ax.arrow(mean_phase, 0, 0, R, alpha=0.9, width=0.1, 
             edgecolor='red', facecolor='red', lw=2, label=f'R={R:.3f}')
    
    ax.set_title(title, pad=20)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_training_curves(train_losses, val_losses=None, order_params_history=None, 
                        save_path=None, title="Training Curves"):
    """
    Plot training curves
    
    Args:
        train_losses: training loss list
        val_losses: validation loss list
        order_params_history: order-parameter history
        save_path: save path
        title: plot title
    """
    n_plots = 1 + (val_losses is not None) + (order_params_history is not None)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    idx = 0
    
    # Loss curves
    ax = axes[idx]
    ax.plot(train_losses, label='Train Loss', linewidth=2)
    if val_losses is not None:
        ax.plot(val_losses, label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    idx += 1
    
    # Order-parameter curve
    if order_params_history is not None:
        ax = axes[idx]
        if isinstance(order_params_history, np.ndarray):
            R_vals = order_params_history
        elif isinstance(order_params_history, list):
            R_vals = np.array(order_params_history)
        else:
            R_vals = order_params_history.detach().cpu().numpy()
        
        ax.plot(R_vals, linewidth=2, color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Final Order Parameter')
        ax.set_title('Synchronization Level')
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_network_topology(adjacency, phases=None, save_path=None, title="Network Topology"):
    """
    Plot network topology
    
    Args:
        adjacency: (N, N) adjacency matrix
        phases: (N,) phases (used for coloring)
        save_path: save path
        title: plot title
    """
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX is required for network visualization")
        return
    
    if isinstance(adjacency, np.ndarray):
        A = adjacency
    else:
        A = adjacency.detach().cpu().numpy()
    
    G = nx.from_numpy_array(A)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    if phases is not None:
        if isinstance(phases, np.ndarray):
            phase_values = phases
        else:
            phase_values = phases.detach().cpu().numpy()
        
        # Map phase to color
        colors = plt.cm.hsv((phase_values % (2*np.pi)) / (2*np.pi))
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=300, ax=ax)
    else:
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    
    ax.set_title(title)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_attention_weights(attention_matrix, save_path=None, title="Attention Weights"):
    """
    Plot attention-weight heatmap
    
    Args:
        attention_matrix: (T, T) or (N, N) attention weight matrices
        save_path: save path
        title: plot title
    """
    if isinstance(attention_matrix, np.ndarray):
        C = attention_matrix
    else:
        C = attention_matrix.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(C, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    ax.set_xlabel('Key Index')
    ax.set_ylabel('Query Index')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig
