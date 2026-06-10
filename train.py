"""
Main training script
Train synchronization/desynchronization controllers
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Avoid OpenMP duplicate runtime warnings

import argparse
import torch
import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sync_transformer import SynchronizationTransformer
from utils.networks import generate_watts_strogatz, generate_fully_connected, network_summary
from utils.trainer import Trainer
from configs.default_config import Config, SyncConfig, SyncSelfConfig, DesyncConfig, DesyncNeighborConfig, TestConfig


def set_seed(seed=42):
    """Set random seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Choose one of the sync/desync and neighbor/self configuration combinations.
    config_map = {
        'sync': SyncConfig,           # sync + neighbor
        'sync_self': SyncSelfConfig,  # sync + self
        'desync': DesyncConfig,       # desync + self
        'desync_neighbor': DesyncNeighborConfig,  # desync + neighbor
        'test': TestConfig,
        'default': Config
    }
    config = config_map.get(args.config, Config)()
    
    # Override configuration
    if args.task:
        config.TASK = args.task
    if args.n_oscillators:
        config.N_OSCILLATORS = args.n_oscillators
    if args.epochs:
        config.N_EPOCHS = args.epochs
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.save_dir:
        config.SAVE_DIR = args.save_dir
    
    # Print configuration
    config.print_config()
    
    # Generate network topology
    print("\nGenerate network topology...")
    if config.NETWORK_TYPE == 'ws':
        # Watts-Strogatz small-world network
        spatial_network = generate_watts_strogatz(
            config.N_OSCILLATORS,
            config.K_NEIGHBORS,
            config.REWIRING_PROB,
            seed=args.seed
        )
        print(f"Using Watts-Strogatz network (N={config.N_OSCILLATORS}, k={config.K_NEIGHBORS}, p={config.REWIRING_PROB})")
    elif config.NETWORK_TYPE == 'fc':
        # fully connected network
        spatial_network = generate_fully_connected(config.N_OSCILLATORS)
        print(f"Using fully connected network (N={config.N_OSCILLATORS})")
    else:
        raise ValueError(f"Unknown NETWORK_TYPE: {config.NETWORK_TYPE}")
    
    # Choose attention network according to ATTENTION_TYPE (independent of TASK)
    if config.ATTENTION_TYPE == 'neighbor':
        # neighbor-DTA: Â = A (same as the spatial network)
        attention_network = spatial_network.clone()
        print("Using neighbor-DTA (A_hat = A)")
    elif config.ATTENTION_TYPE == 'self':
        # self-DTA: Â = I (identity matrix)
        attention_network = torch.eye(config.N_OSCILLATORS)
        print("Using self-DTA (A_hat = I)")
    else:
        raise ValueError(f"Unknown ATTENTION_TYPE: {config.ATTENTION_TYPE}")
    
    print(f"Task type: {config.TASK.upper()} (target: {'R->1' if config.TASK == 'sync' else 'R->0'})")
    
    network_summary(spatial_network)
    
    # Generate natural frequencies（set seed for reproducibility）
    torch.manual_seed(args.seed + 1)  # use a different seed from the network seed
    natural_frequencies = torch.randn(config.N_OSCILLATORS) * config.NATURAL_FREQ_STD
    torch.manual_seed(args.seed)  # restore original seed
    print(f"Natural-frequency distribution: N(0, {config.NATURAL_FREQ_STD}^2)")
    
    # Create model
    print("\nCreate model...")
    model = SynchronizationTransformer(
        n_oscillators=config.N_OSCILLATORS,
        d_model=config.D_MODEL,
        spatial_network=spatial_network,
        attention_network=attention_network,
        natural_frequencies=natural_frequencies,
        coupling_strength=config.COUPLING_STRENGTH,
        noise_strength=config.NOISE_STRENGTH,
        learnable_alpha=config.LEARNABLE_ALPHA,
        alpha_init=config.ALPHA_INIT,
        learnable_w_qk=True,
        w_v_identity=True
    )
    
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Show alpha settings
    alpha_status = "trainable" if config.LEARNABLE_ALPHA else "fixed"
    actual_alpha = torch.sigmoid(torch.tensor(config.ALPHA_INIT)).item()
    print(f"Alpha setting: {alpha_status}, initial value={config.ALPHA_INIT}, actual alpha≈{actual_alpha:.4f}")
    
    # Create trainer
    print(f"\nStart training {config.TASK} controller...")
    trainer = Trainer(
        model=model,
        task=config.TASK,
        lr=config.LEARNING_RATE,
        device=config.DEVICE
    )
    
    # Training
    trainer.train(
        n_epochs=config.N_EPOCHS,
        n_episodes_per_epoch=config.N_EPISODES_PER_EPOCH,
        n_val_episodes=config.N_VAL_EPISODES,
        n_steps=config.N_STEPS,
        n_oscillators=config.N_OSCILLATORS,
        save_dir=config.SAVE_DIR,
        verbose=True,
        spatial_network=spatial_network,
        attention_network=attention_network,
        natural_frequencies=natural_frequencies,
        attention_type=config.ATTENTION_TYPE,
        network_type=config.NETWORK_TYPE,
        natural_freq_std=config.NATURAL_FREQ_STD
    )
    
    print(f"\nTraining complete! Results saved in: {config.SAVE_DIR}")
    
    # Final test
    print("\nFinal test...")
    model.eval()
    with torch.no_grad():
        test_phases = torch.rand(config.N_OSCILLATORS, device=config.DEVICE) * 2 * np.pi
        model.reset_history()
        
        final_phases, order_params, trajectory = model(
            test_phases, 
            n_steps=config.N_STEPS,
            return_trajectory=True
        )
        
        final_R = order_params[-1].item()
        print(f"Test order parameter: initial={order_params[0].item():.4f}, Final={final_R:.4f}")
        print(f"Learned α: {torch.sigmoid(model.alpha).item():.4f}")
        
        if config.TASK == 'sync':
            success = final_R > 0.9
        else:
            success = final_R < 0.3
        
        print(f"Control{'succeeded' if success else 'failed'}!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainingsynchronization/Desynchronization controller')
    
    parser.add_argument('--config', type=str, default='sync',
                       choices=['sync', 'sync_self', 'desync', 'desync_neighbor', 'test', 'default'],
                       help='configuration type (sync/sync_self/desync/desync_neighbor/test/default)')
    parser.add_argument('--task', type=str, default=None,
                       choices=['sync', 'desync'],
                       help='Task type')
    parser.add_argument('--n_oscillators', type=int, default=None,
                       help='Number of oscillators')
    parser.add_argument('--epochs', type=int, default=None,
                       help='number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='random seed')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='output directory')
    
    args = parser.parse_args()
    main(args)
