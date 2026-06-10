"""
Trainer implementation
"""

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import json

from models.controller import SyncController, DesyncController, HybridController
from utils.metrics import evaluate_control_performance
from utils.visualization import plot_training_curves, plot_order_parameter, plot_phase_distribution


class Trainer:
    """
    Controller trainer
    """
    
    def __init__(self, model, task='sync', lr=1e-3, device='cpu'):
        """
        Args:
            model: SynchronizationTransformer model
            task: 'sync' or 'desync'
            lr: learning rate
            device: device
        """
        self.device = device
        self.model = model.to(device)
        self.task = task
        
        # Create controller
        if task == 'sync':
            self.controller = SyncController(self.model)
        elif task == 'desync':
            self.controller = DesyncController(self.model)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        self.controller = self.controller.to(device)
        
        # Optimizer - only optimize learnable parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.order_params_history = []
        
    def train_epoch(self, n_episodes, n_steps, n_oscillators, loss_mode='final'):
        """
        Train one epoch
        
        Args:
            n_episodes: episodes per epoch
            n_steps: steps per episode
            n_oscillators: Number of oscillators
            loss_mode: loss mode ('final', 'mean', 'traj', 'convergence')
        
        Returns:
            avg_loss: average loss
        """
        epoch_losses = []
        
        for episode in range(n_episodes):
            # Sample a random initial condition
            initial_phases = torch.rand(n_oscillators, device=self.device) * 2 * np.pi
            
            # Reset model history
            self.model.reset_history()
            
            # Forward pass
            loss, order_params, final_phases = self.controller(
                initial_phases, n_steps, mode=loss_mode
            )
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping，avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, n_val_episodes, n_steps, n_oscillators):
        """
        Validate
        
        Args:
            n_val_episodes: validation episodes
            n_steps: number of steps
            n_oscillators: Number of oscillators
        
        Returns:
            avg_loss: average loss
            avg_order_param: mean order parameter
        """
        val_losses = []
        order_params_list = []
        
        with torch.no_grad():
            for _ in range(n_val_episodes):
                initial_phases = torch.rand(n_oscillators, device=self.device) * 2 * np.pi
                self.model.reset_history()
                
                loss, order_params, final_phases = self.controller(
                    initial_phases, n_steps, mode='final'
                )
                
                val_losses.append(loss.item())
                order_params_list.append(order_params[-1].item())
        
        avg_loss = np.mean(val_losses)
        avg_order_param = np.mean(order_params_list)
        
        self.val_losses.append(avg_loss)
        self.order_params_history.append(avg_order_param)
        
        return avg_loss, avg_order_param
    
    def train(self, n_epochs, n_episodes_per_epoch, n_val_episodes,
              n_steps, n_oscillators, save_dir=None, verbose=True,
              spatial_network=None, attention_network=None, natural_frequencies=None,
              attention_type='neighbor', network_type='ws', natural_freq_std=0.1,
              loss_mode='final'):
        """
        Full training workflow
        
        Args:
            n_epochs: number of training epochs
            n_episodes_per_epoch: episodes per epoch
            n_val_episodes: validation episodes
            n_steps: simulation steps per episode
            n_oscillators: Number of oscillators
            save_dir: output directory
            verbose: whether to print progress
            loss_mode: loss mode ('final', 'mean', 'traj', 'convergence')
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        iterator = tqdm(range(n_epochs)) if verbose else range(n_epochs)
        
        for epoch in iterator:
            # Training
            train_loss = self.train_epoch(n_episodes_per_epoch, n_steps, n_oscillators, loss_mode)
            
            # Validate
            val_loss, avg_order_param = self.validate(
                n_val_episodes, n_steps, n_oscillators
            )
            
            # Get current alpha value
            alpha = torch.sigmoid(self.model.alpha).item()
            
            if verbose:
                iterator.set_description(
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"R: {avg_order_param:.4f} | "
                    f"α: {alpha:.4f}"
                )
            
            # Save checkpoint
            if save_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'),
                                   spatial_network, attention_network, natural_frequencies, attention_type, network_type, natural_freq_std)
        
        # Save final model
        if save_dir:
            self.save_checkpoint(os.path.join(save_dir, 'final_model.pt'),
                               spatial_network, attention_network, natural_frequencies, attention_type, network_type, natural_freq_std)
            
            # Plot training curves
            plot_training_curves(
                self.train_losses, 
                self.val_losses, 
                self.order_params_history,
                save_path=os.path.join(save_dir, 'training_curves.png')
            )
            
            # Save training history
            history = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'order_params_history': self.order_params_history
            }
            with open(os.path.join(save_dir, 'history.json'), 'w') as f:
                json.dump(history, f)
    
    def save_checkpoint(self, path, spatial_network=None, attention_network=None, natural_frequencies=None, attention_type='neighbor', network_type='ws', natural_freq_std=0.1):
        """Save checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'order_params_history': self.order_params_history,
            'task': self.task,
            'attention_type': attention_type,
            'network_type': network_type,
            'natural_freq_std': natural_freq_std,
            'coupling_strength': self.model.lambda_coupling,
            'noise_strength': self.model.D,
            'spatial_network': spatial_network.cpu() if spatial_network is not None else None,
            'attention_network': attention_network.cpu() if attention_network is not None else None,
            'natural_frequencies': natural_frequencies.cpu() if natural_frequencies is not None else None
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.order_params_history = checkpoint.get('order_params_history', [])


class BatchTrainer:
    """
    Batch trainer - supports multiple initial conditions
    """
    
    def __init__(self, model, task='sync', lr=1e-3, device='cpu'):
        self.device = device
        self.model = model.to(device)
        self.task = task
        
        if task == 'sync':
            self.controller = SyncController(self.model)
        elif task == 'desync':
            self.controller = DesyncController(self.model)
        
        self.controller = self.controller.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.train_losses = []
        
    def train_epoch_batched(self, batch_size, n_steps, n_oscillators):
        """
        Train one batched epoch
        
        Args:
            batch_size: batch size
            n_steps: Simulation steps
            n_oscillators: Number of oscillators
        """
        # Generate batched initial conditions
        initial_phases_batch = torch.rand(batch_size, n_oscillators, device=self.device) * 2 * np.pi
        
        total_loss = 0.0
        
        for i in range(batch_size):
            self.model.reset_history()
            loss, order_params, final_phases = self.controller(
                initial_phases_batch[i], n_steps, mode='final'
            )
            total_loss += loss
        
        # average loss
        avg_loss = total_loss / batch_size
        
        # Backpropagation
        self.optimizer.zero_grad()
        avg_loss.backward()
        self.optimizer.step()
        
        self.train_losses.append(avg_loss.item())
        
        return avg_loss.item()
    
    def train(self, n_epochs, batch_size, n_steps, n_oscillators, save_dir=None):
        """Batch training"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for epoch in tqdm(range(n_epochs), desc="Training"):
            loss = self.train_epoch_batched(batch_size, n_steps, n_oscillators)
            
            if (epoch + 1) % 10 == 0:
                alpha = torch.sigmoid(self.model.alpha).item()
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss:.4f}, α: {alpha:.4f}")
        
        if save_dir:
            self.save_checkpoint(os.path.join(save_dir, 'final_model.pt'))
    
    def save_checkpoint(self, path):
        """Save checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
        }, path)
