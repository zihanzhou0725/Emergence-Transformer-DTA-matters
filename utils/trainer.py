"""
训练器实现
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
    控制器训练器
    """
    
    def __init__(self, model, task='sync', lr=1e-3, device='cpu'):
        """
        Args:
            model: SynchronizationTransformer 模型
            task: 'sync' 或 'desync'
            lr: 学习率
            device: 计算设备
        """
        self.device = device
        self.model = model.to(device)
        self.task = task
        
        # 创建控制器
        if task == 'sync':
            self.controller = SyncController(self.model)
        elif task == 'desync':
            self.controller = DesyncController(self.model)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        self.controller = self.controller.to(device)
        
        # 优化器 - 只优化可学习参数
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.order_params_history = []
        
    def train_epoch(self, n_episodes, n_steps, n_oscillators, loss_mode='final'):
        """
        训练一个epoch
        
        Args:
            n_episodes: 每个epoch的episode数
            n_steps: 每个episode的步数
            n_oscillators: 振子数量
            loss_mode: 损失模式 ('final', 'mean', 'traj', 'convergence')
        
        Returns:
            avg_loss: 平均损失
        """
        epoch_losses = []
        
        for episode in range(n_episodes):
            # 随机采样初始条件
            initial_phases = torch.rand(n_oscillators, device=self.device) * 2 * np.pi
            
            # 重置模型历史
            self.model.reset_history()
            
            # 前向传播
            loss, order_params, final_phases = self.controller(
                initial_phases, n_steps, mode=loss_mode
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, n_val_episodes, n_steps, n_oscillators):
        """
        验证
        
        Args:
            n_val_episodes: 验证episode数
            n_steps: 步数
            n_oscillators: 振子数量
        
        Returns:
            avg_loss: 平均损失
            avg_order_param: 平均序参量
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
        完整训练流程
        
        Args:
            n_epochs: 训练epoch数
            n_episodes_per_epoch: 每个epoch的episode数
            n_val_episodes: 验证episode数
            n_steps: 每个episode的模拟步数
            n_oscillators: 振子数量
            save_dir: 保存目录
            verbose: 是否打印进度
            loss_mode: 损失模式 ('final', 'mean', 'traj', 'convergence')
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        iterator = tqdm(range(n_epochs)) if verbose else range(n_epochs)
        
        for epoch in iterator:
            # 训练
            train_loss = self.train_epoch(n_episodes_per_epoch, n_steps, n_oscillators, loss_mode)
            
            # 验证
            val_loss, avg_order_param = self.validate(
                n_val_episodes, n_steps, n_oscillators
            )
            
            # 获取当前alpha值
            alpha = torch.sigmoid(self.model.alpha).item()
            
            if verbose:
                iterator.set_description(
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"R: {avg_order_param:.4f} | "
                    f"α: {alpha:.4f}"
                )
            
            # 保存检查点
            if save_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'),
                                   spatial_network, attention_network, natural_frequencies, attention_type, network_type, natural_freq_std)
        
        # 保存最终模型
        if save_dir:
            self.save_checkpoint(os.path.join(save_dir, 'final_model.pt'),
                               spatial_network, attention_network, natural_frequencies, attention_type, network_type, natural_freq_std)
            
            # 绘制训练曲线
            plot_training_curves(
                self.train_losses, 
                self.val_losses, 
                self.order_params_history,
                save_path=os.path.join(save_dir, 'training_curves.png')
            )
            
            # 保存训练历史
            history = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'order_params_history': self.order_params_history
            }
            with open(os.path.join(save_dir, 'history.json'), 'w') as f:
                json.dump(history, f)
    
    def save_checkpoint(self, path, spatial_network=None, attention_network=None, natural_frequencies=None, attention_type='neighbor', network_type='ws', natural_freq_std=0.1):
        """保存检查点"""
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
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.order_params_history = checkpoint.get('order_params_history', [])


class BatchTrainer:
    """
    批量训练器 - 支持同时训练多个初始条件
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
        批量训练一个epoch
        
        Args:
            batch_size: 批量大小
            n_steps: 模拟步数
            n_oscillators: 振子数量
        """
        # 生成批量初始条件
        initial_phases_batch = torch.rand(batch_size, n_oscillators, device=self.device) * 2 * np.pi
        
        total_loss = 0.0
        
        for i in range(batch_size):
            self.model.reset_history()
            loss, order_params, final_phases = self.controller(
                initial_phases_batch[i], n_steps, mode='final'
            )
            total_loss += loss
        
        # 平均损失
        avg_loss = total_loss / batch_size
        
        # 反向传播
        self.optimizer.zero_grad()
        avg_loss.backward()
        self.optimizer.step()
        
        self.train_losses.append(avg_loss.item())
        
        return avg_loss.item()
    
    def train(self, n_epochs, batch_size, n_steps, n_oscillators, save_dir=None):
        """批量训练"""
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
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
        }, path)
