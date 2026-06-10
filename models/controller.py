"""
同步/去同步控制器实现
用于学习最优的注意力参数以实现控制目标
"""

import torch
import torch.nn as nn
import numpy as np


class SyncController(nn.Module):
    """
    同步控制器
    目标: 使所有振子相位趋于一致 (序参量 R -> 1)
    """
    
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        
    def compute_loss(self, order_params, target_R=1.0, mode='final', 
                     convergence_weight=0.5, stability_weight=0.1):
        """
        计算同步损失（改进版）
        
        Args:
            order_params: (T,) 序参量序列
            target_R: 目标序参量值 (默认1.0)
            mode: 'final', 'mean', 'traj', 'convergence'
            convergence_weight: 收敛速度权重
            stability_weight: 稳定性权重
        Returns:
            loss: 标量
        """
        if mode == 'final':
            loss = (target_R - order_params[-1]) ** 2
        elif mode == 'mean':
            loss = (target_R - order_params.mean()) ** 2
        elif mode == 'traj':
            weights = torch.linspace(0.5, 1.0, len(order_params), device=order_params.device)
            loss = (weights * (target_R - order_params) ** 2).mean()
        elif mode == 'convergence':
            # 综合优化：最终状态 + 收敛速度 + 稳定性
            final_loss = (target_R - order_params[-1]) ** 2
            
            # 收敛速度损失（鼓励早期就达到高序参量）
            T = len(order_params)
            time_weights = torch.exp(-torch.linspace(0, 3, T, device=order_params.device))
            convergence_loss = (time_weights * (target_R - order_params)).sum()
            
            # 稳定性损失（惩罚下降）
            diff = torch.diff(order_params)
            stability_loss = torch.relu(-diff).mean()
            
            loss = final_loss + convergence_weight * convergence_loss + stability_weight * stability_loss
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return loss
    
    def forward(self, initial_phases, n_steps, **kwargs):
        """前向传播"""
        final_phases, order_params = self.model(initial_phases, n_steps)
        loss = self.compute_loss(order_params, **kwargs)
        return loss, order_params, final_phases


class DesyncController(nn.Module):
    """
    去同步控制器
    目标: 使所有振子相位趋于分散 (序参量 R -> 0)
    """
    
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        
    def compute_loss(self, order_params, target_R=0.0, mode='final', 
                    diversity_weight=0.1, final_phases=None):
        """
        计算去同步损失
        
        Args:
            order_params: (T,) 序参量序列
            target_R: 目标序参量值 (默认0.0)
            mode: 'final' - 只优化最终状态, 'mean' - 优化平均状态
            diversity_weight: 相位多样性奖励权重
            final_phases: 最终相位 (用于计算多样性)
        Returns:
            loss: 标量
        """
        if mode == 'final':
            loss = (order_params[-1] - target_R) ** 2
        elif mode == 'mean':
            loss = (order_params.mean() - target_R) ** 2
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 添加相位多样性奖励
        if diversity_weight > 0 and final_phases is not None:
            # 计算相位之间的差异，鼓励均匀分布
            N = len(final_phases)
            # 将相位排序后计算间隔
            sorted_phases = torch.sort(final_phases)[0]
            phase_diffs = torch.diff(sorted_phases, append=sorted_phases[:1] + 2*np.pi)
            # 理想情况下，相位应该均匀分布，间隔为 2π/N
            ideal_diff = 2 * np.pi / N
            diversity_penalty = ((phase_diffs - ideal_diff) ** 2).mean()
            loss = loss + diversity_weight * diversity_penalty
        
        return loss
    
    def forward(self, initial_phases, n_steps, **kwargs):
        """前向传播"""
        final_phases, order_params = self.model(initial_phases, n_steps)
        loss = self.compute_loss(order_params, final_phases=final_phases, **kwargs)
        return loss, order_params, final_phases


class HybridController(nn.Module):
    """
    混合控制器 - 可以根据任务切换同步/去同步
    """
    
    def __init__(self, base_model, task='sync'):
        super().__init__()
        self.model = base_model
        self.task = task
        
    def set_task(self, task):
        """设置任务类型"""
        assert task in ['sync', 'desync']
        self.task = task
        
    def compute_loss(self, order_params, target_R=None, final_phases=None):
        """
        计算损失
        
        Args:
            order_params: (T,) 序参量序列
            target_R: 目标序参量值 (None则使用默认值)
            final_phases: 最终相位
        """
        if self.task == 'sync':
            target = 1.0 if target_R is None else target_R
            loss = (target - order_params[-1]) ** 2
        else:  # desync
            target = 0.0 if target_R is None else target_R
            loss = (order_params[-1] - target) ** 2
            
            # 对于去同步，添加多样性奖励
            if final_phases is not None:
                N = len(final_phases)
                sorted_phases = torch.sort(final_phases % (2*np.pi))[0]
                phase_diffs = torch.diff(sorted_phases, append=sorted_phases[:1] + 2*np.pi)
                ideal_diff = 2 * np.pi / N
                diversity_penalty = ((phase_diffs - ideal_diff) ** 2).mean()
                loss = loss + 0.1 * diversity_penalty
        
        return loss
    
    def forward(self, initial_phases, n_steps, **kwargs):
        """前向传播"""
        final_phases, order_params = self.model(initial_phases, n_steps)
        loss = self.compute_loss(order_params, final_phases=final_phases, **kwargs)
        return loss, order_params, final_phases
