"""
Synchronization Transformer 核心模型实现
基于论文公式 (S30)-(S36) 的离散时间动力学
历史窗口改为时变：累积所有历史信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SynchronizationTransformer(nn.Module):
    """
    Synchronization Transformer for networked oscillators with Dynamical Temporal Attention (DTA)
    
    核心方程:
    - 相位更新 (S30): θ_T+1 = θ_T + ω + λ*Im[I_T * exp(-iθ_T)] + ξ_T
    - 总信息 (S31): I_T = (1-α) * spatial_coupling + α * attention_coupling
    - 注意力机制 (S33-S36): 使用Transformer风格的注意力计算
    
    历史窗口: 时变，累积所有历史信息 (t=0,1,...,T)
    """
    
    def __init__(self, 
                 n_oscillators: int,
                 d_model: int = 64,
                 spatial_network: torch.Tensor = None,
                 attention_network: torch.Tensor = None,
                 natural_frequencies: torch.Tensor = None,
                 coupling_strength: float = 1.5,
                 noise_strength: float = 0.1,
                 learnable_alpha: bool = True,
                 alpha_init: float = 0.5,
                 learnable_w_qk: bool = True,
                 w_v_identity: bool = True):
        """
        Args:
            n_oscillators: 振子数量 N
            d_model: 特征维度 d
            spatial_network: 空间耦合网络 A (N x N 邻接矩阵)
            attention_network: 注意力网络 Â (N x N 邻接矩阵)
            natural_frequencies: 自然频率 ω (N,)
            coupling_strength: 耦合强度 λ
            noise_strength: 噪声强度 D
            learnable_alpha: 是否学习混合系数 α
            alpha_init: α 的初始值
            learnable_w_qk: 是否学习 W^Q 和 W^K
            w_v_identity: W^V 是否使用单位矩阵
        """
        super().__init__()
        
        self.N = n_oscillators
        self.d = d_model
        self.lambda_coupling = coupling_strength
        self.D = noise_strength
        
        # 注册网络拓扑 (固定参数)
        if spatial_network is None:
            spatial_network = torch.eye(n_oscillators)
        self.register_buffer('A', spatial_network.float())
        
        if attention_network is None:
            attention_network = torch.eye(n_oscillators)
        self.register_buffer('A_hat', attention_network.float())
        
        # 计算度
        self.d_i = self.A.sum(dim=1, keepdim=True)  # (N, 1)
        self.d_hat_i = self.A_hat.sum(dim=1, keepdim=True)  # (N, 1)
        
        # 注册自然频率 (固定参数)
        if natural_frequencies is None:
            natural_frequencies = torch.zeros(n_oscillators)
        self.register_buffer('omega', natural_frequencies.float())
        
        # 可学习参数: 混合系数 α
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        else:
            self.register_buffer('alpha', torch.tensor(alpha_init))
        
        # 可学习参数: 注意力权重矩阵 W^Q, W^K
        if learnable_w_qk:
            # W^Q, W^K: (N, d)
            self.W_Q = nn.Parameter(torch.randn(n_oscillators, d_model) * 0.01)
            self.W_K = nn.Parameter(torch.randn(n_oscillators, d_model) * 0.01)
        else:
            self.register_buffer('W_Q', torch.randn(n_oscillators, d_model) * 0.01)
            self.register_buffer('W_K', torch.randn(n_oscillators, d_model) * 0.01)
        
        # W^V: 通常设为单位矩阵 (N, N)
        if w_v_identity:
            self.register_buffer('W_V', torch.eye(n_oscillators))
        else:
            self.W_V = nn.Parameter(torch.eye(n_oscillators))
        
        # 用于存储历史相位 - 使用列表实现时变窗口（累积所有历史）
        self.phase_history_list = []
        self.current_time = 0
        
    def reset_history(self):
        """重置历史相位缓存"""
        self.phase_history_list = []
        self.current_time = 0
        
    def update_history(self, phases_complex):
        """
        更新相位历史缓存 - 累积所有历史信息（论文核心创新点）
        Args:
            phases_complex: (N,) 复数相位 e^{iθ}
        """
        # 使用 detach() 避免保留计算图历史
        self.phase_history_list.append(phases_complex.detach().clone())
        self.current_time += 1
        
    def compute_attention(self):
        """
        计算注意力输出 M_T (公式 S33-S36)
        使用所有历史信息（时变窗口）
        
        Returns:
            M_T: (N,) 注意力输出 (复数)
        """
        T = len(self.phase_history_list)
        
        if T == 0:
            # 如果没有历史，返回零
            return torch.zeros(self.N, dtype=torch.cfloat, device=self.A.device)
        
        if T == 1:
            # 只有一步历史，直接返回
            return self.phase_history_list[0]
        
        # 将所有历史堆叠成矩阵 Theta_T: (T, N) 复数
        Theta_T = torch.stack(self.phase_history_list, dim=0)  # (T, N)
        
        # 线性投影 (S33)
        # Q, K: (T, d) 复数; V: (T, N) 复数
        W_Q_complex = self.W_Q.to(Theta_T.dtype)
        W_K_complex = self.W_K.to(Theta_T.dtype)
        W_V_complex = self.W_V.to(Theta_T.dtype)
        
        Q = Theta_T @ W_Q_complex  # (T, d)
        K = Theta_T @ W_K_complex  # (T, d)
        V = Theta_T @ W_V_complex  # (T, N)
        
        # 计算注意力矩阵 (S34)
        # |Q K^T|: (T, d) @ (d, T) = (T, T)
        scores = torch.abs(Q @ K.T) / np.sqrt(self.d)  # (T, T)
        
        # Softmax 按行 (S34)
        C = F.softmax(scores, dim=-1)  # (T, T)
        
        # 只取最后一行作为当前时刻的注意力权重 (S36)
        C_T = C[-1]  # (T,)
        
        # 计算注意力输出 (S35-S36)
        # M_T = sum_k C_Tk * v_k
        M_T = torch.sum(C_T.unsqueeze(-1) * V, dim=0)  # (N,) 复数
        
        return M_T
    
    def compute_spatial_coupling(self, phases_complex):
        """
        计算传统空间耦合项
        Args:
            phases_complex: (N,) 复数相位
        Returns:
            spatial_term: (N,) 空间耦合信息
        """
        # 将A和d_i转换为与phases_complex相同的设备和类型
        A_complex = self.A.to(phases_complex.dtype).to(phases_complex.device)
        d_i = self.d_i.to(phases_complex.device)
        spatial_term = (A_complex @ phases_complex) / d_i.squeeze(-1)
        return spatial_term
    
    def compute_attention_coupling(self, M_T):
        """
        计算注意力耦合项
        Args:
            M_T: (N,) 注意力输出
        Returns:
            attention_term: (N,) 注意力耦合信息
        """
        # 将A_hat和d_hat_i转换为与M_T相同的设备和类型
        A_hat_complex = self.A_hat.to(M_T.dtype).to(M_T.device)
        d_hat_i = self.d_hat_i.to(M_T.device)
        attention_term = (A_hat_complex @ M_T) / d_hat_i.squeeze(-1)
        return attention_term
    
    def forward_step(self, phases, noise=None):
        """
        单步前向传播 (公式 S30-S31)
        
        Args:
            phases: (N,) 当前相位 θ_T (弧度)
            noise: (N,) 噪声项 (可选)
        Returns:
            new_phases: (N,) 更新后的相位 θ_{T+1}
            order_param: 标量 序参量 R
        """
        # 设置模拟时间间隔 dt
        dt = 0.05

        # 转换为复数表示
        phases_complex = torch.exp(1j * phases)  # e^{iθ}
        
        # 更新历史
        self.update_history(phases_complex)
        
        # 计算空间耦合
        spatial_term = self.compute_spatial_coupling(phases_complex)
        
        # 计算注意力耦合
        M_T = self.compute_attention()
        attention_term = self.compute_attention_coupling(M_T)
        
        # 总信息 I_T (S31)
        # 使用 sigmoid 约束 alpha 在 [0, 1]
        alpha = torch.sigmoid(self.alpha)
        I_T = (1 - alpha) * spatial_term + alpha * attention_term  # (N,) 复数
        
        # 相位更新 (S30)
        # θ_T+1 = θ_T + ω + λ * Im[I_T * exp(-iθ_T)] + noise
        coupling_term = self.lambda_coupling * (I_T * phases_complex.conj()).imag
        
        # 添加噪声
        if noise is None:
            noise = torch.randn_like(phases) * np.sqrt(2 * self.D * dt)
        
        new_phases = phases + (self.omega + coupling_term) * dt + noise
        
        # 计算序参量 R
        order_param = torch.abs(phases_complex.mean())
        
        return new_phases, order_param
    
    def forward(self, initial_phases, n_steps, return_trajectory=False):
        """
        多步前向模拟
        
        Args:
            initial_phases: (N,) 初始相位
            n_steps: 模拟步数
            return_trajectory: 是否返回完整轨迹
        Returns:
            final_phases: (N,) 最终相位
            order_params: (n_steps,) 序参量序列
            phases_trajectory: (n_steps, N) 相位轨迹 (如果return_trajectory=True)
        """
        self.reset_history()
        
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


class SimplifiedDTA(nn.Module):
    """
    简化的DTA模型，直接使用衰减记忆机制
    对应于论文中的指数衰减自相互作用 (S24-S29)
    """
    
    def __init__(self, 
                 n_oscillators: int,
                 decay_rate: float = 0.01,
                 learnable_decay: bool = False):
        super().__init__()
        
        self.N = n_oscillators
        
        if learnable_decay:
            self.beta = nn.Parameter(torch.tensor(decay_rate))
        else:
            self.register_buffer('beta', torch.tensor(decay_rate))
        
        # 记忆状态 M_t (S29)
        self.register_buffer('M_t', torch.zeros(n_oscillators, dtype=torch.cfloat))
        
    def reset_memory(self):
        """重置记忆状态"""
        self.M_t.zero_()
    
    def forward(self, phases_complex):
        """
        更新记忆并返回注意力信息
        Args:
            phases_complex: (N,) 复数相位
        Returns:
            M_t: (N,) 记忆状态
        """
        # dM_t = β * (e^{iθ_t} - M_t) * dt
        beta = torch.abs(self.beta)
        self.M_t = self.M_t + beta * (phases_complex - self.M_t)
        return self.M_t
