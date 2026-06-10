"""
评估指标计算工具
"""

import torch
import numpy as np


def compute_order_parameter(phases):
    """
    计算序参量 R (Order Parameter)
    R = |<e^{iθ}>| ∈ [0, 1]
    R = 1: 完全同步
    R = 0: 完全去同步 (均匀分布)
    
    Args:
        phases: (N,) 或 (batch, N) 相位 (弧度)
    
    Returns:
        R: 标量或 (batch,) 序参量
    """
    if isinstance(phases, np.ndarray):
        phases = torch.from_numpy(phases)
    
    # e^{iθ}
    complex_phases = torch.exp(1j * phases)
    
    # 计算平均值
    if phases.dim() == 1:
        mean_complex = complex_phases.mean()
        R = torch.abs(mean_complex)
    else:
        mean_complex = complex_phases.mean(dim=-1)
        R = torch.abs(mean_complex)
    
    return R


def compute_phase_coherence(phases):
    """
    计算相位相干性 (与序参量相同)
    
    Args:
        phases: (N,) 相位
    
    Returns:
        coherence: 标量 相干性
    """
    return compute_order_parameter(phases)


def compute_synchronizability(adjacency, coupling_strength=None):
    """
    估计网络的同步能力
    基于拉普拉斯矩阵的特征值
    
    Args:
        adjacency: (N, N) 邻接矩阵
        coupling_strength: 耦合强度 λ (可选)
    
    Returns:
        synchronizability: 同步能力指标
        lambda_c: 估计的临界耦合强度
    """
    if isinstance(adjacency, torch.Tensor):
        adjacency = adjacency.numpy()
    
    N = adjacency.shape[0]
    
    # 计算度矩阵
    degrees = adjacency.sum(axis=1)
    D = np.diag(degrees)
    
    # 拉普拉斯矩阵 L = D - A
    L = D - adjacency
    
    # 计算特征值
    eigenvalues = np.linalg.eigvalsh(L)
    
    # 同步能力指标: 非零最小特征值与最大特征值的比
    # λ_2 / λ_N (Fiedler值与最大特征值的比)
    nonzero_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(nonzero_eigenvalues) > 0:
        lambda_2 = nonzero_eigenvalues.min()
        lambda_N = eigenvalues.max()
        sync_metric = lambda_2 / lambda_N if lambda_N > 0 else 0
    else:
        sync_metric = 0
        lambda_2 = 0
    
    # 估计临界耦合强度 (基于Kuramoto模型)
    # λ_c ≈ 2D / (π * g(0)) 对于全连接网络
    # 这里使用简化估计
    if coupling_strength is not None:
        lambda_c = 2 * coupling_strength / (lambda_2 + 1e-10)
    else:
        lambda_c = 2.0 / (lambda_2 + 1e-10)
    
    return sync_metric, lambda_c


def compute_phase_difference(phases):
    """
    计算相位差异矩阵
    
    Args:
        phases: (N,) 相位
    
    Returns:
        diff_matrix: (N, N) 相位差异矩阵
    """
    if isinstance(phases, torch.Tensor):
        phases = phases.cpu().numpy()
    
    N = len(phases)
    diff_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            diff = np.abs(phases[i] - phases[j])
            # 考虑周期性
            diff = min(diff, 2 * np.pi - diff)
            diff_matrix[i, j] = diff
    
    return diff_matrix


def compute_synchronization_error(phases, target_phases=None):
    """
    计算同步误差
    
    Args:
        phases: (N,) 当前相位
        target_phases: (N,) 目标相位 (None 表示完全同步)
    
    Returns:
        error: 同步误差
    """
    if target_phases is None:
        # 目标是完全同步，使用序参量的补数
        R = compute_order_parameter(phases)
        return 1 - R
    else:
        # 计算与目标相位的差异
        if isinstance(phases, torch.Tensor):
            phases = phases.cpu().numpy()
        if isinstance(target_phases, torch.Tensor):
            target_phases = target_phases.cpu().numpy()
        
        diff = np.abs(phases - target_phases)
        diff = np.minimum(diff, 2 * np.pi - diff)
        return diff.mean()


def compute_entrainment(phases, target_phase=0.0):
    """
    计算相位锁定程度 (与特定相位的偏差)
    
    Args:
        phases: (N,) 相位
        target_phase: 目标相位
    
    Returns:
        entrainment: 锁定程度 [0, 1]
    """
    if isinstance(phases, torch.Tensor):
        phases = phases.cpu().numpy()
    
    # 计算每个相位与目标相位的差异
    diff = np.abs(phases - target_phase)
    diff = np.minimum(diff, 2 * np.pi - diff)
    
    # 转换为余弦相似度
    similarity = np.cos(diff)
    
    return similarity.mean()


def evaluate_control_performance(order_params_history, task='sync', threshold=0.9):
    """
    评估控制性能
    
    Args:
        order_params_history: (T,) 序参量历史
        task: 'sync' 或 'desync'
        threshold: 成功阈值
    
    Returns:
        metrics: 字典包含各项性能指标
    """
    if isinstance(order_params_history, torch.Tensor):
        order_params_history = order_params_history.detach().cpu().numpy()
    
    metrics = {}
    
    # 最终序参量
    final_R = order_params_history[-1]
    metrics['final_order_param'] = float(final_R)
    
    # 平均序参量
    mean_R = order_params_history.mean()
    metrics['mean_order_param'] = float(mean_R)
    
    # 收敛时间 (达到阈值的步数)
    if task == 'sync':
        converged = order_params_history >= threshold
        if converged.any():
            convergence_time = np.where(converged)[0][0]
        else:
            convergence_time = len(order_params_history)
        metrics['convergence_time'] = int(convergence_time)
        metrics['success'] = bool(final_R >= threshold)
    else:  # desync
        converged = order_params_history <= (1 - threshold)
        if converged.any():
            convergence_time = np.where(converged)[0][0]
        else:
            convergence_time = len(order_params_history)
        metrics['convergence_time'] = int(convergence_time)
        metrics['success'] = bool(final_R <= (1 - threshold))
    
    # 序参量的标准差 (稳定性)
    metrics['stability'] = float(order_params_history[-100:].std()) if len(order_params_history) >= 100 else float(order_params_history.std())
    
    return metrics
