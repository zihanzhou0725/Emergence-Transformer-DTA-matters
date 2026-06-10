"""
对照实验: Natural Memory 模型 vs 纯传统耦合 (α=0)
对比指数衰减记忆模型与传统耦合的差异
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.sync_transformer import SynchronizationTransformer
from utils.networks import generate_watts_strogatz, generate_fully_connected


class NaturalMemoryModel(nn.Module):
    """
    自然记忆模型 (Natural Memory Model)
    基于指数衰减记忆核: dM/dt = beta * (e^(i*theta) - M)
    总信息: I = (1-alpha) * spatial_coupling + alpha * memory_coupling
    
    参数:
        alpha: 记忆占比 (0-1)
        beta: 指数衰减率
    """
    
    def __init__(self, n_oscillators, spatial_network, attention_network, natural_frequencies,
                 coupling_strength=1.5, noise_strength=0.05,
                 alpha=0.5, beta=0.1, device='cpu'):
        super().__init__()
        
        self.N = n_oscillators
        self.device = device
        self.lambda_coupling = coupling_strength
        self.D = noise_strength
        self.alpha = alpha  # 记忆占比
        self.beta = beta    # 衰减率
        self.dt = 0.05      # 时间步长
        
        # 注册网络参数
        self.register_buffer('A', spatial_network.float())
        self.register_buffer('A_hat', attention_network.float())  # 记忆耦合网络
        self.register_buffer('omega', natural_frequencies.float())
        
        # 计算度
        self.d_i = self.A.sum(dim=1, keepdim=True)
        self.d_hat_i = self.A_hat.sum(dim=1, keepdim=True)
        
        # 初始化记忆状态
        self.reset_memory()
        
    def reset_memory(self):
        """重置记忆状态 M_t"""
        self.M_t = torch.zeros(self.N, dtype=torch.cfloat, device=self.device)
        
    def compute_spatial_coupling(self, phases_complex):
        """计算传统空间耦合项"""
        A_complex = self.A.to(phases_complex.dtype).to(phases_complex.device)
        d_i = self.d_i.to(phases_complex.device)
        spatial_term = (A_complex @ phases_complex) / d_i.squeeze(-1)
        return spatial_term
    
    def compute_memory_coupling(self, M_t):
        """
        计算记忆耦合项
        memory_coupling = A_hat @ M_t / d_hat_i
        """
        A_hat_complex = self.A_hat.to(M_t.dtype).to(M_t.device)
        d_hat_i = self.d_hat_i.to(M_t.device)
        memory_term = (A_hat_complex @ M_t) / d_hat_i.squeeze(-1)
        return memory_term
    
    def update_memory(self, phases_complex):
        """
        更新记忆状态: dM/dt = beta * (e^(i*theta) - M)
        离散化: M_{t+1} = M_t + beta * dt * (e^(i*theta_t) - M_t)
        """
        self.M_t = self.M_t + self.beta * self.dt * (phases_complex - self.M_t)
        
    def forward_step(self, phases, noise=None):
        """单步前向传播"""
        phases_complex = torch.exp(1j * phases)
        
        # 更新记忆状态
        self.update_memory(phases_complex)
        
        # 计算空间耦合
        spatial_term = self.compute_spatial_coupling(phases_complex)
        
        # 计算记忆耦合
        memory_term = self.compute_memory_coupling(self.M_t)
        
        # 总信息 I = (1-alpha) * spatial + alpha * memory
        I_t = (1 - self.alpha) * spatial_term + self.alpha * memory_term
        
        # 相位更新
        coupling_term = self.lambda_coupling * (I_t * phases_complex.conj()).imag
        
        if noise is None:
            noise = torch.randn_like(phases) * np.sqrt(2 * self.D * self.dt)
        
        new_phases = phases + (self.omega + coupling_term) * self.dt + noise
        order_param = torch.abs(phases_complex.mean())
        
        return new_phases, order_param
    
    def forward(self, initial_phases, n_steps, return_trajectory=False):
        """多步前向模拟"""
        self.reset_memory()
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


def load_model_for_comparison(checkpoint_path, device='cpu'):
    """
    从 checkpoint 加载模型信息
    """
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取模型结构参数
    if 'model_state_dict' in checkpoint and 'W_Q' in checkpoint['model_state_dict']:
        d_model = checkpoint['model_state_dict']['W_Q'].shape[1]
        print(f"[OK] D_MODEL: {d_model}")
    else:
        d_model = 64
        print(f"[WARNING] 使用默认 D_MODEL: {d_model}")
    
    # 获取基本参数
    N = checkpoint.get('spatial_network', torch.eye(100)).shape[0]
    task = checkpoint.get('task', 'sync')
    coupling_strength = checkpoint.get('coupling_strength', 1.5)
    noise_strength = checkpoint.get('noise_strength', 0.05)
    attention_type = checkpoint.get('attention_type', 'neighbor')
    
    # 从 checkpoint 加载网络
    if 'spatial_network' in checkpoint and checkpoint['spatial_network'] is not None:
        spatial_network = checkpoint['spatial_network'].to(device)
        natural_frequencies = checkpoint['natural_frequencies'].to(device)
        
        # 设置注意力网络
        if attention_type == 'neighbor':
            attention_network = spatial_network.clone()
            print(f"[OK] 网络类型: neighbor (Â=A)")
        else:
            attention_network = torch.eye(N, device=device)
            print(f"[OK] 网络类型: self (Â=I)")
    else:
        print("[WARNING] 使用默认网络结构")
        spatial_network = generate_watts_strogatz(N, 4, 0.1, seed=42)
        attention_network = spatial_network.clone()
        natural_frequencies = torch.randn(N) * 0.1
    
    print(f"[INFO] 振子数量 N: {N}")
    print(f"[INFO] 任务类型: {task}")
    print(f"[INFO] 耦合强度 λ: {coupling_strength}")
    print(f"[INFO] 噪声强度 D: {noise_strength}")
    
    return {
        'N': N,
        'd_model': d_model,
        'spatial_network': spatial_network,
        'attention_network': attention_network,
        'natural_frequencies': natural_frequencies,
        'coupling_strength': coupling_strength,
        'noise_strength': noise_strength,
        'task': task,
        'device': device
    }


def create_memory_model(model_info, memory_alpha, memory_beta, device='cpu',
                        coupling_strength=None, noise_strength=None):
    """
    创建自然记忆模型（alpha=0 时即为传统耦合）
    """
    coupling_strength = model_info['coupling_strength'] if coupling_strength is None else coupling_strength
    noise_strength = model_info['noise_strength'] if noise_strength is None else noise_strength

    model = NaturalMemoryModel(
        n_oscillators=model_info['N'],
        spatial_network=model_info['spatial_network'],
        attention_network=model_info['attention_network'],
        natural_frequencies=model_info['natural_frequencies'],
        coupling_strength=coupling_strength,
        noise_strength=noise_strength,
        alpha=memory_alpha,
        beta=memory_beta,
        device=device
    ).to(device)
    model.eval()
    
    if memory_alpha == 0:
        print(f"\n[创建传统耦合 Baseline]")
        print(f"  Alpha: {memory_alpha} (纯空间耦合)")
    else:
        print(f"\n[创建自然记忆模型]")
        print(f"  Alpha: {memory_alpha}, Beta: {memory_beta}")
    
    print(f"  网络类型: {'neighbor (Â=A)' if torch.allclose(model_info['attention_network'], model_info['spatial_network']) else 'self (Â=I)'}")
    print(f"  耦合强度 λ: {coupling_strength}")
    print(f"  噪声强度 D: {noise_strength}")
    
    return model


def generate_trajectory(model, initial_phases, n_steps, device='cpu', return_trajectory=False):
    """为模型生成轨迹（适用于 Memory Model）"""
    with torch.no_grad():
        model.reset_memory()
        initial_phases = initial_phases.to(device)
        result = model(initial_phases, n_steps, return_trajectory=return_trajectory)

    if return_trajectory:
        final_phases, order_params, trajectory = result
    else:
        final_phases, order_params = result
        trajectory = None

    return final_phases, order_params.cpu().numpy(), trajectory


def plot_comparison(order_params_traditional, order_params_memory,
                   initial_phases, task, save_path, case_id,
                   memory_alpha, memory_beta):
    """绘制对比图：传统耦合 vs 自然记忆模型（每隔100个点画一个点）"""
    plt.figure(figsize=(12, 7))
    
    initial_R = order_params_traditional[0]
    n_points = len(order_params_traditional)
    
    # 每隔100个点取一个点
    step = 100
    indices = list(range(0, n_points, step))
    if indices[-1] != n_points - 1:
        indices.append(n_points - 1)  # 确保包含最后一个点
    
    traditional_sampled = order_params_traditional[indices]
    memory_sampled = order_params_memory[indices]
    
    # 绘制两条曲线（每隔100个点画一个点）
    plt.plot(indices, traditional_sampled, linewidth=2.5, color='red',
            label='Traditional Coupling (α=0)', linestyle='-')
    plt.plot(indices, memory_sampled, linewidth=2.5, color='green',
            label=f'Natural Memory (α={memory_alpha}, β={memory_beta})', linestyle='-')
    
    # 参考线
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='Perfect Sync (R=1)')
    plt.axhline(y=0.0, color='gray', linestyle='--', alpha=0.3, linewidth=1.5)
    
    # 填充优势区域（用完整数据）
    plt.fill_between(range(n_points),
                    order_params_memory, order_params_traditional,
                    alpha=0.2, color='purple', 
                    label=f'Memory Advantage (ΔR={order_params_memory[-1] - order_params_traditional[-1]:.3f})')
    
    # 设置标签和标题
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Order Parameter R', fontsize=14)
    plt.title(f'Case {case_id}: Natural Memory vs Traditional Coupling\n'
             f'Task: {task.upper()}, Initial R={initial_R:.3f}\n'
             f'Final R: Traditional={order_params_traditional[-1]:.3f} | Memory={order_params_memory[-1]:.3f}',
             fontsize=13)
    
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.05, 1.05])
    plt.xlim([0, n_points-1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存: {save_path}")


def plot_average_comparison(all_traditional, all_memory, task, save_path,
                           memory_alpha, memory_beta):
    """绘制平均对比图（每隔100个点画一个点）"""
    mean_traditional = np.mean(all_traditional, axis=0)
    std_traditional = np.std(all_traditional, axis=0)
    mean_memory = np.mean(all_memory, axis=0)
    std_memory = np.std(all_memory, axis=0)
    
    plt.figure(figsize=(13, 7))
    
    n_points = len(mean_traditional)
    
    # 每隔100个点取一个点
    step = 100
    indices = list(range(0, n_points, step))
    if indices[-1] != n_points - 1:
        indices.append(n_points - 1)
    
    mean_traditional_sampled = mean_traditional[indices]
    mean_memory_sampled = mean_memory[indices]
    std_traditional_sampled = std_traditional[indices]
    std_memory_sampled = std_memory[indices]
    
    # 绘制平均曲线（每隔100个点画一个点）
    plt.plot(indices, mean_traditional_sampled, linewidth=3, color='red',
            label=f'Traditional Coupling (α=0) - Mean')
    plt.plot(indices, mean_memory_sampled, linewidth=3, color='green',
            label=f'Natural Memory (α={memory_alpha}, β={memory_beta}) - Mean')
    
    # 绘制标准差阴影（每隔100个点）
    plt.fill_between(indices,
                    mean_traditional_sampled - std_traditional_sampled, 
                    mean_traditional_sampled + std_traditional_sampled,
                    alpha=0.15, color='red')
    plt.fill_between(indices,
                    mean_memory_sampled - std_memory_sampled, 
                    mean_memory_sampled + std_memory_sampled,
                    alpha=0.15, color='green')
    
    # 参考线
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='Perfect Sync (R=1)')
    plt.axhline(y=0.0, color='gray', linestyle='--', alpha=0.3, linewidth=1.5)
    
    # 填充优势区域（用完整数据）
    plt.fill_between(range(n_points),
                    mean_memory, mean_traditional,
                    alpha=0.2, color='purple', 
                    label=f'Memory Advantage (ΔR={mean_memory[-1] - mean_traditional[-1]:.3f})')
    
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Order Parameter R', fontsize=14)
    plt.title(f'Average Comparison: Natural Memory vs Traditional Coupling\n'
             f'Task: {task.upper()}, {len(all_traditional)} Random Initial Conditions\n'
             f'Final R: Traditional={mean_traditional[-1]:.3f} | Memory={mean_memory[-1]:.3f}',
             fontsize=13)
    
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.05, 1.05])
    plt.xlim([0, n_points-1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"平均对比图已保存: {save_path}")


def save_matlab_data_sampled(all_traditional, all_memory, task, save_dir, memory_alpha, memory_beta):
    """
    保存采样后的数据为 MATLAB .mat 文件（每100个点采样，节省空间）
    
    变量说明:
    - timestep: 采样后的时间步序列
    - mean_traditional: 传统模型的平均序参量（采样后）
    - std_traditional: 传统模型的标准差（采样后）
    - mean_memory: 记忆模型的平均序参量（采样后）
    - std_memory: 记忆模型的标准差（采样后）
    - traditional_upper/lower: 传统模型上下界
    - memory_upper/lower: 记忆模型上下界
    """
    mean_traditional = np.mean(all_traditional, axis=0)
    std_traditional = np.std(all_traditional, axis=0)
    mean_memory = np.mean(all_memory, axis=0)
    std_memory = np.std(all_memory, axis=0)
    
    # 采样（每100个点）
    n_points = len(mean_traditional)
    step = 100
    indices = list(range(0, n_points, step))
    if indices[-1] != n_points - 1:
        indices.append(n_points - 1)
    
    timestep = np.array(indices)
    mean_traditional_sampled = mean_traditional[indices]
    std_traditional_sampled = std_traditional[indices]
    mean_memory_sampled = mean_memory[indices]
    std_memory_sampled = std_memory[indices]
    
    # 计算上下界
    traditional_upper = mean_traditional_sampled + std_traditional_sampled
    traditional_lower = mean_traditional_sampled - std_traditional_sampled
    memory_upper = mean_memory_sampled + std_memory_sampled
    memory_lower = mean_memory_sampled - std_memory_sampled
    
    # 构建数据字典（只保存采样后的数据）
    mat_data = {
        'timestep': timestep,
        'mean_traditional': mean_traditional_sampled,
        'std_traditional': std_traditional_sampled,
        'mean_memory': mean_memory_sampled,
        'std_memory': std_memory_sampled,
        'traditional_upper': traditional_upper,
        'traditional_lower': traditional_lower,
        'memory_upper': memory_upper,
        'memory_lower': memory_lower,
        'memory_alpha': memory_alpha,
        'memory_beta': memory_beta,
        'n_trials': len(all_traditional),
        'task': task,
        'sampling_info': 'Every 100 points sampled to reduce file size'
    }
    
    # 构建文件名
    filename = f"memory_compare_{task}_alpha{memory_alpha}_beta{memory_beta}.mat"
    filepath = os.path.join(save_dir, filename)
    
    # 保存为.mat文件
    savemat(filepath, mat_data)
    print(f"\n[OK] MATLAB数据已保存: {filepath}")
    print(f"    采样点数: {len(timestep)} / {n_points} (每100点采样)")
    print(f"    包含变量: timestep, mean_traditional, std_traditional, mean_memory, std_memory")
    print(f"    以及上下界: traditional_upper/lower, memory_upper/lower")
    
    return filepath


def parse_args():
    parser = argparse.ArgumentParser(description='Natural Memory vs 传统耦合对照实验')
    parser.add_argument('--checkpoint', type=str, default='results/sync_self_ws/final_model.pt',
                       help='用于读取网络、频率和任务信息的 checkpoint')
    parser.add_argument('--memory_alpha', type=float, default=0.5,
                       help='记忆占比 alpha')
    parser.add_argument('--memory_beta', type=float, default=0.01,
                       help='指数衰减率 beta')
    parser.add_argument('--n_steps', type=int, default=100000,
                       help='模拟步数')
    parser.add_argument('--n_cases', type=int, default=100,
                       help='测试案例数')
    parser.add_argument('--coupling_strength', type=float, default=None,
                       help='覆盖 checkpoint 中的耦合强度 λ')
    parser.add_argument('--noise_strength', type=float, default=None,
                       help='覆盖 checkpoint 中的噪声强度 D')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'auto'],
                       help='运行设备，默认 CPU 以避免显存不足')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='结果保存目录；默认根据 alpha/beta 自动生成')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA 不可用，改用 CPU")
        device = 'cpu'

    print(f"[INFO] 使用 {device.upper()} 运行")

    CHECKPOINT_PATH = args.checkpoint
    MEMORY_ALPHA = args.memory_alpha
    MEMORY_BETA = args.memory_beta
    N_STEPS = args.n_steps
    N_CASES = args.n_cases
    
    # 检查 checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"错误: 找不到模型文件 {CHECKPOINT_PATH}")
        return
    
    # 加载模型信息
    print("\n" + "="*70)
    print("加载模型信息")
    print("="*70)
    model_info = load_model_for_comparison(CHECKPOINT_PATH, device)
    
    # 创建传统耦合 Baseline (alpha=0, beta 任意)
    traditional_model = create_memory_model(
        model_info,
        memory_alpha=0,
        memory_beta=0,
        device=device,
        coupling_strength=args.coupling_strength,
        noise_strength=args.noise_strength
    )
    
    # 创建自然记忆模型 (alpha>0)
    memory_model = create_memory_model(
        model_info,
        MEMORY_ALPHA,
        MEMORY_BETA,
        device,
        coupling_strength=args.coupling_strength,
        noise_strength=args.noise_strength
    )
    
    # 创建保存目录
    save_dir = args.save_dir or f'results/memory_comparison_alpha{MEMORY_ALPHA}_beta{MEMORY_BETA}'
    os.makedirs(save_dir, exist_ok=True)
    
    N = model_info['N']
    
    print("\n" + "="*70)
    print("开始对照实验")
    print("="*70)
    print(f"模拟步数: {N_STEPS}")
    print(f"测试案例数: {N_CASES}")
    print(f"振子数量: {N}")
    
    all_traditional = []
    all_memory = []
    
    for i in range(N_CASES):
        # 相同的随机初始条件
        initial_phases = torch.rand(N) * 2 * np.pi
        
        # 运行传统耦合模型
        final_traditional, order_params_traditional, traj_traditional = generate_trajectory(
            traditional_model, initial_phases, N_STEPS, device
        )
        
        # 运行自然记忆模型（相同初始条件）
        final_memory, order_params_memory, traj_memory = generate_trajectory(
            memory_model, initial_phases, N_STEPS, device
        )
        
        # 保存结果用于平均
        all_traditional.append(order_params_traditional)
        all_memory.append(order_params_memory)
        
        # 打印结果
        print(f"\n案例 {i+1}:")
        print(f"  初始 R: {order_params_traditional[0]:.4f}")
        print(f"  传统耦合最终 R: {order_params_traditional[-1]:.4f}")
        print(f"  自然记忆最终 R: {order_params_memory[-1]:.4f}")
        print(f"  记忆优势 (ΔR): {order_params_memory[-1] - order_params_traditional[-1]:+.4f}")
        
        # 绘制对比图
        save_path = os.path.join(save_dir, f'comparison_case_{i+1}.png')
        plot_comparison(order_params_traditional, order_params_memory,
                       initial_phases, model_info['task'],
                       save_path, i+1, MEMORY_ALPHA, MEMORY_BETA)
    
    # 绘制平均对比图
    print("\n生成平均对比图...")
    plot_average_comparison(all_traditional, all_memory, model_info['task'],
                           os.path.join(save_dir, 'average_comparison.png'),
                           MEMORY_ALPHA, MEMORY_BETA)
    
    # 统计汇总
    print("\n" + "="*70)
    print("对照实验统计汇总")
    print("="*70)
    
    final_traditional_all = [traj[-1] for traj in all_traditional]
    final_memory_all = [traj[-1] for traj in all_memory]
    improvements = [m - t for m, t in zip(final_memory_all, final_traditional_all)]
    
    print(f"记忆模型参数: α={MEMORY_ALPHA}, β={MEMORY_BETA}")
    print("-"*70)
    print(f"传统耦合平均最终 R: {np.mean(final_traditional_all):.4f} ± {np.std(final_traditional_all):.4f}")
    print(f"自然记忆平均最终 R: {np.mean(final_memory_all):.4f} ± {np.std(final_memory_all):.4f}")
    print(f"平均改进 (ΔR): {np.mean(improvements):+.4f} ± {np.std(improvements):.4f}")
    if np.mean(final_traditional_all) > 0.01:
        print(f"相对提升: {(np.mean(improvements) / np.mean(final_traditional_all) * 100):+.1f}%")
    print("="*70)
    
    print(f"\n所有结果保存在: {save_dir}/")
    print("\n提示：")
    print(f"  - 修改 MEMORY_ALPHA 和 MEMORY_BETA 可调整记忆模型参数")
    print(f"  - 当前参数: α={MEMORY_ALPHA}, β={MEMORY_BETA}")
    
    # 保存 MATLAB 数据文件（采样后，节省空间）
    print("\n" + "="*70)
    print("保存MATLAB数据文件（采样后）...")
    print("="*70)
    mat_filepath = save_matlab_data_sampled(
        all_traditional, all_memory, 
        model_info['task'], save_dir,
        MEMORY_ALPHA, MEMORY_BETA
    )
    print("="*70)


if __name__ == '__main__':
    main()
