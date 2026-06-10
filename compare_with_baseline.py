"""
对照实验: 训练好的DTA模型 vs 纯传统耦合 (α=0)
在同一张图上对比，凸显DTA的优势
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决OpenMP警告

import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.io import savemat  # 用于保存MATLAB格式文件

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.sync_transformer import SynchronizationTransformer
from utils.networks import generate_watts_strogatz, generate_fully_connected


def load_trained_model(checkpoint_path, device='cpu'):
    """加载训练好的模型"""
    print(f"加载训练好的模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 从checkpoint加载网络结构和自然频率（确保与训练时相同）
    if 'spatial_network' in checkpoint and checkpoint['spatial_network'] is not None:
        spatial_network = checkpoint['spatial_network'].to(device)
        natural_frequencies = checkpoint['natural_frequencies'].to(device)
        print("[OK] 从checkpoint加载训练时的网络结构和自然频率")
        
        # 加载 ATTENTION_TYPE (如果保存了)
        if 'attention_type' in checkpoint:
            attention_type = checkpoint['attention_type']
            print(f"[OK] ATTENTION_TYPE: {attention_type}")
        else:
            # 兼容旧版本：根据任务推断
            attention_type = 'neighbor' if checkpoint.get('task') == 'sync' else 'self'
            print(f"[WARNING] 使用默认ATTENTION_TYPE: {attention_type}")
        
        # 加载 NETWORK_TYPE (如果保存了)
        if 'network_type' in checkpoint:
            network_type = checkpoint['network_type']
            print(f"[OK] NETWORK_TYPE: {network_type}")
        else:
            network_type = 'ws'  # 兼容旧版本默认WS网络
            print(f"[WARNING] 使用默认NETWORK_TYPE: {network_type}")
        
        # 加载 NATURAL_FREQ_STD (如果保存了)
        if 'natural_freq_std' in checkpoint:
            natural_freq_std = checkpoint['natural_freq_std']
            print(f"[OK] NATURAL_FREQ_STD: {natural_freq_std}")
        else:
            natural_freq_std = 0.1  # 兼容旧版本默认值
            print(f"[WARNING] 使用默认NATURAL_FREQ_STD: {natural_freq_std}")
        
        # 加载 COUPLING_STRENGTH (如果保存了)
        if 'coupling_strength' in checkpoint:
            coupling_strength = checkpoint['coupling_strength']
            print(f"[OK] COUPLING_STRENGTH: {coupling_strength}")
        else:
            coupling_strength = 1.5  # 兼容旧版本默认值
            print(f"[WARNING] 使用默认COUPLING_STRENGTH: {coupling_strength}")

        noise_strength = checkpoint.get('noise_strength', 0.05)
        print(f"[OK] NOISE_STRENGTH: {noise_strength}")
        
        # 根据 ATTENTION_TYPE 设置 attention_network
        if attention_type == 'neighbor':
            attention_network = spatial_network.clone()
        else:  # 'self'
            attention_network = torch.eye(spatial_network.shape[0], device=device)
    else:
        # 兼容旧版本
        print("[WARNING] 使用默认网络结构（可能与训练时不一致）")
        N = 20
        spatial_network = generate_watts_strogatz(N, k_neighbors=2, rewiring_prob=0.1, seed=42)
        attention_network = spatial_network.clone()
        natural_frequencies = torch.randn(N) * 0.1
        coupling_strength = 1.5  # 默认值
        noise_strength = 0.05
        attention_type = 'neighbor'  # 默认值
        network_type = 'ws'  # 默认值
        natural_freq_std = 0.1  # 默认值
    
    N = spatial_network.shape[0]
    task = checkpoint.get('task', 'sync')
    
    # 从权重推断 d_model
    if 'model_state_dict' in checkpoint and 'W_Q' in checkpoint['model_state_dict']:
        d_model = checkpoint['model_state_dict']['W_Q'].shape[1]
        print(f"[OK] D_MODEL: {d_model} (从 checkpoint 权重推断)")
    else:
        d_model = 64  # 默认值
        print(f"[WARNING] 使用默认 D_MODEL: {d_model}")
    
    # 创建模型（累积所有历史，论文核心创新点）
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
    print(f"学习到的 α: {alpha_learned:.4f}")
    print(f"耦合强度 λ: {model.lambda_coupling}")
    
    # 收集配置信息
    config_info = {
        'task': task,
        'attention_type': attention_type if 'attention_type' in locals() else 'neighbor',
        'network_type': network_type if 'network_type' in locals() else 'ws',
        'natural_freq_std': natural_freq_std if 'natural_freq_std' in locals() else 0.1
    }
    
    return model, task, alpha_learned, config_info


def create_baseline_model(trained_model, device='cpu'):
    """
    创建对照模型：与训练模型相同的参数，但α=0（纯传统耦合）
    """
    N = trained_model.N
    
    # 复制训练模型的参数，但设置α=0
    baseline_model = SynchronizationTransformer(
        n_oscillators=N,
        d_model=trained_model.d,
        spatial_network=trained_model.A.clone(),
        attention_network=trained_model.A_hat.clone(),
        natural_frequencies=trained_model.omega.clone(),
        coupling_strength=trained_model.lambda_coupling,
        noise_strength=trained_model.D,
        learnable_alpha=False,  # 不学习
        alpha_init=-10.0,       # 经过 sigmoid 后 ≈ 0，纯传统耦合
        learnable_w_qk=False,   # 不学习注意力矩阵
        w_v_identity=True
    )
    
    # 复制W^Q和W^K（虽然α=0时不会用到，但为了保持一致）
    baseline_model.W_Q.data = trained_model.W_Q.data.clone()
    baseline_model.W_K.data = trained_model.W_K.data.clone()
    
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    
    return baseline_model


def generate_trajectory(model, initial_phases, n_steps, device='cpu'):
    """生成轨迹"""
    with torch.no_grad():
        model.reset_history()
        initial_phases = initial_phases.to(device)
        final_phases, order_params, trajectory = model(
            initial_phases, n_steps, return_trajectory=True
        )
    
    # 清理 CUDA 缓存
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    print(f"[调试] generate_trajectory: n_steps={n_steps}, 返回 order_params 长度={len(order_params)}")
    return final_phases, order_params.cpu().numpy(), trajectory


def plot_comparison(order_params_dta, order_params_baseline, 
                   initial_phases, alpha_value, task,
                   save_path, case_id):
    """
    绘制对比图：DTA vs 纯传统耦合
    """
    print(f"[调试] plot_comparison Case {case_id}: DTA长度={len(order_params_dta)}, Baseline长度={len(order_params_baseline)}")
    
    plt.figure(figsize=(12, 7))
    
    # 计算初始R
    initial_R = order_params_dta[0]
    
    # 绘制两条曲线
    plt.plot(order_params_dta, linewidth=2.5, color='blue', 
            label=f'DTA Model (α={alpha_value:.3f})', marker='o', markersize=3, markevery=10)
    plt.plot(order_params_baseline, linewidth=2.5, color='red', 
            label='Traditional Coupling Only (α=0)', marker='s', markersize=3, markevery=10)
    
    # 参考线
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Perfect Sync (R=1)')
    plt.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Desync (R=0)')
    
    # 填充区域显示差距
    plt.fill_between(range(len(order_params_dta)), 
                    order_params_dta, order_params_baseline,
                    alpha=0.2, color='purple', 
                    label=f'DTA Advantage (ΔR={order_params_dta[-1] - order_params_baseline[-1]:.3f})')
    
    # 设置标签和标题
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
    
    print(f"对比图已保存: {save_path}")


def plot_average_comparison(all_dta, all_baseline, alpha_value, task, save_path):
    """绘制平均对比图"""
    mean_dta = np.mean(all_dta, axis=0)
    std_dta = np.std(all_dta, axis=0)
    mean_baseline = np.mean(all_baseline, axis=0)
    std_baseline = np.std(all_baseline, axis=0)
    
    plt.figure(figsize=(13, 7))
    
    # 绘制平均曲线
    plt.plot(mean_dta, linewidth=3, color='blue', 
            label=f'DTA Model (α={alpha_value:.3f}) - Mean of {len(all_dta)} trials')
    plt.plot(mean_baseline, linewidth=3, color='red', 
            label='Traditional Coupling (α=0) - Mean')
    
    # 绘制标准差阴影
    plt.fill_between(range(len(mean_dta)), 
                    mean_dta - std_dta, mean_dta + std_dta,
                    alpha=0.2, color='blue')
    plt.fill_between(range(len(mean_baseline)), 
                    mean_baseline - std_baseline, mean_baseline + std_baseline,
                    alpha=0.2, color='red')
    
    # 参考线
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Perfect Sync (R=1)')
    plt.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # 填充优势区域
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
    
    print(f"平均对比图已保存: {save_path}")
    
    return mean_dta, std_dta, mean_baseline, std_baseline


def save_matlab_data(all_dta, all_baseline, task, network_type, attention_type, alpha_value, save_dir):
    """
    保存数据为MATLAB .mat文件
    
    变量说明:
    - timestep: 时间步序列 (0, 1, 2, ..., T-1)
    - mean_dta: DTA模型的平均序参量
    - std_dta: DTA模型的标准差
    - mean_baseline: 对照模型的平均序参量  
    - std_baseline: 对照模型的标准差
    - dta_upper: DTA模型上界 (mean + std)
    - dta_lower: DTA模型下界 (mean - std)
    - baseline_upper: 对照模型上界
    - baseline_lower: 对照模型下界
    """
    mean_dta = np.mean(all_dta, axis=0)
    std_dta = np.std(all_dta, axis=0)
    mean_baseline = np.mean(all_baseline, axis=0)
    std_baseline = np.std(all_baseline, axis=0)
    
    # 计算上下界
    dta_upper = mean_dta + std_dta
    dta_lower = mean_dta - std_dta
    baseline_upper = mean_baseline + std_baseline
    baseline_lower = mean_baseline - std_baseline
    
    # 时间步
    timestep = np.arange(len(mean_dta))
    
    # 构建数据字典
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
    
    # 构建文件名: results/compare_<task>_<attention_type>_<network_type>.mat
    filename = f"compare_{task}_{attention_type}_{network_type}.mat"
    filepath = os.path.join(save_dir, filename)
    
    # 保存为.mat文件
    savemat(filepath, mat_data)
    print(f"\n[OK] MATLAB数据已保存: {filepath}")
    print(f"    包含变量: timestep, mean_dta, std_dta, mean_baseline, std_baseline")
    print(f"    以及上下界: dta_upper/lower, baseline_upper/lower")
    
    return filepath


def main():
    # 如果 GPU 内存不足，可以强制使用 CPU:
    # device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        print(f"[INFO] 使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"[INFO] 如果显存不足，请修改代码: device = 'cpu'")
    else:
        print("[INFO] 使用 CPU")
    
    checkpoint_path = 'results/sync_self_ws/final_model.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到模型文件 {checkpoint_path}")
        return
    
    # 加载训练好的模型
    trained_model, task, alpha_learned, config_info = load_trained_model(checkpoint_path, device)
    attention_type = config_info['attention_type']
    network_type = config_info['network_type']
    
    # 创建对照模型（α=0）
    print("\n创建对照模型 (α=0, 纯传统耦合)...")
    baseline_model = create_baseline_model(trained_model, device)
    
    # 创建保存目录
    save_dir = 'results/comparison_dta_vs_baseline'
    os.makedirs(save_dir, exist_ok=True)
    
    # 参数
    n_steps = 2000
    N = trained_model.N  # 从模型获取节点数，而不是硬编码
    
    print(f"\n[调试] 模拟步数 n_steps = {n_steps}")
    print(f"[调试] 振子数量 N = {N}")
    
    # 测试多个案例
    print("\n生成对比案例...")
    n_cases = 20
    
    all_dta = []
    all_baseline = []
    
    for i in range(n_cases):
        # 相同的随机初始条件
        initial_phases = torch.rand(N) * 2 * np.pi
        
        # 运行DTA模型
        final_dta, order_params_dta, traj_dta = generate_trajectory(
            trained_model, initial_phases, n_steps, device
        )
        
        # 运行对照模型（相同初始条件）
        final_baseline, order_params_baseline, traj_baseline = generate_trajectory(
            baseline_model, initial_phases, n_steps, device
        )
        
        # 保存结果用于平均
        all_dta.append(order_params_dta)
        all_baseline.append(order_params_baseline)
        
        # 每5个案例清理一次显存
        if device == 'cuda' and (i + 1) % 5 == 0:
            torch.cuda.empty_cache()
            print(f"  [清理显存] 已完成 {i+1}/{n_cases} 个案例")
        
        # 打印结果
        print(f"\n案例 {i+1}:")
        print(f"  初始 R: {order_params_dta[0]:.4f}")
        print(f"  DTA 最终 R: {order_params_dta[-1]:.4f}")
        print(f"  传统耦合最终 R: {order_params_baseline[-1]:.4f}")
        print(f"  DTA优势 (ΔR): {order_params_dta[-1] - order_params_baseline[-1]:+.4f}")
        
        # 绘制对比图
        save_path = os.path.join(save_dir, f'comparison_case_{i+1}.png')
        plot_comparison(order_params_dta, order_params_baseline,
                       initial_phases, alpha_learned, task,
                       save_path, i+1)
    
    # 绘制平均对比图
    print("\n生成平均对比图...")
    mean_dta, std_dta, mean_baseline, std_baseline = plot_average_comparison(
        all_dta, all_baseline, alpha_learned, task,
        os.path.join(save_dir, 'average_comparison.png')
    )
    
    # 统计汇总
    print("\n" + "="*60)
    print("对照实验统计汇总")
    print("="*60)
    
    final_dta_all = [traj[-1] for traj in all_dta]
    final_baseline_all = [traj[-1] for traj in all_baseline]
    improvements = [d - b for d, b in zip(final_dta_all, final_baseline_all)]
    
    print(f"DTA模型平均最终R: {np.mean(final_dta_all):.4f} ± {np.std(final_dta_all):.4f}")
    print(f"传统耦合平均最终R: {np.mean(final_baseline_all):.4f} ± {np.std(final_baseline_all):.4f}")
    print(f"平均改进 (ΔR): {np.mean(improvements):+.4f} ± {np.std(improvements):.4f}")
    print(f"相对提升: {(np.mean(improvements) / np.mean(final_baseline_all) * 100):+.1f}%")
    print("="*60)
    
    print(f"\n所有对比图保存在: {save_dir}/")
    print("文件说明:")
    print("  - comparison_case_X.png: 单个案例对比")
    print("  - average_comparison.png: 平均轨迹对比")
    
    # 保存 MATLAB 数据文件
    print("\n" + "="*60)
    print("保存MATLAB数据文件...")
    print("="*60)
    mat_filepath = save_matlab_data(
        all_dta, all_baseline, 
        task, network_type, attention_type, 
        alpha_learned, save_dir
    )
    print("="*60)


if __name__ == '__main__':
    main()
