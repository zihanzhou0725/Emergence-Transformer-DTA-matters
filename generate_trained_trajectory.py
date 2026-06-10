"""
加载训练好的模型，生成 R-timestep 演化图
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决OpenMP警告

import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.sync_transformer import SynchronizationTransformer
from utils.networks import generate_watts_strogatz
from utils.visualization import plot_order_parameter
from utils.metrics import compute_order_parameter

def load_trained_model(checkpoint_path, device='cpu'):
    """加载训练好的模型"""
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 使用默认配置重建模型
    N = 20  # 根据test配置
    T_context = 10
    
    spatial_network = generate_watts_strogatz(N, k_neighbors=4, rewiring_prob=0.1, seed=42)
    
    # 判断任务类型
    task = checkpoint.get('task', 'sync')
    if task == 'sync':
        attention_network = spatial_network.clone()
        print("任务类型: 同步 (Sync)")
    else:
        attention_network = torch.eye(N)
        print("任务类型: 去同步 (Desync)")
    
    natural_frequencies = torch.randn(N) * 0.1
    
    model = SynchronizationTransformer(
        n_oscillators=N,
        d_model=64,
        spatial_network=spatial_network,
        attention_network=attention_network,
        natural_frequencies=natural_frequencies,
        coupling_strength=1.5,
        noise_strength=0.05,
        learnable_alpha=True,
        learnable_w_qk=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 显示学习到的参数
    alpha_learned = torch.sigmoid(model.alpha).item()
    print(f"学习到的 α: {alpha_learned:.4f}")
    
    return model, task


def generate_trajectory(model, initial_phases, n_steps, device='cpu'):
    """生成轨迹"""
    with torch.no_grad():
        model.reset_history()
        initial_phases = initial_phases.to(device)
        
        final_phases, order_params, trajectory = model(
            initial_phases, n_steps, return_trajectory=True
        )
    
    return final_phases, order_params, trajectory


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模型路径
    checkpoint_path = 'results/test_experiment/final_model.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到模型文件 {checkpoint_path}")
        print("请先运行训练: python train.py --config test")
        return
    
    # 加载模型
    model, task = load_trained_model(checkpoint_path, device)
    
    # 创建保存目录
    save_dir = 'results/trained_trajectories'
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成多个测试案例
    n_test_cases = 5
    n_steps = 100
    N = 20
    
    print(f"\n生成 {n_test_cases} 个测试案例的轨迹...")
    
    for i in range(n_test_cases):
        # 随机初始条件
        initial_phases = torch.rand(N) * 2 * np.pi
        
        # 生成轨迹
        final_phases, order_params, trajectory = generate_trajectory(
            model, initial_phases, n_steps, device
        )
        
        # 打印结果
        initial_R = order_params[0].item()
        final_R = order_params[-1].item()
        print(f"\n案例 {i+1}:")
        print(f"  初始 R: {initial_R:.4f}")
        print(f"  最终 R: {final_R:.4f}")
        print(f"  变化: {final_R - initial_R:+.4f}")
        
        # 绘制并保存
        save_path = os.path.join(save_dir, f'trajectory_case_{i+1}.png')
        plot_order_parameter(
            order_params.cpu().numpy(),
            save_path=save_path,
            title=f'Trained Model - Case {i+1}: R(0)={initial_R:.3f} → R(T)={final_R:.3f}'
        )
        print(f"  图表已保存: {save_path}")
    
    # 绘制平均轨迹
    print("\n生成平均轨迹...")
    all_order_params = []
    for i in range(20):  # 更多样本计算平均
        initial_phases = torch.rand(N) * 2 * np.pi
        _, order_params, _ = generate_trajectory(model, initial_phases, n_steps, device)
        all_order_params.append(order_params.cpu().numpy())
    
    mean_order_params = np.mean(all_order_params, axis=0)
    std_order_params = np.std(all_order_params, axis=0)
    
    # 保存平均轨迹图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(mean_order_params, linewidth=2, color='blue', label='Mean R')
    plt.fill_between(
        range(len(mean_order_params)),
        mean_order_params - std_order_params,
        mean_order_params + std_order_params,
        alpha=0.3, color='blue', label='±1 Std Dev'
    )
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Sync (R=1)')
    plt.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Desync (R=0)')
    plt.xlabel('Time Step')
    plt.ylabel('Order Parameter R')
    plt.title(f'Trained Model - Average Trajectory (20 samples)\nTask: {task}, Final R={mean_order_params[-1]:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.05, 1.05])
    
    save_path = os.path.join(save_dir, 'average_trajectory.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"平均轨迹图已保存: {save_path}")
    
    print(f"\n所有结果保存在: {save_dir}/")


if __name__ == '__main__':
    main()
