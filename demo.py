"""
演示脚本
快速展示Synchronization Transformer的功能
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决OpenMP警告

import torch
import numpy as np
import os

from models.sync_transformer import SynchronizationTransformer
from utils.networks import generate_watts_strogatz, network_summary
from utils.metrics import compute_order_parameter
from utils.visualization import plot_order_parameter, plot_phase_distribution


def demo_sync_task():
    """演示同步任务"""
    print("=" * 60)
    print("演示: 同步任务 (Sync Task)")
    print("=" * 60)
    
    # 参数设置
    N = 50
    n_steps = 100
    
    # 生成网络
    print("\n1. 生成网络拓扑...")
    spatial_network = generate_watts_strogatz(N, k_neighbors=4, rewiring_prob=0.1, seed=42)
    network_summary(spatial_network)
    
    # 同步任务: 使用 neighbor-DTA
    attention_network = spatial_network.clone()
    
    # 创建模型
    print("\n2. 创建模型...")
    model = SynchronizationTransformer(
        n_oscillators=N,
        d_model=64,
        spatial_network=spatial_network,
        attention_network=attention_network,
        natural_frequencies=torch.randn(N) * 0.1,
        coupling_strength=1.5,
        noise_strength=0.05,
        learnable_alpha=True,
        alpha_init=0.3
    )
    
    # 模拟
    print("\n3. 运行模拟...")
    initial_phases = torch.rand(N) * 2 * np.pi
    
    with torch.no_grad():
        model.reset_history()
        final_phases, order_params, trajectory = model(
            initial_phases, n_steps, return_trajectory=True
        )
    
    # 结果
    print(f"\n4. 结果:")
    print(f"   初始序参量 R(0): {order_params[0].item():.4f}")
    print(f"   最终序参量 R(T): {order_params[-1].item():.4f}")
    print(f"   当前 α 值: {torch.sigmoid(model.alpha).item():.4f}")
    
    # 可视化
    os.makedirs('./results/demo', exist_ok=True)
    plot_order_parameter(
        order_params.numpy(),
        save_path='./results/demo/sync_order_param.png',
        title='Sync Task: Order Parameter Evolution'
    )
    print(f"   图表已保存到: ./results/demo/sync_order_param.png")


def demo_desync_task():
    """演示去同步任务"""
    print("\n" + "=" * 60)
    print("演示: 去同步任务 (Desync Task)")
    print("=" * 60)
    
    # 参数设置
    N = 50
    n_steps = 100
    
    # 生成网络
    print("\n1. 生成网络拓扑...")
    spatial_network = generate_watts_strogatz(N, k_neighbors=4, rewiring_prob=0.1, seed=42)
    
    # 去同步任务: 使用 self-DTA
    attention_network = torch.eye(N)
    
    # 创建模型
    print("\n2. 创建模型...")
    model = SynchronizationTransformer(
        n_oscillators=N,
        d_model=64,
        spatial_network=spatial_network,
        attention_network=attention_network,
        natural_frequencies=torch.randn(N) * 0.1,
        coupling_strength=2.0,
        noise_strength=0.05,
        learnable_alpha=True,
        alpha_init=0.7
    )
    
    # 模拟
    print("\n3. 运行模拟...")
    initial_phases = torch.ones(N) * np.pi / 2  # 初始几乎同步
    
    with torch.no_grad():
        model.reset_history()
        final_phases, order_params, trajectory = model(
            initial_phases, n_steps, return_trajectory=True
        )
    
    # 结果
    print(f"\n4. 结果:")
    print(f"   初始序参量 R(0): {order_params[0].item():.4f}")
    print(f"   最终序参量 R(T): {order_params[-1].item():.4f}")
    print(f"   当前 α 值: {torch.sigmoid(model.alpha).item():.4f}")
    
    # 可视化
    plot_order_parameter(
        order_params.numpy(),
        save_path='./results/demo/desync_order_param.png',
        title='Desync Task: Order Parameter Evolution'
    )
    print(f"   图表已保存到: ./results/demo/desync_order_param.png")


def demo_attention_mechanism():
    """演示注意力机制"""
    print("\n" + "=" * 60)
    print("演示: 注意力机制 (Attention Mechanism)")
    print("=" * 60)
    
    N = 20
    
    # 创建简单网络
    spatial_network = torch.eye(N)
    attention_network = torch.eye(N)
    
    model = SynchronizationTransformer(
        n_oscillators=N,
        d_model=32,
        spatial_network=spatial_network,
        attention_network=attention_network,
        learnable_alpha=True,
        learnable_w_qk=True
    )
    
    print("\n1. 可学习参数:")
    print(f"   W_Q shape: {model.W_Q.shape}")
    print(f"   W_K shape: {model.W_K.shape}")
    print(f"   W_V shape: {model.W_V.shape}")
    print(f"   α (混合系数): {torch.sigmoid(model.alpha).item():.4f}")
    
    print("\n2. 模拟相位演化...")
    initial_phases = torch.rand(N) * 2 * np.pi
    
    with torch.no_grad():
        model.reset_history()
        final_phases, order_params, trajectory = model(
            initial_phases, n_steps=50, return_trajectory=True
        )
    
    print(f"\n3. 序参量变化:")
    print(f"   初始: {order_params[0].item():.4f}")
    print(f"   中间: {order_params[25].item():.4f}")
    print(f"   最终: {order_params[-1].item():.4f}")
    
    print("\n4. 历史缓存状态:")
    print(f"   历史长度: {len(model.phase_history_list)}")
    print(f"   当前时间步: {model.current_time}")


def main():
    print("\n")
    print("#" * 60)
    print("# Synchronization Transformer 演示")
    print("# 基于论文: 'Synchronization Transformer: Dynamical")
    print("#          Temporal Attention Matters'")
    print("#" * 60)
    
    # 运行演示
    demo_sync_task()
    demo_desync_task()
    demo_attention_mechanism()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    print("\n使用说明:")
    print("  1. 训练同步控制器: python train.py --config sync")
    print("  2. 训练去同步控制器: python train.py --config desync")
    print("  3. 快速测试: python train.py --config test")
    print("  4. 评估模型: python evaluate.py --checkpoint <path>")
    print("=" * 60)


if __name__ == '__main__':
    main()
