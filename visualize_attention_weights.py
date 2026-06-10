"""
可视化注意力权重矩阵 W^Q 和 W^K
从 checkpoint 加载并绘制热力图
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.io import savemat


def load_checkpoint(checkpoint_path):
    """加载 checkpoint"""
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 提取权重矩阵
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Checkpoint 中没有 model_state_dict")
    
    state_dict = checkpoint['model_state_dict']
    
    # 获取 W_Q 和 W_K
    W_Q = state_dict['W_Q'].cpu().numpy()  # (N, d)
    W_K = state_dict['W_K'].cpu().numpy()  # (N, d)
    
    # 获取 alpha
    alpha_learned = torch.sigmoid(state_dict['alpha']).item() if 'alpha' in state_dict else None
    
    # 获取配置信息
    config_info = {
        'task': checkpoint.get('task', 'unknown'),
        'attention_type': checkpoint.get('attention_type', 'unknown'),
        'network_type': checkpoint.get('network_type', 'unknown'),
        'natural_freq_std': checkpoint.get('natural_freq_std', 0.1),
        'alpha_learned': alpha_learned,
        'N': W_Q.shape[0],
        'd': W_Q.shape[1]
    }
    
    return W_Q, W_K, config_info


def visualize_weights(W_Q, W_K, config_info, save_dir='./results/attention_weights'):
    """可视化 W_Q 和 W_K"""
    N, d = W_Q.shape
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 构建文件名前缀
    task = config_info['task']
    att_type = config_info['attention_type']
    net_type = config_info['network_type']
    prefix = f"{task}_{att_type}_{net_type}"
    
    # 计算统计信息
    print(f"\n{'='*60}")
    print("权重矩阵统计信息")
    print(f"{'='*60}")
    print(f"矩阵形状: ({N}, {d}) = (振子数量, 特征维度)")
    print(f"W_Q: mean={W_Q.mean():.6f}, std={W_Q.std():.6f}")
    print(f"W_K: mean={W_K.mean():.6f}, std={W_K.std():.6f}")
    print(f"W_Q 范围: [{W_Q.min():.6f}, {W_Q.max():.6f}]")
    print(f"W_K 范围: [{W_K.min():.6f}, {W_K.max():.6f}]")
    print(f"{'='*60}")
    
    # ========== 图1: W_Q 和 W_K 热力图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # W_Q 热力图
    im1 = axes[0].imshow(W_Q, cmap='RdBu_r', aspect='auto', 
                         vmin=-np.abs(W_Q).max(), vmax=np.abs(W_Q).max())
    axes[0].set_title(f'W^Q (Query Matrix)\nShape: {N}×{d}', fontsize=14)
    axes[0].set_xlabel('Feature Dimension (d)', fontsize=12)
    axes[0].set_ylabel('Oscillator Index (N)', fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='Weight Value')
    
    # W_K 热力图
    im2 = axes[1].imshow(W_K, cmap='RdBu_r', aspect='auto',
                         vmin=-np.abs(W_K).max(), vmax=np.abs(W_K).max())
    axes[1].set_title(f'W^K (Key Matrix)\nShape: {N}×{d}', fontsize=14)
    axes[1].set_xlabel('Feature Dimension (d)', fontsize=12)
    axes[1].set_ylabel('Oscillator Index (N)', fontsize=12)
    plt.colorbar(im2, ax=axes[1], label='Weight Value')
    
    plt.suptitle(f'Attention Weight Matrices\n{prefix}, α={config_info["alpha_learned"]:.4f}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{prefix}_WQ_WK_heatmap.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] 热力图已保存: {save_path}")
    
    # ========== 图2: 权重分布直方图 ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # W_Q 分布
    axes[0, 0].hist(W_Q.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('W^Q Distribution', fontsize=12)
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(W_Q.mean(), color='red', linestyle='--', label=f'Mean={W_Q.mean():.4f}')
    axes[0, 0].legend()
    
    # W_K 分布
    axes[0, 1].hist(W_K.flatten(), bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('W^K Distribution', fontsize=12)
    axes[0, 1].set_xlabel('Weight Value')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(W_K.mean(), color='red', linestyle='--', label=f'Mean={W_K.mean():.4f}')
    axes[0, 1].legend()
    
    # W_Q 按行平均
    row_mean_Q = W_Q.mean(axis=1)
    axes[1, 0].plot(range(N), row_mean_Q, 'b-o', markersize=3)
    axes[1, 0].set_title('W^Q Row Mean (per oscillator)', fontsize=12)
    axes[1, 0].set_xlabel('Oscillator Index')
    axes[1, 0].set_ylabel('Mean Weight')
    axes[1, 0].grid(True, alpha=0.3)
    
    # W_K 按行平均
    row_mean_K = W_K.mean(axis=1)
    axes[1, 1].plot(range(N), row_mean_K, 'g-o', markersize=3)
    axes[1, 1].set_title('W^K Row Mean (per oscillator)', fontsize=12)
    axes[1, 1].set_xlabel('Oscillator Index')
    axes[1, 1].set_ylabel('Mean Weight')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Weight Distribution Analysis\n{prefix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{prefix}_weight_distribution.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] 分布图已保存: {save_path}")
    
    # ========== 图3: 相关性分析 ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # W_Q 和 W_K 的相关性
    correlation = np.corrcoef(W_Q.flatten(), W_K.flatten())[0, 1]
    axes[0].scatter(W_Q.flatten(), W_K.flatten(), alpha=0.5, s=1)
    axes[0].set_xlabel('W^Q Value')
    axes[0].set_ylabel('W^K Value')
    axes[0].set_title(f'W^Q vs W^K Correlation\nr = {correlation:.4f}')
    axes[0].grid(True, alpha=0.3)
    
    # W_Q 和 W_K 的差值
    diff = W_Q - W_K
    im = axes[1].imshow(diff, cmap='RdBu_r', aspect='auto',
                        vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[1].set_title('W^Q - W^K Difference', fontsize=12)
    axes[1].set_xlabel('Feature Dimension (d)')
    axes[1].set_ylabel('Oscillator Index (N)')
    plt.colorbar(im, ax=axes[1], label='Difference')
    
    plt.suptitle(f'Correlation Analysis\n{prefix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{prefix}_correlation_analysis.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] 相关图已保存: {save_path}")
    
    # ========== 保存 MATLAB 数据 ==========
    mat_data = {
        'W_Q': W_Q,
        'W_K': W_K,
        'W_Q_mean': W_Q.mean(),
        'W_Q_std': W_Q.std(),
        'W_K_mean': W_K.mean(),
        'W_K_std': W_K.std(),
        'correlation_QK': correlation,
        'alpha_learned': config_info['alpha_learned'],
        'N': N,
        'd': d,
        'task': config_info['task'],
        'attention_type': config_info['attention_type'],
        'network_type': config_info['network_type']
    }
    
    mat_path = os.path.join(save_dir, f'{prefix}_attention_weights.mat')
    savemat(mat_path, mat_data)
    print(f"[OK] MATLAB数据已保存: {mat_path}")
    
    print(f"\n所有结果保存在: {save_dir}/")
    
    return save_dir


def main():
    parser = argparse.ArgumentParser(description='可视化注意力权重矩阵 W^Q 和 W^K')
    parser.add_argument('--checkpoint', type=str, 
                       default='results/sync_self_ws/final_model.pt',
                       help='Checkpoint 文件路径')
    parser.add_argument('--save_dir', type=str, 
                       default='./results/attention_weights',
                       help='保存目录')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"错误: 找不到 checkpoint 文件 {args.checkpoint}")
        print("\n请确保先运行训练，或指定正确的 checkpoint 路径")
        return
    
    # 加载 checkpoint
    W_Q, W_K, config_info = load_checkpoint(args.checkpoint)
    
    # 可视化
    save_dir = visualize_weights(W_Q, W_K, config_info, args.save_dir)
    
    print("\n可视化完成！")


if __name__ == '__main__':
    main()
