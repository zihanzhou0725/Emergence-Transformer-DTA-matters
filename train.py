"""
主训练脚本
用于训练同步/去同步控制器
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决OpenMP警告

import argparse
import torch
import numpy as np
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sync_transformer import SynchronizationTransformer
from utils.networks import generate_watts_strogatz, generate_fully_connected, network_summary
from utils.trainer import Trainer
from configs.default_config import Config, SyncConfig, SyncSelfConfig, DesyncConfig, DesyncNeighborConfig, TestConfig


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 选择配置 (4种组合: sync/desync × neighbor/self)
    config_map = {
        'sync': SyncConfig,           # sync + neighbor
        'sync_self': SyncSelfConfig,  # sync + self
        'desync': DesyncConfig,       # desync + self
        'desync_neighbor': DesyncNeighborConfig,  # desync + neighbor
        'test': TestConfig,
        'default': Config
    }
    config = config_map.get(args.config, Config)()
    
    # 覆盖配置
    if args.task:
        config.TASK = args.task
    if args.n_oscillators:
        config.N_OSCILLATORS = args.n_oscillators
    if args.epochs:
        config.N_EPOCHS = args.epochs
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.save_dir:
        config.SAVE_DIR = args.save_dir
    
    # 打印配置
    config.print_config()
    
    # 生成网络拓扑
    print("\n生成网络拓扑...")
    if config.NETWORK_TYPE == 'ws':
        # Watts-Strogatz 小世界网络
        spatial_network = generate_watts_strogatz(
            config.N_OSCILLATORS,
            config.K_NEIGHBORS,
            config.REWIRING_PROB,
            seed=args.seed
        )
        print(f"使用 Watts-Strogatz 网络 (N={config.N_OSCILLATORS}, k={config.K_NEIGHBORS}, p={config.REWIRING_PROB})")
    elif config.NETWORK_TYPE == 'fc':
        # 全连接网络
        spatial_network = generate_fully_connected(config.N_OSCILLATORS)
        print(f"使用 全连接网络 (N={config.N_OSCILLATORS})")
    else:
        raise ValueError(f"未知的 NETWORK_TYPE: {config.NETWORK_TYPE}")
    
    # 根据 ATTENTION_TYPE 选择注意力网络 (与 TASK 独立)
    if config.ATTENTION_TYPE == 'neighbor':
        # neighbor-DTA: Â = A (与空间网络相同)
        attention_network = spatial_network.clone()
        print("使用 neighbor-DTA (A_hat = A)")
    elif config.ATTENTION_TYPE == 'self':
        # self-DTA: Â = I (单位矩阵)
        attention_network = torch.eye(config.N_OSCILLATORS)
        print("使用 self-DTA (A_hat = I)")
    else:
        raise ValueError(f"未知的 ATTENTION_TYPE: {config.ATTENTION_TYPE}")
    
    print(f"任务类型: {config.TASK.upper()} (目标: {'R->1' if config.TASK == 'sync' else 'R->0'})")
    
    network_summary(spatial_network)
    
    # 生成自然频率（设置种子确保可复现）
    torch.manual_seed(args.seed + 1)  # 使用不同的种子避免与网络相同
    natural_frequencies = torch.randn(config.N_OSCILLATORS) * config.NATURAL_FREQ_STD
    torch.manual_seed(args.seed)  # 恢复原始种子
    print(f"本征频率分布: N(0, {config.NATURAL_FREQ_STD}^2)")
    
    # 创建模型
    print("\n创建模型...")
    model = SynchronizationTransformer(
        n_oscillators=config.N_OSCILLATORS,
        d_model=config.D_MODEL,
        spatial_network=spatial_network,
        attention_network=attention_network,
        natural_frequencies=natural_frequencies,
        coupling_strength=config.COUPLING_STRENGTH,
        noise_strength=config.NOISE_STRENGTH,
        learnable_alpha=config.LEARNABLE_ALPHA,
        alpha_init=config.ALPHA_INIT,
        learnable_w_qk=True,
        w_v_identity=True
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 显示 alpha 设置
    alpha_status = "可训练" if config.LEARNABLE_ALPHA else "固定"
    actual_alpha = torch.sigmoid(torch.tensor(config.ALPHA_INIT)).item()
    print(f"Alpha 设置: {alpha_status}, 初始值={config.ALPHA_INIT}, 实际alpha≈{actual_alpha:.4f}")
    
    # 创建训练器
    print(f"\n开始训练 {config.TASK} 控制器...")
    trainer = Trainer(
        model=model,
        task=config.TASK,
        lr=config.LEARNING_RATE,
        device=config.DEVICE
    )
    
    # 训练
    trainer.train(
        n_epochs=config.N_EPOCHS,
        n_episodes_per_epoch=config.N_EPISODES_PER_EPOCH,
        n_val_episodes=config.N_VAL_EPISODES,
        n_steps=config.N_STEPS,
        n_oscillators=config.N_OSCILLATORS,
        save_dir=config.SAVE_DIR,
        verbose=True,
        spatial_network=spatial_network,
        attention_network=attention_network,
        natural_frequencies=natural_frequencies,
        attention_type=config.ATTENTION_TYPE,
        network_type=config.NETWORK_TYPE,
        natural_freq_std=config.NATURAL_FREQ_STD
    )
    
    print(f"\n训练完成! 结果保存在: {config.SAVE_DIR}")
    
    # 最终测试
    print("\n最终测试...")
    model.eval()
    with torch.no_grad():
        test_phases = torch.rand(config.N_OSCILLATORS, device=config.DEVICE) * 2 * np.pi
        model.reset_history()
        
        final_phases, order_params, trajectory = model(
            test_phases, 
            n_steps=config.N_STEPS,
            return_trajectory=True
        )
        
        final_R = order_params[-1].item()
        print(f"测试序参量: 初始={order_params[0].item():.4f}, 最终={final_R:.4f}")
        print(f"学习到的 α: {torch.sigmoid(model.alpha).item():.4f}")
        
        if config.TASK == 'sync':
            success = final_R > 0.9
        else:
            success = final_R < 0.3
        
        print(f"控制{'成功' if success else '失败'}!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练同步/去同步控制器')
    
    parser.add_argument('--config', type=str, default='sync',
                       choices=['sync', 'sync_self', 'desync', 'desync_neighbor', 'test', 'default'],
                       help='配置文件类型 (sync/sync_self/desync/desync_neighbor/test/default)')
    parser.add_argument('--task', type=str, default=None,
                       choices=['sync', 'desync'],
                       help='任务类型')
    parser.add_argument('--n_oscillators', type=int, default=None,
                       help='振子数量')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练epoch数')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='保存目录')
    
    args = parser.parse_args()
    main(args)
