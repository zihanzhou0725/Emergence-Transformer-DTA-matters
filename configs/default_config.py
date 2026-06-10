"""
默认配置文件
"""

import torch
import numpy as np


class Config:
    """基础配置类"""
    
    # 设备配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 网络配置
    NETWORK_TYPE = 'ws'         # 网络类型: 'ws' (Watts-Strogatz) 或 'fc' (Fully Connected)
    N_OSCILLATORS = 100         # 振子数量 N
    K_NEIGHBORS = 4             # 最近邻节点数 k (仅WS网络使用)
    REWIRING_PROB = 0.1         # 重连概率 p (仅WS网络使用)
    
    # 本征频率配置
    NATURAL_FREQ_STD = 0.1      # 本征频率高斯分布的标准差 σ
    
    # 模型配置
    CONTEXT_LENGTH = 20         # 历史窗口长度 T
    D_MODEL = 32                # 特征维度 d
    COUPLING_STRENGTH = 1.0     # 耦合强度 λ
    NOISE_STRENGTH = 0.05       # 噪声强度 D
    
    # Alpha 配置
    LEARNABLE_ALPHA = True      # 是否训练 alpha (False则固定)
    ALPHA_INIT = 0.0            # 初始alpha值 (实际alpha=sigmoid(ALPHA_INIT))
    # 提示: 若LEARNABLE_ALPHA=False且要alpha≈1, 设ALPHA_INIT=5.0 (sigmoid(5)≈0.99)
    
    # 训练配置
    TASK = 'sync'               # 任务类型: 'sync' 或 'desync' (控制目标)
    ATTENTION_TYPE = 'neighbor' # 注意力类型: 'neighbor' (Â=A) 或 'self' (Â=I)
    N_EPOCHS = 100              # 训练epoch数
    N_EPISODES_PER_EPOCH = 10  # 每个epoch的episode数
    N_VAL_EPISODES = 5          # 验证episode数
    N_STEPS = 200               # 每个episode的模拟步数（时变历史窗口）
    LEARNING_RATE = 1e-3        # 学习率
    
    # 保存配置（使用属性自动生成，包含网络类型）
    @property
    def SAVE_DIR(self):
        return f'./results/{self.TASK}_{self.ATTENTION_TYPE}_{self.NETWORK_TYPE}'
    
    @classmethod
    def to_dict(cls):
        """转换为字典"""
        return {
            k: v for k, v in cls.__dict__.items() 
            if not k.startswith('_') and not callable(v)
        }
    
    @classmethod
    def print_config(cls):
        """打印配置"""
        print("=" * 50)
        print("配置信息")
        print("=" * 50)
        for k, v in cls.to_dict().items():
            print(f"{k}: {v}")
        print("=" * 50)


class SyncConfig(Config):
    """同步任务配置"""
    TASK = 'sync'
    ATTENTION_TYPE = 'neighbor'
    COUPLING_STRENGTH = 2.0
    ALPHA_INIT = 0.0
    
    @property
    def SAVE_DIR(self):
        return f'./results/sync_neighbor_{self.NETWORK_TYPE}'


class SyncSelfConfig(Config):
    """同步任务 + self-DTA"""
    TASK = 'sync'
    ATTENTION_TYPE = 'self'
    COUPLING_STRENGTH = 2.0
    ALPHA_INIT = 0.0
    
    @property
    def SAVE_DIR(self):
        return f'./results/sync_self_{self.NETWORK_TYPE}'


class DesyncConfig(Config):
    """去同步任务配置"""
    TASK = 'desync'
    ATTENTION_TYPE = 'self'
    COUPLING_STRENGTH = 2.0
    ALPHA_INIT = 0.0
    
    @property
    def SAVE_DIR(self):
        return f'./results/desync_self_{self.NETWORK_TYPE}'


class DesyncNeighborConfig(Config):
    """去同步任务 + neighbor-DTA"""
    TASK = 'desync'
    ATTENTION_TYPE = 'neighbor'
    COUPLING_STRENGTH = 2.0
    ALPHA_INIT = 0.0
    
    @property
    def SAVE_DIR(self):
        return f'./results/desync_neighbor_{self.NETWORK_TYPE}'


class TestConfig(Config):
    """测试配置 (小规模快速测试)"""
    N_OSCILLATORS = 20
    CONTEXT_LENGTH = None  # 时变历史窗口，无需预设
    N_EPOCHS = 20
    N_EPISODES_PER_EPOCH = 5
    N_VAL_EPISODES = 3
    N_STEPS = 300  # 增加到300步
    
    @property
    def SAVE_DIR(self):
        return f'./results/test_{self.NETWORK_TYPE}'
