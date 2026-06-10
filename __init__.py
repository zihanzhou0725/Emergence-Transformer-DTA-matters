"""
DTA Platform: Numerical Experiment Framework for Synchronization/Desynchronization Control
基于机器学习的振子同步/去同步控制器数值实验框架

基于论文:
- "Synchronization Transformer: Dynamical Temporal Attention Matters"

主要组件:
- models: Synchronization Transformer 模型
- utils: 网络生成、可视化、评估指标
- configs: 配置文件
"""

__version__ = "1.0.0"
__author__ = "DTA Platform"

from .models import SynchronizationTransformer, SyncController, DesyncController
from .utils.networks import generate_watts_strogatz, network_summary

__all__ = [
    'SynchronizationTransformer',
    'SyncController',
    'DesyncController',
    'generate_watts_strogatz',
    'network_summary',
]
