"""
网络拓扑生成工具
包括 Watts-Strogatz 小世界网络等
"""

import torch
import numpy as np
import networkx as nx


def generate_watts_strogatz(n_nodes, k_neighbors, rewiring_prob, seed=None):
    """
    生成 Watts-Strogatz 小世界网络
    
    Args:
        n_nodes: 节点数量 N
        k_neighbors: 每个节点连接的最近邻节点数 (必须为偶数)
        rewiring_prob: 重连概率 p
        seed: 随机种子
    
    Returns:
        adjacency: (N, N) 邻接矩阵 (PyTorch tensor)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 使用 NetworkX 生成 WS 网络
    G = nx.watts_strogatz_graph(n_nodes, k_neighbors, rewiring_prob, seed=seed)
    
    # 转换为邻接矩阵
    adjacency = nx.to_numpy_array(G)
    
    return torch.from_numpy(adjacency).float()


def generate_fully_connected(n_nodes):
    """
    生成全连接网络
    
    Args:
        n_nodes: 节点数量 N
    
    Returns:
        adjacency: (N, N) 邻接矩阵
    """
    adjacency = torch.ones(n_nodes, n_nodes) - torch.eye(n_nodes)
    return adjacency


def generate_ring_network(n_nodes, k_neighbors=2):
    """
    生成环形网络 (WS 网络在 p=0 时的特例)
    
    Args:
        n_nodes: 节点数量 N
        k_neighbors: 每个节点连接的最近邻节点数
    
    Returns:
        adjacency: (N, N) 邻接矩阵
    """
    adjacency = torch.zeros(n_nodes, n_nodes)
    half_k = k_neighbors // 2
    
    for i in range(n_nodes):
        for j in range(1, half_k + 1):
            neighbor_right = (i + j) % n_nodes
            neighbor_left = (i - j) % n_nodes
            adjacency[i, neighbor_right] = 1
            adjacency[i, neighbor_left] = 1
    
    return adjacency


def generate_erdos_renyi(n_nodes, connection_prob, seed=None):
    """
    生成 Erdős-Rényi 随机网络
    
    Args:
        n_nodes: 节点数量 N
        connection_prob: 连接概率 p
        seed: 随机种子
    
    Returns:
        adjacency: (N, N) 邻接矩阵
    """
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.erdos_renyi_graph(n_nodes, connection_prob, seed=seed)
    adjacency = nx.to_numpy_array(G)
    
    return torch.from_numpy(adjacency).float()


def generate_barabasi_albert(n_nodes, m_edges, seed=None):
    """
    生成 Barabási-Albert 无标度网络
    
    Args:
        n_nodes: 节点数量 N
        m_edges: 每个新节点连接的边数
        seed: 随机种子
    
    Returns:
        adjacency: (N, N) 邻接矩阵
    """
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.barabasi_albert_graph(n_nodes, m_edges, seed=seed)
    adjacency = nx.to_numpy_array(G)
    
    return torch.from_numpy(adjacency).float()


def compute_average_shortest_path_length(adjacency):
    """
    计算网络的平均最短路径长度 (ASPL)
    
    Args:
        adjacency: (N, N) 邻接矩阵
    
    Returns:
        aspl: 平均最短路径长度
    """
    if isinstance(adjacency, torch.Tensor):
        adjacency = adjacency.numpy()
    
    G = nx.from_numpy_array(adjacency)
    
    if not nx.is_connected(G):
        # 如果不连通，返回最大连通分量的ASPL
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    return nx.average_shortest_path_length(G)


def compute_clustering_coefficient(adjacency):
    """
    计算网络的平均聚类系数
    
    Args:
        adjacency: (N, N) 邻接矩阵
    
    Returns:
        cc: 平均聚类系数
    """
    if isinstance(adjacency, torch.Tensor):
        adjacency = adjacency.numpy()
    
    G = nx.from_numpy_array(adjacency)
    return nx.average_clustering(G)


def compute_degree_distribution(adjacency):
    """
    计算网络的度分布
    
    Args:
        adjacency: (N, N) 邻接矩阵
    
    Returns:
        degrees: (N,) 各节点的度
    """
    if isinstance(adjacency, torch.Tensor):
        adjacency = adjacency.numpy()
    
    return adjacency.sum(axis=1)


def network_summary(adjacency):
    """
    打印网络统计信息
    
    Args:
        adjacency: (N, N) 邻接矩阵
    """
    if isinstance(adjacency, torch.Tensor):
        adjacency_np = adjacency.numpy()
    else:
        adjacency_np = adjacency
    
    n_nodes = adjacency_np.shape[0]
    n_edges = adjacency_np.sum() / 2  # 无向图
    
    G = nx.from_numpy_array(adjacency_np)
    is_connected = nx.is_connected(G)
    
    print("=" * 50)
    print("网络统计信息")
    print("=" * 50)
    print(f"节点数量: {n_nodes}")
    print(f"边数量: {int(n_edges)}")
    print(f"是否连通: {is_connected}")
    
    if is_connected:
        aspl = nx.average_shortest_path_length(G)
        print(f"平均最短路径长度 (ASPL): {aspl:.4f}")
    
    avg_degree = adjacency_np.sum(axis=1).mean()
    print(f"平均度: {avg_degree:.4f}")
    
    cc = nx.average_clustering(G)
    print(f"平均聚类系数: {cc:.4f}")
    print("=" * 50)
