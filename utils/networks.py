"""
Network topology generation utilities
Includes Watts-Strogatz small-world networks and related graphs
"""

import torch
import numpy as np
import networkx as nx


def generate_watts_strogatz(n_nodes, k_neighbors, rewiring_prob, seed=None):
    """
    Generate a Watts-Strogatz small-world network
    
    Args:
        n_nodes: number of nodes N
        k_neighbors: number of nearest neighbors per node (must be even)
        rewiring_prob: rewiring probability p
        seed: random seed
    
    Returns:
        adjacency: (N, N) adjacency matrix (PyTorch tensor)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Use NetworkX to generate the WS network
    G = nx.watts_strogatz_graph(n_nodes, k_neighbors, rewiring_prob, seed=seed)
    
    # Convert to adjacency matrix
    adjacency = nx.to_numpy_array(G)
    
    return torch.from_numpy(adjacency).float()


def generate_fully_connected(n_nodes):
    """
    Generate a fully connected network
    
    Args:
        n_nodes: number of nodes N
    
    Returns:
        adjacency: (N, N) adjacency matrix
    """
    adjacency = torch.ones(n_nodes, n_nodes) - torch.eye(n_nodes)
    return adjacency


def generate_ring_network(n_nodes, k_neighbors=2):
    """
    Generate a ring network (special case of a WS network when p=0)
    
    Args:
        n_nodes: number of nodes N
        k_neighbors: number of nearest neighbors per node
    
    Returns:
        adjacency: (N, N) adjacency matrix
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
    Generate an Erdos-Renyi random network
    
    Args:
        n_nodes: number of nodes N
        connection_prob: connection probability p
        seed: random seed
    
    Returns:
        adjacency: (N, N) adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.erdos_renyi_graph(n_nodes, connection_prob, seed=seed)
    adjacency = nx.to_numpy_array(G)
    
    return torch.from_numpy(adjacency).float()


def generate_barabasi_albert(n_nodes, m_edges, seed=None):
    """
    Generate a Barabasi-Albert scale-free network
    
    Args:
        n_nodes: number of nodes N
        m_edges: number of edges for each new node
        seed: random seed
    
    Returns:
        adjacency: (N, N) adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.barabasi_albert_graph(n_nodes, m_edges, seed=seed)
    adjacency = nx.to_numpy_array(G)
    
    return torch.from_numpy(adjacency).float()


def compute_average_shortest_path_length(adjacency):
    """
    Compute average shortest path length (ASPL)
    
    Args:
        adjacency: (N, N) adjacency matrix
    
    Returns:
        aspl: Average shortest path length
    """
    if isinstance(adjacency, torch.Tensor):
        adjacency = adjacency.numpy()
    
    G = nx.from_numpy_array(adjacency)
    
    if not nx.is_connected(G):
        # If disconnected，return ASPL for the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    return nx.average_shortest_path_length(G)


def compute_clustering_coefficient(adjacency):
    """
    Compute average clustering coefficient
    
    Args:
        adjacency: (N, N) adjacency matrix
    
    Returns:
        cc: Average clustering coefficient
    """
    if isinstance(adjacency, torch.Tensor):
        adjacency = adjacency.numpy()
    
    G = nx.from_numpy_array(adjacency)
    return nx.average_clustering(G)


def compute_degree_distribution(adjacency):
    """
    Compute degree distribution
    
    Args:
        adjacency: (N, N) adjacency matrix
    
    Returns:
        degrees: (N,) node degrees
    """
    if isinstance(adjacency, torch.Tensor):
        adjacency = adjacency.numpy()
    
    return adjacency.sum(axis=1)


def network_summary(adjacency):
    """
    Print network summary
    
    Args:
        adjacency: (N, N) adjacency matrix
    """
    if isinstance(adjacency, torch.Tensor):
        adjacency_np = adjacency.numpy()
    else:
        adjacency_np = adjacency
    
    n_nodes = adjacency_np.shape[0]
    n_edges = adjacency_np.sum() / 2  # undirected graph
    
    G = nx.from_numpy_array(adjacency_np)
    is_connected = nx.is_connected(G)
    
    print("=" * 50)
    print("Network summary")
    print("=" * 50)
    print(f"number of nodes: {n_nodes}")
    print(f"Number of edges: {int(n_edges)}")
    print(f"Connected: {is_connected}")
    
    if is_connected:
        aspl = nx.average_shortest_path_length(G)
        print(f"Average shortest path length (ASPL): {aspl:.4f}")
    
    avg_degree = adjacency_np.sum(axis=1).mean()
    print(f"Average degree: {avg_degree:.4f}")
    
    cc = nx.average_clustering(G)
    print(f"Average clustering coefficient: {cc:.4f}")
    print("=" * 50)
