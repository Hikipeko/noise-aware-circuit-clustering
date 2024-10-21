import random
import copy
import networkx as nx
import numpy as np
from clusterparser import ClusterParser


def generate_circuit_graph(cp: ClusterParser) -> nx.Graph:
    """Generate circuit graph based on ClusterParser. Hypergraph is transferred to a normal graph by using the star model.

    Args:
        cp (ClusterParser): the ClusterParser

    Returns:
        nx.Graph: circuit graph
    """
    edges = []

    for net in cp.net_nodes:
        if len(net) <= 1:
            continue
        out_node = None
        for node_id, _, _, dir in net:
            if dir == "O":
                out_node = node_id
                break
        if out_node is None:
            out_node = net[0][0]
        for in_node, _, _, dir in net:
            if in_node != out_node:
                edges.append((out_node, in_node))

    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def avg_path_length(G, N, labels, mode=None):
    """Find L coefficient

    Args:
        G (Graph): graph
        N (int): number of samples
        labels (list): clustering labels
        mode (string): "icn"/"cluster"/None
    """
    distance = []
    disconnect_cnt = 0
    nodes = list(G)
    if not mode:
        l = np.random.randint(0, len(G.nodes), (N, 2))
    elif mode == "icn":
        indices = [i for i, label in enumerate(labels) if label == -1]
        l = [random.sample(indices, 2) for _ in range(N)]
    else:
        indices = [i for i, label in enumerate(labels) if label != -1]
        l = [random.sample(indices, 2) for _ in range(N)]
    for i, j in l:
        try:
            shortest_path_length = nx.shortest_path_length(
                G, source=nodes[i], target=nodes[j]
            )
        except nx.NetworkXNoPath:
            disconnect_cnt += 1
            continue
        except Exception as e:
            print(e)
            continue
        distance.append(shortest_path_length)
    return sum(distance) / len(distance), disconnect_cnt


def avg_clustering_icn(G, N, labels, icn):
    """Find L coefficient

    Args:
        G (nx.Graph): graph
        N (int): number of samples
        labels (list): clustering labels
        icn (boolean): true or false
    """
    triangles = 0
    nodes = list(G)
    if icn:
        indices_with_neg1 = [i for i, label in enumerate(labels) if label == -1]
    else:
        indices_with_neg1 = [i for i, label in enumerate(labels) if label != -1]
    random_indices = random.choices(indices_with_neg1, k=N)
    for i in random_indices:
        if i >= len(nodes):
            continue
        nbrs = list(G[nodes[i]])
        if len(nbrs) < 2:
            continue
        u, v = random.sample(nbrs, 2)
        if u in G[v]:
            triangles += 1
    return triangles / N


def noise_cell_proportion(G, N, labels):
    """Estimate the noise cell proportion on shortest paths by random sampling.

    Args:
        G (nx.Graph): _description_
        N (int): number of samples
        labels (list): clustering labels

    Returns:
        _type_: proportion (float), number of failure (disconnected pairs) (int)
    """
    l = np.random.randint(0, len(G.nodes), (N, 2))
    noise_cnt = 0
    total_cnt = 0
    disconnect_cnt = 0
    for i, j in l:
        try:
            shortest_path = nx.shortest_path(G, source=i, target=j)
        except nx.NetworkXNoPath:
            disconnect_cnt += 1
            continue
        except Exception as e:
            print(e)
            continue
        for node in shortest_path:
            if labels[node] == -1:
                noise_cnt += 1
            total_cnt += 1
    return noise_cnt / total_cnt, disconnect_cnt


def remove_ICN_experiment(G, N, labels):
    """Test change of L if ICNs are removed, compared with a control group.

    Args:
        G (nx.Graph): _description_
        N (int): number of samples
        labels (list): clustering labels
    """
    # Remove ICN
    G_copy = copy.deepcopy(G)
    nodes_to_remove = [i for i in G_copy.nodes if labels[i] == -1]
    G_copy.remove_nodes_from(nodes_to_remove)
    L_remove_icn, fail_remove_icn = avg_path_length(G_copy, N, labels)

    # Random remove
    G_copy_random = copy.deepcopy(G)
    nodes_to_remove_count = sum(1 for i in G.nodes if labels[i] == -1)
    nodes_to_remove_random = random.sample(
        list(G_copy_random.nodes), nodes_to_remove_count
    )
    G_copy_random.remove_nodes_from(nodes_to_remove_random)
    L_remove_random, fail_remove_random = avg_path_length(G_copy_random, N, labels)

    print(f"L_remove_icn: {L_remove_icn:.2f}, fail: {fail_remove_icn}/{N}")
    print(f"L_remove_contrast: {L_remove_random:.2f}, fail: {fail_remove_random}/{N}")
