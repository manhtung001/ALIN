import os
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CoraFull, Coauthor
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.nn import GCNConv
from torch_sparse import coalesce, SparseTensor
import random
import networkx as nx
from partition import GraphPartition


def load_data(name="cora",
              read=False,
              save=False,
              transform=T.ToSparseTensor(),
              seed=0,
              PERCENT_TO_DELETE_EDGE=0.15,
              pre_compute=True, verbose=False):

    assert name in ["cora", "pubmed", "citeseer", "corafull",
                    "cs", "physics", 'arxiv']

    path = os.path.join("data", name)
    if name == 'cora' or name == 'pubmed' or name == 'citeseer':
        dataset = Planetoid(root=path, name=name, transform=transform)
    elif name == "corafull":
        dataset = CoraFull(root=path, transform=transform)
    elif name == "cs" or name == "physics":
        dataset = Coauthor(root=path, name=name, transform=transform)
    elif name == 'arxiv':
        dataset = PygNodePropPredDataset(
            root=path, name='ogbn-' + name, transform=transform)
    else:
        raise NotImplemented

    data = dataset[0]
    if not hasattr(data, 'num_classes'):
        data.num_classes = dataset.num_classes
    data.adj_t = data.adj_t.to_symmetric() if not isinstance(
        data.adj_t, torch.Tensor) else data.adj_t
    data.max_part = data.num_classes

    try:
        if name == 'cora':
            data.max_part = 7
            data.params = {'age': [0.05, 0.05, 0.9]}
        elif name == 'pubmed':
            data.max_part = 8
            data.params = {'age': [0.15, 0.15, 0.7]}
        elif name == 'citeseer':
            data.max_part = 14
            data.params = {'age': [0.35, 0.35, 0.3]}
        elif name == 'corafull':
            data.max_part = 7
            data.params = {'age': [0.1, 0.1, 0.8]}
        elif name == 'cs':
            data.max_part = 6
            data.params = {'age': [0.1, 0.1, 0.8]}
        elif name == 'physics':
            data.max_part = 5
            data.params = {'age': [0.1, 0.1, 0.8]}
        elif name == 'arxiv':
            data.max_part = 9
            data.params = {'age': [0.1, 0.1, 0.8]}
        else:
            raise NotImplemented

        if not hasattr(data, 'g'):
            edges = [(int(i), int(j)) for i, j in zip(data.adj_t.storage._row,
                                                      data.adj_t.storage._col)]
            data.g = nx.Graph()
            data.g.add_edges_from(edges)

        random.seed(seed)

        num_edges = len(data.g.edges())
        num_edges_to_delete = int(PERCENT_TO_DELETE_EDGE * num_edges)

        edges = list(data.g.edges())
        random.shuffle(edges)
        deleted_edges = edges[:num_edges_to_delete]
        data.g.remove_edges_from(deleted_edges)
        deleted_edges = np.array(deleted_edges)
        data.g.deleted_edges = deleted_edges

        if read:
            filename = "data/partitions.json"
        else:
            graph = data.g.to_undirected()
            graph_part = GraphPartition(graph, data.x, data.max_part)

            communities = graph_part.clauset_newman_moore(weight=None)
            sizes = ([len(com) for com in communities])
            threshold = 1/3
            if min(sizes) * len(sizes) / len(data.x) < threshold:
                print("if load_data")
                data.partitions = graph_part.agglomerative_clustering(
                    communities)
            else:
                print("else load_data")
                sorted_communities = sorted(
                    communities, key=lambda c: len(c), reverse=True)
                data.partitions = {}
                data.partitions[len(sizes)] = torch.zeros(
                    data.x.shape[0], dtype=torch.int)
                for i, com in enumerate(sorted_communities):
                    data.partitions[len(sizes)][com] = i

        # Convert the networkx graph to PyG format
        edges = list(data.g.edges())
        # Add reverse edges
        edges_reverse = [(j, i) for i, j in edges]
        # Add self-loops
        edges_selfLoop = [(int(i), int(i)) for i in range(data.num_nodes)]
        data.g.add_edges_from(edges_selfLoop)
        edges += edges_reverse
        edges += edges_selfLoop
        edge_index = torch.tensor(edges).t().contiguous()
        edge_index, _ = coalesce(
            edge_index, None, data.num_nodes, data.num_nodes)
        # Create the adjacency sparse tensor
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(
            data.num_nodes, data.num_nodes))
        data.adj_t = adj_t

        if pre_compute:
            feat_dim = data.x.size(1)
            conv = GCNConv(feat_dim, feat_dim, cached=True, bias=False)
            conv.lin.weight = torch.nn.Parameter(torch.eye(feat_dim))
            with torch.no_grad():
                data.aggregated = conv(data.x, data.adj_t)
                data.aggregated = conv(data.aggregated, data.adj_t)

    except UserWarning:
        pass

    return data, num_edges_to_delete, deleted_edges, num_edges
