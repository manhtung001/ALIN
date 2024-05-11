from __future__ import division
from __future__ import print_function
import numpy as np
from sklearn import cluster
from dgl import function as fn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch.nn import Linear


class Cluster:
    """
    Kmeans Clustering
    """

    def __init__(self, n_clusters, n_dim, seed,
                 implementation='sklearn',
                 init='k-means++',
                 device=torch.cuda.is_available()):

        assert implementation in ['sklearn', 'faiss', 'cuml']
        assert init in ['k-means++', 'random']

        self.n_clusters = n_clusters
        self.n_dim = n_dim
        self.implementation = implementation
        self.initialization = init
        self.model = None

        if implementation == 'sklearn':
            self.model = cluster.KMeans(
                n_clusters=n_clusters, init=init, random_state=seed)
        elif implementation == 'faiss':
            import faiss
            self.model = faiss.Kmeans(
                n_dim, n_clusters, niter=20, nredo=10, seed=seed, gpu=device != 'cpu')
        elif implementation == 'cuml':
            import cuml
            if init == 'k-means++':
                init = 'scalable-kmeans++'
            self.model = cuml.KMeans(
                n_dim, n_clusters, random_state=seed, init=init, output_type='numpy')
        else:
            raise NotImplemented

    def train(self, x):
        if self.implementation == 'sklearn':
            self.model.fit(x)
        elif self.implementation == 'faiss':
            if self.initialization == 'kmeans++':
                init_centroids = self._kmeans_plusplus(
                    x, self.n_clusters).cpu().numpy()
            else:
                init_centroids = None
            self.model.train(x, init_centroids=init_centroids)
        elif self.implementation == 'cuml':
            self.model.fit(x)
        else:
            raise NotImplemented

    def predict(self, x):
        if self.implementation == 'sklearn':
            return self.model.predict(x)
        elif self.implementation == 'faiss':
            _, labels = self.model.index.search(x, 1)
            return labels
        else:
            raise NotImplemented

    def get_centroids(self):
        if self.implementation == 'sklearn':
            return self.model.cluster_centers_
        elif self.implementation == 'faiss':
            return self.model.centroids
        elif self.implementation == 'cuml':
            return self.model.cluster_centers_
        else:
            raise NotImplemented

    def get_inertia(self):
        if self.implementation == 'sklearn':
            return self.model.inertia_
        else:
            raise NotImplemented

    @staticmethod
    def _kmeans_plusplus(X, n_clusters):
        """
        K-means++ initialization in PyTorch for Faiss.

        Modified from sklearn version of implementation.
        https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/cluster/_kmeans.py
        """

        n_samples, n_features = X.shape

        # Set the number of local seeding trials if none is given
        n_local_trials = 2 + int(np.log(n_clusters))

        # Pick first center randomly and track index of point
        center_id = torch.randint(n_samples, (1,)).item()
        centers = [X[center_id]]

        # Initialize list of closest distances and calculate current potential
        closest_dist_sq = torch.cdist(
            X, X[center_id].unsqueeze(dim=0)).pow(2).squeeze()
        current_pot = closest_dist_sq.sum()

        # Pick the remaining n_clusters-1 points
        for c in range(1, n_clusters):
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            rand_vals = torch.rand(n_local_trials).to(
                current_pot.device) * current_pot
            candidate_ids = torch.searchsorted(torch.cumsum(
                closest_dist_sq.flatten(), dim=0), rand_vals)

            # Numerical imprecision can result in a candidate_id out of range
            torch.clip(candidate_ids, min=None,
                       max=closest_dist_sq.shape[0] - 1, out=candidate_ids)

            # Compute distances to center candidates
            distance_to_candidates = torch.cdist(
                X[candidate_ids].unsqueeze(dim=0), X).pow(2).squeeze()

            # update closest distances squared and potential for each candidate
            torch.minimum(closest_dist_sq, distance_to_candidates,
                          out=distance_to_candidates)
            candidates_pot = distance_to_candidates.sum(dim=1)

            # Decide which candidate is the best
            best_candidate = torch.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            # Permanently add best center candidate found in local tries
            centers.append(X[best_candidate])

        centers = torch.stack(centers, dim=0).to(dtype=X.dtype)
        return centers


class GCN(torch.nn.Module):
    def __init__(self, dataset):

        super(GCN, self).__init__()
        torch.manual_seed(68)

        hidden_channel1 = 128
        hidden_channel2 = 64

        if dataset == 'cora':
            num_features = 1433
            num_classes = 7
        elif dataset == 'citeseer':
            num_features = 3703
            num_classes = 6
        elif dataset == 'pubmed':
            num_features = 500
            num_classes = 3
        elif dataset == 'cs':
            num_features = 6805
            num_classes = 15
        elif dataset == 'physics':
            num_features = 8415
            num_classes = 5
        elif dataset == 'arxiv':
            num_features = 128
            num_classes = 40
        elif dataset == 'corafull':
            num_features = 8710
            num_classes = 70
        self.conv1 = GCNConv(num_features,  hidden_channel1)
        self.conv2 = GCNConv(hidden_channel1, hidden_channel2)
        self.conv3 = GCNConv(hidden_channel2, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, adj_t)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, adj_t)
        return x

    def embed(self, x, adj_t):
        emb = self.forward(x, adj_t)
        return emb


class SAGE(torch.nn.Module):
    def __init__(self, dataset):

        super(SAGE, self).__init__()
        torch.manual_seed(68)

        hidden_channel1 = 128
        hidden_channel2 = 64

        if dataset == 'cora':
            num_features = 1433
            num_classes = 7
        elif dataset == 'citeseer':
            num_features = 3703
            num_classes = 6
        elif dataset == 'pubmed':
            num_features = 500
            num_classes = 3
        elif dataset == 'cs':
            num_features = 6805
            num_classes = 15
        elif dataset == 'physics':
            # num_features = 2000
            num_features = 8415
            num_classes = 5
        elif dataset == 'arxiv':
            num_features = 128
            num_classes = 40
        elif dataset == 'corafull':
            num_features = 8710
            num_classes = 70

        self.conv1 = SAGEConv(num_features,  hidden_channel1)
        self.conv2 = SAGEConv(hidden_channel1, hidden_channel2)
        self.out = Linear(hidden_channel2, num_classes)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)

        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def decode(self, z):
        y = F.softmax(self.out(z), dim=1)
        return y

    def decode_edge(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x, adj_t):
        x = self.encode(x, adj_t)
        x = self.decode(x)
        return x

    def embed(self, x, adj_t):
        emb = self.encode(x, adj_t)
        return emb


class GAT(torch.nn.Module):
    def __init__(self, dataset):

        super(GAT, self).__init__()
        torch.manual_seed(68)

        hidden_channel1 = 128
        hidden_channel2 = 64

        if dataset == 'cora':
            num_features = 1433
            num_classes = 7
        elif dataset == 'citeseer':
            num_features = 3703
            num_classes = 6
        elif dataset == 'pubmed':
            num_features = 500
            num_classes = 3
        elif dataset == 'cs':
            num_features = 6805
            num_classes = 15
        elif dataset == 'physics':
            num_features = 8415
            num_classes = 5
        elif dataset == 'arxiv':
            num_features = 128
            num_classes = 40
        elif dataset == 'corafull':
            num_features = 8710
            num_classes = 70

        num_heads = 8
        hidden_channel1 = int(hidden_channel1 / num_heads)
        hidden_channel2 = int(hidden_channel2 / num_heads)

        self.conv1 = GATConv(num_features, hidden_channel1,
                             heads=num_heads, bias=False)
        self.conv2 = GATConv(hidden_channel1*num_heads,
                             hidden_channel2, heads=num_heads, bias=False)
        self.conv3 = GATConv(hidden_channel2*num_heads,
                             num_classes, heads=num_heads, bias=False)
        self.out = Linear(hidden_channel2*num_heads, num_classes)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def decode(self, z):
        y = F.softmax(self.out(z), dim=1)
        return y

    def decode_edge(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x, adj_t):
        x = self.encode(x, adj_t)
        x = self.decode(x)
        return x

    def embed(self, x, adj_t):
        emb = self.encode(x, adj_t)
        return emb
