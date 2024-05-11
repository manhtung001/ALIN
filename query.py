import numpy as np
from copy import deepcopy
from timeit import default_timer as timer
import sklearn.metrics as metrics
import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_sparse import coalesce, SparseTensor
from torch_geometric.utils import negative_sampling, degree
from models import Cluster
from sklearn.cluster import KMeans
from collections import defaultdict
from partition import GraphPartition
import matplotlib.pyplot as plt
import time
import math
import os


class ActiveLearning:
    """
    An active learning framework that...
    * queries from an oracle;
    * updates its known set,
    * trains the GNN model, and
    * evaluate the Macro F-1 score.
    """

    def __init__(self, data, model, seed, args):
        self.round = 0
        self.data = data
        self.model = model
        self.seed = seed
        self.args = args
        self.retrain = args.retrain
        self.clf = None
        self.aggregated = None
        self.num_centers = args.num_centers
        self.num_parts = -1

    def query(self, b):
        pass

    def update(self, train_mask):
        self.data.train_mask = train_mask
        self.round += 1

    def update_edges(self, train_mask):
        indices = torch.nonzero(train_mask).flatten()
        deleted_edges = torch.tensor(self.data.g.deleted_edges.T)
        edges_to_add_back = torch.tensor([])

        for node_selected in indices:
            check_idxes = None
            if node_selected in deleted_edges[0]:
                check_idxes = (
                    deleted_edges[0] == node_selected).nonzero().squeeze().view(-1)
            if node_selected in deleted_edges[1]:
                if check_idxes is not None:
                    check_idxes = torch.cat(
                        (check_idxes, (deleted_edges[1] == node_selected).nonzero().squeeze().view(-1)), 0)
                else:
                    check_idxes = (
                        deleted_edges[1] == node_selected).nonzero().squeeze().view(-1)
            if check_idxes is not None:
                edges_to_add_back = torch.cat(
                    (edges_to_add_back, deleted_edges.T[check_idxes]), 0)
                edges_to_add_back = edges_to_add_back.to(torch.int)

        edges_to_add_back = edges_to_add_back.numpy()
        edges_to_add_back = [(int(i), int(j)) for i, j in edges_to_add_back]
        edges_to_add_back_reverse = [(j, i) for i, j in edges_to_add_back]
        edges = list(self.data.g.edges())
        edges_reverse = [(j, i) for i, j in edges]
        edges += edges_reverse
        edges += edges_to_add_back
        edges += edges_to_add_back_reverse
        edge_index = torch.tensor(edges).t().contiguous()
        edge_index, _ = coalesce(
            edge_index, None, self.data.num_nodes, self.data.num_nodes)
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=None, sparse_sizes=(
            self.data.num_nodes, self.data.num_nodes))
        self.data.adj_t = adj_t
        self.data.g.add_edges_from(edges_to_add_back)
        self.data.adj_t = self.data.adj_t.to(self.args.device)

    def caclulate_list_edges_add_back(self, train_mask):
        indices = torch.nonzero(train_mask).flatten()
        deleted_edges = torch.tensor(self.data.g.deleted_edges.T)
        edges_to_add_back = torch.tensor([])

        for node_selected in indices:
            check_idxes = None
            if node_selected in deleted_edges[0]:
                check_idxes = (
                    deleted_edges[0] == node_selected).nonzero().squeeze().view(-1)
            if node_selected in deleted_edges[1]:
                if check_idxes is not None:
                    check_idxes = torch.cat(
                        (check_idxes, (deleted_edges[1] == node_selected).nonzero().squeeze().view(-1)), 0)
                else:
                    check_idxes = (
                        deleted_edges[1] == node_selected).nonzero().squeeze().view(-1)
            if check_idxes is not None:
                edges_to_add_back = torch.cat(
                    (edges_to_add_back, deleted_edges.T[check_idxes]), 0)
                edges_to_add_back = edges_to_add_back.to(torch.int)

        edges_to_add_back = edges_to_add_back.numpy()
        edges_to_add_back = [(int(i), int(j)) for i, j in edges_to_add_back]
        edges_to_add_back_reverse = [(j, i) for i, j in edges_to_add_back]
        edges_to_add_back += edges_to_add_back_reverse
        return edges_to_add_back

    def get_arg(self, arg, baseline):
        if baseline in ['ALIN', 'ALINFar']:
            epochs = 230
        else:
            epochs = self.args.epochs
        return epochs, arg.lr, arg.weight_decay

    def update_partition(self):
        graph = self.data.g.to_undirected()
        graph_part = GraphPartition(graph, self.data.x, self.data.max_part)
        communities = graph_part.clauset_newman_moore(weight=None)
        sizes = ([len(com) for com in communities])
        threshold = 1/3
        if min(sizes) * len(sizes) / len(self.data.x) < threshold:
            self.data.partitions = graph_part.agglomerative_clustering(
                communities)
        else:
            sorted_communities = sorted(
                communities, key=lambda c: len(c), reverse=True)
            self.data.partitions = {}
            self.data.partitions[len(sizes)] = torch.zeros(
                self.data.x.shape[0], dtype=torch.int)
            for i, com in enumerate(sorted_communities):
                self.data.partitions[len(sizes)][com] = i

    def init_clf(self):
        self.clf = deepcopy(self.model).to(self.args.device)

    def train(self, baseline, query_i_th, kind_decay):
        if self.retrain:
            self.clf = deepcopy(self.model).to(self.args.device)
        else:
            self.clf = self.model.to(self.args.device)
        epochs, lr, weight_decay = self.get_arg(self.args, baseline)
        optimizer = optim.Adam(
            self.clf.parameters(), lr=lr,
            weight_decay=weight_decay)
        criterion_ec = torch.nn.BCEWithLogitsLoss()
        folder_log_train = 'log_train_test'
        if not os.path.exists(folder_log_train):
            os.makedirs(folder_log_train)
        f = open(folder_log_train + '/' +
                 str(baseline) + '_log_train.txt', 'a')
        f.write("query_i_th: " + str(query_i_th) + "\n")
        list_loss_ec = []
        list_loss_nc = []
        list_loss = []

        def step_decay(initial_weight, decay_rate, decay_steps, total_epochs, epoch):
            drop = 0.00
            epochs_drop = 80
            return initial_weight * np.power(drop, np.floor((1 + epoch) / epochs_drop))

        def exponential_decay(initial_weight, decay_rate, decay_steps, total_epochs, epoch):
            return initial_weight * np.exp(-decay_rate * epoch / decay_steps)

        def inverse_time_decay(initial_weight, decay_rate, decay_steps, total_epochs, epoch):
            return initial_weight / (1 + decay_rate * epoch / decay_steps)

        def cosine_annealing(initial_weight, decay_rate, decay_steps, total_epochs, epoch):
            return initial_weight * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

        def custom_decay(initial_weight, decay_rate, decay_steps, total_epochs, epoch):
            x = (epoch / decay_steps) * 0.4
            return initial_weight - initial_weight / (1 + math.exp(-8*x + 85))

        decay_rate = 0.1
        decay_steps = 3
        for epoch in range(epochs):
            self.clf.train()
            optimizer.zero_grad()
            labels = self.data.y
            if len(labels.shape) > 1:
                labels = labels.squeeze(1)
            logits = self.clf(self.data.x, self.data.adj_t)
            f.write("epoch: " + str(epoch) + "\n")
            if baseline in ['ALIN', 'ALINFar']:
                alpha_combine_los = self.args.alpha_combine_los
                if kind_decay == 'exponential_decay':
                    alpha_combine_los = exponential_decay(
                        1 - alpha_combine_los, decay_rate, decay_steps, epochs, epoch)
                elif kind_decay == 'inverse_time_decay':
                    alpha_combine_los = inverse_time_decay(
                        1 - alpha_combine_los, decay_rate, decay_steps, epochs, epoch)
                elif kind_decay == 'cosine_annealing':
                    alpha_combine_los = cosine_annealing(
                        1 - alpha_combine_los, decay_rate, decay_steps, epochs, epoch)
                alpha_combine_los = 1 - alpha_combine_los
                edges_current_src = self.data.adj_t.storage._row.clone().tolist()
                edges_current_dst = self.data.adj_t.storage._col.clone().tolist()
                edges_current = torch.tensor(
                    [edges_current_src, edges_current_dst])
                edges_current_labels = torch.ones(edges_current.shape[1])
                num_neg_samples = int(edges_current.shape[1])
                num_nodes = logits.shape[0]
                train_edges_pos_neg = torch.cat([edges_current, negative_sampling(
                    edges_current, num_nodes, num_neg_samples, method='sparse')], dim=1)
                train_edges_pos_neg_label = torch.cat(
                    [edges_current_labels, torch.zeros(num_neg_samples)], dim=0).to(self.args.device)
                outs_edge_raw = torch.sum(
                    logits[train_edges_pos_neg[0]] * logits[train_edges_pos_neg[1]], dim=1)
                train_edges_pos_neg_label = train_edges_pos_neg_label.float()
                loss_ec = criterion_ec(
                    outs_edge_raw, train_edges_pos_neg_label)
                f.write("loss_ec lib: " + str(loss_ec.item()) + "\n")
            else:
                alpha_combine_los = 1
                loss_ec = 0
            if baseline in ['ALIN', 'ALINFar'] and epoch > 80:
                alpha_combine_los = 1
                loss_ec = 0
            softmax_logits = F.softmax(logits, dim=1)
            neg_log_softmax = - \
                torch.log(
                    softmax_logits[self.data.train_mask, labels[self.data.train_mask]])
            loss_cross_entropy_manual = torch.mean(neg_log_softmax)
            loss_nc = loss_cross_entropy_manual
            f.write("loss_nc handle: " + str(loss_nc.item()) + "\n")
            loss = alpha_combine_los * loss_nc + \
                (1 - alpha_combine_los) * loss_ec
            f.write("loss: " + str(loss.item()) + "\n")
            f.write("\n")
            if type(loss_ec) == int:
                list_loss_ec.append(loss_ec)
            else:
                list_loss_ec.append(loss_ec.item())
            list_loss_nc.append(loss_nc.item())
            list_loss.append(loss.item())
            if self.args.verbose == 2:
                print('Epoch {:03d}: Training loss: {:.4f}'.format(
                    epoch, loss))
            loss.backward()
            optimizer.step()
        return (list_loss_ec, list_loss_nc, list_loss)

    def evaluate(self):
        self.clf.eval()
        logits = self.clf(self.data.x, self.data.adj_t)
        y_pred = logits.max(1)[1].cpu()
        y_true = self.data.y.cpu()
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        acc = metrics.f1_score(y_true, y_pred, average='micro')
        if self.args.verbose == 2:
            print('Macro-f1 score: {:.4f}'.format(f1))
            print('Micro-f1 score: {:.4f}'.format(acc))
        return f1, acc

    def get_node_representation(self, rep='aggregation', encoder='gcn'):
        if rep == 'aggregation':
            if self.aggregated is None:
                feat_dim = self.data.x.size(1)
                if encoder == 'sage':
                    conv = SAGEConv(feat_dim, feat_dim, bias=False)
                    conv.lin_l.weight = torch.nn.Parameter(torch.eye(feat_dim))
                    conv.lin_r.weight = torch.nn.Parameter(torch.eye(feat_dim))
                else:
                    conv = GCNConv(feat_dim, feat_dim, cached=True, bias=False)
                    conv.lin.weight = torch.nn.Parameter(torch.eye(feat_dim))
                conv.to(self.args.device)
                with torch.no_grad():
                    self.aggregated = conv(self.data.x, self.data.adj_t)
                    self.aggregated = conv(self.aggregated, self.data.adj_t)
            return self.aggregated
        elif rep == 'embedding':
            with torch.no_grad():
                embed = self.clf.embed(self.data.x, self.data.adj_t)
            return embed
        else:
            return self.data.x

    def split_cluster(self, b, partitions, x_embed=None, method='default'):
        if method == 'inertia':
            part_size = []
            for i in range(self.num_parts):
                part_id = np.where(partitions == i)[0]
                x = x_embed[part_id]
                kmeans = Cluster(
                    n_clusters=1, n_dim=x_embed.shape[1], seed=self.seed, device=self.args.device)
                kmeans.train(x.cpu())
                inertia = kmeans.get_inertia()
                part_size.append(inertia)
            part_size = np.rint(b * np.array(part_size) /
                                sum(part_size)).astype(int)
            part_size = np.maximum(self.num_centers, part_size)
            i = 0
            while part_size.sum() - b != 0:
                if part_size.sum() - b > 0:
                    i = self.num_parts - 1 if i <= 0 else i
                    while part_size[i] <= 1:
                        i -= 1
                    part_size[i] -= 1
                    i -= 1
                else:
                    i = 0 if i >= self.num_parts else i
                    part_size[i] += 1
                    i += 1
        elif method == 'size':
            part_size = []
            for i in range(self.num_parts):
                part_size.append(len(np.where(partitions == i)[0]))
            part_size = np.rint(b * np.array(part_size) /
                                sum(part_size)).astype(int)
            part_size = np.maximum(self.num_centers, part_size)
            i = 0
            while part_size.sum() - b != 0:
                if part_size.sum() - b > 0:
                    i = self.num_parts - 1 if i <= 0 else i
                    while part_size[i] <= 1:
                        i -= 1
                    part_size[i] -= 1
                    i -= 1
                else:
                    i = 0 if i >= self.num_parts else i
                    part_size[i] += 1
                    i += 1
        else:
            part_size = [b // self.num_parts for _ in range(self.num_parts)]
            for i in range(b % self.num_parts):
                part_size[i] += 1
        return part_size

    def __str__(self):
        return "Active Learning Agent (uninitialized)"


class Random(ActiveLearning):
    """
    Random:
    The Random Sampling method chooses nodes uniformly at random,
    similarly as the commonly used semi-supervised learning experiment setting for GCN.
    """

    def __init__(self, data, model, seed, args):
        super(Random, self).__init__(data, model, seed, args)

    def query(self, b):
        indice = np.random.choice(
            np.where(self.data.train_mask == 0)[0], b, replace=False
        )
        return torch.tensor(indice)

    def __str__(self):
        return "Random"


class Density(ActiveLearning):
    """
    Density:
    The Density method first performs a clustering algorithm on the hidden representations of the nodes,
    and then chooses nodes with maximum density score, which is (approximately) inversely proportional to
    the L2-distance between each node and its cluster center.
    """

    def __init__(self, data, model, seed, args):
        super(Density, self).__init__(data, model, seed, args)

    def query(self, b):
        # Get propagated nodes
        x_embed = self.get_node_representation('embedding').cpu()

        # Perform K-Means as approximation
        kmeans = Cluster(
            n_clusters=b, n_dim=x_embed.shape[1], seed=self.seed, device=self.args.device)
        kmeans.train(x_embed)

        # Calculate density
        centers = kmeans.get_centroids()
        label = kmeans.predict(x_embed)
        centers = centers[label]
        dist_map = torch.linalg.norm(x_embed - centers, dim=1)
        density = 1 / (1 + dist_map)

        density[np.where(self.data.train_mask != 0)[0]] = 0
        _, indices = torch.topk(density, k=b)

        return indices

    def __str__(self):
        return "Density"


class Uncertainty(ActiveLearning):
    """
    Uncertainty:
    The Uncertainty method chooses the nodes with maximum entropy on the predicted class distribution.
    """

    def __init__(self, data, model, seed, args):
        super(Uncertainty, self).__init__(data, model, seed, args)

    def query(self, b):
        logits = self.clf(self.data.x, self.data.adj_t)
        entropy = -torch.sum(F.softmax(logits, dim=1) *
                             F.log_softmax(logits, dim=1), dim=1)
        entropy[np.where(self.data.train_mask != 0)[0]] = 0
        _, indices = torch.topk(entropy, k=b)
        return indices

    def __str__(self):
        return "Uncertainty"


class CoreSetGreedy(ActiveLearning):
    """
    CoreSet:
    The CoreSet method performs a K-Center clustering over the hidden representations of nodes.
    A time-efficient greedy approximation version by choosing node closest to the cluster centers.
    """

    def __init__(self, data, model, seed, args):
        super(CoreSetGreedy, self).__init__(data, model, seed, args)

    def query(self, b):

        embed = self.get_node_representation('embedding').cpu()
        indices = list(np.where(self.data.train_mask != 0)[0])

        for i in range(b):
            dist = metrics.pairwise_distances(
                embed, embed[indices], metric='euclidean')
            min_distances = torch.min(torch.tensor(dist), dim=1)[0]
            new_index = min_distances.argmax()
            indices.append(int(new_index))
        return indices

    def __str__(self):
        return "Core Set (Greedy)"


class CoreSetMIP(ActiveLearning):
    """
    CoreSet:
    The CoreSet method performs a K-Center clustering over the hidden representations of nodes.
    Optimized by gurobipy MIP.
    """

    def __init__(self, data, model, seed, args):
        super(CoreSetMIP, self).__init__(data, model, seed, args)

    def query(self, b):
        import gurobipy

        # Get distance matrix
        embed = self.get_node_representation('embedding')
        dist_mat = embed.matmul(embed.t())
        sq = dist_mat.diagonal().reshape(self.data.num_nodes, 1)
        dist_mat = torch.sqrt(-dist_mat * 2 + sq + sq.t())

        # Perform greedy K-center
        mask = self.data.train_mask.copy()
        mat = dist_mat[~mask, :][:, mask]
        _, indices = mat.min(dim=1)[0].topk(k=b)
        indices = torch.arange(self.data.num_nodes)[~mask][indices]
        mask[indices] = True

        # Robust approximation
        opt = mat.min(dim=1)[0].max()
        ub = opt
        lb = opt / 2.0
        xx, yy = np.where(dist_mat <= opt)
        dd = dist_mat[xx, yy]

        flag = self.data.train_mask.copy()
        subset = np.where(flag == 0)[0].tolist()

        # Solve MIP for fac_loc
        x = {}
        y = {}
        z = {}
        n = self.data.num_nodes
        m = len(xx)

        model = gurobipy.Model("k-center")
        for i in range(n):
            z[i] = model.addVar(
                obj=1, ub=0.0, vtype="B", name="z_{}".format(i))

        for i in range(m):
            _x = xx[i]
            _y = yy[i]
            if _y not in y:
                if _y in subset:
                    y[_y] = model.addVar(
                        obj=0, ub=1.0, lb=1.0, vtype="B", name="y_{}".format(_y))
                else:
                    y[_y] = model.addVar(
                        obj=0, vtype="B", name="y_{}".format(_y))
            x[_x, _y] = model.addVar(
                obj=0, vtype="B", name="x_{},{}".format(_x, _y))
        model.update()

        coef = [1 for j in range(n)]
        var = [y[j] for j in range(n)]
        model.addConstr(
            gurobipy.LinExpr(coef, var), "=", rhs=b + len(subset), name="k_center")

        for i in range(m):
            _x = xx[i]
            _y = yy[i]
            model.addConstr(
                x[_x, _y], "<", y[_y], name="Strong_{},{}".format(_x, _y))

        yyy = {}
        for v in range(m):
            _x = xx[v]
            _y = yy[v]
            if _x not in yyy:
                yyy[_x] = []
            if _y not in yyy[_x]:
                yyy[_x].append(_y)

        for _x in yyy:
            coef = []
            var = []
            for _y in yyy[_x]:
                coef.append(1)
                var.append(x[_x, _y])
            coef.append(1)
            var.append(z[_x])
            model.addConstr(
                gurobipy.LinExpr(coef, var), "=", 1, name="Assign{}".format(_x))

        # Approximate
        delta = 1e-7
        sol_file = None
        while ub - lb > delta:
            cur_r = (ub + lb) / 2.0
            viol = np.where(dd > cur_r)
            new_max_d = torch.min(dd[dd >= cur_r])
            new_min_d = torch.max(dd[dd <= cur_r])
            for v in viol[0]:
                x[xx[v], yy[v]].UB = 0

            model.update()
            r = model.optimize()
            if model.getAttr(gurobipy.GRB.Attr.Status) == gurobipy.GRB.INFEASIBLE:
                failed = True
                print("Infeasible")
            elif sum([z[i].X for i in range(len(z))]) > 0:
                failed = True
                print("Failed")
            else:
                failed = False
            if failed:
                lb = max(cur_r, new_max_d)
                for v in viol[0]:
                    x[xx[v], yy[v]].UB = 1
            else:
                print("sol founded", cur_r, lb, ub)
                ub = min(cur_r, new_min_d)
                sol_file = "s_{}_solution_{}.sol".format(b, cur_r)
                model.write(sol_file)

        # Process results
        if sol_file is not None:
            results = open(sol_file).read().split('\n')
            results_nodes = filter(lambda x1: 'y' in x1,
                                   filter(lambda x1: '#' not in x1, results))

            def string_to_id(x1): return (
                int(x1.split(' ')[0].split('_')[1]),
                int(x1.split(' ')[1]))
            result_node_ids = map(string_to_id, results_nodes)
            centers = []
            for node_result in result_node_ids:
                if node_result[1] > 0:
                    centers.append(node_result[0])
            return torch.tensor(centers)
        else:
            return None

    def __str__(self):
        return "Core Set (MIP)"


class Degree(ActiveLearning):
    """
    Centrality:
    The Centrality method chooses nodes with the largest graph centrality metric value.
    This framework chooses node degree as the metric.
    """

    def __init__(self, data, model, seed, args):
        super(Degree, self).__init__(data, model, seed, args)

    def query(self, b):

        if hasattr(self.data.adj_t.storage, '_row'):
            degree = self.data.adj_t.sum(dim=0)
        else:
            indice = torch.cat([self.data.adj_t[0].unsqueeze(dim=0),
                                self.data.adj_t[1].unsqueeze(dim=0)], dim=0)
            values = torch.ones(
                self.data.adj_t.shape[1], device=self.args.device)
            adj = torch.sparse_coo_tensor(
                indice, values, [self.data.num_nodes, self.data.num_nodes]).to_dense()
            degree = adj.sum(dim=0)

        degree[np.where(self.data.train_mask != 0)[0]] = 0
        _, indices = torch.topk(degree, k=b)
        return indices

    def __str__(self):
        return "Centrality (Degree)"


class PageRank(ActiveLearning):
    """
    PageRank:
    The Centrality method chooses nodes with the largest graph centrality metric value.
    This framework chooses node degree as the metric.
    """

    def __init__(self, data, model, seed, args):
        super(PageRank, self).__init__(data, model, seed, args)

    def query(self, b):
        page = torch.tensor(list(pagerank(self.data.g).values()))
        page[np.where(self.data.train_mask != 0)[0]] = 0
        _, indices = torch.topk(page, k=b)
        return indices

    def __str__(self):
        return "Centrality (PageRank)"


class AGE(ActiveLearning):
    """
    AGE:
    AGE defines the informativeness of nodes by linearly combining three metrics:
    centrality, density and uncertainty.
    It further chooses nodes with the highest scores.
    """

    def __init__(self, data, model, seed, args):
        super(AGE, self).__init__(data, model, seed, args)

    def query(self, b):
        # Get entropy
        logits = self.clf(self.data.x, self.data.adj_t)
        entropy = -torch.sum(F.softmax(logits, dim=1) *
                             F.log_softmax(logits, dim=1), dim=1)

        # Get centrality
        page = torch.tensor(list(pagerank(self.data.g).values()),
                            dtype=logits.dtype, device=self.args.device)

        # Get density
        x = self.get_node_representation('embedding').cpu()
        N = x.shape[0]

        kmeans = Cluster(
            n_clusters=b, n_dim=x.shape[1], seed=self.seed, device=self.args.device)
        kmeans.train(x)
        centers = kmeans.get_centroids()
        label = kmeans.predict(x)

        x = x.to(logits.device)
        centers = torch.tensor(centers[label], dtype=x.dtype, device=x.device)
        dist_map = torch.linalg.norm(x - centers, dim=1).to(logits.dtype)
        density = 1 / (1 + dist_map)

        # Get percentile
        percentile = (torch.arange(N, dtype=logits.dtype,
                      device=self.args.device) / N)
        id_sorted = density.argsort(descending=False)
        density[id_sorted] = percentile
        id_sorted = entropy.argsort(descending=False)
        entropy[id_sorted] = percentile
        id_sorted = page.argsort(descending=False)
        page[id_sorted] = percentile

        # Get linear combination
        alpha, beta, gamma = self.data.params['age']
        age_score = alpha * entropy + beta * density + gamma * page
        age_score[np.where(self.data.train_mask != 0)[0]] = 0
        _, indices = torch.topk(age_score, k=b)
        return indices


class ClusterBased(ActiveLearning):
    """
    Cluster:
    The cluster method first performs clustering (K-Means as approximation of K-Medoids)
    on the aggregated node features and then choose the nodes closest to the K-means centers.

    rep {'feature', 'embedding', 'aggregation'}
    init {‘k-means++’, ‘random’}
    """

    def __init__(self, data, model, seed, args,
                 representation='aggregation',
                 encoder='gcn',
                 initialization='k-means++'):
        super(ClusterBased, self).__init__(data, model, seed, args)
        self.representation = representation
        self.encoder = encoder
        self.initialization = None if initialization != 'k-means++' else initialization

    def query(self, b):

        # Get node representations
        x = self.get_node_representation(self.representation, self.encoder)

        # Perform K-Means clustering:
        kmeans = Cluster(
            n_clusters=b, n_dim=x.shape[1], seed=self.seed, device=self.args.device)
        kmeans.train(x.cpu().numpy())
        centers = torch.tensor(kmeans.get_centroids(),
                               dtype=x.dtype, device=x.device)

        # Obtain the centers
        indices = list(np.where(self.data.train_mask != 0)[0])
        for center in centers:
            center = center.to(dtype=x.dtype, device=x.device)
            dist_map = torch.linalg.norm(x - center, dim=1)
            dist_map[indices] = torch.tensor(
                np.infty, dtype=dist_map.dtype, device=dist_map.device)
            idx = int(torch.argmin(dist_map))
            indices.append(idx)

        return torch.tensor(indices)


class PartitionBased(ActiveLearning):
    """
    Partition:
    Our method, which first partitions the graph into communities, and
    performs clustering over each graph community on the aggregated node features.

    rep {'none', 'embed', 'prop'}
    init {‘k-means++’, ‘random’}
    compensation {float: 0 - 1}
    """

    def __init__(self, data, model, seed, args,
                 representation='aggregation',
                 encoder='gcn',
                 initialization='k-means++',
                 compensation=1):
        super(PartitionBased, self).__init__(data, model, seed, args)
        self.representation = representation
        self.encoder = encoder
        self.initialization = None if initialization != 'k-means++' else initialization
        self.compensation = compensation

    def query(self, b):
        # Perform graph partition (preprocessed)
        self.num_parts = int(np.ceil(b / self.num_centers))
        compensation = 0
        if self.num_parts > self.data.max_part:
            self.num_parts = self.data.max_part
            compensation = self.compensation

        partitions = np.array(self.data.partitions[self.num_parts].cpu())

        # Get node representations
        x = self.get_node_representation(self.representation, self.encoder)

        # Determine the number of partitions and number of centers
        part_size = self.split_cluster(b, partitions, x)

        # Iterate over each partition
        indices = list(np.where(self.data.train_mask != 0)[0])

        for i in range(self.num_parts):
            part_id = np.where(partitions == i)[0]
            masked_id = [i for i, x in enumerate(part_id) if x in indices]
            xi = x[part_id]

            n_clusters = part_size[i]
            if n_clusters <= 0:
                continue

            # Perform K-Means clustering:
            kmeans = Cluster(
                n_clusters=n_clusters, n_dim=xi.shape[1], seed=self.seed, device=self.args.device)
            kmeans.train(xi.cpu().numpy())
            centers = kmeans.get_centroids()

            # Compensating for the interference across partitions
            dist = None
            if self.compensation > 0:
                dist_to_center = torch.ones(
                    x.shape[0], dtype=x.dtype, device=x.device) * np.infty
                for idx in indices:
                    dist_to_center = torch.minimum(
                        dist_to_center, torch.linalg.norm(x - x[idx], dim=1))
                dist = dist_to_center[part_id]

            # Obtain the centers
            for center in centers:
                center = torch.tensor(center, dtype=x.dtype, device=x.device)
                dist_map = torch.linalg.norm(xi - center, dim=1)
                if self.compensation > 0:
                    dist_map -= dist * compensation
                dist_map[masked_id] = torch.tensor(
                    np.infty, dtype=dist_map.dtype, device=dist_map.device)
                idx = int(torch.argmin(dist_map))
                masked_id.append(idx)
                indices.append(part_id[idx])
        return torch.tensor(indices)


class ALIN(ActiveLearning):
    """
    """

    def __init__(self, data, model, seed, args,
                 representation='aggregation',
                 encoder='gcn',
                 initialization='k-means++',
                 compensation=1):
        super(ALIN, self).__init__(data, model, seed, args)
        self.representation = representation
        self.encoder = encoder
        self.initialization = None if initialization != 'k-means++' else initialization
        self.compensation = compensation

    def query_first_time(self, b):
        self.num_parts = int(np.ceil(b / self.num_centers))
        compensation = 0
        if self.num_parts > self.data.max_part:
            self.num_parts = self.data.max_part
            compensation = self.compensation
        partitions = np.array(self.data.partitions[self.num_parts].cpu())
        x = self.get_node_representation(self.representation, self.encoder)
        part_size = self.split_cluster(b, partitions, x)
        indices = list(np.where(self.data.train_mask != 0)[0])
        for i in range(self.num_parts):
            part_id = np.where(partitions == i)[0]
            masked_id = [i for i, x in enumerate(part_id) if x in indices]
            xi = x[part_id]
            n_clusters = part_size[i]
            if n_clusters <= 0:
                continue
            kmeans = Cluster(
                n_clusters=n_clusters, n_dim=xi.shape[1], seed=self.seed, device=self.args.device)
            kmeans.train(xi.cpu().numpy())
            centers = kmeans.get_centroids()
            dist = None
            if self.compensation > 0:
                dist_to_center = torch.ones(
                    x.shape[0], dtype=x.dtype, device=x.device) * np.infty
                for idx in indices:
                    dist_to_center = torch.minimum(
                        dist_to_center, torch.linalg.norm(x - x[idx], dim=1))
                dist = dist_to_center[part_id]
            for center in centers:
                center = torch.tensor(center, dtype=x.dtype, device=x.device)
                dist_map = torch.linalg.norm(xi - center, dim=1)
                if self.compensation > 0:
                    dist_map -= dist * compensation
                dist_map[masked_id] = torch.tensor(
                    np.infty, dtype=dist_map.dtype, device=dist_map.device)
                idx = int(torch.argmin(dist_map))
                masked_id.append(idx)
                indices.append(part_id[idx])
        return torch.tensor(indices)

    def query(self, b):
        logits = self.clf(self.data.x, self.data.adj_t)
        entropy = -torch.sum(F.softmax(logits, dim=1) *
                             F.log_softmax(logits, dim=1), dim=1)
        entropy[np.where(self.data.train_mask != 0)[0]] = 0
        node_score = torch.clone(entropy)
        node_score_normalized = (
            node_score - node_score.min()) / (node_score.max() - node_score.min())
        batch_size = 8192
        num_nodes = self.data.num_nodes
        with torch.no_grad():
            logits_t = logits @ logits.t()
            edge_scores = torch.zeros(num_nodes).to(self.args.device)
            for batch_start in range(0, num_nodes, batch_size):
                batch_end = min(batch_start + batch_size, num_nodes)
                current_batch = torch.arange(
                    batch_start, batch_end).to(self.args.device)
                # Ensure both tensors are on the same device
                tmp_nodes = torch.arange(num_nodes).to(self.args.device)
                edges = torch.cartesian_prod(
                    current_batch, tmp_nodes).to(self.args.device)
                outs_edge = logits_t[edges[:, 0], edges[:, 1]]
                sum_probs = torch.sum(outs_edge.view(
                    batch_end - batch_start, num_nodes), dim=-1)
                edge_scores[batch_start:batch_end] = sum_probs
            edge_scores -= degree(self.data.adj_t.storage._row,
                                  num_nodes, dtype=torch.long)
            edge_score_normalized = (
                edge_scores - edge_scores.min()) / (edge_scores.max() - edge_scores.min())
            edge_score_normalized[self.data.train_mask != 0] = 0
        score = self.args.gamma_combine*node_score_normalized + \
            (1 - self.args.gamma_combine)*edge_score_normalized
        _, indices = torch.topk(score, k=b)
        return indices
