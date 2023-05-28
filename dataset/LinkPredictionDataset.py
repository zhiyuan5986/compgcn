import dgl
import math
import random
import numpy as np
import torch
from dgl.data.knowledge_graph import load_data
from abc import ABC, ABCMeta, abstractmethod
from dgl.data.utils import load_graphs
from dataset.base_dataset import BaseDataset
# from . import AcademicDataset, HGBDataset, OHGBDataset
from utils.utils import add_reverse_edges
# from dgl.data.utils import load_graphs

# __all__ = ['LinkPredictionDataset', 'HGB_LinkPrediction']


class LinkPredictionDataset(BaseDataset):
    """
    metric: Accuracy, multi-label f1 or multi-class f1. Default: `accuracy`
    """

    def __init__(self, *args, **kwargs):
        super(LinkPredictionDataset, self).__init__(*args, **kwargs)
        self.target_link = None
        self.target_link_r = None

    def get_split(self, val_ratio=0.1, test_ratio=0.2):
        """
        Get subgraphs for train, valid and test.
        Generally, the original will have train_mask and test_mask in edata, or we will split it automatically.

        If the original graph do not has the train_mask in edata, we default that there is no valid_mask and test_mask.
        So we will split the edges of the original graph into train/valid/test 0.7/0.1/0.2.

        The dataset has not validation_mask, so we split train edges randomly.
        Parameters
        ----------
        val_ratio : int
            The ratio of validation. Default: 0.1
        test_ratio : int
            The ratio of test. Default: 0.2

        Returns
        -------
        train_hg
        """

        val_edge_dict = {}
        test_edge_dict = {}
        out_ntypes = []
        train_graph = self.g

        # # NOTE: myself
        # ##########################
        # return train_graph, None, None, None, None
        # ##########################
        for i, etype in enumerate(self.target_link):
            num_edges = self.g.num_edges(etype)
            if 'train_mask' not in self.g.edges[etype].data:
                """
                split edges into train/valid/test.
                """
                random_int = torch.randperm(num_edges)
                val_index = random_int[:int(num_edges * val_ratio)]
                val_edge = self.g.find_edges(val_index, etype)
                test_index = random_int[int(num_edges * val_ratio):int(num_edges * (test_ratio + val_ratio))]
                test_edge = self.g.find_edges(test_index, etype)

                val_edge_dict[etype] = val_edge
                test_edge_dict[etype] = test_edge
                out_ntypes.append(etype[0])
                out_ntypes.append(etype[2])
                train_graph = dgl.remove_edges(train_graph, torch.cat((val_index, test_index)), etype)
                # train_graph = dgl.remove_edges(train_graph, val_index, etype)
                if self.target_link_r is None:
                    pass
                else:
                    reverse_edge = self.target_link_r[i]
                    train_graph = dgl.remove_edges(train_graph, torch.arange(train_graph.num_edges(reverse_edge)),
                                                   reverse_edge)
                    edges = train_graph.edges(etype=etype)
                    train_graph = dgl.add_edges(train_graph, edges[1], edges[0], etype=reverse_edge)

            else:
                if 'valid_mask' not in self.g.edges[etype].data:
                    train_idx = self.g.edges[etype].data['train_mask']
                    random_int = torch.randperm(int(train_idx.sum()))
                    val_index = random_int[:int(train_idx.sum() * val_ratio)]
                    val_edge = self.g.find_edges(val_index, etype)

                else:
                    val_mask = self.g.edges[etype].data['valid_mask'].squeeze()
                    val_index = torch.nonzero(val_mask).squeeze()
                    val_edge = self.g.find_edges(val_index, etype)

                test_mask = self.g.edges[etype].data['test_mask'].squeeze()
                test_index = torch.nonzero(test_mask).squeeze()
                test_edge = self.g.find_edges(test_index, etype)

                val_edge_dict[etype] = val_edge
                test_edge_dict[etype] = test_edge
                out_ntypes.append(etype[0])
                out_ntypes.append(etype[2])
                #self.val_label = train_graph.edges[etype[1]].data['label'][val_index]
                self.test_label = train_graph.edges[etype[1]].data['label'][test_index]
                train_graph = dgl.remove_edges(train_graph, torch.cat((val_index, test_index)), etype)

        # train_graph = dgl.remove_edges(train_graph, torch.cat((val_index, test_index)), 'item-user')
        self.out_ntypes = set(out_ntypes)
        val_graph = dgl.heterograph(val_edge_dict,
                                    {ntype: self.g.number_of_nodes(ntype) for ntype in set(out_ntypes)})
        test_graph = dgl.heterograph(test_edge_dict,
                                     {ntype: self.g.number_of_nodes(ntype) for ntype in set(out_ntypes)})

        # todo: val/test negative graphs should be created before training rather than
        #  create them dynamically in every evaluation.
        return self.g, val_graph, test_graph, None, None


class DEMO_LinkPrediction(LinkPredictionDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(DEMO_LinkPrediction, self).__init__(*args, **kwargs)
        self.g = self.load_demo(dataset_name)

    def load_link_pred(self, path):
        u_list = []
        v_list = []
        label_list = []
        with open(path) as f:
            for i in f.readlines():
                u, v, label = i.strip().split(', ')
                u_list.append(int(u))
                v_list.append(int(v))
                label_list.append(int(label))
        return u_list, v_list, label_list

    def load_demo(self, dataset_name):
        self.dataset_name = dataset_name
        # data_path = "/home/users/lqa/Documents/DataScience/FinalProject/preprocess_datasets_for_openhgnn/demo_graph.bin"
        # breakpoint()
        # print(dataset_name)
        data_path = f'./graph/{dataset_name}.bin'
        # data_path = './openhgnn/dataset/demo_graph.bin'
        g, _ = dgl.load_graphs(data_path)
        g = g[0].long()
        self.has_feature = True
        self.target_link = [('author', 'ref', 'paper')]
        self.target_link_r = None# [('paper', 'refed_by', 'author')]
        self.node_type = ['author', 'paper']
        # NOTE: for HetGNN
        # self.category = 'author'
        # v1
        # self.meta_paths_dict = {
        #     'APA': [("author", "ref", "paper"), ("paper", "refed_by", "author")],
        #     'APPA': [("author", "ref", "paper"), ('paper', 'cite', 'paper'),("paper", "refed_by", "author")],
        #     'PAAP':[('paper', 'refed_by', 'author'),('author', 'co_author', 'author'), ('author', 'ref', 'paper')],
        #     'PAP':[('paper', 'refed_by', 'author'), ('author', 'ref', 'paper')]
        # }
        # v2
        # self.meta_paths_dict = {
        #     'APA': [("author", "ref", "paper"), ("paper", "refed_by", "author")],
        #     'PAP':[('paper', 'refed_by', 'author'), ('author', 'ref', 'paper')],
        #     'AA':["author", "co_author", "author"],
        #     'PP':[("paper", "cite", "paper")],
        # }
        # v3
        # self.meta_paths_dict = {
        #     'APA': [("author", "ref", "paper"), ("paper", "refed_by", "author")],
        #     'APAPA':[("author", "ref", "paper"),('paper', 'refed_by', 'author'), ('author', 'ref', 'paper'), ('paper', 'refed_by', 'author')],
        #     'APPA':[("author", "ref", "paper"),("paper", "cite", "paper"), ('paper', 'refed_by', 'author')],
        #     'AA':[("author", "co_author", "author")],
        #     # 'PP':[("paper", "cite", "paper")],

        # }
        # v4
        # self.meta_paths_dict = {
        #     # 'PAP':[('paper', 'refed_by', 'author'), ('author', 'ref', 'paper')],
        #     'PAAP':[('paper', 'refed_by', 'author'),('author', 'co_author', 'author'), ('author', 'ref', 'paper')],
        #     'PP':[("paper", "cite", "paper")],
        # }
        return g
    
    def get_split(self, val_ratio=0.2, test_ratio=0.1):
        return super(DEMO_LinkPrediction, self).get_split(val_ratio, test_ratio)


# def build_graph_from_triplets(num_nodes, num_rels, triplets):
#     """ Create a DGL graph. The graph is bidirectional because RGCN authors
#         use reversed relations.
#         This function also generates edge type and normalization factor
#         (reciprocal of node incoming degree)
#     """
#     g = dgl.graph(([], []))
#     g.add_nodes(num_nodes)
#     src, rel, dst = triplets
#     src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
#     rel = np.concatenate((rel, rel + num_rels))
#     edges = sorted(zip(dst, src, rel))
#     dst, src, rel = np.array(edges).transpose()
#     g.add_edges(src, dst)
#     norm = comp_deg_norm(g)
#     print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
#     return g, rel.astype('int64'), norm.astype('int64')


# def comp_deg_norm(g):
#     g = g.local_var()
#     in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
#     norm = 1.0 / in_deg
#     norm[np.isinf(norm)] = 0
#     return norm


# def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
#     """Sample edges by neighborhool expansion.
#     This guarantees that the sampled edges form a connected graph, which
#     may help deeper GNNs that require information from more than one hop.
#     """
#     edges = np.zeros((sample_size), dtype=np.int32)

#     # initialize
#     sample_counts = np.array([d for d in degrees])
#     picked = np.array([False for _ in range(n_triplets)])
#     seen = np.array([False for _ in degrees])

#     for i in range(0, sample_size):
#         weights = sample_counts * seen

#         if np.sum(weights) == 0:
#             weights = np.ones_like(weights)
#             weights[np.where(sample_counts == 0)] = 0

#         probabilities = (weights) / np.sum(weights)
#         chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
#                                          p=probabilities)
#         chosen_adj_list = adj_list[chosen_vertex]
#         seen[chosen_vertex] = True

#         chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
#         chosen_edge = chosen_adj_list[chosen_edge]
#         edge_number = chosen_edge[0]

#         while picked[edge_number]:
#             chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
#             chosen_edge = chosen_adj_list[chosen_edge]
#             edge_number = chosen_edge[0]

#         edges[i] = edge_number
#         other_vertex = chosen_edge[1]
#         picked[edge_number] = True
#         sample_counts[chosen_vertex] -= 1
#         sample_counts[other_vertex] -= 1
#         seen[other_vertex] = True

#     return edges


# def sample_edge_uniform(adj_list, degrees, n_triplets, sample_size):
#     """Sample edges uniformly from all the edges."""
#     all_edges = np.arange(n_triplets)
#     return np.random.choice(all_edges, sample_size, replace=False)
