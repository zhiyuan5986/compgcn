
import torch
import os
import pickle as pkl
torch.manual_seed(0)
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn
from dgl.nn import HeteroGraphConv, GATConv, SAGEConv, GraphConv
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    negative_sampler,
    NeighborSampler,
)
import tqdm
import copy
# https://openhgnn.readthedocs.io/en/latest/advanced_materials/developer_guide.html#evaluate-a-new-dataset

# 读入边


def read_txt(file):
    res_list = list()
    with open(file, "r") as f:
        line_list = f.readlines()
    for line in line_list:
        res_list.append(list(map(int, line.strip().split(' '))))      
    return res_list

base_path = "/kaggle/input/cs3319-02-project-1-graph-based-recommendation"


cite_file = "paper_file_ann.txt"
train_ref_file = "bipartite_train_ann.txt"
test_ref_file = "bipartite_test_ann.txt"
coauthor_file = "author_file_ann.txt"
feature_file = "feature.pkl"

citation = read_txt(os.path.join(base_path, cite_file))
existing_refs = read_txt(os.path.join(base_path, train_ref_file))
refs_to_pred = read_txt(os.path.join(base_path, test_ref_file))
coauthor = read_txt(os.path.join(base_path, coauthor_file))

feature_file = os.path.join(base_path, feature_file)
with open(feature_file, 'rb') as f:
      paper_feature = pkl.load(f)
        


print("Number of citation edges: {}\n\
Number of existing references: {}\n\
Number of author-paper pairs to predict: {}\n\
Number of coauthor edges: {}\n\
Shape of paper features: {}"
.format(len(citation), len(existing_refs), len(refs_to_pred), len(coauthor), paper_feature.shape))

# 转化为Dataframe

cite_edges = pd.DataFrame(citation, columns=['source', 'target'])
cite_edges = cite_edges.set_index(
    "c-" + cite_edges.index.astype(str)
)

ref_edges = pd.DataFrame(existing_refs, columns=['source', 'target'])
ref_edges = ref_edges.set_index(
    "r-" + ref_edges.index.astype(str)
)

pred_ref_edges = pd.DataFrame(refs_to_pred, columns=['source', 'target'])
pred_ref_edges = pred_ref_edges.set_index(
    "pr-" + pred_ref_edges.index.astype(str)
)


coauthor_edges = pd.DataFrame(coauthor, columns=['source', 'target'])
coauthor_edges = coauthor_edges.set_index(
    "a-" + coauthor_edges.index.astype(str)
)

print(cite_edges.head())
# ref_edges.head()
# coauthor_edges.head()
node_tmp = pd.concat([cite_edges.loc[:, 'source'], cite_edges.loc[:, 'target'], ref_edges.loc[:, 'target']])
node_papers = pd.DataFrame(index=pd.unique(node_tmp))


node_tmp = pd.concat([ref_edges['source'], coauthor_edges['source'], coauthor_edges['target']])
node_authors = pd.DataFrame(index=pd.unique(node_tmp))

print("Number of paper nodes: {}, number of author nodes: {}".format(len(node_papers), len(node_authors)))

# 构建异构图：三种不同的边
coauthor_torch_edges = torch.from_numpy(coauthor_edges.values)
coauthor_torch_edges = torch.cat((coauthor_torch_edges, torch.flip(coauthor_torch_edges, dims=[1])), dim=0)

cite_torch_edges = torch.from_numpy(cite_edges.values) 
ref_torch_edges = torch.from_numpy(ref_edges.values) 

pred_ref_torch_edges = torch.from_numpy(pred_ref_edges.values) 

graph_data = {
   ('author', 'co_author', 'author'): (coauthor_torch_edges[:, 0], coauthor_torch_edges[:, 1]),
   ('author', 'ref', 'paper'): (ref_torch_edges[:, 0], ref_torch_edges[:, 1]),
   ('paper', 'cite', 'paper'): (cite_torch_edges[:, 0], cite_torch_edges[:, 1]),
   ('paper', 'refed_by', 'author'): (ref_torch_edges[:, 1], ref_torch_edges[:, 0]),
   # ('paper', 'cited_by', 'paper'): (cite_torch_edges[:, 1], cite_torch_edges[:, 0]),
}
g = dgl.heterograph(graph_data)
print(g)
g.nodes['paper'].data['h'] = paper_feature
g.nodes['author'].data['h'] = torch.randn(len(node_authors), paper_feature.shape[1])
dgl.save_graphs("demo_graph.bin", g)

pred_network = dgl.heterograph({
   ('author', 'ref', 'paper'): (pred_ref_torch_edges[:, 0], pred_ref_torch_edges[:, 1]),
   # ('paper', 'refed_by', 'author'): (pred_ref_torch_edges[:, 1], pred_ref_torch_edges[:, 0]),
})
pred_network.nodes['paper'].data['h'] = paper_feature
pred_network.nodes['author'].data['h'] = g.nodes['author'].data['h']
dgl.save_graphs("test_graph.bin", pred_network)




