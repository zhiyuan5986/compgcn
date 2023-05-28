import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

# from tasks import build_task
# from ..layers.HeteroLinear import HeteroFeature
from utils.utils import get_nodes_dict
from tasks.link_prediction import LinkPrediction
from dgl.nn import HeteroEmbedding, HeteroLinear
class HeteroFeature(nn.Module):
    r"""
    This is a feature preprocessing component which is dealt with various heterogeneous feature situation.

    In general, we will face the following three situations.

        1. The dataset has not feature at all.

        2. The dataset has features in every node type.

        3. The dataset has features of a part of node types.

    To deal with that, we implement the HeteroFeature.In every situation, we can see that

        1. We will build embeddings for all node types.

        2. We will build linear layer for all node types.

        3. We will build embeddings for parts of node types and linear layer for parts of node types which have original feature.

    Parameters
    ----------
    h_dict: dict
        Input heterogeneous feature dict,
        key of dict means node type,
        value of dict means corresponding feature of the node type.
        It can be None if the dataset has no feature.
    n_nodes_dict: dict
        Key of dict means node type,
        value of dict means number of nodes.
    embed_size: int
        Dimension of embedding, and used to assign to the output dimension of Linear which transform the original feature.
    need_trans: bool, optional
        A flag to control whether to transform original feature linearly. Default is ``True``.
    act : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

args.dataset,
    -----------
    embed_dict : nn.ParameterDict
        store the embeddings

    hetero_linear : HeteroLinearLayer
        A heterogeneous linear layer to transform original feature.
    """

    def __init__(self, h_dict, n_nodes_dict, embed_size, act=None, need_trans=True, all_feats=True):
        super(HeteroFeature, self).__init__()
        self.n_nodes_dict = n_nodes_dict
        self.embed_size = embed_size
        self.h_dict = h_dict
        self.need_trans = need_trans

        self.type_node_num_sum = [0]
        self.all_type = []
        for ntype, type_num in n_nodes_dict.items():
            num_now = self.type_node_num_sum[-1]
            num_now += type_num
            self.type_node_num_sum.append(num_now)
            self.all_type.append(ntype)
        self.type_node_num_sum = torch.tensor(self.type_node_num_sum)

        linear_dict = {}
        embed_dict = {}
        for ntype, n_nodes in self.n_nodes_dict.items():
            h = h_dict.get(ntype)
            if h is None:
                if all_feats:
                    embed_dict[ntype] = n_nodes
            else:
                linear_dict[ntype] = h.shape[1]
        self.embes = HeteroEmbedding(embed_dict, embed_size)
        if need_trans:
            self.linear = HeteroLinear(linear_dict, embed_size)
        self.act = act  # activate

    def forward(self):
        out_dict = {}
        out_dict.update(self.embes.weight)
        tmp = self.linear(self.h_dict)
        if self.act:  # activate
            for x, y in tmp.items():
                tmp.update({x: self.act(y)})
        out_dict.update(tmp)
        return out_dict

    def forward_nodes(self, id_dict):
        # Turn "id_dict" into a dictionary if "id_dict" is a tensor, and record the corresponding relationship in "to_pos"
        id_tensor = None
        if torch.is_tensor(id_dict):
            device = id_dict.device
        else:
            device = id_dict.get(next(iter(id_dict))).device

        if torch.is_tensor(id_dict):
            id_tensor = id_dict
            self.type_node_num_sum = self.type_node_num_sum.to(device)
            id_dict = {}
            to_pos = {}
            for i, x in enumerate(id_tensor):
                tmp = torch.where(self.type_node_num_sum <= x)[0]
                if len(tmp) > 0:
                    tmp = tmp.max()
                    now_type = self.all_type[tmp]
                    now_id = x - self.type_node_num_sum[tmp]
                    if now_type not in id_dict.keys():
                        id_dict[now_type] = []
                    id_dict[now_type].append(now_id)
                    if now_type not in to_pos.keys():
                        to_pos[now_type] = []
                    to_pos[now_type].append(i)
            for ntype in id_dict.keys():
                id_dict[ntype] = torch.tensor(id_dict[ntype], device=device)

        embed_id_dict = {}
        linear_id_dict = {}
        for entype, id in id_dict.items():
            if self.h_dict.get(entype) is None:
                embed_id_dict[entype] = id
            else:
                linear_id_dict[entype] = id
        out_dict = {}
        tmp = self.embes(embed_id_dict)
        out_dict.update(tmp)
        # for key in self.h_dict:
        #     self.h_dict[key] = self.h_dict[key].to(device)
        h_dict = {}
        for key in linear_id_dict:
            linear_id_dict[key] = linear_id_dict[key].to('cpu')
        for key in linear_id_dict:
            h_dict[key] = self.h_dict[key][linear_id_dict[key]].to(device)
        tmp = self.linear(h_dict)
        if self.act:  # activate
            for x, y in tmp.items():
                tmp.update({x: self.act(y)})
        for entype in linear_id_dict:
            out_dict[entype] = tmp[entype]

        # The result corresponds to the original position according to the corresponding relationship
        if id_tensor is not None:
            out_feat = [None] * len(id_tensor)
            for ntype, feat_list in out_dict.items():
                for i, feat in enumerate(feat_list):
                    now_pos = to_pos[ntype][i]
                    out_feat[now_pos] = feat.data
            out_dict = torch.stack(out_feat, dim=0)

        return out_dict

class BaseFlow(ABC):
    candidate_optimizer = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        'Adadelta': torch.optim.Adadelta
    }

    def __init__(self, args):
        """

        Parameters
        ----------
        args

        Attributes
        -------------
        evaluate_interval: int
            the interval of evaluation in validation
        """
        super(BaseFlow, self).__init__()
        self.evaluator = None
        self.evaluate_interval = 1
        if hasattr(args, '_checkpoint'):
            self._checkpoint = os.path.join(args._checkpoint, f"{args.model_name}_{args.dataset_name}.pt")
        else:
            if hasattr(args, 'load_from_pretrained'):
                self._checkpoint = os.path.join(args.output_dir,
                                                f"{args.model_name}_{args.dataset_name}_{args.task}.pt")
            else:
                self._checkpoint = None

        # stage flags: whether to run the corresponding stages
        # todo: only take effects in node classification trainer flow

        # args.training_flag = getattr(args, 'training_flag', True)
        # args.validation_flag = getattr(args, 'validation_flag', True)
        args.test_flag = getattr(args, 'test_flag', True)
        args.prediction_flag = getattr(args, 'prediction_flag', False)
        args.use_uva = getattr(args, 'use_uva', False)

        self.args = args
        self.logger = self.args.logger
        self.model_name = args.model_name
        self.model = args.model
        self.device = args.device
        self.task = LinkPrediction(args)
        # self.task = build_task(args)
        if self.args.use_uva:
            self.hg = self.task.get_graph()
        else:
            self.hg = self.task.get_graph().to(self.device)
        # self.args.meta_paths_dict = self.task.dataset.meta_paths_dict
        # self.patience = args.patience
        self.max_epoch = args.max_epoch
        self.optimizer = None
        self.loss_fn = self.task.get_loss_fn()

    def preprocess(self):
        r"""
        Every trainerflow should run the preprocess_feature if you want to get a feature preprocessing.
        The Parameters in input_feature will be added into optimizer and input_feature will be added into the model.

        Attributes
        -----------
        input_feature : HeteroFeature
            It will return the processed feature if call it.

        """
        if hasattr(self.args, 'activation'):
            if hasattr(self.args.activation, 'weight'):
                import torch.nn as nn
                act = nn.PReLU()
            else:
                act = self.args.activation
        else:
            act = None
        # useful type selection
        if hasattr(self.args, 'feat'):
            pass
        else:
            # Default 0, nothing to do.
            self.args.feat = 0
        self.feature_preprocess(act)
        self.optimizer.add_param_group({'params': self.input_feature.parameters()})
        # for early stop, load the model with input_feature module.
        self.model.add_module('input_feature', self.input_feature)
        self.load_from_pretrained()

    def feature_preprocess(self, act):
        """
        Feat
            0, 1 ,2
        Node feature
            1 node type & more than 1 node types
            no feature

        Returns
        -------

        """

        if self.hg.ndata.get('h', {}) == {} or self.args.feat == 2:
            if self.hg.ndata.get('h', {}) == {}:
                self.logger.feature_info('Assign embedding as features, because hg.ndata is empty.')
            else:
                self.logger.feature_info('feat2, drop features!')
                self.hg.ndata.pop('h')
            self.input_feature = HeteroFeature({}, get_nodes_dict(self.hg), self.args.hidden_dim,
                                            act=act).to(self.device)
        elif self.args.feat == 0:
            self.input_feature = self.init_feature(act)
        elif self.args.feat == 1:
            if self.args.task != 'node_classification':
                self.logger.feature_info('\'feat 1\' is only for node classification task, set feat 0!')
                self.input_feature = self.init_feature(act)
            else:
                h_dict = self.hg.ndata.pop('h')
                self.logger.feature_info('feat1, preserve target nodes!')
                self.input_feature = HeteroFeature({self.category: h_dict[self.category]}, get_nodes_dict(self.hg), self.args.hidden_dim,
                                                act=act).to(self.device)

    def init_feature(self, act):
        self.logger.feature_info("Feat is 0, nothing to do!")
        if isinstance(self.hg.ndata['h'], dict):
            # The heterogeneous contains more than one node type.
            input_feature = HeteroFeature(self.hg.ndata['h'], get_nodes_dict(self.hg),
                                            self.args.hidden_dim, act=act).to(self.device)
        elif isinstance(self.hg.ndata['h'], torch.Tensor):
            # The heterogeneous only contains one node type.
            input_feature = HeteroFeature({self.hg.ntypes[0]: self.hg.ndata['h']}, get_nodes_dict(self.hg),
                                            self.args.hidden_dim, act=act).to(self.device)
        return input_feature

    @abstractmethod
    def train(self):
        pass

    def _full_train_step(self):
        r"""
        Train with a full_batch graph
        """
        raise NotImplementedError

    def _mini_train_step(self):
        r"""
        Train with a mini_batch seed nodes graph
        """
        raise NotImplementedError

    def _full_test_step(self):
        r"""
        Test with a full_batch graph
        """
        raise NotImplementedError

    def _mini_test_step(self):
        r"""
        Test with a mini_batch seed nodes graph
        """
        raise NotImplementedError

    def load_from_pretrained(self):
        if hasattr(self.args, 'load_from_pretrained') and self.args.load_from_pretrained:
            try:
                ck_pt = torch.load(self._checkpoint)
                self.model.load_state_dict(ck_pt)
                self.logger.info('[Load Model] Load model from pretrained model:' + self._checkpoint)
            except FileNotFoundError:
                self.logger.info('[Load Model] Do not load the model from pretrained, '
                                      '{} doesn\'t exists'.format(self._checkpoint))
        # return self.model

    def save_checkpoint(self):
        if self._checkpoint and hasattr(self.model, "_parameters()"):
            torch.save(self.model.state_dict(), self._checkpoint)
