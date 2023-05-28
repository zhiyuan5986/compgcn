import dgl
import torch
import torch.nn.functional as F
from dgl.dataloading.negative_sampler import Uniform, GlobalUniform
from dgl.sampling import global_uniform_negative_sampling
from compgcnforkaggle.tasks.base_task import BaseTask
# from dataset import build_dataset
from compgcnforkaggle.dataset.LinkPredictionDataset import DEMO_LinkPrediction
from compgcnforkaggle.utils.evaluator import Evaluator


class LinkPrediction(BaseTask):
    r"""
    Link prediction tasks.

    Attributes
    -----------
    dataset : NodeClassificationDataset
        Task-related dataset

    evaluator : Evaluator
        offer evaluation metric

    Methods
    ---------
    get_graph :
        return a graph
    get_loss_fn :
        return a loss function
    """

    def __init__(self, args):
        super(LinkPrediction, self).__init__()
        self.name_dataset = args.dataset
        self.logger = args.logger
        self.dataset = DEMO_LinkPrediction(args.dataset, logger=self.logger)
        # self.dataset = build_dataset(args.dataset, 'link_prediction', logger=self.logger)
        # self.evaluator = Evaluator()
        self.train_hg, self.val_hg, self.test_hg, self.neg_val_graph, self.neg_test_graph = self.dataset.get_split()
        self.pred_hg = getattr(self.dataset, 'pred_graph', None)
        if self.val_hg is None and self.test_hg is None:
            pass
        else:
            self.val_hg = self.val_hg.to(args.device)
            self.test_hg = self.test_hg.to(args.device)
        self.evaluator = Evaluator(args.seed)
        if not hasattr(args, 'score_fn'):
            self.ScorePredictor = HeteroDistMultPredictor()
            args.score_fn = 'distmult'
        elif args.score_fn == 'dot-product':
            self.ScorePredictor = HeteroDotProductPredictor()
        elif args.score_fn == 'distmult':
            self.ScorePredictor = HeteroDistMultPredictor()
        # deprecated, new score predictor of these score_fn are in their model
        # elif args.score_fn in ['transe', 'transh', 'transr', 'transd'] :
        #     self.ScorePredictor = HeteroTransXPredictor(args.dis_norm)

        # NOTE: decrease
        self.negative_sampler = Uniform(1)
        # self.negative_sampler = GlobalUniform(50)
        # self.negative_sampler = global_uniform_negative_sampling(self.train_hg, 50) # Not True

        self.evaluation_metric = getattr(args, 'evaluation_metric', 'roc_auc')  # default evaluation_metric is roc_auc
        if args.dataset in ['wn18', 'FB15k', 'FB15k-237']:
            self.evaluation_metric = 'mrr'
            self.filtered = args.filtered
            if hasattr(args, "valid_percent"):
                self.dataset.modify_size(args.valid_percent, 'valid')
            if hasattr(args, "test_percent"):
                self.dataset.modify_size(args.test_percent, 'test')

        args.logger.info('[Init Task] The task: link prediction, the dataset: {}, the evaluation metric is {}, '
                         'the score function: {} '.format(self.name_dataset, self.evaluation_metric, args.score_fn))

    def get_out_ntype(self):
        ntype = []
        for l in self.dataset.target_link:
            ntype.append(l[0])
            ntype.append(l[2])
        return set(ntype)

    def get_graph(self):
        return self.dataset.g

    def get_loss_fn(self):
        return F.binary_cross_entropy_with_logits

    def get_evaluator(self, name):
        if name == 'acc':
            return self.evaluator.author_link_prediction
        elif name == 'mrr':
            return self.evaluator.mrr_
        elif name == 'academic_lp':
            return self.evaluator.author_link_prediction
        elif name == 'roc_auc':
            return self.evaluator.cal_roc_auc

    def evaluate(self, n_embedding, r_embedding=None, mode='test'):
        r"""

        Parameters
        ----------
        n_embedding: torch.Tensor
            the embedding of nodes
        r_embedding: torch.Tensor
            the embedding of relation types
        mode: str
            the evaluation mode, train/valid/test
        Returns
        -------

        """
        if self.evaluation_metric == 'acc':
            acc = self.evaluator.author_link_prediction
            return dict(Accuracy=acc)
        elif self.evaluation_metric == 'mrr':
            mrr_matrix = self.evaluator.mrr_(n_embedding, r_embedding,
                                             self.dataset.train_triplets, self.dataset.valid_triplets,
                                             self.dataset.test_triplets,
                                             score_predictor=self.ScorePredictor, hits=[1, 3, 10],
                                             filtered=getattr(self, 'filtered', 'filtered'), eval_mode=mode)
            return mrr_matrix
        elif self.evaluation_metric == 'roc_auc':
            if mode == 'test':
                eval_hg = self.test_hg
                neg_hg = self.neg_val_graph
            elif mode == 'valid':
                eval_hg = self.val_hg
                neg_hg = self.neg_val_graph
            else:
                raise ValueError('Mode error, supported test and valid.')
            if neg_hg is None:
                neg_hg = self.construct_negative_graph(eval_hg)
            p_score = torch.sigmoid(self.ScorePredictor(eval_hg, n_embedding, r_embedding))
            n_score = torch.sigmoid(self.ScorePredictor(neg_hg, n_embedding, r_embedding))
            p_label = torch.ones(len(p_score), device=p_score.device)
            n_label = torch.zeros(len(n_score), device=p_score.device)
            roc_auc = self.evaluator.cal_roc_auc(torch.cat((p_label, n_label)).cpu(), torch.cat((p_score, n_score)).cpu())
            loss = F.binary_cross_entropy_with_logits(torch.cat((p_score, n_score)), torch.cat((p_label, n_label)))
            return dict(roc_auc=roc_auc, loss=loss)
        else:
            return self.evaluator.link_prediction

    def predict(self, n_embedding, r_embedding, **kwargs):
        score = torch.sigmoid(self.ScorePredictor(self.pred_hg, n_embedding, r_embedding))
        indices = self.pred_hg.edges()
        return indices, score

    def tranX_predict(self):
        pred_triples_T = self.dataset.pred_triples.T
        score = torch.sigmoid(self.ScorePredictor(pred_triples_T[0], pred_triples_T[1], pred_triples_T[2]))
        indices = self.pred_hg.edges()
        return indices, score

    def downstream_evaluate(self, logits, evaluation_metric):
        if evaluation_metric == 'academic_lp':
            auc, macro_f1, micro_f1 = self.evaluator.author_link_prediction(logits, self.dataset.train_batch,
                                                                            self.dataset.test_batch)
            return dict(AUC=auc, Macro_f1=macro_f1, Mirco_f1=micro_f1)

    def get_batch(self):
        return self.dataset.train_batch, self.dataset.test_batch

    def get_train(self):
        return self.train_hg

    def get_labels(self):
        return self.dataset.get_labels()

    def dict2emd(self, r_embedding):
        r_emd = []
        for i in range(self.dataset.num_rels):
            r_emd.append(r_embedding[str(i)])
        return torch.stack(r_emd).squeeze()

    def construct_negative_graph(self, hg):
        e_dict = {
            etype: hg.edges(etype=etype, form='eid')
            for etype in hg.canonical_etypes}
        neg_srcdst = self.negative_sampler(hg, e_dict)
        neg_pair_graph = dgl.heterograph(neg_srcdst,
                                         {ntype: hg.number_of_nodes(ntype) for ntype in hg.ntypes})
        return neg_pair_graph


class HeteroDotProductPredictor(torch.nn.Module):
    """
    References: `documentation of dgl <https://docs.dgl.ai/guide/training-link.html#heterogeneous-graphs>_`
    
    """

    def forward(self, edge_subgraph, x, *args, **kwargs):
        """
        Parameters
        ----------
        edge_subgraph: dgl.Heterograph
            the prediction graph only contains the edges of the target link
        x: dict[str: torch.Tensor]
            the embedding dict. The key only contains the nodes involving with the target link.
    
        Returns
        -------
        score: torch.Tensor
            the prediction of the edges in edge_subgraph
        """

        with edge_subgraph.local_scope():
            for ntype in edge_subgraph.ntypes:
                edge_subgraph.nodes[ntype].data['x'] = x[ntype]
                for etype in edge_subgraph.canonical_etypes:
                    edge_subgraph.apply_edges(
                        dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            score = edge_subgraph.edata['score']
            if isinstance(score, dict):
                result = []
                for _, value in score.items():
                    result.append(value)
                score = torch.cat(result)
            return score.squeeze()


class HeteroDistMultPredictor(torch.nn.Module):

    def forward(self, edge_subgraph, x, r_embedding, *args, **kwargs):
        """
        DistMult factorization (Yang et al. 2014) as the scoring function,
        which is known to perform well on standard link prediction benchmarks when used on its own.
    
        In DistMult, every relation r is associated with a diagonal matrix :math:`R_{r} \in \mathbb{R}^{d \times d}`
        and a triple (s, r, o) is scored as
    
        .. math::
            f(s, r, o)=e_{s}^{T} R_{r} e_{o}
    
        Parameters
        ----------
        edge_subgraph: dgl.Heterograph
            the prediction graph only contains the edges of the target link
        x: dict[str: torch.Tensor]
            the node embedding dict. The key only contains the nodes involving with the target link.
        r_embedding: torch.Tensor
            the all relation types embedding
    
        Returns
        -------
        score: torch.Tensor
            the prediction of the edges in edge_subgraph
        """
        with edge_subgraph.local_scope():
            for ntype in edge_subgraph.ntypes:
                edge_subgraph.nodes[ntype].data['x'] = x[ntype]
            for etype in edge_subgraph.canonical_etypes:
                e = r_embedding[etype[1]]
                n = edge_subgraph.num_edges(etype)
                if 1 == len(edge_subgraph.canonical_etypes):
                    edge_subgraph.edata['e'] = e.expand(n, -1)
                else:
                    edge_subgraph.edata['e'] = {etype: e.expand(n, -1)}
                edge_subgraph.apply_edges(
                    dgl.function.u_mul_e('x', 'e', 's'), etype=etype)
                edge_subgraph.apply_edges(
                    dgl.function.e_mul_v('s', 'x', 'score'), etype=etype)

            score = edge_subgraph.edata['score']
            if isinstance(score, dict):
                result = []
                for _, value in score.items():
                    result.append(torch.sum(value, dim=1))
                score = torch.cat(result)
            else:
                score = torch.sum(score, dim=1)
            return score

# class HeteroTransXPredictor(torch.nn.Module):
#     def __init__(self, dis_norm):
#         super(HeteroTransXPredictor, self).__init__()
#         self.dis_norm = dis_norm

#     def forward(self, h, r, t):
#         h = F.normalize(h, 2, -1)
#         r = F.normalize(r, 2, -1)
#         t = F.normalize(t, 2, -1)
#         dist = torch.norm(h+r-t, self.dis_norm, dim=-1)
#         return dist
