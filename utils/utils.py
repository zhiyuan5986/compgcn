import dgl
import torch
import numpy as np
class EarlyStopping(object):
    def __init__(self, patience=10, save_path=None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_loss = None
        self.early_stop = False
        if save_path is None:
            self.best_model = None
        self.save_path = save_path

    def step(self, loss, score, model):
        if isinstance(score, tuple):
            score = score[0]
        if self.best_loss is None:
            self.best_score = score
            self.best_loss = loss
            self.save_model(model)
        elif (loss > self.best_loss) and (score < self.best_score):
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (score >= self.best_score) and (loss <= self.best_loss):
                self.save_model(model)

            self.best_loss = np.min((loss, self.best_loss))
            self.best_score = np.max((score, self.best_score))
            self.counter = 0
        return self.early_stop

    def step_score(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_model(model)
        elif score < self.best_score:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if score >= self.best_score:
                self.save_model(model)

            self.best_score = np.max((score, self.best_score))
            self.counter = 0
        return self.early_stop

    def loss_step(self, loss, model):
        """
        
        Parameters
        ----------
        loss Float or torch.Tensor
        
        model torch.nn.Module

        Returns
        -------

        """
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if self.best_loss is None:
            self.best_loss = loss
            self.save_model(model)
        elif loss >= self.best_loss:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if loss < self.best_loss:
                self.save_model(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.counter = 0
        return self.early_stop

    def save_model(self, model):
        if self.save_path is None:
            self.best_model = copy.deepcopy(model)
        else:
            model.eval()
            torch.save(model.state_dict(), self.save_path)

    def load_model(self, model):
        if self.save_path is None:
            return self.best_model
        else:
            model.load_state_dict(torch.load(self.save_path))

def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

def ccorr(a, b):
    """
    Compute circular correlation of two tensors.
    Parameters
    ----------
    a: Tensor, 1D or 2D
    b: Tensor, 1D or 2D
    Notes
    -----
    Input a and b should have the same dimensions. And this operation supports broadcasting.
    Returns
    -------
    Tensor, having the same dimension as the input a.
    """
    try:
        from torch import irfft
        from torch import rfft
    except ImportError:
        from torch.fft import irfft2
        from torch.fft import rfft2

        def rfft(x, d):
            t = rfft2(x, dim=(-d))
            return torch.stack((t.real, t.imag), -1)

        def irfft(x, d, signal_sizes):
            return irfft2(torch.complex(x[:, :, 0], x[:, :, 1]), s=signal_sizes, dim=(-d))

    return irfft(com_mult(conj(rfft(a, 1)), rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def edata_in_out_mask(hg):

    """
    An API for CompGCN which needs identify the edge is IN or OUT.

    :param a heterogeneous graph:
    in_edges_mask means the edge is the original edge.
    out_edges_mask means the edge is the inverse edge.

    :return: hg
    """
    for canonical_etype in hg.canonical_etypes:
        eid = hg.all_edges(form='eid', etype=canonical_etype)
        if canonical_etype[1][:4] == 'rev-':
            hg.edges[canonical_etype].data['in_edges_mask'] = torch.zeros(eid.shape[0], device=hg.device).bool()
            hg.edges[canonical_etype].data['out_edges_mask'] = torch.ones(eid.shape[0], device=hg.device).bool()
        else:
            hg.edges[canonical_etype].data['out_edges_mask'] = torch.zeros(eid.shape[0], device=hg.device).bool()
            hg.edges[canonical_etype].data['in_edges_mask'] = torch.ones(eid.shape[0], device=hg.device).bool()

    return hg

def get_nodes_dict(hg):
    n_dict = {}
    for n in hg.ntypes:
        n_dict[n] = hg.num_nodes(n)
    return n_dict

def add_reverse_edges(hg, copy_ndata=True, copy_edata=True, ignore_one_type=True):
    # get node cnt for each ntype

    canonical_etypes = hg.canonical_etypes
    num_nodes_dict = {ntype: hg.number_of_nodes(ntype) for ntype in hg.ntypes}

    edge_dict = {}
    for etype in canonical_etypes:
        u, v = hg.edges(form='uv', order='eid', etype=etype)
        edge_dict[etype] = (u, v)
        edge_dict[(etype[2], etype[1] + '-rev', etype[0])] = (v, u)
    new_hg = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)

    # handle features
    if copy_ndata:
        node_frames = dgl.utils.extract_node_subframes(hg, None)
        dgl.utils.set_new_frames(new_hg, node_frames=node_frames)

    if copy_edata:
        for etype in canonical_etypes:
            edge_frame = hg.edges[etype].data
            for data_name, value in edge_frame.items():
                new_hg.edges[etype].data[data_name] = value
    return new_hg

def get_ntypes_from_canonical_etypes(canonical_etypes=None):
    ntypes = set()
    for etype in canonical_etypes:
        src = etype[0]
        dst = etype[2]
        ntypes.add(src)
        ntypes.add(dst)
    return ntypes