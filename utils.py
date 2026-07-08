import numpy as np
import torch
import random
import wandb
import omegaconf
import time

def set_seed(seed=0):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_wandb(cfg):
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if cfg.experiments.general.wandb == 'None':
        return
    kwargs = {'name': f'{ts}', 'project': cfg.experiments.general.project_name, 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.experiments.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')

def one_hot_encoding(X, n_bins):
    batch_encode = []
    for x in X:
        one_hot_emb = np.zeros(n_bins)
        one_hot_emb[x] = 1
        batch_encode.append(one_hot_emb)
    return np.array(batch_encode)

def to_tf_device(x, device, dtype=torch.float32):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)  # convert from numpy or list
    if device is not None:
        return x.to(dtype=dtype, device=device)
    else:
        return x.to(dtype=dtype)

class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.X[node_mask == 0] = - 1
            self.E = self.E * e_mask1 * e_mask2
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
        return self

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'