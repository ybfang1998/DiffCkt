import numpy as np
import torch
from torch.nn import functional as F

import utils


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()

def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5       # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)

def sample_discrete_features(probX, probE, node_mask):
    ''' Sample features from multinomial distribution with given probabilities (probU, probV, probE)
        :param probU: bs, F, n, du_out        device features
        :param probV: bs, F, n, dv_out        net features
        :param probE: bs, n, n, de_out     edge features
    '''
    bs, n, _ = probX.shape
    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)       # (bs * n, dx_out)

    # Sample X
    X_t = probX.multinomial(1)                                  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)     # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    # Sample E
    E_t = torch.bernoulli(probE[..., 1]).reshape(bs, n, n, 5, 5)

    upper_mask = torch.triu(torch.ones(n, n, device=E_t.device), diagonal=1).bool().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    E_t_upper = torch.where(upper_mask, E_t, torch.zeros_like(E_t))
    E_t_lower = E_t_upper.permute(0, 2, 1, 4, 3)
    E_t = (E_t_upper + E_t_lower).reshape(bs, n, n, 25)

    return utils.PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))

def sample_discrete_feature_noise(limit_dist, node_mask):
    """ Sample from the limit distribution of the diffusion process"""
    bs, n_max = node_mask.shape
    long_mask = node_mask.long()

    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    y_limit = limit_dist.y[None, :].expand(bs, -1)

    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_X = U_X.type_as(long_mask)
    U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()

    U_y = torch.empty((bs, 0))
    U_y = U_y.type_as(long_mask)

    U_E = torch.bernoulli(e_limit).type_as(long_mask).float()
    # Get upper triangular part of edge noise, without main diagonal
    U_E = U_E.reshape(bs, n_max, n_max, 5, 5)
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :, :] = 1
    U_E = U_E * upper_triangular_mask
    U_E = (U_E + U_E.permute(0, 2, 1, 4, 3)).reshape(bs, n_max, n_max, 25)
    return utils.PlaceHolder(X=U_X, E=U_E, y=U_y).mask(node_mask)

def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb, edge_loss_fn='CE'):
    """ M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
        X_t: bs, n, dt          or bs, n, n, dt  or bs, n, n, n_edge_types, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
    """
    # Flatten feature tensors
    # Careful with this line. It does nothing if X is a node feature. If X is an edge features it maps to
    # bs x (n ** 2) x d for the case of CE edge loss, and bs x (n ** 2 * n_edge_types) x d for the case of BCE edge loss. The transition matrices should be computed accordingly.
    if edge_loss_fn == 'BCE':
        X_t = X_t.to(torch.float32)
        X_t = torch.stack([1.0 - X_t, X_t], dim=-1)
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)            # bs x N x dt

    Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
    left_term = X_t @ Qt_T                      # bs, N, d_t-1
    left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

    right_term = Qsb.unsqueeze(1)               # bs, 1, d0, d_t-1
    numerator = left_term * right_term          # bs, N, d0, d_t-1

    X_t_transposed = X_t.transpose(-1, -2)      # bs, dt, N

    prod = Qtb @ X_t_transposed                 # bs, d0, N
    prod = prod.transpose(-1, -2)               # bs, N, d0
    denominator = prod.unsqueeze(-1)            # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator
    return out