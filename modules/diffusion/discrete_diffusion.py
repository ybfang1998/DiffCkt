import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.diffusion import diffusion_utils
from modules.diffusion.transformer import GraphTransformer
import utils
from modules.matrics.train_metrics import TrainLossDiscrete

import pytorch_lightning as pl
import time
import os

class DiscreteDiffusion(pl.LightningModule):
    def __init__(self,
                 cfg,
                 log_every_steps,
                 node_dist):
        super().__init__()

        self.cfg = cfg
        self.T = cfg.model.T
        self.output_dims = cfg.datasets.Diffckt.output_dims
        self.input_dims = cfg.datasets.Diffckt.input_dims

        self.conditional = cfg.model.conditional

        # if not conditional, only t as condition
        if self.conditional:
            self.input_dims['y'] += 1
        else:
            self.input_dims['y'] = 1

        self.log_every_steps = log_every_steps

        self.noise_schedule = PredefinedNoiseScheduleDiscrete('cosine',
                                                              timesteps=self.T)
        # The dimension of the transition matrix for E is 2 because we only have two classes for each channel of edge: 0 and 1.
        self.transition_model = DiscreteUniformTransition(self.output_dims['X'], 2, self.output_dims['y'])
        #self.transition_model = DiscreteUniformTransition(self.output_dims['X'], self.output_dims['E'], self.output_dims['y'])


        cfg_graph_transformer = cfg.model.GraphTransformer
        self.model = GraphTransformer(cfg_graph_transformer.n_layers,
                                      self.input_dims,
                                      cfg_graph_transformer.hidden_mlp_dims,
                                      cfg_graph_transformer.hidden_dims,
                                      self.output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())
        
        x_limit = torch.ones(self.output_dims['X']) / self.output_dims['X']
        e_limit = torch.ones(self.output_dims['E']) * 0.5
        y_limit = torch.ones(self.output_dims['y']) / self.output_dims['y']
        self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        self.node_dist = node_dist

        # Loss and Metrics
        self.edge_loss_fn = cfg.experiments.train.edge_loss
        self.edge_activation = cfg.experiments.train.edge_activation

        self.train_loss = TrainLossDiscrete(cfg.experiments.train.lambda_train, self.edge_loss_fn, self.edge_activation, device=self.device)
        self.save_hyperparameters()

    def training_step(self, data, i):
        if self.edge_loss_fn == 'CE':
            data = {k: utils.to_tf_device(v, self.device, dtype=torch.float32) for k, v in data.items()}
        elif self.edge_loss_fn == 'BCE':
            data = {k: utils.to_tf_device(v, self.device, dtype=torch.float32) if k != 'E'
                    else utils.to_tf_device(v, self.device, dtype=torch.long) for k, v in data.items()}
        
        node_mask = data['node_mask'].to(dtype=torch.bool)
        data = utils.PlaceHolder(data['X'], data['E'], data['y']).mask(node_mask)
        X, E, y = data.X, data.E, data.y
        noisy_data = self.apply_noise(X, E, node_mask, y)
        if self.conditional:
            y = torch.cat((y, noisy_data['t']), dim=1)
        else:
            y = noisy_data['t']

        pred = self.model(noisy_data['X_t'], noisy_data['E_t'], y, node_mask)
        # Loss
        loss = self.train_loss(pred.X, pred.E, X, utils.to_tf_device(E, self.device), log=i % self.log_every_steps == 0, node_mask=node_mask)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.experiments.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.experiments.train.weight_decay)

    def on_fit_start(self) -> None:
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        # self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.log("train_epoch/total_CE", to_log['train_epoch/total_CE'], sync_dist=True)
        self.print(f"Epoch {self.current_epoch}: total_CE: {to_log['train_epoch/total_CE'] :.3f}")
        self.print(f" -- X_CE: {to_log['train_epoch/X_CE'] :.3f} --"
                   f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                   f" -- {time.time() - self.start_epoch_time:.1f}s ")

    def predict_step(self, data, i, num_nodes=None):
        G = self.sample_batch(num_nodes)
        return {'G': G}

    def on_predict_start(self):
        # available after trainer has setup
        self.total_batches = self.trainer.num_predict_batches[0]  # first dataloader
        self.processed_batches = 0
        print(f"Total prediction batches = {self.total_batches}")

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.processed_batches += 1
        proportion = self.processed_batches / self.total_batches
        print(f"Predicted batch {self.processed_batches}/{self.total_batches} "
              f"({proportion:.2%} done)")


    def apply_noise(self, X, E, node_mask, y):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(E.size(0), 1), device=self.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, dx_in, dx_out), (bs, de_in, de_out)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n_d, dx_d_out)
        probE = utils.to_tf_device(F.one_hot(E, num_classes=2), self.device) @ Qtb.E.unsqueeze(1).unsqueeze(1)  # (bs, n_d, n_n, de_out, 2)

        sampled_t = diffusion_utils.sample_discrete_features(probX, probE, node_mask)

        #One-hot Embedding
        X_t = F.one_hot(sampled_t.X, num_classes=self.output_dims['X'])
        assert X.shape == X_t.shape
        E_t = sampled_t.E
        assert E.shape == E_t.shape

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E,
                      'node_mask': node_mask}
        return noisy_data

    @torch.no_grad()
    def sample_batch(self, num_nodes=None):
        """
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        batch_size = self.cfg.experiments.test.batch_size

        if num_nodes is None:
            n_nodes = self.node_dist.sample((batch_size,)).to(self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes

        n_max = 22
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(X)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        G = []
        for i in range(batch_size):
            g = {}
            n_node = n_nodes[i].item()
            g['X'] = X[i].cpu()
            g['E'] = E[i].cpu()
            g["node_mask"] = node_mask[i].cpu()
            G.append(g)
        return G

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape

        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        if self.conditional:
            y = torch.cat((y, t), dim=1)
        else:
            y = t

        pred = self.model(X_t, E_t, y, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)
        pred_E = F.sigmoid(pred.E)


        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)  # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E,
                                                                                           edge_loss_fn=self.edge_loss_fn)
        pred_E_two_hot = torch.stack([1.0 - pred_E, pred_E], dim=-1)
        pred_E_flat = pred_E_two_hot.flatten(start_dim=1, end_dim=-2)
        weighted_E = pred_E_flat.unsqueeze(-1) * p_s_and_t_given_0_E
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, 25, 2)

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask)


        X_s = F.one_hot(sampled_s.X, num_classes=self.output_dims['X']).float()
        assert X_s.shape == X_t.shape

        E_s = sampled_s.E
        assert E_t.shape == E_s.shape

        out_one_hot = utils.PlaceHolder(X_s, E_s, y=torch.zeros(y.shape[0], 0))
        out_discrete = utils.PlaceHolder(X_s, E_s, y=torch.zeros(y.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y), out_discrete.mask(node_mask, collapse=True).type_as(y)


class DiscreteUniformTransition:
    def __init__(self, x_classes: int, e_classes: int, y_classes: int):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes

        self.u_e = torch.ones(1, self.E_classes, self.E_classes)
        if self.E_classes > 0:
            self.u_e = self.u_e / self.E_classes

        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            betas = diffusion_utils.cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == 'custom':
            betas = diffusion_utils.custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer('betas', torch.from_numpy(betas).float())

        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)
        # print(f"[Noise schedule: {noise_schedule}] alpha_bar:", self.alphas_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()]