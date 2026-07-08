import torch
from torch import Tensor
import torch.nn as nn
import wandb
from modules.matrics.abstract_metrics import CrossEntropyMetric, BinaryCrossEntropyMetric

class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train, edge_loss_fn, edge_activation, pos_weight=None, device=None):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        if edge_loss_fn == 'CE':
            self.edge_loss = CrossEntropyMetric()
        elif edge_loss_fn == 'BCE':
            self.edge_loss = BinaryCrossEntropyMetric(pos_weight)
        # Keep torchmetrics metrics on the same device as the model/inputs.
        # (In Lightning the whole module is moved automatically, but this prevents mismatches
        # when `device` is provided and metrics are used early.)
        if device is not None:
            self.node_loss.to(device)
            self.edge_loss.to(device)
        self.lambda_train = lambda_train
        self.edge_loss_fn = edge_loss_fn
        self.edge_activation = edge_activation
        self.pos_weight = pos_weight
    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, log: bool, node_mask=None):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        log : boolean. """

        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n_u, 10)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n_u * n_v, n_edge_types)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n_u, 10)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))  # (bs * n_u * n_v, n_edge_types)

        # Remove masked rows
        mask_X_flatten = torch.reshape(node_mask, (-1,))
        if self.edge_loss_fn == 'CE':
            mask_E_flatten = (true_E != 0.).any(dim=-1)
        elif self.edge_loss_fn == 'BCE':
            mask_E = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
            mask_E_flatten = torch.reshape(mask_E, (-1,))

        flat_true_X = true_X[mask_X_flatten, :]
        flat_pred_X = masked_pred_X[mask_X_flatten, :]

        flat_true_E = true_E[mask_E_flatten, :]
        flat_pred_E = masked_pred_E[mask_E_flatten, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0

        loss_total = self.lambda_train[0] * loss_X + self.lambda_train[1] * loss_E

        if log:
            to_log = {"train_loss/batch_CE": (loss_total).detach(),
                      "train_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1}

            if wandb.run:
                wandb.log(to_log, commit=True)

        return loss_total

    def reset(self):
        for metric in [self.node_loss, self.edge_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1

        to_log = {"train_epoch/total_CE": epoch_node_loss + epoch_edge_loss,
                  "train_epoch/X_CE": epoch_node_loss,
                  "train_epoch/E_CE": epoch_edge_loss}
        if wandb.run:
            wandb.log(to_log, commit=True)

        return to_log
