import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric
 
class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction='sum')
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples
    
class BinaryCrossEntropyMetric(Metric):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.add_state('total_bce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        # Important: `pos_weight` must live on the same device as `preds/target`.
        # Register as a buffer so Lightning/nn.Module `.to(device)` will move it.
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight)
        self.register_buffer("pos_weight", pos_weight)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n * n, d)
            target: Ground truth values     (bs * n * n, d). """
        # BCE expects float targets (0/1). `pos_weight` is optional.
        kwargs = {}
        if self.pos_weight is not None:
            kwargs["pos_weight"] = self.pos_weight
        output = F.binary_cross_entropy_with_logits(preds, target.float(), reduction='sum', **kwargs)
        self.total_bce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_bce / self.total_samples
