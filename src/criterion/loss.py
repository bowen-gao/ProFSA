import torch
import torch.nn.functional as F
from torch import nn


class InBatchSoftmax(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        logits_per_pocket: torch.Tensor,
        logits_per_mol: torch.Tensor,
        **kwargs,
    ):
        n_sample = logits_per_pocket.size(0)
        target = torch.arange(n_sample, dtype=torch.long).view(-1).cuda()

        probs_pocket = F.log_softmax(logits_per_pocket.float(), dim=-1)
        loss_pocket = F.nll_loss(probs_pocket, target, reduction=self.reduction)

        probs_mol = F.log_softmax(logits_per_mol.float(), dim=-1)
        loss_mol = F.nll_loss(probs_mol, target, reduction=self.reduction)

        loss = 0.5 * loss_pocket + 0.5 * loss_mol

        result = {
            "loss": loss,
            "loss_pocket": loss_pocket,
            "loss_mol": loss_mol,
            "target": target,
        }
        return result
