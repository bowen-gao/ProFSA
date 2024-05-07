import logging
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.timetool import with_time
from src.utils.torchtool import is_rank_zero

from .components.metrics import AUCROC, BEDROC, Accuracy, TopkAccuracy
from .loss import InBatchSoftmax

logger = logging.getLogger(__name__)


class ScreeningCriterion(nn.Module):
    def __init__(self, loss: InBatchSoftmax):
        super().__init__()
        self.loss = loss

        self.acc = Accuracy()
        self.aucroc = AUCROC()
        self.top3_acc = TopkAccuracy(topk=3)
        self.bedroc = BEDROC()

        # name, metric, step_compute
        self.train_metrics = []

        self.val_metrics = [
            ("Acc", self.acc, False),
            ("Top3Acc", self.top3_acc, False),
            ("AUCROC", self.aucroc, False),
            ("BEDROC", self.bedroc, False),
        ]

    def reset(self):
        for _, metric, _ in self.train_metrics + self.val_metrics:
            metric.reset()
        self.samples = []

    def forward(self, outputs: Dict[str, torch.Tensor]):
        result = {}

        # prepare loss inputs & update loss
        probs = F.softmax(outputs["logits_per_pocket"].float(), dim=-1)
        preds = probs.argmax(dim=-1)
        loss_inputs = {
            "logits_per_pocket": outputs["logits_per_pocket"],
            "logits_per_mol": outputs["logits_per_mol"],
        }

        loss_dict = self.loss(**loss_inputs)
        result.update(loss_dict)

        # prepare metric inputs & update metrics
        metric_inputs = {
            "preds": preds,
            "probs": probs,
            "target": result["target"],
            # "affinities": outputs["affinities"] if not self.training else None,
        }
        metrics = (
            self.train_metrics
            if self.training
            else self.train_metrics + self.val_metrics
        )
        for name, metric, is_compute in metrics:
            metric.update(**metric_inputs)
            if is_compute:
                result[name] = metric.compute()
        del result["target"]

        return result

    def compute(self, verbose=False):
        results = {}
        metrics = (
            self.train_metrics
            if self.training
            else self.train_metrics + self.val_metrics
        )
        for name, metric, _ in metrics:
            if metric._update_called:
                if verbose:
                    value, time_cost = with_time(
                        metric.compute, pretty_time=True
                    )()
                    if is_rank_zero():
                        logger.info(f"- {name}: {value} ({time_cost})")
                else:
                    value = metric.compute()
                if value is not None:
                    results[name] = value
        return results
