import logging
from typing import Dict

import torch
from torch import nn
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)

from src.utils.timetool import with_time
from src.utils.torchtool import is_rank_zero

from .components.metrics import RMSE

logger = logging.getLogger(__name__)


class LBACriterion(nn.Module):
    def __init__(self, loss: nn.MSELoss):
        super().__init__()
        self.loss = loss

        self.spearman = SpearmanCorrCoef()
        self.pearson = PearsonCorrCoef()
        self.mse = MeanSquaredError()
        self.rmse = RMSE()
        self.mae = MeanAbsoluteError()

        # name, metric, step_compute
        self.train_metrics = [
            ("MSE", self.mse, True),
            ("RMSE", self.rmse, True),
            ("MAE", self.mae, True),
        ]

        self.val_metrics = [
            ("Pearson", self.pearson, False),
            ("Spearman", self.spearman, False),
        ]

    def reset(self):
        for _, metric, _ in self.train_metrics + self.val_metrics:
            metric.reset()
        self.samples = []

    def forward(self, outputs: Dict[str, torch.Tensor]):
        result = {}

        # prepare loss inputs & update loss
        loss_inputs = {
            "input": outputs["logit"],
            "target": outputs["target"],
        }

        loss = self.loss(**loss_inputs)
        result.update({"loss": loss})

        # prepare metric inputs & update metrics
        metric_inputs = {
            "preds": outputs["logit"],
            "target": outputs["target"],
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
