import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.timetool import with_time
from src.utils.torchtool import is_rank_zero

from .components.metrics import (
    AUROC_GR,
    BEDROC_GR,
    EF_GR,
    RE_GR,
    Accuracy,
    Perplexity,
    SeqAccuracy,
)

logger = logging.getLogger(__name__)


class RetrievalCriterion(nn.Module):
    def __init__(self):
        super().__init__()

        self.acc = SeqAccuracy()
        self.token_acc = Accuracy()
        self.retri_acc = SeqAccuracy(type_filter=0)
        self.retri_token_acc = Accuracy(type_filter=0)
        self.index_acc = SeqAccuracy(type_filter=1)
        self.index_token_acc = Accuracy(type_filter=1)
        self.ppl = Perplexity(ignore_index=-100)
        self.bedroc = BEDROC_GR()
        self.auroc = AUROC_GR()
        self.ef = EF_GR()
        self.re = RE_GR()

        # name, metric, step_compute
        self.train_metrics = [
            ("Acc", self.acc, True),
            ("TokenAcc", self.token_acc, True),
            ("RetriAcc", self.retri_acc, True),
            ("RetriTokenAcc", self.retri_token_acc, True),
            ("IndexAcc", self.index_acc, True),
            ("IndexTokenAcc", self.index_token_acc, True),
        ]

        self.val_metrics = [
            ("Acc", self.acc, False),
            ("TokenAcc", self.token_acc, False),
            ("RetriAcc", self.retri_acc, False),
            ("RetriTokenAcc", self.retri_token_acc, False),
            ("IndexAcc", self.index_acc, False),
            ("IndexTokenAcc", self.index_token_acc, False),
            # virtual screening metrics
            ("BEDROC", self.bedroc, False),
            ("AUROC", self.auroc, False),
            ("EF", self.ef, False),
            ("RE", self.re, False),
        ]

        self.test_metrics = [
            # virtual screening metrics
            ("BEDROC", self.bedroc, False),
            ("AUROC", self.auroc, False),
            ("EF", self.ef, False),
            ("RE", self.re, False),            
        ]

        self.virtual_screening_metrics = ["BEDROC", "AUROC", "EF", "RE"]

    def reset(self):
        for _, metric, _ in self.train_metrics + self.val_metrics + self.test_metrics:
            metric.reset()
        self.samples = []

    def forward(self, outputs: Dict[str, torch.Tensor]):
        result = {}

        # prepare loss inputs & update loss
        probs = F.softmax(outputs["logits"].float(), dim=-1)
        preds = probs.argmax(dim=-1)  # [batch_size, seq_len]

        # prepare metric inputs & update metrics
        metric_inputs = {
            "preds": preds,
            "logits": outputs["logits"],
            "target": outputs["labels"],
            "type": outputs["type"],
        }
        metrics = self.train_metrics
        for name, metric, is_compute in metrics:
            metric.update(**metric_inputs)
            if is_compute:
                result[name] = metric.compute()

        return result

    def compute(self, verbose=False):
        results = {}
        metrics = self.train_metrics
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

    def gen_forward(self, outputs: Dict[str, Any], valid: bool=False, return_acc: bool=False):
        # prepare metric inputs & update metrics
        vs_metric_inputs = {
            "binary_labels": outputs["binary_labels"],
            "sequences_scores": outputs["sequences_scores"],
            "batch_size": outputs["batch_size"],
            "type": outputs["type"],
        }
        if return_acc:
            acc_metric_inputs = {
                "preds": outputs["sequences"],
                "target": outputs["labels"],
                "type": outputs["type"],
            }
        metrics = self.val_metrics if valid else self.test_metrics
        for name, metric, _ in metrics:
            if return_acc and (name not in self.virtual_screening_metrics):
                metric.update(**acc_metric_inputs)
                continue
            metric.update(**vs_metric_inputs)

    def gen_compute(self, verbose=False, valid=False):
        results = {}
        metrics = self.val_metrics if valid else self.test_metrics
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
