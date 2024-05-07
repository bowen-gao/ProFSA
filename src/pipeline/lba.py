import logging
from typing import Any, List

from .screening import ScreeningLitModule

logger = logging.getLogger(__name__)


class LBALitModule(ScreeningLitModule):
    def step(self, batch: Any):
        inputs = batch["net_input"]
        outputs = self.model(**inputs)

        outputs["target"] = batch["target"]["finetune_target"]
        outputs = self.criterion(outputs)

        return outputs

    def test_step(self, batch: Any, batch_idx: int):
        outputs = self.step(batch)
        return outputs

    def on_test_epoch_end(self):
        metrics = self.criterion.compute(verbose=True)
        for name, value in metrics.items():
            self.log(
                f"test/{name}",
                value,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        self.criterion.reset()
