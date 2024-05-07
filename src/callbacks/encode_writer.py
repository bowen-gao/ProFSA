from lightning import LightningModule, Trainer
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
import os
from typing import Any, Optional, Sequence

class EncodeWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval='epoch'):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(prediction, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))

    
    def write_on_epoch_end(self, trainer: Trainer, pl_module: LightningModule, predictions: Sequence[Any], batch_indices: Sequence[Any]) -> None:
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))