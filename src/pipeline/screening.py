import logging
from typing import Any, Dict

from hydra.utils import instantiate
from lightning import LightningModule

from src.utils.torchtool import get_model_size_mb, is_rank_zero

logger = logging.getLogger(__name__)


class ScreeningLitModule(LightningModule):
    def __init__(self, cfg: Dict[str, Any] = None):
        super().__init__()

        self.save_hyperparameters(cfg)
        self.model = instantiate(self.hparams.model)
        self.criterion = instantiate(self.hparams.criterion)

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optim, self.parameters())
        scheduler = instantiate(
            self._set_num_training_steps(self.hparams.scheduler), optimizer
        )
        # torch's schedulers are epoch-based, but transformers' are step-based
        interval = (
            "step"
            if self.hparams.scheduler._target_.startswith("transformers")
            else "epoch"
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def _set_num_training_steps(self, scheduler_cfg):
        if "num_training_steps" in scheduler_cfg:
            scheduler_cfg = dict(scheduler_cfg)
            if is_rank_zero():
                logger.info("Computing number of training steps...")
            scheduler_cfg[
                "num_training_steps"
            ] = self.trainer.estimated_stepping_batches

            if is_rank_zero():
                logger.info(
                    f"Training steps: {scheduler_cfg['num_training_steps']}"
                )
        return scheduler_cfg

    def on_train_start(self):
        self.criterion.reset()
        self.log(
            "model_size/total",
            get_model_size_mb(self.model),
            rank_zero_only=True,
            logger=True,
        )
        if hasattr(self.model, "mol_model"):
            self.log(
                "model_size/mol_model",
                get_model_size_mb(self.model.mol_model),
                rank_zero_only=True,
                sync_dist=True,
                logger=True,
            )
        if hasattr(self.model, "pocket_model"):
            self.log(
                "model_size/pocket_model",
                get_model_size_mb(self.model.pocket_model),
                rank_zero_only=True,
                sync_dist=True,
                logger=True,
            )

    def step(self, batch: Any):
        """The choice of one forward step:

        - computing loss
        - update evaluation metrics
            - exclude evaluation metrics
        - generate from scratch
        """
        inputs = batch["net_input"]
        outputs = self.model(**inputs)

        # Compute loss and metrics, input keys and values are reserved
        # if not self.training:
        #     outputs["affinities"] = batch["target"]["finetune_target"]
        outputs = self.criterion(outputs)

        return outputs

    def on_train_epoch_start(self):
        self.trainer.train_dataloader.dataset.set_epoch(self.current_epoch)

    def training_step(self, batch: Any, batch_idx: int):
        outputs = self.step(batch)

        for key, value in outputs.items():
            self.log(
                f"train/{key}",
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
            )
        return outputs["loss"]

    def on_train_epoch_end(self):
        metrics = self.criterion.compute(verbose=True)
        for name, value in metrics.items():
            self.log(
                f"train/{name}",
                value,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        self.criterion.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        outputs = self.step(batch)
        return outputs

    def on_validation_epoch_end(self):
        metrics = self.criterion.compute(verbose=True)
        for name, value in metrics.items():
            self.log(
                f"val/{name}",
                value,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        self.criterion.reset()

    def predict_step(self, batch):
        net_input=batch["net_input"]
        output=self.model(**net_input)
        return {
            "pocket_emb":output["pocket_emb"],
            "mol_emb":output["mol_emb"],
            "pocket_name":batch["pocket_name"],
        }