import logging
from pathlib import Path

import dotenv
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from lightning.pytorch.utilities.memory import get_model_size_mb
from omegaconf import open_dict

from src.utils.exptool import print_config, register_omegaconf_resolver

logger = logging.getLogger(__name__)

register_omegaconf_resolver()
dotenv.load_dotenv(override=True)


def test_model(tmpdir: Path):
    with initialize(config_path="../conf"):
        cfg = compose(config_name="train")
        with open_dict(cfg):
            cfg.paths.output_dir = str(tmpdir)
            cfg.paths.work_dir = str(tmpdir)
            cfg.hydra = None

    print_config(cfg)

    datamodule = instantiate(cfg.dataset)
    train_loader = datamodule.train_dataloader()
    train_batch = next(iter(train_loader))

    model = instantiate(cfg.model).cuda()
    logger.info(f"Model size: {get_model_size_mb(model)} MB")

    logger.info("Input shape:")
    net_input = {
        key: val.cuda() for key, val in train_batch["net_input"].items()
    }
    for key, val in net_input.items():
        logger.info(f"- {key}: {val.shape}")
    outputs = model(**net_input)

    batch_size = cfg.dataset.batch_size.train
    assert outputs["logits_per_pocket"].shape == torch.Size(
        [batch_size, batch_size]
    )
    assert outputs["logits_per_mol"].shape == torch.Size(
        [batch_size, batch_size]
    )
