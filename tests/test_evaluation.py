import logging
from pathlib import Path

import dotenv
import pytest
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import open_dict

from src.utils.exptool import print_config, register_omegaconf_resolver

logger = logging.getLogger(__name__)

register_omegaconf_resolver()
dotenv.load_dotenv(override=True)


@pytest.mark.parametrize("training", [True, False])
def test_criterion(training, tmpdir: Path):
    with initialize(version_base="1.3", config_path="../conf"):
        cfg = compose(config_name="train", return_hydra_config=True)
        with open_dict(cfg):
            cfg.paths.output_dir = str(tmpdir)
            cfg.paths.work_dir = str(tmpdir)
            cfg.hydra = None

    print_config(cfg)

    model = instantiate(cfg.model).cuda()
    datamodule = instantiate(cfg.dataset)
    criterion = instantiate(cfg.criterion)
    criterion.cuda()
    criterion.reset()
    model.train(training)
    criterion.train(training)
    dataloader = (
        datamodule.train_dataloader()
        if training
        else datamodule.val_dataloader()
    )
    batch = next(iter(dataloader))
    net_input = {key: val.cuda() for key, val in batch["net_input"].items()}
    outputs = model(**net_input)

    logger.info("Criterion inputs:")
    for key, value in outputs.items():
        logger.info(f"- {key}: {value.shape} {value.dtype}")

    results = criterion(outputs)

    assert "loss" in results
    assert "loss_pocket" in results
    assert "loss_mol" in results

    logger.info("Criterion outputs:")
    for key, value in results.items():
        logger.info(f"- {key}: {value} {value.shape} {value.dtype}")

    if not training:
        logger.info(f"Metrics:")
        metrics = criterion.compute(verbose=True)
        assert "Acc" in metrics
        assert "Top3Acc" in metrics
        assert "AUCROC" in metrics
        assert "BEDROC" in metrics

    else:
        results["loss"].backward()
