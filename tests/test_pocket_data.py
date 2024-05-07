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


@pytest.mark.parametrize("name", ["kahraman", "tough_m1"])
def test_datamodule(name: str, tmpdir: Path):
    with initialize(config_path="../conf"):
        cfg = compose(config_name="train", overrides=[f"dataset={name}"])
        with open_dict(cfg):
            cfg.paths.output_dir = str(tmpdir)
            cfg.paths.work_dir = str(tmpdir)
            cfg.hydra = None

    print_config(cfg)

    datamodule = instantiate(cfg.dataset)

    test_loader = datamodule.test_dataloader()

    assert test_loader

    logger.info(f"Testing samples: {len(test_loader.dataset)}")

    test_batch = next(iter(test_loader))

    if name == "kahraman":
        outer_keys = ["net_input", "pocket_name"]
    elif name == "tough_m1":
        outer_keys = ["pocket_a", "pocket_b", "label"]

    # test keys and shape
    for key in outer_keys:
        assert key in test_batch

        if key not in ["net_input", "pocket_a", "pocket_b"]:
            assert len(test_batch[key]) == cfg.dataset.batch_size

    if name == "kahraman":
        group_keys = ["net_input"]
    elif name == "tough_m1":
        group_keys = ["pocket_a", "pocket_b"]

    for group_key in group_keys:
        logger.info(f"Testing {group_key} samples:")
        for key in [
            "pocket_src_tokens",
            "pocket_src_distance",
            "pocket_src_edge_type",
            "pocket_src_coord",
        ]:
            assert key in test_batch[group_key]

            assert len(test_batch[group_key][key]) == cfg.dataset.batch_size

            val = test_batch[group_key][key]
            logger.info(f"- {key}: {val.shape} {val.dtype}")

    if name == "tough_m1":
        val = test_batch["label"]
        logger.info(f"* label: {val.shape} {val.dtype}")
