import logging
from pathlib import Path

import dotenv
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import open_dict

from src.utils.exptool import print_config, register_omegaconf_resolver

# from IPython import embed

logger = logging.getLogger(__name__)

register_omegaconf_resolver()
dotenv.load_dotenv(override=True)


def test_datamodule(tmpdir: Path):
    with initialize(config_path="../conf"):
        cfg = compose(config_name="train")
        with open_dict(cfg):
            cfg.paths.output_dir = str(tmpdir)
            cfg.paths.work_dir = str(tmpdir)
            cfg.hydra = None

    print_config(cfg)

    datamodule = instantiate(cfg.dataset)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    assert train_loader
    assert val_loader

    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    # x = val_loader.dataset.__getitem__(0)
    # embed()

    # test keys and shape
    for key in ["net_input", "target", "smi_name", "pocket_name"]:
        assert key in train_batch
        assert key in val_batch

        if key not in ["net_input", "target"]:
            assert len(train_batch[key]) == cfg.dataset.batch_size.train
            assert len(val_batch[key]) == cfg.dataset.batch_size.val

    for key in [
        "mol_src_tokens",
        "mol_src_distance",
        "mol_src_edge_type",
        "pocket_src_tokens",
        "pocket_src_distance",
        "pocket_src_edge_type",
        "pocket_src_coord",
        "mol_len",
        "pocket_len",
    ]:
        assert key in train_batch["net_input"]
        assert key in val_batch["net_input"]

        assert (
            len(train_batch["net_input"][key]) == cfg.dataset.batch_size.train
        )
        assert len(val_batch["net_input"][key]) == cfg.dataset.batch_size.val

        logger.info(
            f"- {key}: {val_batch['net_input'][key].shape} "
            f"{val_batch['net_input'][key].dtype}"
        )

    assert "finetune_target" in train_batch["target"]
    assert "finetune_target" in val_batch["target"]
    assert (
        len(train_batch["target"]["finetune_target"])
        == cfg.dataset.batch_size.train
    )
    assert (
        len(val_batch["target"]["finetune_target"])
        == cfg.dataset.batch_size.val
    )

    # test random
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    train_dataset.set_epoch(0)
    val_dataset.set_epoch(0)
    train_epoch0_sample0 = train_dataset[0]
    val_epoch0_sample0 = val_dataset[0]

    train_dataset.set_epoch(1)
    val_dataset.set_epoch(1)
    train_epoch1_sample0 = train_dataset[0]
    val_epoch1_sample0 = val_dataset[0]

    assert train_epoch0_sample0["smi_name"] != train_epoch1_sample0["smi_name"]
    assert val_epoch0_sample0["smi_name"] == val_epoch1_sample0["smi_name"]
