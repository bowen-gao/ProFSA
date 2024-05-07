from pathlib import Path

import dotenv
from hydra import compose, initialize
from omegaconf import open_dict

from src.utils.exptool import print_config, register_omegaconf_resolver

register_omegaconf_resolver()

dotenv.load_dotenv(override=True)


def test_conf(tmpdir: Path):
    with initialize(version_base="1.3", config_path="../conf"):
        cfg = compose(config_name="train")
        with open_dict(cfg):
            cfg.paths.output_dir = str(tmpdir)
            cfg.paths.work_dir = str(tmpdir)
            cfg.hydra = None

    print_config(cfg)
