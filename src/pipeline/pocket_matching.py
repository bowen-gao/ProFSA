import logging
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List

import torch
from einops import einsum
from hydra.utils import instantiate
from lightning import LightningModule
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class KahramanTestModule(LightningModule):
    def __init__(self, cfg: Dict[str, Any] = None):
        super().__init__()

        self.save_hyperparameters(cfg)
        self.model = instantiate(self.hparams.model)
        self.test_step_outputs = []

    def test_step(self, batch: Any, batch_idx: int):
        inputs = batch["net_input"]
        features = self.model.forward_pocket(**inputs, return_rep=True)
        pocket_names = batch["pocket_name"]
        result = {
            "features": features,
            "pocket_names": pocket_names,
        }
        self.test_step_outputs.append(result)
        return result

    def load_data(self):
        with open(self.hparams.dataset.dataset_cfg.label_path, "rb") as f:
            pairs, labels = pkl.load(f)
        logger.info(f"Total number of pairs: {len(pairs)}")
        logger.info(f"Total number of labels: {len(labels)}")
        return pairs, labels

    def on_test_epoch_end(self):
        pairs, labels = self.load_data()

        labels_map = {}
        for (pocket_1, pocket_2), label in zip(pairs, labels):
            p1_name = pocket_1["pocket"].split("/")[-1].split("_")[0]
            p2_name = pocket_2["pocket"].split("/")[-1].split("_")[0]
            label_name = f"{p1_name}_{p2_name}"
            labels_map[label_name] = label

        pocket_features = torch.cat([x["features"] for x in self.test_step_outputs])
        pocket_names = [
            item
            for sublist in [x["pocket_names"] for x in self.test_step_outputs]
            for item in sublist
        ]
        matching_matrix = pocket_features @ pocket_features.T
        logger.info(
            f"Total number of pocket features: {pocket_features.shape[0]}"
        )
        logger.info(f"Total number of pocket names: {len(pocket_names)}")
        logger.info(f"Total number of pairs: {matching_matrix.numel()}")

        pred_list = []
        label_list = []

        for i in range(matching_matrix.shape[0]):
            for j in range(matching_matrix.shape[1]):
                p1_name = pocket_names[i]
                p2_name = pocket_names[j]
                label_name = f"{p1_name}_{p2_name}"
                pred_list.append(matching_matrix[i, j].item())
                label_list.append(labels_map[label_name])

        auc = roc_auc_score(label_list, pred_list)
        self.log("Kahraman_AUC", auc, prog_bar=True)

        result_root_dir = Path("results") / "pocket_matching"
        result_root_dir.mkdir(exist_ok=True, parents=True)

        with open(result_root_dir / "Kahraman_pred.pkl", "wb") as f:
            pkl.dump(pred_list, f)

        with open(result_root_dir / "Kahraman_AUC.txt", "w") as f:
            f.write(f"{auc}")

        return {
            "Kahraman_AUC": auc,
        }


class TOUGHM1TestModule(LightningModule):
    def __init__(self, cfg: Dict[str, Any] = None):
        super().__init__()

        self.save_hyperparameters(cfg)
        self.model = instantiate(self.hparams.model)
        self.test_step_outputs = []

    def test_step(self, batch: Any, batch_idx: int):
        pocket_a_inputs = batch["pocket_a"]
        pocket_b_inputs = batch["pocket_b"]
        features_a = self.model.forward_pocket(
            **pocket_a_inputs, return_rep=True
        )
        features_b = self.model.forward_pocket(
            **pocket_b_inputs, return_rep=True
        )
        matching_score = einsum(features_a, features_b, "b i, b i -> b")
        label = batch["label"]

        result = {
            "score": matching_score.squeeze(),
            "label": label,
        }
        self.test_step_outputs.append(result)
        return result

    def on_test_epoch_end(self):
        score = torch.cat([x["score"] for x in self.test_step_outputs]).tolist()
        label = torch.cat([x["label"] for x in self.test_step_outputs]).tolist()

        logger.info(f"Total number of pairs: {len(score)}")

        auc = roc_auc_score(label, score)
        self.log("TOUGHM1_AUC", auc, prog_bar=True)

        result_root_dir = Path("results") / "pocket_matching"
        result_root_dir.mkdir(exist_ok=True, parents=True)

        with open(result_root_dir / "TOUGHM1_pred.pkl", "wb") as f:
            pkl.dump(score, f)

        with open(result_root_dir / "TOUGHM1_AUC.txt", "w") as f:
            f.write(f"{auc}")

        return {
            "TOUGHM1_AUC": auc,
        }
