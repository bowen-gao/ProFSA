import logging
from pathlib import Path
from typing import Any, Dict

import torch
from torch import Tensor, nn

from .components.unimol.dictionary import Dictionary
from .components.unimol.unimol import NonLinearHead, UniMolModel

logger = logging.getLogger(__name__)


class DrugCLIP(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.cfg = cfg

        self.mol_dict, self.pocket_dict = self.load_dictionary()
        self.logit_scale = cfg.logit_scale

        self.mol_model = UniMolModel(
            cfg.mol, self.mol_dict.pad(), len(self.mol_dict)
        )
        self.pocket_model = UniMolModel(
            cfg.pocket, self.pocket_dict.pad(), len(self.pocket_dict)
        )

        self.mol_project_fake = NonLinearHead(
            cfg.mol.encoder_embed_dim, 128, "relu"
        )
        self.pocket_project_fake = NonLinearHead(
            cfg.pocket.encoder_embed_dim, 128, "relu"
        )

    def load_dictionary(self):
        mol_dict_path = Path(self.cfg.data_dir) / self.cfg.mol_dict_file
        mol_dict = Dictionary.load(str(mol_dict_path))
        mol_dict.add_symbol("[MASK]", is_special=True)

        pocket_dict_path = Path(self.cfg.data_dir) / self.cfg.pocket_dict_file
        pocket_dict = Dictionary.load(str(pocket_dict_path))
        pocket_dict.add_symbol("[MASK]", is_special=True)

        return mol_dict, pocket_dict

    def get_dist_features(self, dist, et, flag):
        n_node = dist.size(-1)
        if flag == "mol":
            gbf_feature = self.mol_model.gbf(dist, et)
            gbf_result = self.mol_model.gbf_proj(gbf_feature)
        else:
            gbf_feature = self.pocket_model.gbf(dist, et)
            gbf_result = self.pocket_model.gbf_proj(gbf_feature)
        graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        return graph_attn_bias

    def forward_pocket(
        self,
        pocket_src_tokens: torch.Tensor,
        pocket_src_distance: torch.Tensor,
        pocket_src_edge_type: torch.Tensor,
        return_rep: bool = False,
        normalize_rep: bool = True,
        **kwargs,
    ):
        pocket_padding_mask = pocket_src_tokens.eq(
            self.pocket_model.padding_idx
        )
        pocket_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        pocket_graph_attn_bias = self.get_dist_features(
            pocket_src_distance, pocket_src_edge_type, "pocket"
        )
        pocket_outputs = self.pocket_model.encoder(
            pocket_x,
            padding_mask=pocket_padding_mask,
            attn_mask=pocket_graph_attn_bias,
        )
        pocket_encoder_rep = pocket_outputs[0]
        pocket_rep = pocket_encoder_rep[:, 0, :]

        if return_rep:
            if normalize_rep:
                pocket_rep = pocket_rep / pocket_rep.norm(dim=1, keepdim=True)
            return pocket_rep

        pocket_emb = self.pocket_project_fake(pocket_rep)
        pocket_emb = pocket_emb / pocket_emb.norm(dim=1, keepdim=True)

        return pocket_emb

    def forward_mol(
        self,
        mol_src_tokens: torch.Tensor,
        mol_src_distance: torch.Tensor,
        mol_src_edge_type: torch.Tensor,
        return_rep: bool = False,
        normalize_rep: bool = True,
        **kwargs,
    ):
        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = self.get_dist_features(
            mol_src_distance, mol_src_edge_type, "mol"
        )
        mol_outputs = self.mol_model.encoder(
            mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias
        )
        mol_encoder_rep = mol_outputs[0]
        mol_rep = mol_encoder_rep[:, 0, :]

        if return_rep:
            if normalize_rep:
                mol_rep = mol_rep / mol_rep.norm(dim=1, keepdim=True)
            return mol_rep

        mol_emb = self.mol_project_fake(mol_rep)
        mol_emb = mol_emb / mol_emb.norm(dim=1, keepdim=True)

        return mol_emb

    def forward(
        self,
        mol_src_tokens: torch.Tensor,
        mol_src_distance: torch.Tensor,
        mol_src_edge_type: torch.Tensor,
        pocket_src_tokens: torch.Tensor,
        pocket_src_distance: torch.Tensor,
        pocket_src_edge_type: torch.Tensor,
        **kwargs,
    ):
        mol_emb = self.forward_mol(
            mol_src_tokens, mol_src_distance, mol_src_edge_type, **kwargs
        )
        pocket_emb = self.forward_pocket(
            pocket_src_tokens,
            pocket_src_distance,
            pocket_src_edge_type,
            **kwargs,
        )

        logits_per_pocket = pocket_emb @ mol_emb.t()
        logits_per_pocket = logits_per_pocket * self.logit_scale
        logits_per_mol = logits_per_pocket.t()

        return {
            "logits_per_pocket": logits_per_pocket,
            "logits_per_mol": logits_per_mol,
            "pocket_emb": pocket_emb,
            "mol_emb": mol_emb,
        }


class DrugCLIPReg(DrugCLIP):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.load_pretrained()

        self.classification_head = nn.Sequential(
            nn.Linear(
                cfg.pocket.encoder_embed_dim + cfg.pocket.encoder_embed_dim,
                1024,
            ),
            nn.Dropout(p=cfg.dropout),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(p=cfg.dropout),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=cfg.dropout),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(p=cfg.dropout),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def load_pretrained(self):
        if self.cfg.pretrained_weights is not None:
            logger.info(
                f"Loading pretrained weights from {self.cfg.pretrained_weights}"
            )
            model = torch.load(self.cfg.pretrained_weights, map_location="cpu")[
                "state_dict"
            ]
            state_dict = {}
            for key, value in model.items():
                if key.startswith("model."):
                    state_dict[key[6:]] = value
                else:
                    state_dict[key] = value
            self.load_state_dict(state_dict, strict=False)

        if self.cfg.pretrained_fixed:
            logger.info("Freezing pretrained weights")
            for param in self.parameters():
                param.requires_grad = False

    def forward(
        self,
        mol_src_tokens: Tensor,
        mol_src_distance: Tensor,
        mol_src_edge_type: Tensor,
        pocket_src_tokens: Tensor,
        pocket_src_distance: Tensor,
        pocket_src_edge_type: Tensor,
        **kwargs,
    ):

        mol_rep = self.forward_mol(
            mol_src_tokens,
            mol_src_distance,
            mol_src_edge_type,
            return_rep=True,
            normalize_rep=False,
            **kwargs,
        )
        pocket_rep = self.forward_pocket(
            pocket_src_tokens,
            pocket_src_distance,
            pocket_src_edge_type,
            return_rep=True,
            normalize_rep=False,
            **kwargs,
        )
        logit = self.classification_head(
            torch.cat([mol_rep.detach(), pocket_rep.detach()], dim=-1)
        ).squeeze(-1)
        return {"logit": logit}
