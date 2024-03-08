# ProFSA - Self-supervised Pocket Pretraining via Protein Fragment-Surroundings Alignment

<br>

Official repository for ["Self-supervised Pocket Pretraining via Protein Fragment-Surroundings Alignment"](https://github.com/bowen-gao/ProFSA).

<!-- ![A fancy image here](docs/_static/imgs/logo.svg) -->
<img src="imgs/MainFigure_v3.png" width="800">

**Figure:** *An illustration of protein fragment-surroundings alignment framework.*

> **Self-supervised Pocket Pretraining via Protein Fragment-Surroundings Alignment** <br>
> Bowen Gao, Yinjun Jia, Yuanle Mo, Yuyan Ni, Wei-Ying Ma, Zhi-Ming Ma, Yanyan Lan <br>
> *Published on The Twelfth International Conference on Learning Representations (ICLR 2024)*

<!-- [![](imgs/project.svg)](https://hongxin2019.github.io/TVR/) -->
[![](https://img.shields.io/badge/-code-green?style=flat-square&logo=github&labelColor=gray)](https://github.com/bowen-gao/ProFSA)
[![](https://img.shields.io/badge/arXiv-2310.07229-b31b1b?style=flat-square)](https://arxiv.org/pdf/2310.07229.pdf)
[![](https://img.shields.io/badge/PyTorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![DeepCodebase](https://img.shields.io/badge/Deep-Codebase-2d50a5.svg?style=flat-square)](https://github.com/hughplay/DeepCodebase)
<!-- [![](https://img.shields.io/badge/TRANCE-dataset-blue?style=flat-square&labelColor=gray)](https://hongxin2019.github.io/TVR/dataset)
[![](https://img.shields.io/badge/TRANCE-explore_dataset-blue?style=flat-square&labelColor=gray)](https://hongxin2019.github.io/TVR/explore) -->



<br>

## Description

Pocket representations play a vital role in various biomedical applications, such as druggability estimation, ligand affinity prediction, and de novo drug design. While existing geometric features and pretrained representations have demonstrated promising results, they usually treat pockets independent of ligands, neglecting the fundamental interactions between them. However, the limited pocket-ligand complex structures available in the PDB database (less than 100 thousand non-redundant pairs) hampers large-scale pretraining endeavors for interaction modeling. To address this constraint, we propose a novel pocket pretraining approach that leverages knowledge from high-resolution atomic protein structures, assisted by highly effective pretrained small molecule representations. By segmenting protein structures into drug-like fragments and their corresponding pockets, we obtain a reasonable simulation of ligand-receptor interactions, resulting in the generation of over 5 million complexes. Subsequently, the pocket encoder is trained in a contrastive manner to align with the representation of pseudo-ligand furnished by some pretrained small molecule encoders. Our method, named ProFSA, achieves state-of-the-art performance across various tasks, including pocket druggability prediction, pocket matching, and ligand binding affinity prediction. Notably, ProFSA surpasses other pretraining methods by a substantial margin. Moreover, our work opens up a new avenue for mitigating the scarcity of protein-ligand complex data through the utilization of high-quality and diverse protein structure databases.


## Dataset

https://drive.google.com/file/d/1_Md8akleucwATBXo1Dc4ei3fxvVulkXG/view

If you find this code useful, please consider star this repo and cite us:

```
@inproceedings{gao2023self,
  title={Self-supervised Pocket Pretraining via Protein Fragment-Surroundings Alignment},
  author={Gao, Bowen and Jia, Yinjun and Mo, YuanLe and Ni, Yuyan and Ma, Wei-Ying and Ma, Zhi-Ming and Lan, Yanyan},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```
