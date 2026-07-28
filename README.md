# Beyond Linearity in Attention Projections
### The Case for Nonlinear Queries - Community extension and verification

A spinoff from the Eleuther AI community discord aiming to extend the results and explore the effects of the theoretically driven nonlinear queries. Paper under preparation.




The work is based on:
Official implementation and model weights for the paper: **"Beyond Linearity in Attention Projections: The Case for Nonlinear Queries"** ([arXiv:2603.13381](https://arxiv.org/abs/2603.13381)).
This repository implements nonlinear query projections of the form $Q(X) = (X + f_\theta(X))/2$, where $f_\theta$ is a bottleneck MLP, replacing the standard linear $W_Q$ at the same parameter budget. Building on the algebraic redundancy of $W_Q$ established by [Karbevski and Mijoski (2025)](https://arxiv.org/abs/2510.23912)), we show that nonlinear queries consistently improve validation loss over baseline, comfortably outperforming a model with 12.5% more non-embedding parameters.

To be presented at the ICLR 2026 Workshop on Geometry-grounded Representation Learning and Generative Modeling (GRaM)).

---
## 🚀 Quick Start
### 1. Model Checkpoints
Pre-trained checkpoints and training losses from our runs are available for download:
* **[Download from Google Drive](https://drive.google.com/drive/folders/1JNlDCGk1Rw-kfsgmDBtLOkjl9iyGXCtt?usp=drive_link)**

You can explore the losses using `explore losses.ipynb`
### 2. Data Preparation

We utilize the **OpenWebText** dataset. Follow these steps after preparing the `uv` environment:
1. **Dataset Acquisition:** Run `Data_Handling.ipynb` to download and preprocess the raw data.
2. **Reproducibility:** Run `Generate_Indices.ipynb` to ensure consistent data shuffling and splitting.
3. **Configuration:** Plenty configurations can be found in the configs folder. Creating a new one is relatively simple following the examples.

You might want to modify `Generate_Indices.ipynb` if you want to create differently sized batches or a train run that runs for longer than 600k steps.
### 3. Training

To initiate training on a specific GPU (e.g., GPU 0), use the following command:
`python train.py _a_config_file_ --gpu {gpus_to_use}`

For example:
`python train.py configs/configs_tied/config_tiedw_original.py --gpu 0`

Note: The repo has been created for a single GPU training, but it should not be difficult to modify it for DDP training as well, following Karpathy's example.
---
## 🛠 Architecture 

The attention mechanism has been modified to support nonlinear query projections: the standard linear $W_Q$ is replaced with a residual bottleneck MLP $Q(X) = (X + f_\theta(X))/2$, where $f_\theta(X) = \text{LN}(\text{GELU}(\text{RMSNorm}(X)W_1)W_2)$ with $W_1 \in \mathbb{R}^{d \times r}$, $W_2 \in \mathbb{R}^{r \times d}$, and $r = d/2$. 

Keys and values remain standard linear projections.
---
## 📝 Citation [OLD, TO BE CHANGED]
If you find this work useful in your research, please cite:
```bibtex
@article{karbevski2026beyond,
  title={Beyond Linearity in Attention Projections: The Case for Nonlinear Queries},
  author={Karbevski, Marko},
  journal={arXiv preprint arXiv:2603.13381},
  year={2026},
  note={Presented at the ICLR 2026 Workshop on Geometry-grounded Representation Learning and Generative Modeling (GRaM)}
}
```
---
## 🙏 Acknowledgments
We are grateful to the anonymous reviewers for their constructive feedback, and to Nils Graef, Yiping Ji, Haris Mandal, and Antonij Mijoski for valuable discussions. This codebase builds on the [nanoGPT](https://github.com/karpathy/nanoGPT) repository by Andrej Karpathy. We thank Eleuther AI for nurturing a vibrant community.

---
## 🤝 Collaboration & Contributing (Open-Source & Commercial)

This repository represents an independent research initiative focused on establishing the mathematical foundations and structural validity of Nonlinear Residual Queries (NRQ). 

To isolate the core architectural mechanics cleanly, development has been focused entirely on delivering a verified implementation, rather than an industrialized production framework. As a solo researcher, I highly welcome both open-source and commercial collaborations to improve the work.
---
The code has been tested on Python version `3.12.11` using a single Nvidia 5090 RTX GPU.


Contributors: [To be added]
