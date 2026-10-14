# Vision Transformer (ViT) Paper Implementation

From-scratch PyTorch implementation of the Vision Transformer from [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020), built step by step in notebooks using only basic PyTorch layers.

<p align="center">
  <img src="assets/vit.gif" alt="Schematic ViT forward pass: an aerial image is split into patches (3x3 shown for clarity), the patches are linearly projected, combined with position embeddings and a learnable class token, passed through the Transformer Encoder, and the class token goes through the MLP head to predict the class. The encoder block (Norm, Multi-Head Attention, MLP and residual connections, repeated L times) is shown on the right, as in Figure 1 of the paper" width="100%">
</p>

## Table of Contents

- [About](#about)
- [Progress](#progress)
- [Setup](#setup)
  - [Environment](#environment)
  - [Data](#data)
- [Usage](#usage)
- [Implementation](#implementation)
  - [Patch embedding](#patch-embedding)
  - [Multi-head attention](#multi-head-attention)
- [Citation](#citation)

## About

The goal of this repository is to understand the Vision Transformer by rebuilding it piece by piece, mapping each part of the code back to the equations in the paper. Every component is implemented and explained in its own notebook, without relying on `torchvision`'s or `timm`'s ready-made ViT modules.

## Progress

- ✅ Image patching
- ✅ Linear patch projection
- ✅ Class token
- ✅ Learnable position embedding
- ✅ Multi-head self-attention
- ⬜ MLP block
- ⬜ Transformer encoder
- ⬜ Classification head
- ⬜ Training and evaluation

## Setup

### Environment

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
git clone git@github.com:zvonimirkonjevic/vit-pytorch.git
cd vit-pytorch
uv sync
```

Open the notebooks in Jupyter or VS Code and select the `.venv` interpreter as the kernel.

### Data

The notebooks use the [UC Merced Land Use dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html): 2,100 aerial RGB images of roughly 256x256 pixels (44 are a few pixels off), split evenly across 21 land-use classes. Download it and place the class folders under `data/raw/`:

```
data/
└── raw/
    ├── agricultural/
    │   ├── agricultural00.tif
    │   └── ...
    ├── airplane/
    └── ...
```

`data/` is git-ignored.

## Usage

| Notebook | Description |
| --- | --- |
| [`02_embedding_patches.ipynb`](notebooks/02_embedding_patches.ipynb) | Splits images into patches, flattens them, projects them to $D$ dimensions, and prepends the class token and adds position embeddings. Also generates the patch embedding GIF. |
| [`03_multi_head_attention.ipynb`](notebooks/03_multi_head_attention.ipynb) | Implements scaled dot-product attention and multi-head self-attention on the embedded patches of one image ($\mathbf{z}_0$, loaded from `embedded_patches_single_image.pt`), and checks the result against `nn.MultiheadAttention`. |

## Implementation

### Patch embedding

<p align="center">
  <img src="assets/patch_embedding.gif" alt="An image is cut into 16x16 patches, each patch is flattened into a 768-value row and linearly projected to the embedding dimension" width="100%">
</p>

The transformer expects a 1D sequence of token embeddings, so an image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ is reshaped into a sequence of $N = HW / P^2$ flattened patches $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$. Eq. 1 of the paper then builds the input sequence:

$$
\mathbf{z}_0 = [\mathbf{x}_\text{class};\, \mathbf{x}^1_p \mathbf{E};\, \mathbf{x}^2_p \mathbf{E};\, \cdots;\, \mathbf{x}^N_p \mathbf{E}] + \mathbf{E}_{pos}, \quad \mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}, \quad \mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}
$$

| Paper | Code (`PatchEmbedding`) | Shape for a 256x256 RGB image, $P = 16$, $D = 768$ |
| --- | --- | --- |
| $\mathbf{x}$ | `images` | `(B, 3, 256, 256)` |
| $\mathbf{x}_p$ | `patchify(images)` | `(B, 256, 768)` |
| $\mathbf{x}_p \mathbf{E}$ | `linear(...)` | `(B, 256, 768)` |
| $\mathbf{x}_\text{class}$ | `class_token` | `(1, 1, 768)` |
| $\mathbf{E}_{pos}$ | `position_embedding` | `(1, 257, 768)` |
| $\mathbf{z}_0$ | `forward(images)` | `(B, 257, 768)` |

`patchify` avoids Python loops by reshaping the image into a grid of patches and permuting the axes so each patch becomes one contiguous block.

The paper does not specify how $\mathbf{x}_\text{class}$ and $\mathbf{E}_{pos}$ are initialized, so they follow the [official implementation](https://github.com/google-research/vision_transformer): zeros for the class token and $\mathcal{N}(0, 0.02^2)$ for the position embedding.

### Multi-head attention

<p align="center">
  <img src="assets/multi_head_attention.gif" alt="Step-by-step walk through the Scaled Dot-Product Attention and Multi-Head Attention diagrams, showing the tensor each block produces" width="100%">
</p>

Each encoder layer starts with multi-head self-attention (MSA) on the layer-normalized tokens, followed by a residual connection (Eq. 2):

$$
\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}
$$

MSA follows [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Each of the $h$ heads projects the tokens with its own $W_i^Q, W_i^K, W_i^V$ and runs scaled dot-product attention. The heads run in parallel, and their outputs are concatenated and projected back to $D$ by $W^O$:

$$
\text{head}_i = \text{softmax}\left(\frac{\mathbf{z}W_i^Q (\mathbf{z}W_i^K)^\top}{\sqrt{d_k}}\right)\mathbf{z}W_i^V, \qquad \text{MSA}(\mathbf{z}) = [\text{head}_1; \dots; \text{head}_h]\,W^O
$$

| Paper | Code | Shape for $N + 1 = 257$ tokens, $D = 768$, $h = 12$, $d_k = d_v = 64$ |
| --- | --- | --- |
| $\mathbf{z}$ | `x` | `(B, 257, 768)` |
| $\mathbf{z}W_i^Q,\ \mathbf{z}W_i^K,\ \mathbf{z}W_i^V$ | `W_q(x)`, `W_k(x)`, `W_v(x)` in `ScaleDotProductAttention` | `(B, 257, 64)` each |
| $QK^\top / \sqrt{d_k}$ | `attention_scores` | `(B, 257, 257)` |
| $\text{softmax}(\cdot)$ | `attention_weights` | `(B, 257, 257)` |
| $\text{head}_i$ | `ScaleDotProductAttention(x)` | `(B, 257, 64)` |
| $[\text{head}_1; \dots; \text{head}_h]$ | `torch.cat(head_outputs, dim=-1)` in `MultiHeadAttention` | `(B, 257, 768)` |
| $\text{MSA}(\mathbf{z})$ | `W_o(...)` | `(B, 257, 768)` |

With the weights copied over, `MultiHeadAttention` produces the same output as PyTorch's `nn.MultiheadAttention`.

## Citation

```bibtex
@inproceedings{dosovitskiy2021image,
  author     = {Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  title      = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  booktitle  = {International Conference on Learning Representations},
  eventtitle = {ICLR 2021},
  date       = {2021},
  eprint     = {2010.11929},
  eprinttype = {arxiv},
  url        = {https://openreview.net/forum?id=YicbFdNTTy}
}

@inproceedings{vaswani2017attention,
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  title     = {Attention Is All You Need},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017},
  eprint    = {1706.03762},
  url       = {https://arxiv.org/abs/1706.03762}
}
```
