"""ViT model components, in the order they appear in the paper (Section 3.1)."""

import torch
import torch.nn as nn


LAYERS = 12 # L
HEIGHT = 256 # H
WIDTH = 256 # W
HIDDEN_DIM = 768 # D
MLP_DIM = 3072 # MLP_DIM
HEADS = 12 # h
PATCH_SIZE = 16 # P


class PatchEmbedding(nn.Module):
    """Patchifies a batch of images and projects each patch to embed_dim dimensions, the x_p E term of Eq. 1.

    Args:
        patch_size: Side length P of each square patch.
        color_channels: Number of image channels C.
        embed_dim: Embedding dimension D.
    """

    def __init__(self, patch_size: int = 16, color_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.linear = nn.Linear(in_features=patch_size * patch_size * color_channels, out_features=embed_dim)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """Splits a batch of images into flattened non-overlapping patches without looping.

        Args:
            images: Tensor of shape (B, C, H, W).

        Returns:
            Tensor of shape (B, N, P*P*C), same patch and pixel order as the loop-based patchify.
        """
        batch_size, color_channels, height, width = images.shape
        assert height % self.patch_size == 0 and width % self.patch_size == 0, (
            f"image size {height}x{width} not divisible by patch_size={self.patch_size}"
        )
        p = self.patch_size

        # Split H into (patch_row, row_inside_patch) and W into (patch_col, col_inside_patch)
        patches = images.reshape(batch_size, color_channels, height // p, p, width // p, p)
        # (B, C, patch_row, r, patch_col, c) -> (B, patch_row, patch_col, r, c, C) so each patch is one block,
        # with channels last inside a patch to keep the loop-based pixel order
        patches = patches.permute(0, 2, 4, 3, 5, 1)
        return patches.reshape(batch_size, -1, p * p * color_channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Computes the patch embeddings of a batch of images.

        Args:
            images: Float tensor of shape (B, C, H, W).

        Returns:
            Tensor of shape (B, N, D).
        """
        patch_embeddings = self.linear(self.patchify(images))
        return patch_embeddings


class ScaleDotProductAttention(nn.Module):
    """Single attention head computing softmax(QK^T / sqrt(d_k))V over a sequence of tokens.

    Args:
        embedding_dim: Dimension D of the input tokens.
        d_k: Dimension of the queries and keys.
        d_v: Dimension of the values.
    """

    def __init__(self, embedding_dim: int = 768, d_k: int = 64, d_v: int = 64):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.W_q = nn.Linear(embedding_dim, d_k)
        self.W_k = nn.Linear(embedding_dim, d_k)
        self.W_v = nn.Linear(embedding_dim, d_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes single-head self-attention.

        Args:
            x: Tensor of shape (B, N, embedding_dim).

        Returns:
            Tensor of shape (B, N, d_v).
        """
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output


class MultiHeadAttention(nn.Module):
    """Runs h attention heads in parallel, concatenates their outputs and projects them back to embedding_dim.

    Args:
        embedding_dim: Dimension D of the input tokens.
        d_k: Dimension of the queries and keys of each head.
        d_v: Dimension of the values of each head.
        h: Number of attention heads.
    """

    def __init__(self, embedding_dim: int = 768, d_k: int = 64, d_v: int = 64, h: int = 12):
        super().__init__()
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.attention_heads = nn.ModuleList([ScaleDotProductAttention(embedding_dim, d_k, d_v) for _ in range(h)])
        self.W_o = nn.Linear(h * d_v, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes multi-head self-attention.

        Args:
            x: Tensor of shape (B, N, embedding_dim).

        Returns:
            Tensor of shape (B, N, embedding_dim).
        """
        head_outputs = [head(x) for head in self.attention_heads]
        multiple_head_output = torch.cat(head_outputs, dim=-1)
        output = self.W_o(multiple_head_output)
        return output


class MultiHeadAttentionBlock(nn.Module):
    """Layer normalization followed by multi-head attention, the MSA(LN(z)) term of Eq. 2.

    Args:
        embedding_dim: Dimension D of the input tokens.
        d_k: Dimension of the queries and keys of each head.
        d_v: Dimension of the values of each head.
        h: Number of attention heads.
        dropout_rate: Dropout probability applied after the output projection W_o.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        d_k: int = 64,
        d_v: int = 64,
        h: int = 12,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.multi_head_attention = MultiHeadAttention(embedding_dim, d_k, d_v, h)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies layer normalization, multi-head self-attention, then dropout.

        The residual connection is added by the encoder block, not here.

        Args:
            x: Tensor of shape (B, N, embedding_dim).

        Returns:
            Tensor of shape (B, N, embedding_dim).
        """
        normalized = self.layer_norm(x)
        attention_output = self.multi_head_attention(normalized)
        output = self.dropout(attention_output)
        return output


class MultiLayerPerceptronBlock(nn.Module):
    """Layer normalization followed by a multi-layer perceptron, the MLP(LN(z)) term of Eq. 3.

    Args:
        embedding_dim: Dimension D of the input tokens.
        mlp_dim: Dimension of the hidden layer in the MLP.
        dropout_rate: Dropout probability applied after each linear layer.
    """

    def __init__(self, embedding_dim: int = 768, mlp_dim: int = 3072, dropout_rate: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies layer normalization, then a multi-layer perceptron.

        The residual connection is added by the encoder block, not here.

        Args:
            x: Tensor of shape (B, N, embedding_dim).

        Returns:
            Tensor of shape (B, N, embedding_dim).
        """
        normalized = self.layer_norm(x)
        output = self.mlp(normalized)
        return output


class TransformerEncoderBlock(nn.Module):
    """A single block of the Transformer encoder, consisting of multi-head attention and a multi-layer perceptron.

    Args:
        embedding_dim: Dimension D of the input tokens.
        h: Number of attention heads.
        mlp_dim: Dimension of the hidden layer in the MLP.
        dropout_rate: Dropout probability applied after the attention output projection and each linear layer of the MLP.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        h: int = 12,
        mlp_dim: int = 3072,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        assert embedding_dim % h == 0, f"embedding_dim must be divisible by h: embedding_dim={embedding_dim}, h={h}"
        d_k = embedding_dim // h
        d_v = embedding_dim // h
        self.mha_block = MultiHeadAttentionBlock(embedding_dim, d_k, d_v, h, dropout_rate)
        self.mlp_block = MultiLayerPerceptronBlock(embedding_dim, mlp_dim, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies multi-head attention followed by a multi-layer perceptron, each with a residual connection.

        Args:
            x: Tensor of shape (B, N, embedding_dim).

        Returns:
            Tensor of shape (B, N, embedding_dim).
        """
        # Multi-head attention with residual connection
        mha_output = self.mha_block(x)
        x = x + mha_output

        # MLP with residual connection
        mlp_output = self.mlp_block(x)
        output = x + mlp_output

        return output


class ViT(nn.Module):
    """Vision Transformer (ViT) model. The defaults build ViT-Base/16 (Table 1) for 256x256 RGB images.

    Args:
        image_size: Side length H=W of the input images.
        patch_size: Side length P of each square patch.
        color_channels: Number of image channels C.
        embedding_dim: Dimension D of the token embeddings.
        num_transformer_layers: Number of Transformer encoder blocks L.
        num_heads: Number of attention heads h.
        mlp_dim: Dimension of the hidden layer in the MLP.
        num_classes: Number of output classes for classification.
        mlp_dropout: Dropout probability applied after the attention output projection and each linear layer of the MLP.
        embedding_dropout: Dropout probability applied after adding the position embeddings to the patch embeddings.
    """

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        color_channels: int = 3,
        embedding_dim: int = 768,
        num_transformer_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        num_classes: int = 21,
        mlp_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
    ):
        super().__init__()
        assert image_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {image_size}, patch size: {patch_size}."

        self.num_patches = (image_size * image_size) // patch_size**2

        self.class_embedding = nn.Parameter(torch.zeros((1, 1, embedding_dim), requires_grad=True))
        self.position_embedding = nn.Parameter(torch.randn((1, self.num_patches + 1, embedding_dim), requires_grad=True) * 0.02)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = PatchEmbedding(patch_size, color_channels, embedding_dim)

        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim, num_heads, mlp_dim, mlp_dropout) for _ in range(num_transformer_layers)])

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the output logits of the ViT model.

        Args:
            x: Tensor of shape (B, C, H, W).

        Returns:
            Tensor of shape (B, num_classes).
        """
        batch_size = x.shape[0]
        patch_embeddings = self.patch_embedding(x)
        class_embeddings = self.class_embedding.expand(batch_size, -1, -1)
        embeddings = torch.cat((class_embeddings, patch_embeddings), dim=1)
        embeddings += self.position_embedding
        embeddings = self.embedding_dropout(embeddings)

        transformer_output = self.transformer_encoder(embeddings)
        class_token_output = transformer_output[:, 0]
        logits = self.classifier(class_token_output)

        return logits