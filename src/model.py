"""
SIFTERN model architecture.

Three separate classes to support future multi-partition / transfer-learning
workflows:

  SIFTERNEncoder   - shared transformer encoder (embedding + 5-block encoder +
                     global average pool)
  ClassifierHead   - partition-specific linear classifier (one per sky partition
                     in the full system)
  SIFTERN          - convenience wrapper that composes the two above

To add a second sky partition later:
  1. Instantiate a new ClassifierHead(d_model, n_classes_partition_2).
  2. Call model.freeze_encoder() so only the new head is trained.
  3. Replace model.head with the new head and fine-tune.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SIFTERNEncoder(nn.Module):
    """
    Transformer encoder that maps a variable-length set of 2D centroids to a
    single fixed-size context vector via global average pooling.

    Parameters
    ----------
    d_model : int
        Embedding / model dimension (paper: 256).
    nhead : int
        Number of attention heads (paper: 8).
    n_blocks : int
        Number of transformer encoder layers (paper: 5).
    dim_feedforward : int
        Hidden size of the MLP sub-block inside each encoder layer.
        Defaults to 4 * d_model.
    dropout : float
        Dropout probability (0.0 = disabled, sensible for small datasets).
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        n_blocks: int = 5,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.embedding = nn.Linear(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="relu",
            dropout=dropout,
            batch_first=True,  # input shape: (B, seq_len, d_model)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_blocks,
        )

    def forward(self, x: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, n, 2)
            Batch of re-projected centroid lists (zero-padded).
        src_key_padding_mask : Tensor, shape (B, n), dtype=bool
            True where positions are padding (ignored by attention).

        Returns
        -------
        context : Tensor, shape (B, d_model)
            Global-average-pooled encoder output, computed only over valid
            (non-padding) tokens.
        """
        # (B, n, 2) -> (B, n, d_model)
        h = self.embedding(x)

        # (B, n, d_model) — padding positions attend but do not contribute
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        # Global average pool over valid tokens only
        # padding_mask is True where padding -> invert for valid positions
        valid_mask = ~src_key_padding_mask  # (B, n), True = valid
        valid_counts = valid_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
        # Zero out padded positions before summing
        h = h * valid_mask.unsqueeze(-1).float()
        context = h.sum(dim=1) / valid_counts  # (B, d_model)
        return context


class ClassifierHead(nn.Module):
    """
    Linear classifier that maps an encoder context vector to class logits.

    One instance per sky partition in the full multi-partition system.

    Parameters
    ----------
    d_model : int
        Input dimension (must match SIFTERNEncoder.d_model).
    n_classes : int
        Number of identifiable catalog stars in this partition.
    """

    def __init__(self, d_model: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, context: Tensor) -> Tensor:
        return self.fc(context)


class SIFTERN(nn.Module):
    """
    Full SIFTERN model: encoder + single classifier head.

    For the MVP we use one sky partition.  In the full system, the encoder is
    shared across partitions and each partition has its own ClassifierHead.

    Parameters
    ----------
    n_classes : int
        Number of identifiable catalog stars in the target sky partition.
    d_model : int
        Embedding dimension (paper: 256).
    nhead : int
        Number of attention heads (paper: 8).
    n_blocks : int
        Number of transformer encoder layers (paper: 5).
    """

    def __init__(
        self,
        n_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        n_blocks: int = 5,
    ) -> None:
        super().__init__()
        self.encoder = SIFTERNEncoder(d_model=d_model, nhead=nhead, n_blocks=n_blocks)
        self.head = ClassifierHead(d_model=d_model, n_classes=n_classes)

    def forward(
        self, x: Tensor, src_key_padding_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor, shape (B, n, 2)
        src_key_padding_mask : Tensor, shape (B, n), dtype=bool

        Returns
        -------
        logits : Tensor, shape (B, n_classes)
        context : Tensor, shape (B, d_model)
            The encoder's context vector — useful for downstream tasks
            (e.g. GuidedMatch, multi-head inference) without re-running the
            encoder.
        """
        context = self.encoder(x, src_key_padding_mask)
        logits = self.head(context)
        return logits, context

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True
