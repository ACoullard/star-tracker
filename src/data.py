"""
StarTrackerDataset: loads star scene data from a JSON file and prepares
(centroid list, label) pairs for SIFTERN training.

JSON format per sample:
  {
    "att": [ra, dec, roll],
    "centroids": [[x0, y0], [x1, y1], ...],
    "stars": [catalog_id, ...],
    "mags": [magnitude, ...]
  }

Preprocessing:
  1. Find the guide star: centroid closest to the image center.
  2. Re-project all centroids so the guide star sits at the origin.
  3. Map the guide star's catalog ID to a contiguous class index.

The `transform` parameter is an augmentation pipeline hook. It receives the
raw centroid list (list of [x, y] pairs, already re-projected) and returns a
modified list. Leave as None for the MVP; noise augmentation plugs in here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset


class StarTrackerDataset(Dataset):
    """
    Parameters
    ----------
    path : str | Path
        Path to the JSON data file.
    image_size : int
        Sensor side length in pixels. The image center is taken to be
        (image_size / 2, image_size / 2).
    transform : callable, optional
        Augmentation pipeline applied to the re-projected centroid list
        before it is converted to a tensor. Signature:
            transform(centroids: list[list[float]]) -> list[list[float]]
        Pass None (default) to skip augmentation.
    """

    def __init__(
        self,
        path: str | Path,
        image_size: int = 512,
        transform: Callable | None = None,
    ) -> None:
        self.image_size = image_size
        self.transform = transform
        self._center = image_size / 2.0

        with open(path, "r") as f:
            self._samples = json.load(f)

        # Build a stable class mapping from the unique guide-star catalog IDs
        # found in the dataset.  Sorted for reproducibility.
        self.star_id_to_idx: dict[int, int] = {}
        self.idx_to_star_id: dict[int, int] = {}
        self._build_class_mapping()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def n_classes(self) -> int:
        return len(self.star_id_to_idx)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        sample = self._samples[idx]
        centroids: list[list[float]] = sample["centroids"]
        star_ids: list[int] = sample["stars"]

        # Step 1: find guide star (closest centroid to image center)
        guide_idx = self._find_guide_star(centroids)
        guide_x, guide_y = centroids[guide_idx]

        # Step 2: re-project all centroids so guide star is at origin
        reprojected = [
            [x - guide_x, y - guide_y] for x, y in centroids
        ]

        # Step 3: apply augmentation (no-op for MVP)
        if self.transform is not None:
            reprojected = self.transform(reprojected)

        # Step 4: build tensor and label
        centroids_tensor = torch.tensor(reprojected, dtype=torch.float32)
        label = self.star_id_to_idx[star_ids[guide_idx]]

        return centroids_tensor, label

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_guide_star(self, centroids: list[list[float]]) -> int:
        """Return the index of the centroid closest to the image center."""
        c = self._center
        best_idx = 0
        best_dist_sq = float("inf")
        for i, (x, y) in enumerate(centroids):
            dist_sq = (x - c) ** 2 + (y - c) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = i
        return best_idx

    def _build_class_mapping(self) -> None:
        """Scan the dataset once to collect all guide-star catalog IDs."""
        guide_star_ids: set[int] = set()
        for sample in self._samples:
            centroids = sample["centroids"]
            star_ids = sample["stars"]
            guide_idx = self._find_guide_star(centroids)
            guide_star_ids.add(star_ids[guide_idx])

        for class_idx, star_id in enumerate(sorted(guide_star_ids)):
            self.star_id_to_idx[star_id] = class_idx
            self.idx_to_star_id[class_idx] = star_id


def collate_fn(
    batch: list[tuple[Tensor, int]],
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Pad a batch of variable-length centroid lists to a common length.

    Returns
    -------
    padded : Tensor, shape (B, max_n, 2)
        Centroid coordinates, zero-padded to the longest scene in the batch.
    padding_mask : Tensor, shape (B, max_n), dtype=bool
        True where a position is padding (for use as `src_key_padding_mask`
        in nn.TransformerEncoder).
    labels : Tensor, shape (B,), dtype=long
    """
    centroids_list, labels = zip(*batch)
    batch_size = len(centroids_list)
    max_n = max(c.shape[0] for c in centroids_list)

    padded = torch.zeros(batch_size, max_n, 2, dtype=torch.float32)
    padding_mask = torch.ones(batch_size, max_n, dtype=torch.bool)

    for i, c in enumerate(centroids_list):
        n = c.shape[0]
        padded[i, :n] = c
        padding_mask[i, :n] = False  # False = valid token

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, padding_mask, labels_tensor
