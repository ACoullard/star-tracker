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

Class mapping is driven by filter-catalog.csv (the same catalog used to
generate the JSON), filtered to the sky partition defined by dec_range.
This makes class indices stable across partitions and runs.

The `transform` parameter is an augmentation pipeline hook. It receives the
re-projected centroid list (list of [x, y] pairs, guide star at [0.0, 0.0])
and returns a modified list. See CentroidAugmenter in augment.py.
"""

from __future__ import annotations

import csv
import json
import math
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
    catalog_path : str | Path
        Path to filter-catalog.csv.  Format: ID, spat_x, spat_y, spat_z, mag
    dec_range : tuple[float, float]
        (min_deg, max_deg) declination bounds of this sky partition.  The class
        mapping includes all catalog stars with declination in
        [min_deg - half_fov, max_deg + half_fov) to cover every star that can
        appear as a guide star for attitudes within the partition.
    half_fov : float
        Half the camera field of view in degrees (default 6.0 for a 12° FOV).
        Used to pad dec_range when filtering the catalog.
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
        catalog_path: str | Path,
        dec_range: tuple[float, float],
        half_fov: float = 6.0,
        image_size: int = 512,
        transform: Callable | None = None,
    ) -> None:
        self.image_size = image_size
        self.transform = transform
        self._center = image_size / 2.0
        self._catalog_path = Path(catalog_path)
        self._dec_range = dec_range
        self._half_fov = half_fov

        with open(path, "r") as f:
            self._samples = json.load(f)

        self.star_id_to_idx: dict[int, int] = {}
        self.idx_to_star_id: dict[int, int] = {}
        self._build_class_mapping(catalog_path, dec_range, half_fov)

        # Precompute guide star index for every sample once at init.
        # Avoids recomputing in __getitem__ on every training step.
        self._guide_indices: list[int] = [
            self._find_guide_star(s["centroids"]) for s in self._samples
        ]

    def with_transform(self, transform: Callable | None) -> "StarTrackerDataset":
        """Return a shallow copy of this dataset with a different transform.

        All heavy data (_samples, _guide_indices, catalog mappings) is shared
        in memory — only the transform attribute differs.
        """
        import copy
        ds = copy.copy(self)
        ds.transform = transform
        return ds

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

        guide_idx = self._guide_indices[idx]
        guide_x, guide_y = centroids[guide_idx]

        reprojected = [
            [x - guide_x, y - guide_y] for x, y in centroids
        ]

        if self.transform is not None:
            reprojected = self.transform(reprojected)

        centroids_tensor = torch.tensor(reprojected, dtype=torch.float32)
        label = self.star_id_to_idx[star_ids[guide_idx]]

        return centroids_tensor, label

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

    def _build_class_mapping(
        self,
        catalog_path: str | Path,
        dec_range: tuple[float, float],
        half_fov: float,
    ) -> None:
        """
        Build star_id → class_index mapping from filter-catalog.csv.

        Only catalog stars with declination in
        [dec_range[0] - half_fov, dec_range[1] + half_fov) are included.
        Sorted by star ID for a deterministic, reproducible mapping.
        """
        dec_lo = dec_range[0] - half_fov
        dec_hi = dec_range[1] + half_fov

        star_ids: list[int] = []
        with open(catalog_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                star_id = int(row[0])
                spat_z = float(row[3])
                dec_deg = math.degrees(math.asin(spat_z))
                if dec_lo <= dec_deg < dec_hi:
                    star_ids.append(star_id)

        for class_idx, star_id in enumerate(sorted(star_ids)):
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
