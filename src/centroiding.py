from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label, find_objects
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Internal blob representation
# ---------------------------------------------------------------------------

@dataclass
class _BlobROI:
    x0: int
    y0: int
    x1: int
    y1: int
    patch: np.ndarray  # float32, shape (y1-y0, x1-x0)


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CentroiderParams:
    threshold: float = 0.05   # intensity threshold as fraction of [0, 1]
    min_area: int = 2         # minimum blob area in pixels
    max_area: int = 200       # maximum blob area in pixels (rejects large artifacts)
    border_margin: int = 5    # exclude blobs whose bbox touches this border


@dataclass
class CoGParams(CentroiderParams):
    pass


@dataclass
class GaussianGridParams(CentroiderParams):
    fit_radius: int = 5       # half-width of the Gaussian fit window in pixels
    sigma_init: float = 1.5   # initial sigma guess for curve_fit
    max_iter: int = 50        # max function evaluations for curve_fit


# ---------------------------------------------------------------------------
# Gaussian helper (module-level so it is pickleable)
# ---------------------------------------------------------------------------

def _gaussian_2d(xy, A, cx, cy, sx, sy, B):
    x, y = xy
    return A * np.exp(-((x - cx) ** 2 / (2 * sx ** 2) + (y - cy) ** 2 / (2 * sy ** 2))) + B


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class Centroider(ABC):
    def __init__(self, params: CentroiderParams) -> None:
        self.params = params

    def extract(self, image) -> list[list[float]]:
        """
        Extract centroids from a star image.

        Accepts:
          - numpy ndarray uint8 or float32: (H, W) or (H, W, C)
          - torch.Tensor: (H, W), (1, H, W), or (H, W, C)

        Returns list of [x, y] pixel coordinates (float), compatible with
        StarTrackerDataset centroid format.
        """
        gray = self._to_gray(image)
        blobs = self._detect_blobs(gray)
        centroids = []
        for blob in blobs:
            try:
                cx, cy = self._refine(blob)
                centroids.append([cx, cy])
            except (RuntimeError, ValueError):
                continue
        return centroids

    def _to_gray(self, image) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 3:
            image = image.mean(axis=2)
        if image.max() > 1.0:
            image = image / 255.0
        return image

    def _detect_blobs(self, gray: np.ndarray) -> list[_BlobROI]:
        p = self.params
        binary = (gray > p.threshold).astype(np.uint8)
        labeled, _ = label(binary)

        blobs: list[_BlobROI] = []
        H, W = gray.shape
        m = p.border_margin

        for region_slice in find_objects(labeled):
            if region_slice is None:
                continue
            row_slice, col_slice = region_slice
            y0, y1 = row_slice.start, row_slice.stop
            x0, x1 = col_slice.start, col_slice.stop

            area = (y1 - y0) * (x1 - x0)
            if not (p.min_area <= area <= p.max_area):
                continue
            if x0 < m or y0 < m or x1 > W - m or y1 > H - m:
                continue

            patch = gray[y0:y1, x0:x1].copy()
            blobs.append(_BlobROI(x0=x0, y0=y0, x1=x1, y1=y1, patch=patch))

        return blobs

    @abstractmethod
    def _refine(self, blob: _BlobROI) -> tuple[float, float]:
        """Return (x, y) centroid in full-image pixel coordinates."""
        ...


# ---------------------------------------------------------------------------
# Center of Gravity
# ---------------------------------------------------------------------------

class CenterOfGravity(Centroider):
    """
    Intensity-weighted centroid.
      cx = sum(I_ij * j) / sum(I_ij)
      cy = sum(I_ij * i) / sum(I_ij)
    Fast (pure numpy), no fit failure modes.
    """

    def __init__(self, params: CoGParams | None = None) -> None:
        super().__init__(params or CoGParams())

    def _refine(self, blob: _BlobROI) -> tuple[float, float]:
        patch = blob.patch
        total = patch.sum()
        if total == 0:
            raise ValueError("Empty blob patch")

        rows = np.arange(blob.y0, blob.y1)
        cols = np.arange(blob.x0, blob.x1)
        col_grid, row_grid = np.meshgrid(cols, rows)

        cx = float((patch * col_grid).sum() / total)
        cy = float((patch * row_grid).sum() / total)
        return cx, cy


# ---------------------------------------------------------------------------
# Gaussian Grid
# ---------------------------------------------------------------------------

class GaussianGrid(Centroider):
    """
    Fits a 2D Gaussian to each blob using scipy.optimize.curve_fit.
    Uses a CoG pre-estimate to initialize and crop the fit window,
    giving sub-pixel accuracy and stable convergence.
    """

    def __init__(self, params: GaussianGridParams | None = None) -> None:
        super().__init__(params or GaussianGridParams())

    def _refine(self, blob: _BlobROI) -> tuple[float, float]:
        p: GaussianGridParams = self.params  # type: ignore[assignment]
        patch = blob.patch

        total = patch.sum()
        if total == 0:
            raise ValueError("Empty blob patch")

        # CoG pre-estimate to initialise Gaussian center
        rows = np.arange(blob.y0, blob.y1)
        cols = np.arange(blob.x0, blob.x1)
        col_grid, row_grid = np.meshgrid(cols, rows)
        cx0 = float((patch * col_grid).sum() / total)
        cy0 = float((patch * row_grid).sum() / total)

        # Crop a tight window around the pre-estimate
        r = p.fit_radius
        y_lo = max(blob.y0, int(cy0) - r)
        y_hi = min(blob.y1, int(cy0) + r + 1)
        x_lo = max(blob.x0, int(cx0) - r)
        x_hi = min(blob.x1, int(cx0) + r + 1)

        subpatch = blob.patch[y_lo - blob.y0: y_hi - blob.y0,
                               x_lo - blob.x0: x_hi - blob.x0]

        if subpatch.size == 0:
            return cx0, cy0

        sub_cols = np.arange(x_lo, x_hi)
        sub_rows = np.arange(y_lo, y_hi)
        col_g, row_g = np.meshgrid(sub_cols, sub_rows)
        xdata = np.vstack([col_g.ravel(), row_g.ravel()])
        ydata = subpatch.ravel()

        A0 = float(ydata.max())
        B0 = float(ydata.min())
        p0 = [A0, cx0, cy0, p.sigma_init, p.sigma_init, B0]
        bounds = (
            [0,          x_lo, y_lo, 0.5, 0.5, 0],
            [1.5 * A0 + 1e-6, x_hi, y_hi, r,   r,   A0],
        )

        popt, _ = curve_fit(
            _gaussian_2d, xdata, ydata,
            p0=p0, bounds=bounds, max_nfev=p.max_iter,
        )
        _, cx_fit, cy_fit, *_ = popt
        return float(cx_fit), float(cy_fit)


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

CENTROIDER_REGISTRY: dict[str, tuple[type[Centroider], type[CentroiderParams]]] = {
    "cog": (CenterOfGravity, CoGParams),
    "gaussian_grid": (GaussianGrid, GaussianGridParams),
}


def get_centroider(
    name: str,
    params: CentroiderParams | dict | None = None,
) -> Centroider:
    """
    Factory function.

    Usage:
        get_centroider("cog")
        get_centroider("cog", {"threshold": 0.08, "max_stars": 20})
        get_centroider("gaussian_grid", GaussianGridParams(fit_radius=7))
    """
    if name not in CENTROIDER_REGISTRY:
        known = list(CENTROIDER_REGISTRY.keys())
        raise KeyError(f"Unknown centroider '{name}'. Known algorithms: {known}")

    cls, params_cls = CENTROIDER_REGISTRY[name]
    if params is None:
        resolved = params_cls()
    elif isinstance(params, dict):
        resolved = params_cls(**params)
    else:
        resolved = params
    return cls(resolved)
