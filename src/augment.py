from __future__ import annotations

import random


class CentroidAugmenter:
    """
    Applies three perturbations to a re-projected centroid list to simulate
    non-ideal sensor conditions, matching the SIFTER paper augmentation scheme.

    The guide star is always at [0.0, 0.0] in the re-projected frame.
    It receives position noise but is never dropped.

    Parameters
    ----------
    image_size : int
        Sensor side length in pixels. Used to derive half_size = image_size / 2
        for bounds checking and false-star placement.
    centroid_sigma : float
        Std dev of Gaussian position noise applied to every centroid (pixels).
    max_false_stars : int
        Upper bound on injected false stars. k ~ Uniform[0, max_false_stars].
    drop_prob : float
        Per-star probability of dropping each non-guide centroid, simulating
        stars lost to magnitude cutoff or obscuration. Default 0 (disabled).
    seed : int | None
        Optional seed for a private RNG. When None, uses the global random state.
    """

    def __init__(
        self,
        image_size: int = 512,
        centroid_sigma: float = 5.0,
        max_false_stars: int = 5,
        drop_prob: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.half_size = image_size / 2.0
        self.centroid_sigma = centroid_sigma
        self.max_false_stars = max_false_stars
        self.drop_prob = drop_prob
        self._rng = random.Random(seed)

    def __call__(
        self, centroids: list[list[float]]
    ) -> list[list[float]]:
        """
        Parameters
        ----------
        centroids : list of [x, y]
            Re-projected centroid list. Guide star is at [0.0, 0.0].

        Returns
        -------
        list of [x, y]
            Perturbed centroid list (length may differ from input).
            Guide star is always the first element.
        """
        # Separate guide star (identified by exact [0.0, 0.0] in the input
        # before any noise is applied) from the rest.
        guide_idx = next(
            (i for i, (x, y) in enumerate(centroids) if x == 0.0 and y == 0.0),
            0,
        )
        guide = centroids[guide_idx]
        others = [c for i, c in enumerate(centroids) if i != guide_idx]

        noised_guide = self._noise(guide, bounds_check=False)
        noised_others = [
            noised
            for c in others
            for noised in (self._noise(c, bounds_check=True),)
            if noised is not None
        ]

        if self.drop_prob > 0.0:
            noised_others = [
                c for c in noised_others if self._rng.random() >= self.drop_prob
            ]

        false_stars = self._make_false_stars()

        return [noised_guide] + noised_others + false_stars

    # ------------------------------------------------------------------

    def _noise(
        self, centroid: list[float], bounds_check: bool
    ) -> list[float] | None:
        """Apply Gaussian noise. Returns None if out of bounds (and bounds_check=True)."""
        nx = centroid[0] + self._rng.gauss(0.0, self.centroid_sigma)
        ny = centroid[1] + self._rng.gauss(0.0, self.centroid_sigma)
        if bounds_check and (abs(nx) > self.half_size or abs(ny) > self.half_size):
            return None
        return [nx, ny]

    def _make_false_stars(self) -> list[list[float]]:
        k = self._rng.randint(0, self.max_false_stars)
        h = self.half_size
        return [
            [self._rng.uniform(-h, h), self._rng.uniform(-h, h)]
            for _ in range(k)
        ]
