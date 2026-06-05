"""Vectorized spatial k-nearest-neighbour construction for BANKSY.

The original pipeline fits one ``NearestNeighbors(algorithm="ball_tree")`` object
and issues a separate ``kneighbors`` query per azimuthal order ``m`` (requesting
``k_geom * (m + 1)`` neighbours each time). We mirror that exactly: fit once,
query per order.

This per-order querying is deliberate. On gridded data (e.g. ``bin20``) many
neighbours are equidistant, so the k-NN set is tie-ambiguous. Issuing a single
large query and slicing its head would pick different tie-winners than the
original's separate per-order queries, changing the neighbour-mean block. Direct
per-order queries reproduce the original's neighbour selection.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class SpatialNeighborhood:
    """A fitted spatial k-NN model that issues distance-sorted, self-excluded queries."""

    def __init__(self, coords: np.ndarray) -> None:
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("coords must have shape (n_obs, n_dims>=2)")
        self.coords = coords
        self.n_obs = int(coords.shape[0])
        self._nn = NearestNeighbors(algorithm="ball_tree").fit(coords)

    def query(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(indices, distances, deltas)`` for the ``k`` nearest neighbours.

        ``deltas`` are ``coord[nbr] - coord[self]`` over the first two spatial
        dimensions, shape ``(n_obs, k, 2)`` (for azimuthal angles). The cell
        itself is excluded (``X=None`` semantics), matching the original.
        """
        if k < 1:
            raise ValueError("k must be positive")
        if k > self.n_obs - 1:
            raise ValueError(
                f"k={k} exceeds available neighbours (n_obs - 1 = {self.n_obs - 1})"
            )
        distances, indices = self._nn.kneighbors(X=None, n_neighbors=k)
        deltas = self.coords[indices, :2] - self.coords[:, None, :2]
        return indices, distances, deltas
