"""Vectorized neighbour-weight construction matching the original BANKSY.

Replaces the per-row Python loops in ``main.generate_spatial_weights_fixed_nbrs``
(``scaled_gaussian`` branch, ``row_normalize``, ``theta_from_spatial_graph``)
with array operations over the dense ``(n_obs, k)`` neighbour graph.

All per-row reductions (median, exp, row-normalize) are order-independent, so
operating on the distance-sorted dense layout yields identical weight values to
the original CSR-data loops.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def scaled_gaussian_weights(distances: np.ndarray) -> np.ndarray:
    """Row-normalized ``scaled_gaussian`` weights for an ``(n_obs, k)`` block.

    Matches the original::

        median_r = np.median(row_distances)
        weights  = np.exp(-(row_distances / median_r) ** 2)
        weights /= weights.sum()   # row_normalize

    The median is taken over each cell's ``k`` neighbour distances.
    """
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2 or distances.shape[1] == 0:
        raise ValueError("distances must have shape (n_obs, k>=1)")

    median_r = np.median(distances, axis=1, keepdims=True)
    weights = np.exp(-((distances / median_r) ** 2))

    row_sums = weights.sum(axis=1, keepdims=True)
    # Original row_normalize leaves all-zero rows untouched (divides only when
    # row_sum != 0); replicate that to avoid introducing NaNs.
    np.divide(weights, row_sums, out=weights, where=row_sums != 0)
    return weights


def azimuthal_angles(deltas: np.ndarray) -> np.ndarray:
    """Per-edge azimuthal angles ``arctan2(dy, dx)`` for an ``(n_obs, k, 2)`` block.

    Matches ``theta_from_spatial_graph``: ``arctan2(relative_y, relative_x)``.
    """
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 3 or deltas.shape[2] < 2:
        raise ValueError("deltas must have shape (n_obs, k, >=2)")
    return np.arctan2(deltas[:, :, 1], deltas[:, :, 0])


def edge_adjacency(
    indices: np.ndarray, data: np.ndarray, n_obs: int
) -> sparse.csr_matrix:
    """Build an ``(n_obs, n_obs)`` CSR adjacency from a dense edge block.

    ``indices`` and ``data`` both have shape ``(n_obs, k)``; row ``i`` connects
    to columns ``indices[i]`` with values ``data[i]``.
    """
    k = indices.shape[1]
    rows = np.repeat(np.arange(n_obs), k)
    cols = indices.reshape(-1)
    values = data.reshape(-1)
    return sparse.csr_matrix((values, (rows, cols)), shape=(n_obs, n_obs))
