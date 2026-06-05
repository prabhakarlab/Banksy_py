"""Unified, vectorized BANKSY-matrix construction.

:func:`build_banksy_matrix` replaces the original two-step
``initialize_banksy`` + ``generate_banksy_matrix`` pipeline with a single
function. It reproduces the original's *numerical* output (given correct
float coordinates) while replacing every per-cell Python loop with sparse
linear algebra.

Algebraic equivalences used
---------------------------
Neighbour mean (m = 0)::

    nbr_mean = W0 @ X               W0 = row-normalized gaussian weights

Azimuthal Gabor filter (m >= 1)::

    AGF_m = | W @ X |               W = g * e^{i m theta}  (complex CSR)

This turns the original O(n_obs) Python loop into a single complex sparse
mat-mul per order.

Note on centering: the original ``create_nbr_matrix`` has a ``center=True``
default that subtracts the neighbourhood mean before the filter, but the
production entry point ``generate_banksy_matrix`` calls it positionally as
``create_nbr_matrix(adata, banksy_dict, nbr_weight_decay, max_m, variance_balance)``
-- so ``variance_balance`` (``False``) lands in the ``center`` slot and the
filter actually runs *un-centered*. This backend reproduces that production
behaviour (no centering).
"""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
from scipy import sparse

from banksy_ilgwg.neighbors import SpatialNeighborhood
from banksy_ilgwg.weights import (
    azimuthal_angles,
    edge_adjacency,
    scaled_gaussian_weights,
)

DEFAULT_COORD_KEYS = ("x", "y", "spatial")


def build_banksy_matrix(
    adata: anndata.AnnData,
    coord_keys: tuple[str, ...] = DEFAULT_COORD_KEYS,
    *,
    k_geom: int = 15,
    max_m: int = 1,
    lambda_param: float = 0.2,
    nbr_weight_decay: str = "scaled_gaussian",
) -> anndata.AnnData:
    """Build the BANKSY neighbour-augmented matrix for a single ``lambda``.

    Parameters
    ----------
    adata:
        Cell-by-gene AnnData (already HVG-subset if desired). ``adata.X`` and
        ``adata.obsm[coord_keys[-1]]`` are read but never mutated.
    coord_keys:
        Tuple whose last element names the ``obsm`` spatial-coordinate key.
    k_geom:
        Base number of spatial neighbours (order ``m`` uses ``k_geom*(m+1)``).
    max_m:
        Maximum azimuthal Gabor order (``0`` = neighbour mean only).
    lambda_param:
        Neighbourhood contribution in ``[0, 1]``.
    nbr_weight_decay:
        Only ``"scaled_gaussian"`` is supported by this backend.

    Returns
    -------
    AnnData
        Shape ``(n_obs, (max_m + 2) * n_genes)`` dense matrix with the original
        ``obs`` and a ``var`` carrying ``is_nbr`` / ``k`` annotations and
        ``_nbr_{k}`` suffixes, matching the original ``concatenate_all`` layout.
    """
    if nbr_weight_decay != "scaled_gaussian":
        raise NotImplementedError(
            f"banksy_ilgwg supports only 'scaled_gaussian', got {nbr_weight_decay!r}"
        )
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param must be in [0, 1]")
    if max_m < 0:
        raise ValueError("max_m must be non-negative")

    coord_key = coord_keys[-1]
    if coord_key not in adata.obsm:
        raise KeyError(f"coord key {coord_key!r} not found in adata.obsm")

    # Correct angles require float coords (original underflows on uint coords).
    coords = np.asarray(adata.obsm[coord_key], dtype=np.float64)
    x_dense = _as_dense(adata.X)
    n_obs = adata.n_obs

    neighborhood = SpatialNeighborhood(coords)

    # Order 0: weighted neighbour mean. Query k_geom neighbours directly (slicing
    # a larger query would reshuffle ties on gridded data).
    idx0, dist0, _ = neighborhood.query(k_geom)
    w0 = edge_adjacency(idx0, scaled_gaussian_weights(dist0), n_obs)
    nbr_matrices: list[np.ndarray] = [np.asarray(w0 @ x_dense)]

    # Orders >= 1: azimuthal Gabor filter magnitude (each over k_geom*(m+1) nbrs).
    for m in range(1, max_m + 1):
        idx, dist, deltas = neighborhood.query(k_geom * (m + 1))
        nbr_matrices.append(_agf_magnitude(idx, dist, deltas, m, x_dense, n_obs))

    mat_list = [x_dense, *nbr_matrices]
    scale_factors = _scale_factors(
        num_neighbour_matrices=len(nbr_matrices), lambda_param=lambda_param
    )

    scaled = [scale_factors[n] * _zscore_columns(mat) for n, mat in enumerate(mat_list)]
    concatenated = np.concatenate(scaled, axis=1)

    return anndata.AnnData(
        concatenated, obs=adata.obs, var=_build_var(adata, len(nbr_matrices))
    )


def _agf_magnitude(
    idx: np.ndarray,
    dist: np.ndarray,
    deltas: np.ndarray,
    m: int,
    x_dense: np.ndarray,
    n_obs: int,
) -> np.ndarray:
    """Vectorized (un-centered) azimuthal Gabor filter magnitude for order ``m``.

    Equivalent to the original's complex weighting ``|sum_j g_j e^{i m theta_j} x_j|``
    via a single complex sparse mat-mul.
    """
    g = scaled_gaussian_weights(dist)
    theta = azimuthal_angles(deltas)

    weights = edge_adjacency(idx, g * np.exp(1j * m * theta), n_obs)
    return np.abs(np.asarray(weights @ x_dense))


def _scale_factors(*, num_neighbour_matrices: int, lambda_param: float) -> np.ndarray:
    """Per-block scale factors matching the original ``concatenate_all``.

    Own block gets ``sqrt(1 - lambda)``; the ``num_k`` neighbour blocks split
    ``lambda`` with each higher order receiving half the previous order's share.
    """
    num_k = num_neighbour_matrices
    squared = np.zeros(num_k + 1)
    squared[0] = 1.0 - lambda_param
    denom = sum(1.0 / (2 ** (k + 1)) for k in range(num_k))
    for k in range(num_k):
        squared[k + 1] = (1.0 / (2 ** (k + 1))) / denom * lambda_param
    return np.sqrt(squared)


def _zscore_columns(matrix: np.ndarray) -> np.ndarray:
    """Feature-wise z-score replicating the original ``main.zscore`` exactly.

    Uses population variance via ``E[x^2] - E[x]^2`` (not ``np.std``) and
    ``nan_to_num`` for zero-variance columns, matching the original byte-for-byte
    in arithmetic form.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    e_x = mat.mean(axis=0)
    e_x2 = np.square(mat).mean(axis=0)
    variance = e_x2 - np.square(e_x)
    zscored = (mat - e_x) / np.sqrt(variance)
    return np.nan_to_num(zscored)


def _build_var(adata: anndata.AnnData, num_k: int) -> pd.DataFrame:
    """Construct the concatenated ``var`` frame matching ``concatenate_all``."""
    var_original = adata.var.copy()
    var_original["is_nbr"] = False
    var_original["k"] = -1

    var_list = [var_original]
    for k in range(num_k):
        var_nbrs = adata.var.copy()
        var_nbrs["is_nbr"] = True
        var_nbrs["k"] = k
        var_nbrs.index = var_nbrs.index + f"_nbr_{k}"
        var_list.append(var_nbrs)
    return pd.concat(var_list)


def _as_dense(x) -> np.ndarray:
    """Dense float64 view of a (possibly sparse) expression matrix."""
    if sparse.issparse(x):
        return x.toarray().astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)
