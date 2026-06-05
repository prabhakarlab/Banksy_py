"""Equivalence tests: banksy_ilgwg reproduces the original BANKSY matrix.

The original pipeline is fed *float* coordinates so that its azimuthal angles
are correct (on uint coords it underflows; see banksy_ilgwg docs). Under that
condition the new vectorized backend must match the original's matrix to within
floating-point reordering tolerance.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from banksy.embed_banksy import generate_banksy_matrix
from banksy.initialize_banksy import initialize_banksy
from banksy_ilgwg import build_banksy_matrix

COORD_KEYS = ("x", "y", "spatial")


def _make_adata(n_obs: int = 200, n_genes: int = 40, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(1.0, size=(n_obs, n_genes)).astype(np.float64)
    coords = rng.uniform(0.0, 1000.0, size=(n_obs, 2)).astype(np.float64)

    adata = ad.AnnData(sparse.csr_matrix(counts))
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.obsm["spatial"] = coords
    return adata


def _original_matrix(
    adata: ad.AnnData, *, k_geom: int, max_m: int, lambda_param: float
) -> ad.AnnData:
    banksy_dict = initialize_banksy(
        adata,
        COORD_KEYS,
        num_neighbours=k_geom,
        nbr_weight_decay="scaled_gaussian",
        max_m=max_m,
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_agf_angles=False,
        plt_theta=False,
    )
    _, banksy_matrix = generate_banksy_matrix(
        adata, banksy_dict, lambda_list=[lambda_param], max_m=max_m, verbose=False
    )
    return banksy_matrix


@pytest.mark.parametrize("max_m", [0, 1, 2])
@pytest.mark.parametrize("lambda_param", [0.2, 0.8])
def test_matrix_matches_original(max_m: int, lambda_param: float) -> None:
    adata = _make_adata()
    k_geom = 6

    expected = _original_matrix(
        adata, k_geom=k_geom, max_m=max_m, lambda_param=lambda_param
    )
    actual = build_banksy_matrix(
        adata, COORD_KEYS, k_geom=k_geom, max_m=max_m, lambda_param=lambda_param
    )

    assert actual.shape == expected.shape
    # The original stores azimuthal angles as float32 (main.theta_from_spatial_graph),
    # so the AGF block carries ~1e-7 angle error; this backend uses float64 and is
    # strictly more accurate. Tolerance reflects the original's float32 residual.
    np.testing.assert_allclose(
        np.asarray(actual.X), np.asarray(expected.X), atol=1e-6, rtol=1e-5
    )


def test_var_and_obs_layout_matches_original() -> None:
    adata = _make_adata()
    expected = _original_matrix(adata, k_geom=6, max_m=1, lambda_param=0.2)
    actual = build_banksy_matrix(adata, COORD_KEYS, k_geom=6, max_m=1, lambda_param=0.2)

    assert list(actual.var_names) == list(expected.var_names)
    assert list(actual.obs_names) == list(expected.obs_names)
    np.testing.assert_array_equal(
        actual.var["is_nbr"].to_numpy(), expected.var["is_nbr"].to_numpy()
    )
    np.testing.assert_array_equal(
        actual.var["k"].to_numpy(), expected.var["k"].to_numpy()
    )


def test_does_not_mutate_input() -> None:
    adata = _make_adata()
    x_before = adata.X.toarray().copy()
    coords_before = adata.obsm["spatial"].copy()

    build_banksy_matrix(adata, COORD_KEYS, k_geom=6, max_m=1, lambda_param=0.2)

    np.testing.assert_array_equal(adata.X.toarray(), x_before)
    np.testing.assert_array_equal(adata.obsm["spatial"], coords_before)


def test_rejects_unsupported_decay() -> None:
    adata = _make_adata()
    with pytest.raises(NotImplementedError):
        build_banksy_matrix(adata, COORD_KEYS, nbr_weight_decay="reciprocal")
