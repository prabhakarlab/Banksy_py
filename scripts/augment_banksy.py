#!/usr/bin/env python
"""Create an augmented BANKSY matrix using the optimized ``banksy_ilgwg`` backend.

Drop-in replacement for the ``initialize_banksy`` + ``generate_banksy_matrix``
stage of ``augment_c04494a6_banksy.py``. The slow two-step, per-cell-loop
pipeline is replaced by a single vectorized call to
:func:`banksy_ilgwg.build_banksy_matrix`.

The produced ``banksy_matrix`` is numerically equivalent to the original
pipeline *with correct (float) coordinates*. Note that the original on this
dataset silently corrupts the AGF block via a uint32 coordinate underflow; this
backend computes the correct angles, so its AGF columns intentionally differ
from the legacy (buggy) output.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-banksy")

import anndata as ad
import numpy as np

from banksy_ilgwg import build_banksy_matrix

COORD_KEYS = ("x", "y", "spatial")
NBR_WEIGHT_DECAY = "scaled_gaussian"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/C04494A6.bin20_1.0.h5ad",
        help="Input AnnData .h5ad file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/C04494A6_banksy",
        help="Directory for the BANKSY matrix backup.",
    )
    parser.add_argument("--lambda-param", type=float, default=0.2)
    parser.add_argument("--pca-dims", type=int, default=20)
    parser.add_argument("--k-geom", type=int, default=15)
    parser.add_argument("--max-m", type=int, default=1)
    parser.add_argument(
        "--n-top-hvg",
        type=int,
        default=1000,
        help="Use this many highly variable genes, ranked by dispersions_norm when available.",
    )
    parser.add_argument(
        "--banksy-matrix-compression",
        choices=("gzip", "lzf", "none"),
        default="gzip",
        help="Compression codec for the BANKSY matrix .h5ad backup.",
    )
    parser.add_argument(
        "--banksy-matrix-compression-level",
        type=int,
        default=4,
        help="gzip compression level for the BANKSY matrix .h5ad backup.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def default_banksy_matrix_output_path(
    output_dir: Path, args: argparse.Namespace
) -> Path:
    lambda_token = f"{args.lambda_param:g}".replace(".", "p")
    return output_dir / (
        f"C04494A6_banksy_matrix"
        f"_lambda{lambda_token}"
        f"_kgeom{args.k_geom}"
        f"_maxm{args.max_m}"
        f"_hvg{args.n_top_hvg}"
        ".h5ad"
    )


def save_banksy_matrix(
    banksy_matrix: ad.AnnData,
    output_path: Path | str,
    *,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a generated BANKSY matrix as compressed AnnData."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if compression == "none":
        compression = None
    if compression != "gzip":
        compression_opts = None

    banksy_matrix.uns["banksy_matrix_backup"] = {
        "format": "h5ad",
        "compression": compression or "none",
        "backend": "banksy_ilgwg",
        **(metadata or {}),
    }
    banksy_matrix.write_h5ad(
        output_path,
        compression=compression,
        compression_opts=compression_opts,
    )
    return output_path


def load_hvg_subset(input_path: str, n_top_hvg: int) -> ad.AnnData:
    """Read the AnnData, ensure spatial coords, and subset to top HVGs."""
    adata = ad.read_h5ad(input_path)
    adata.var_names_make_unique()

    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = adata.obs.loc[:, ["x", "y"]].to_numpy()

    if "highly_variable" in adata.var:
        hvg_var = adata.var.loc[adata.var["highly_variable"].to_numpy(dtype=bool)]
        if "dispersions_norm" in hvg_var:
            hvg_var = hvg_var.sort_values("dispersions_norm", ascending=False)
        selected_genes = hvg_var.index[:n_top_hvg]
        adata = adata[:, selected_genes].copy()
    else:
        adata = adata.copy()

    adata.obsp.clear()
    adata.uns.pop("neighbors", None)
    return adata


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = load_hvg_subset(args.input, args.n_top_hvg)

    print(f"Running BANKSY on {adata.n_obs} cells x {adata.n_vars} genes", flush=True)
    print(
        "Parameters: "
        f"k_geom={args.k_geom}, max_m={args.max_m}, "
        f"lambda={args.lambda_param}, pca_dims={args.pca_dims}, "
        f"seed={args.seed}, backend=banksy_ilgwg",
        flush=True,
    )

    timer_start = time.perf_counter()
    banksy_matrix = build_banksy_matrix(
        adata,
        COORD_KEYS,
        k_geom=args.k_geom,
        max_m=args.max_m,
        lambda_param=args.lambda_param,
        nbr_weight_decay=NBR_WEIGHT_DECAY,
    )
    augmentation_elapsed_seconds = time.perf_counter() - timer_start
    print(
        "BANKSY augmentation runtime "
        f"(build_banksy_matrix): "
        f"{augmentation_elapsed_seconds:.2f} seconds "
        f"({augmentation_elapsed_seconds / 60:.2f} minutes)",
        flush=True,
    )
    print(
        f"Shape of BANKSY matrix: {banksy_matrix.shape}",
        flush=True,
    )

    banksy_matrix_output = default_banksy_matrix_output_path(output_dir, args)
    saved_banksy_matrix = save_banksy_matrix(
        banksy_matrix,
        banksy_matrix_output,
        compression=args.banksy_matrix_compression,
        compression_opts=args.banksy_matrix_compression_level,
        metadata={
            "input": args.input,
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "lambda_param": float(args.lambda_param),
            "k_geom": int(args.k_geom),
            "max_m": int(args.max_m),
            "n_top_hvg": int(args.n_top_hvg),
            "nbr_weight_decay": NBR_WEIGHT_DECAY,
            "seed": int(args.seed),
            "augmentation_elapsed_seconds": round(augmentation_elapsed_seconds, 6),
        },
    )
    print(f"Wrote BANKSY matrix backup: {saved_banksy_matrix}", flush=True)


if __name__ == "__main__":
    main()
