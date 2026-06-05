#!/usr/bin/env python
"""Created augmented matrix for data/C04494A6.bin20_1.0.h5ad."""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-banksy")

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.sparse import csr_matrix

from banksy.embed_banksy import generate_banksy_matrix
from banksy.initialize_banksy import initialize_banksy
from banksy_utils.color_lists import zeileis_28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/C04494A6.bin20_1.0.h5ad",
        help="Input AnnData .h5ad file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/C04494A6_banksy",
        help="Directory for the plot and labels.",
    )
    parser.add_argument("--lambda-param", type=float, default=0.2)
    parser.add_argument("--resolution", type=float, default=0.7)
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
        "--write-h5ad",
        action="store_true",
        help="Also write an annotated HVG-subset AnnData file.",
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
        **(metadata or {}),
    }
    banksy_matrix.write_h5ad(
        output_path,
        compression=compression,
        compression_opts=compression_opts,
    )
    return output_path


def make_colormap(num_labels: int) -> ListedColormap:
    colors = list(zeileis_28)
    if num_labels > len(colors):
        extra = plt.get_cmap("tab20b").colors + plt.get_cmap("tab20c").colors
        colors.extend(matplotlib.colors.to_hex(color) for color in extra)
    if num_labels > len(colors):
        colors.extend(
            matplotlib.colors.to_hex(plt.get_cmap("hsv")(i / num_labels))
            for i in range(num_labels)
        )
    return ListedColormap(colors[:num_labels])


def plot_spatial_clusters(
    adata: ad.AnnData,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    coords = np.asarray(adata.obsm["spatial"])
    unique_labels = np.array(sorted(np.unique(labels)))
    remapped = (
        pd.Series(labels)
        .map({label: i for i, label in enumerate(unique_labels)})
        .to_numpy()
    )
    cmap = make_colormap(len(unique_labels))

    fig, ax = plt.subplots(figsize=(9, 8))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=remapped,
        cmap=cmap,
        s=max(0.35, 40000 / adata.n_obs),
        linewidths=0,
        alpha=0.9,
        rasterized=True,
    )
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    pad = max(x_max - x_min, y_max - y_min) * 0.03
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if len(unique_labels) <= 30:
        handles, _ = scatter.legend_elements(num=len(unique_labels))
        ax.legend(
            handles,
            [str(label) for label in unique_labels],
            title="BANKSY cluster",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            markerscale=4,
        )
    else:
        ax.text(
            0.99,
            0.01,
            f"{len(unique_labels)} clusters; labels in banksy_clusters.csv",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coord_keys = ("x", "y", "spatial")
    adata = ad.read_h5ad(args.input)
    adata.var_names_make_unique()

    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = adata.obs.loc[:, ["x", "y"]].to_numpy()

    if "highly_variable" in adata.var:
        hvg_var = adata.var.loc[adata.var["highly_variable"].to_numpy(dtype=bool)]
        if "dispersions_norm" in hvg_var:
            hvg_var = hvg_var.sort_values("dispersions_norm", ascending=False)
        selected_genes = hvg_var.index[: args.n_top_hvg]
        adata = adata[:, selected_genes].copy()
    else:
        adata = adata.copy()

    adata.obsp.clear()
    adata.uns.pop("neighbors", None)

    print(f"Running BANKSY on {adata.n_obs} cells x {adata.n_vars} genes", flush=True)
    print(
        "Parameters: "
        f"k_geom={args.k_geom}, max_m={args.max_m}, "
        f"lambda={args.lambda_param}, pca_dims={args.pca_dims}, "
        f"resolution={args.resolution}, seed={args.seed}",
        flush=True,
    )

    timer_start = time.perf_counter()
    banksy_dict = initialize_banksy(
        adata,
        coord_keys,
        num_neighbours=args.k_geom,
        nbr_weight_decay="scaled_gaussian",
        max_m=args.max_m,
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_agf_angles=False,
        plt_theta=False,
    )

    banksy_dict, banksy_matrix = generate_banksy_matrix(
        adata,
        banksy_dict,
        lambda_list=[args.lambda_param],
        max_m=args.max_m,
    )
    augmentation_elapsed_seconds = time.perf_counter() - timer_start
    print(
        "BANKSY augmentation runtime "
        f"(initialize_banksy through generate_banksy_matrix): "
        f"{augmentation_elapsed_seconds:.2f} seconds "
        f"({augmentation_elapsed_seconds / 60:.2f} minutes)",
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
            "nbr_weight_decay": "scaled_gaussian",
            "seed": int(args.seed),
            "augmentation_elapsed_seconds": round(augmentation_elapsed_seconds, 6),
        },
    )
    print(f"Wrote BANKSY matrix backup: {saved_banksy_matrix}", flush=True)


if __name__ == "__main__":
    main()
