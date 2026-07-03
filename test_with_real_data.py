#!/usr/bin/env python3
"""
End-to-end integration test using real project data under numpy 2.x.
Uses the STARmap dataset (1549 cells x 1020 genes) — smallest available dataset.
Covers: data loading → BANKSY init → PCA/UMAP → Leiden clustering → cluster refinement
"""
import warnings
warnings.filterwarnings("ignore")

import os, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.sparse import issparse

# Make sure we use the project's src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as _np
import scipy as _sp
print(f"numpy  : {_np.__version__}")
print(f"scipy  : {_sp.__version__}")
print()

PROJ = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJ, "data", "starmap")
ADATA_FILE = "starmap_BY3_1k.h5ad"

# ── Step 1: Load data ───────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Load STARmap h5ad data")
print("=" * 60)

from banksy_utils.load_data import load_adata

coord_keys = ('x', 'y', 'spatial')
random_seed = 1234
np.random.seed(random_seed)

raw_y, raw_x, adata = load_adata(
    DATA_DIR, True, ADATA_FILE, "", "", coord_keys
)
adata.var_names_make_unique()

# filter to cells with annotations
adata = adata[adata.obs["cluster_name"].notnull()]

# load manual annotations
ann_df = pd.read_csv(os.path.join(DATA_DIR, "Starmap_BY3_1k_meta_annotated_18oct22.csv"))
manual_labels = "smoothed_manual"
annotation_key = "manual_annotations"
adata.obs[annotation_key] = ann_df[manual_labels].values
adata.obs[annotation_key] = adata.obs[annotation_key].astype("category")

# add spatial coords
adata.obsm[coord_keys[2]] = pd.concat(
    [adata.obs[coord_keys[0]], adata.obs[coord_keys[1]]], axis=1
).to_numpy()

print(f"  adata shape : {adata.shape}")
print(f"  X dtype     : {adata.X.dtype if hasattr(adata.X,'dtype') else type(adata.X)}")
print(f"  X is sparse : {issparse(adata.X)}")
print(f"  obs cols    : {list(adata.obs.columns)}")
assert adata.shape[0] > 0 and adata.shape[1] > 0
print("[PASS] Step 1: data loaded\n")

# ── Step 2: Initialize BANKSY ───────────────────────────────────────────────
print("=" * 60)
print("Step 2: Initialize BANKSY (build spatial graph, compute AGF)")
print("=" * 60)

from banksy.initialize_banksy import initialize_banksy

t0 = time.perf_counter()
banksy_dict = initialize_banksy(
    adata,
    coord_keys,
    num_neighbours=15,
    nbr_weight_decay="scaled_gaussian",
    max_m=1,
    plt_edge_hist=False,
    plt_nbr_weights=False,
    plt_agf_angles=False,
    plt_theta=False,
)
print(f"  elapsed: {time.perf_counter()-t0:.1f}s")
assert isinstance(banksy_dict, dict) and len(banksy_dict) > 0
print(f"  banksy_dict keys: {list(banksy_dict.keys())}")
print("[PASS] Step 2: BANKSY initialized\n")

# ── Step 3: Run BANKSY (PCA + Leiden clustering) ────────────────────────────
print("=" * 60)
print("Step 3: Run BANKSY — PCA + Leiden clustering")
print("=" * 60)

from banksy.run_banksy import run_banksy_multiparam
from banksy_utils.color_lists import spagcn_color

t0 = time.perf_counter()
results_df = run_banksy_multiparam(
    adata,
    banksy_dict,
    lambda_list=[0.8],
    resolutions=[0.8],
    color_list=spagcn_color,
    max_m=1,
    filepath=os.path.join(PROJ, "data", "starmap", "tmp_test"),
    key=coord_keys,
    pca_dims=[20],
    annotation_key=annotation_key,
    max_labels=7,
    cluster_algorithm="leiden",
    match_labels=False,
    savefig=False,
    add_nonspatial=False,
    variance_balance=False,
)
print(f"  elapsed: {time.perf_counter()-t0:.1f}s")
assert results_df is not None and len(results_df) > 0
print(f"  results_df shape: {results_df.shape}")
print(f"  results_df:\n{results_df.to_string()}")
print("[PASS] Step 3: clustering complete\n")

# ── Step 4: Refine clusters ─────────────────────────────────────────────────
print("=" * 60)
print("Step 4: Refine clusters")
print("=" * 60)

from banksy_utils.refine_clusters import refine_clusters

t0 = time.perf_counter()
results_df = refine_clusters(
    adata,
    results_df,
    coord_keys=coord_keys,
    color_list=spagcn_color,
    savefig=False,
    output_folder=os.path.join(PROJ, "data", "starmap", "tmp_test"),
    refine_method="once",
    annotation_key=annotation_key,
    num_neigh=6,
)
print(f"  elapsed: {time.perf_counter()-t0:.1f}s")
assert results_df is not None
print(f"  refined results_df:\n{results_df.to_string()}")
print("[PASS] Step 4: cluster refinement complete\n")

# ── Step 5: Verify cluster labels in results_df ─────────────────────────────
print("=" * 60)
print("Step 5: Verify cluster labels and ARI in results_df")
print("=" * 60)

print(f"  results_df columns: {list(results_df.columns)}")
assert 'labels' in results_df.columns, "No 'labels' column in results_df"
assert 'ari' in results_df.columns, "No 'ari' column in results_df"

for idx, row in results_df.iterrows():
    lab = row['labels']
    ari = float(row['ari'])
    refined_ari = float(row['refined_ari']) if 'refined_ari' in results_df.columns else None
    n_cells = lab.num_samples
    n_clusters = lab.num_labels
    dense_arr = lab.dense                          # numpy array of integer labels
    print(f"  [{idx}]")
    print(f"    n_cells    = {n_cells}")
    print(f"    n_clusters = {n_clusters}")
    print(f"    ARI        = {ari:.4f}")
    if refined_ari is not None:
        print(f"    refined ARI= {refined_ari:.4f}")
    print(f"    labels (first 10): {dense_arr[:10]}")
    assert isinstance(dense_arr, np.ndarray), f"dense should be ndarray, got {type(dense_arr)}"
    assert n_cells == adata.shape[0], f"n_cells mismatch: {n_cells} vs {adata.shape[0]}"
    assert ari > 0.0, f"ARI should be positive, got {ari}"
    assert n_clusters >= 2, f"Should have at least 2 clusters, got {n_clusters}"

print("[PASS] Step 5: labels and ARI verified\n")

print("=" * 60)
print(f"ALL STEPS PASSED — numpy {_np.__version__} is fully compatible")
print("=" * 60)
