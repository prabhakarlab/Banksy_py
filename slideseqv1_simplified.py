# This is the simplified python script that demonstrates how to directly run slide-seq v1 dataset with BANKSY
import os, time
import numpy as np
import warnings
from banksy_utils.color_lists import spagcn_color

warnings.filterwarnings("ignore")

import scanpy as sc

sc.logging.print_header()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 1  # errors (0), warnings (1), info (2), hints (3)

import random

# Note that BANKSY itself is deterministic, here the seeds affect the umap clusters and leiden partition
seed = 0
np.random.seed(seed)
random.seed(seed)
start = time.perf_counter_ns()

# %%
# Define File paths
file_path = os.path.join("data", "slide_seq", "v1")
gcm_filename = "Cerebellum_MappedDGEForR.csv"

# (Optional) Arguments for load_data only if annadata is not present
locations_filename = "Cerebellum_BeadLocationsForR.csv"
adata_filename = "slideseqv1_cerebellum_adataraw.h5ad"

# %%
from banksy_utils.load_data import load_adata

# To either load data from .h5ad directly or convert raw data to .h5ad format
load_adata_directly = True

# Keys to specify coordinate indexes in the anndata Object
coord_keys = ('xcoord', 'ycoord', 'coord_xy')

raw_y, raw_x, adata = load_adata(file_path,
                                 load_adata_directly,
                                 adata_filename,
                                 gcm_filename,
                                 locations_filename,
                                 coord_keys)

# %%
adata.var_names_make_unique()
adata.var["mt"] = adata.var_names.str.startswith("MT-")

# Calulates QC metrics and put them in place to the adata object
sc.pp.calculate_qc_metrics(adata,
                           qc_vars=["mt"],
                           log1p=True,
                           inplace=True)

# %%
from banksy_utils.filter_utils import filter_cells

# Filter cells with each respective filters
adata = filter_cells(adata,
                     min_count=40,
                     max_count=1000,
                     MT_filter=20,
                     gene_filter=10)

# %%
from banksy_utils.filter_utils import normalize_total, filter_hvg

# Normalizes the anndata dataset
adata = normalize_total(adata)

# %%
adata, adata_allgenes = filter_hvg(adata,
                                   n_top_genes=2000,
                                   flavor="seurat")

# %%
from banksy.main import median_dist_to_nearest_neighbour

# set params
# ==========
plot_graph_weights = True
k_geom = 15  # only for fixed type
max_m = 1  # azumithal transform up to kth order
nbr_weight_decay = "scaled_gaussian"  # can also be "reciprocal", "uniform" or "ranked"

# Find median distance to closest neighbours, the median distance will be `sigma`
nbrs = median_dist_to_nearest_neighbour(adata, key=coord_keys[2])

from banksy.initialize_banksy import initialize_banksy
from banksy.run_banksy import run_banksy_multiparam

banksy_dict = initialize_banksy(adata,
                                coord_keys,
                                k_geom,
                                nbr_weight_decay=nbr_weight_decay,
                                max_m=max_m,
                                plt_edge_hist=True,
                                plt_nbr_weights=True,
                                plt_agf_angles=False
                                )

from banksy.embed_banksy import generate_banksy_matrix

# The following are the main hyperparameters for BANKSY
resolutions = [0.7]  # clustering resolution for UMAP
pca_dims = [20]  # Dimensionality in which PCA reduces to
lambda_list = [0.2]  # list of lambda parameters

banksy_dict, banksy_matrix = generate_banksy_matrix(adata,
                                                    banksy_dict,
                                                    lambda_list,
                                                    max_m)

output_folder = os.path.join(file_path, 'tmp_png')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

results_df = run_banksy_multiparam(
    adata,
    banksy_dict,
    lambda_list,
    resolutions,
    color_list=spagcn_color,
    max_m=max_m,
    filepath=output_folder,
    key=coord_keys,
    pca_dims=pca_dims,
    annotation_key=None,
    max_labels=None,
    cluster_algorithm='leiden',
    match_labels=False,
    savefig=True,
    add_nonspatial=True,
    variance_balance=False
)

## Save results
results_df.to_csv(os.path.join(output_folder, 'results.csv'))
