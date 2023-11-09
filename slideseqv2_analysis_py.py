# %%
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
sc.logging.print_header()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 1 # errors (0), warnings (1), info (2), hints (3)

import random
# Note that BANKSY itself is deterministic, here the seeds affect the umap clusters and leiden partition
seed = 0
np.random.seed(seed)
random.seed(seed)

# %%
# Define File paths
file_path = os.path.join("data", "slide_seq", "v2")
gcm_filename = "Cerebellum_MappedDGEForR.csv"

# (Optional) Arguments for load_data only if annadata is not present
locations_filename = "Cerebellum_BeadLocationsForR.csv"
adata_filename = "slideseqv2_cerebellum_adataraw.h5ad"

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
from banksy_utils.plot_utils import plot_cell_positions

# %%
from banksy_utils.filter_utils import filter_cells

# Filter cells with each respective filters
adata = filter_cells(adata, 
             min_count=50, 
             max_count=2500, 
             MT_filter=20, 
             gene_filter=10)

# %%
# Filter out beads outside the puck
puck_center = (3330,3180) # (x,y)
puck_radius = 2550
puck_mask = np.sqrt((adata.obs["ycoord"] - puck_center[1])**2 + (adata.obs["xcoord"] - puck_center[0])**2) < puck_radius
adata = adata[puck_mask,:]

# Visualize cell positions in puck
plot_cell_positions(adata,
            raw_x,
            raw_y,
            coord_keys=coord_keys,
            fig_size = (6,6),
            add_circle=True,
            puck_center=puck_center,
            puck_radius=puck_radius)

# %%
from banksy_utils.filter_utils import normalize_total, filter_hvg
# Normalizes the anndata dataset
adata = normalize_total(adata)

# %%
adata, adata_allgenes = filter_hvg(adata,
            n_top_genes = 2000,
            flavor="seurat")

# %%
from banksy.main import median_dist_to_nearest_neighbour

# set params
# ==========
plot_graph_weights = True
k_geom = 15 # only for fixed type
max_m = 1 # azumithal transform up to kth order
nbr_weight_decay = "scaled_gaussian" # can also be "reciprocal", "uniform" or "ranked"

# Find median distance to closest neighbours
nbrs = median_dist_to_nearest_neighbour(adata, key = coord_keys[2])

# %%
from banksy.initialize_banksy import initialize_banksy

banksy_dict = initialize_banksy(adata,
                coord_keys,
                k_geom,
                nbr_weight_decay = nbr_weight_decay,
                max_m = max_m,
                plt_edge_hist = True,
                plt_nbr_weights = True,
                plt_agf_angles = False 
                )


# %%
from banksy.main import concatenate_all
from banksy.embed_banksy import generate_banksy_matrix

# The following are the main hyperparameters for BANKSY
resolutions = [0.7,] # clustering resolution for UMAP
pca_dims = [20] # Dimensionality in which PCA reduces to
lambda_list = [0.2,] # list of lambda parameters

banksy_dict, banksy_matrix = generate_banksy_matrix(adata,
                                                    banksy_dict,
                                                    lambda_list,
                                                    max_m)

# %%
banksy_dict["nonspatial"] = {
    # Here we simply append the nonspatial matrix (adata.X) to obtain the nonspatial clustering results
    0.0: {"adata": concatenate_all([adata.X], 0, adata=adata), }
}

print(banksy_dict['nonspatial'][0.0]['adata'])

# %%
from banksy_utils.umap_pca import pca_umap

pca_umap(banksy_dict,
         pca_dims=pca_dims,
         add_umap=True)

# Note that by default, we recommend resolution-based clustering (i.e., `leiden` or `louvain`) if no prior information on the number of clusters known. However, if the number of clusters is known *a priori*, the user can use `mclust` (gaussian-mixture model) by specifying the number of clusters beforehand.

# %%
from banksy.cluster_methods import run_Leiden_partition

results_df, max_num_labels = run_Leiden_partition(
    banksy_dict,
    resolutions,
    num_nn = 50,
    num_iterations = -1,
    partition_seed= 1234,
    match_labels = True,
)

### Visualize the clustering results from BANKSY, including the clusters from the Umap embbedings

# %%
from banksy.plot_banksy import plot_results

c_map =  'tab20' # specify color map
weights_graph =  banksy_dict['scaled_gaussian']['weights'][0]

plot_results(
    results_df,
    weights_graph,
    c_map,
    match_labels = True,
    coord_keys = coord_keys,
    max_num_labels  =  max_num_labels, 
    save_path = os.path.join(file_path, 'tmp_png'),
    save_fig = False, # Save Spatial Plot Only
    save_fullfig = True # Save Full Plot
)
