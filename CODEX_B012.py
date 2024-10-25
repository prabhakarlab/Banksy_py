# %%
# This is the simplified python script that demonstrates how to directly run slide-seq v1 dataset with BANKSY
import os, time, random, gc
import anndata
from anndata import AnnData
import numpy as np
import pandas as pd
import warnings
from banksy_utils.color_lists import spagcn_color
warnings.filterwarnings("ignore")
import scanpy as sc
sc.logging.print_header()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 1  # errors (0), warnings (1), info (2), hints (3)

seed = 0
np.random.seed(seed)
random.seed(seed)
start = time.perf_counter_ns()

# %% [markdown]
# ### Analysis for CODEX
# Dataset is publically available here [CODEX correct link](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE234713). 
# Download the file `GSM7423_09_CODEX_HuBMAP_alldata_Dryad_merged.csv` (a healthy colon HC sample) and arrange it under `data/CODEX/23_09_CODEX_HuBMAP_alldata_Dryad_merged`

# %%
file_path = os.path.join("data", "CODEX")
metadata_file = "23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv"
meta_df = pd.read_csv(os.path.join(file_path, metadata_file), index_col=0)

# %% [markdown]
# `CODEX` dataset: This dataset contains samples from all donors. Here, we focus on identifying tissue segments from the transverse segment for donor B008. This can be accessed via `meta_df['unique_region'] == B008_Trans`

# %%
# See all unique regions availale for clustering
meta_df['unique_region'].unique()

# %%
unique_regions = ["B012_Ileum", "B012_Right", "B012_Trans"]
meta_df = meta_df.loc[meta_df['unique_region'].isin(unique_regions)]
meta_df

# %% [markdown]
# ### Create AnnData object
# The dataset can be split into the `cell-expression matrix` (indexed from columns 1 to 47) and rest of the metadata (indexed from columns 47 onwards) this also including the position of cells under `XCorr` and `YCorr`.

# %%
adata = AnnData(X = np.array(meta_df.iloc[:, :47]), obs = meta_df.iloc[:, 47:])
adata.var_names_make_unique()
del meta_df
gc.collect()

# %% [markdown]
# ### Pre-process the loaded data
# 1. Normalize the expression by cell count using `sc.pp.normalize_total`
# 2. Add the position array under `adata.obsm['coord_xy']`

# %%
# Normalize
sc.pp.normalize_total(adata, inplace=True)
# Stack coordinate array
coord_keys = ('Xcorr', 'Ycorr', 'coord_xy')
adata.obsm[coord_keys[2]] = np.vstack((adata.obs[coord_keys[0]].values,adata.obs[coord_keys[1]].values)).T

adata.obs

# %% [markdown]
# ### Preliminary Analysis
# - Due to the size of this dataset, each sample is divided into 20 FOVs. We first visualize the spatial distribution of all 20 FOVs.
# - Here we can directly cluster the entire dataset (without performing batch clustering).

# %%
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
cmap = plt.get_cmap('tab20')
sc.pl.scatter(adata, x='Xcorr', y='Ycorr', color='Tissue Segment',  color_map=cmap, title=f"Tissue segments in region {unique_regions}", show = False,size = 5)
plt.show()

# %% [markdown]
# ## Running BANKSY
# Finally we run BANKSY using the following (default) parameters for `cell-typing`:
# - $k_{geom} = 15 $
# -  $\lambda = 0.2$
# - m = 1 (first order azimuthal transform)
# </br>
# 
# From, the BANKSY embeddings, we then
# - Run PCA with 20 PCs
# - Perform Leiden clustering with a resolution parameter of 2.0
# <br>

# %%
from banksy.main import median_dist_to_nearest_neighbour
from banksy.initialize_banksy import initialize_banksy
from banksy.embed_banksy import generate_banksy_matrix
from banksy.run_banksy import run_banksy_multiparam
from banksy.main import concatenate_all

k_geom = 15  # only for fixed type
max_m = 1  # azumithal transform up to kth order
nbr_weight_decay = "scaled_gaussian"  # can also be "reciprocal", "uniform" or "ranked"
resolutions = None  # clustering resolution for leiden algorithm
max_labels = 3 # Number of clusters for tissue segmentation
pca_dims = [20]  # Dimensionality in which PCA reduces to
lambda_list = [0.8]

# %% [markdown]
# ### Initalize the weighted neighbourhood graphs for BANKSY 

# %%
nbrs = median_dist_to_nearest_neighbour(adata, key=coord_keys[2])
banksy_dict = initialize_banksy(adata,
                                coord_keys,
                                k_geom,
                                nbr_weight_decay=nbr_weight_decay,
                                max_m=max_m,
                                plt_edge_hist=False,
                                plt_nbr_weights=True,
                                plt_agf_angles=False,
                                plt_theta=False
                                )


# %% [markdown]
# ## Create BANKSY Matrix

# %%
banksy_dict, banksy_matrix = generate_banksy_matrix(adata,
                                                    banksy_dict,
                                                    lambda_list,
                                                    max_m)

# Create output folder path if not done so
output_folder = os.path.join(file_path, 'BANKSY-Results')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Add nonspatial clustering for comparsion
banksy_dict["nonspatial"] = {
    # Here we simply append the nonspatial matrix (adata.X) to obtain the nonspatial clustering results
    0.0: {"adata": concatenate_all([adata.X], 0, adata=adata), }
}

# %% [markdown]
# ### Perform Dimensionality Reduction
# 1. Perform PCA to produce PCA embeddings for Leiden Clustering
# 2. UMAP (just for visualization)

# %%
from banksy_utils.umap_pca import pca_umap

pca_umap(banksy_dict,
         pca_dims = pca_dims,
         add_umap = True
         )

# %% [markdown]
# ## Run Leiden Parition and plot the results

# %%
from banksy.cluster_methods import run_Leiden_partition

results_df, max_num_labels = run_Leiden_partition(
    banksy_dict,
    resolutions,
    num_nn = 50,
    num_iterations = -1,
    partition_seed = 1234,
    match_labels = True,
    max_labels = max_labels,
)


from banksy.plot_banksy import plot_results

c_map =  'tab20' # specify color map
weights_graph =  banksy_dict['scaled_gaussian']['weights'][1]

plot_results(
    results_df,
    weights_graph,
    c_map,
    match_labels = True,
    coord_keys = coord_keys,
    max_num_labels  =  max_num_labels, 
    save_path = os.path.join(file_path, 'BANKSY-Results'),
    save_fig = False, # Save Spatial Plot Only
    save_fullfig = True, # Save Full Plot
    dataset_name = f"CODEX-{unique_regions}",
    save_labels=True
)
results_df.to_csv(os.path.join(file_path, "Summary.csv"))


