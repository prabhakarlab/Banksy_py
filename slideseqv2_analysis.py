#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, re
import numpy as np
import pandas as pd
from IPython.display import display
import warnings
warnings.filterwarnings("ignore") 

import scipy.sparse as sparse
from scipy.io import mmread
from scipy.stats import pearsonr, pointbiserialr

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import seaborn as sns
import scanpy as sc
sc.logging.print_header()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 1 # errors (0), warnings (1), info (2), hints (3)
plt.rcParams["font.family"] = "Arial"
sns.set_style("white")

import random
# Note that BANKSY itself is deterministic, here the seeds affect the umap clusters and leiden partition
seed = 1234
np.random.seed(seed)
random.seed(seed)


# ## Load Slide-seq Dataset
# Allocate path to folder and the file name of the designated AnnaData Object (in `.h5ad` format) <br>

# In[2]:


# Define File paths
file_path = os.path.join("data", "slide_seq", "v2")
gcm_filename = "Cerebellum_MappedDGEForR.csv"

# (Optional) Arguments for load_data only if annadata is not present
locations_filename = "slideseqv2_mc_coords_filtered.csv"
adata_filename = "slideseqv2_cerebellum_adataraw.h5ad"


# ### Loading anndata file or converting raw files to anndata format  (h5ad)
# 
# If the specified anndata file is present in the directory, the annadata file `.h5ad` in the file path can be read directly by setting the `load_adata_directly` to `True` <br>
# Otherwise, `load_data` can convert the raw data of GCM and locations in `.csv` format to `.h5ad` directly if Annadata file is not present

# In[3]:


from banksy_utils.load_data import load_adata, display_adata

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


# ### Display Anndata Object
# Use `display_adata` to show summary of anndata object

# In[4]:


display_adata(adata)


# ### Preprocess AnnData Object 
# We then preprocess the raw adata file with `preprocess_data`, which calculates the QC metrics using `scanpy`'s built-in function `sc.pp.calculate_qc_metrics` and adding them in-place in the anndata object

# In[5]:


adata.var_names_make_unique()
adata.var["mt"] = adata.var_names.str.startswith("MT-")

# Calulates QC metrics and put them in place to the adata object
sc.pp.calculate_qc_metrics(adata, 
                           qc_vars=["mt"], 
                           log1p=True, 
                           inplace=True)


# ### Display the preprocessed adata with added qc_metrics
# 
# Now the anndata contains qc_metric under its variables (`adata.var`)

# In[6]:


display(adata.obs, adata.var)


# ## Visualize the histogram plot of `total_counts` and `n_genes`
# We can visualize the unfiltered and filtered histograms for `total_counts` and `n_genes` variables in the adata object <br> 
# 
# The filtered histograms are plotted after filtering by the upper threshold of total counts `total_counts_cuttoff`, the upper threshold of n_genes `n_genes_high_cutoff`, and the lower threshold of n_genes

# In[7]:


from banksy_utils.plot_utils import plot_qc_hist, plot_cell_positions

# bin options for fomratting histograms
# Here, we set 'auto' for 1st figure, 80 bins for 2nd figure. and so on
hist_bin_options = ['auto', 80, 80, 100]

plot_qc_hist(adata, 
         total_counts_cutoff=150, 
         n_genes_high_cutoff=2500, 
         n_genes_low_cutoff=800,
         bin_options = hist_bin_options)


# ## Do an initial filter of cells by their cell counts, MT count and gene counts
# We can then first filter cells by their counts, MT and gene filters.

# In[8]:


from banksy_utils.filter_utils import filter_cells

# Filter cells with each respective filters
adata = filter_cells(adata, 
             min_count=50, 
             max_count=2500, 
             MT_filter=20, 
             gene_filter=10)


# Show summary of adata after filtering by cell counts

# In[9]:


display_adata(adata)


# ### Plot the Histograms of the filtered dataset

# In[10]:


hist_bin_options = ['auto', 100, 60, 100]

plot_qc_hist(adata,
        total_counts_cutoff=2000,
        n_genes_high_cutoff=2000, 
        n_genes_low_cutoff= 0,
        bin_options = hist_bin_options)


# ### Visualize the position of cells using a scatter plot
# For slideseq_v2, we customize the plot with a puck_center 

# In[11]:


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


# ## Normalize & Filter highly variable genes (HVG)
# 
# ### Normalize data matrix using `normalize_total`
# 

# In[12]:


from banksy_utils.filter_utils import normalize_total, filter_hvg, print_max_min

import anndata
# Normalizes the anndata dataset
adata = normalize_total(adata)


# ### Filter by Highly Variable Genes (HVG) using a specified threshold
# 
# We use the **raw count data** (flavor = `"seurat_v3"`) or **log-transformed data** (flavor = `"seurat"`). Here, we recommend using the default `seurat` flavor from `Scanpy`.
# 
# Here we keep the original AnnData object (`adata_allgenes`) for the analysis of the results produced by BANKSY if nessecary. However, we will mostly be running banksy with the filtered AnnData object (`adata`)

# In[13]:


adata, adata_allgenes = filter_hvg(
    adata,
    n_top_genes=2000,
    flavor="seurat"
)


# ### Check genes and cells annotations
# 
# Display attribute of the filtered data after filtering steps and HVG analysis

# In[14]:


display_adata(adata)


# ## Generate spatial weights graph
# 
# #### In BANKSY, we imagine edges / connections in the graph to represent neighbour relationships between cells
# 
# In doing so, the banksy algorithm requires the following specifications in the main BANKSY algorithm:
# 
# 1. The number of spatial neighbours `num_neighbours`, this is also known as the $k_{geom}$ parameter in the manuscript.
#     
# 2. Assigning weights (dependent on inter-cell spatial distances) to edges in the connected spatial graph. By default, we use the `gaussian decay` option, where weights decay as a function of distnace to the index cell with $\sigma$ = `sigma`. As default, we set $\sigma$ to be the median distance between cells, and do not prescribe any cutoff-radius `p_outside` (no cutoff is conducted).
# 
# 3. The Azumithal Gabor Filter parameter `max_m` which indicates whether to use the AGF (`max_m = 1`) or just the mean expression (`max_m = 0`). By default, we set `max_m = 1`.
#     
#     
# ### Construction of the spatial $k_{geom}$ Nearest-Neighbour graph
# 
# We represent connections between cells and its neighbours in a graph $G = \{N,E,W\}$, comprising of a set of nodes $n \in N$. edges representing connectivity between cells $e \in E$, the edges are be weighted $w \in W$ as a function of the spatial distance between cells.
#     
# Weight of edges can be represented by uniform distance (i.e., a closer neighbour will have a higher weight), or using `reciprocal` ($\frac{1}{r})$. As mentioned above, BANKSY by default applies a gaussian envelope function to map the distance (between nodes) to the `weights` the connection of cell-to-neighbor

# In[15]:


from banksy.main import median_dist_to_nearest_neighbour

# set params
# ==========
plot_graph_weights = True
k_geom = 15 # number of spatial neighbours
max_m = 1 # use both mean and AFT
nbr_weight_decay = "scaled_gaussian" # can also choose "reciprocal", "uniform" or "ranked"

# Find median distance to closest neighbours
nbrs = median_dist_to_nearest_neighbour(adata, key = coord_keys[2])


# ### Generate spatial weights from distance
# 
# Here, we generate the spatial weights using the gaussian decay function from the median distance to the k-th nearest neighbours as specified earlier.
# 
# The utility functions `plot_edge_histograms`, `plot_weights`, `plot_theta_graph` can be used to visualize the characteristics of the edges, weights and theta (from the AGF) respectively.
# 
# #### Optional Visualization of Weights and edge graphs
# 
# (1) **Visualize the edge histogram** to show the histogram of distances between cells and the weights between cells by setting `plt_edge_hist = True`
# 
# (2) **Visualize weights** by plotting the connections. Line thickness is proportional to edge weight (normalized by highest weight across all edges) by setting `plt_weights = True`
# 
# (3) **Visualize weights with Azimuthal angles**. Visualizing the azimuthal angles computed by the AGF by colour, the azimuthal connectivities are annotated in red. Warning: this plot many take a few minutes to compute for large datasets. `plt_agf_angles = False`
# 
# (4) **Visualize angles around random cell**. Plot points around a random index cell, annotated with angles from the index cell. `plot_theta = True`

# In[16]:


from banksy.initialize_banksy import initialize_banksy

banksy_dict = initialize_banksy(
    adata,
    coord_keys,
    k_geom,
    nbr_weight_decay=nbr_weight_decay,
    max_m=max_m,
    plt_edge_hist=True,
    plt_nbr_weights=True,
    plt_agf_angles=False, # takes long time to plot
    plt_theta=True,
)


# ## Generate BANKSY matrix
# 
# To generate the BANKSY matrix, we proceed with the following:
# 
# 1. Matrix multiply sparse CSR weights matrix with cell-gene matrix to get **neighbour matrix** and the **AGF matrix** if `max_m` > 1
# 2. Z-score both matrices along **genes**
# 3. Multiply each matrix by a weighting factor $\lambda$ (We refer to this parameter as `lambda` in our manuscript and code)
# 4. Concatenate the matrices along the genes dimension in the form -> `horizontal_concat(cell_mat, nbr_mat, agf_mat)`
# 
# Here, we save all the results in the dictionary (`banksy_dict`), which contains the results from the subsequent operations for BANKSY. 

# In[17]:


from banksy.embed_banksy import generate_banksy_matrix

# The following are the main hyperparameters for BANKSY
# -----------------------------------------------------
resolutions = [0.7]  # clustering resolution for UMAP
pca_dims = [20]  # Dimensionality in which PCA reduces to
lambda_list = [0.2]  # list of lambda parameters

banksy_dict, banksy_matrix = generate_banksy_matrix(adata,
                                                    banksy_dict,
                                                    lambda_list,
                                                    max_m)


# ### Append Non-spatial results to the `banksy_dict` for comparsion

# In[18]:


from banksy.main import concatenate_all

banksy_dict["nonspatial"] = {
    # Here we simply append the nonspatial matrix (adata.X) to obtain the nonspatial clustering results
    0.0: {"adata": concatenate_all([adata.X], 0, adata=adata), }
}

print(banksy_dict['nonspatial'][0.0]['adata'])


# ## Reduce dimensions of each data matrix
# We utilize two common methods for dimensionality reduction:
# 
# 1. PCA (using `scikit-learn`), we reduce the size of the matrix from $3*N_{genes}$ to `pca_dims`. As a default settings, we reduce to 20 dimensions.
# 2. UMAP (`UMAP` package), which we use to visualize expressions of clusters in the umap space (2-D space).

# In[19]:


from banksy_utils.umap_pca import pca_umap

pca_umap(banksy_dict,
         pca_dims = pca_dims,
         add_umap = True,
         plt_remaining_var = False,
         )


# ### Cluster cells using a partition algorithm
# 
# For the purpose of this dataset, we conduct cluster cells using the `leiden` algorithm (use `!pip install leidenalg` if package missing) partition methods. Other clustering algorithms include `louvain` (another resolution based clustering algorithm), or `mclust` (a clustering based on gaussian mixture model). 
# 
# Note that by default, we recommend resolution-based clustering (i.e., `leiden` or `louvain`) if no prior information on the number of clusters known. However, if the number of clusters is known *a priori*, the user can use `mclust` (gaussian-mixture model) by specifying the number of clusters beforehand.

# In[20]:


from banksy.cluster_methods import run_Leiden_partition

results_df, max_num_labels = run_Leiden_partition(
    banksy_dict,
    resolutions,
    num_nn = 50, # k_expr: number of neighbours in expression (BANKSY embedding or non-spatial) space
    num_iterations = -1, # run to convergenece
    partition_seed = seed,
    match_labels = True,
)


# ## Plot results
# 
# ### Visualize the clustering results from BANKSY, including the clusters from the Umap embbedings

# In[21]:


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
    save_fig = True
)


# In[ ]:





# In[ ]:




