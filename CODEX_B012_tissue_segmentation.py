#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# ### Analysis for CODEX
# - Dataset is publically available here [CODEX dataset](https://datadryad.org/stash/dataset/doi:10.5061/dryad.pk0p2ngrf). 
# - Download the file `GSM7423_09_CODEX_HuBMAP_alldata_Dryad_merged.csv` (a healthy colon HC sample).
# - Arrange it under `data/CODEX/23_09_CODEX_HuBMAP_alldata_Dryad_merged`

# In[2]:


file_path = os.path.join("data", "CODEX","CODEX_csv_data")
metadata_file = "23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv"
meta_df = pd.read_csv(os.path.join(file_path, metadata_file), index_col=0)


# `CODEX` dataset: This dataset contains samples from all donors. Here, we focus on identifying tissue segments from the transverse segment for donor B008. This can be accessed via `meta_df['unique_region'] == B008_Trans`

# In[3]:


# See all unique regions availale for clustering
meta_df['unique_region'].unique()


# In[4]:


unique_regions = ["B012_Ileum", "B012_Right", "B012_Trans"]
meta_df = meta_df.loc[meta_df['unique_region'].isin(unique_regions)]
meta_df


# ### Create AnnData object
# - The dataset can be split into the `cell-expression matrix` (indexed from columns 1 to 47) and rest of the metadata (indexed from columns 47 onwards) this also including the position of cells under `XCorr` and `YCorr`.
# - Here, we will focus on the `Right` region

# In[5]:


right_df =  meta_df.loc[meta_df['unique_region'] == 'B012_Right']
adata = AnnData(X = np.array(right_df.iloc[:, :47]), obs = right_df.iloc[:, 47:])
adata.var_names_make_unique()

gc.collect()


# ### Pre-process the loaded data
# 1. Add the position array under `adata.obsm['coord_xy']`

# In[6]:


# Stack coordinate array
coord_keys = ('Xcorr', 'Ycorr', 'coord_xy')
adata.obsm[coord_keys[2]] = np.vstack((adata.obs[coord_keys[0]].values,adata.obs[coord_keys[1]].values)).T

adata.obs


# ### Preliminary Analysis
# - In CODEX, we want to visualize the communities (8 different classes) in the `B006_ascending` tissue segment
# - We first visualize the spatial distribution of the 8 different `communities`.

# In[7]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
cmap = plt.get_cmap('tab20')
sc.pl.scatter(adata, x='Xcorr', y='Ycorr', color='Tissue Segment',  color_map=cmap, title=f"Tissue segments in region {unique_regions}", show = False,size = 5)
plt.show()


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

# In[8]:


from banksy.main import median_dist_to_nearest_neighbour
from banksy.initialize_banksy import initialize_banksy
from banksy.embed_banksy import generate_banksy_matrix
from banksy.main import concatenate_all

k_geom = 15  # only for fixed type
max_m = 1  # azumithal transform up to kth order
nbr_weight_decay = "scaled_gaussian"  # can also be "reciprocal", "uniform" or "ranked"
resolutions = list(np.arange(0.08, 0.1, 0.005))#[0.08]  # clustering resolution for leiden algorithm
max_labels = None # Number of clusters for tissue segmentation
pca_dims = [20]  # Dimensionality in which PCA reduces to
lambda_list = [0.8]


# ### Initalize the weighted neighbourhood graphs for BANKSY 

# In[9]:


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


# ## Create BANKSY Matrix

# In[10]:


banksy_dict, banksy_matrix = generate_banksy_matrix(adata,
                                                    banksy_dict,
                                                    lambda_list,
                                                    max_m)

# Create output folder path if not done so
output_folder = os.path.join(file_path, 'BANKSY-Results')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# ### Perform Dimensionality Reduction
# 1. Perform PCA to produce PCA embeddings for Leiden Clustering
# 2. UMAP (just for visualization)

# In[11]:


from banksy_utils.umap_pca import pca_umap

pca_umap(banksy_dict,
         pca_dims = pca_dims,
         add_umap = True
         )


# ## Run Leiden Parition and plot the results

# In[12]:


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




# In[13]:


results_df


# In[14]:


results_df[results_df['num_labels']==3]


# In[15]:


from banksy.plot_banksy import plot_results

c_map =  'tab20' # specify color map
weights_graph =  banksy_dict['scaled_gaussian']['weights'][1]

plot_results(
    #results_df,
    results_df[results_df['num_labels']==3],
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


# In[16]:


# We can access the clusters from banksy using the following command
banksy_spatial_clusters = results_df.labels[results_df.index[0]]
banksy_spatial_clusters.dense


# ### BANKSY's labels show good agreement with annotated Neighbourhoods

# In[17]:


from sklearn.metrics import adjusted_rand_score as ari, adjusted_mutual_info_score as ami
from sklearn.metrics import matthews_corrcoef as mcc
def calculate_metrics(cluster_labels, annotated_labels, verbose=True):
    # A custom function to calcualte all metrics
    ari_score  = ari(cluster_labels, annotated_labels)
    ami_score =   ami(cluster_labels, annotated_labels)

    if isinstance(annotated_labels.dtype, pd.CategoricalDtype):
        print("Converting annotations to required 'int' type for computing MCC")
        annotated_labels = annotated_labels.cat.codes

    mcc_score =  mcc(cluster_labels,annotated_labels)
    if verbose:
        print("--- Summarizing metrics ---")
        print(f"ARI: {ari_score:.3f}")
        print(f"AMI: {ami_score:.3f}")
        print(f"MCC: {mcc_score:.3f}")
    return ari_score, ami_score, mcc_score

banksy_ari, banksy_ami, banksy_mcc = calculate_metrics(banksy_spatial_clusters.dense, adata.obs['Tissue Segment'])
#nonspatial_ari, nonspatial_ami, nonspatial_mcc = calculate_metrics(nonspatial_clusters.dense, adata.obs['Community'])


# In[ ]:




