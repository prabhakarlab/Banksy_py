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


# ### `CODEX` dataset Analysis 
# This dataset contains samples from all donors. Here, we focus on identifying tissue segments from the transverse segment for donor B008. This can be accessed via `meta_df['unique_region'] == B006_Ascending`

# In[3]:


# See all unique regions availale for clustering
meta_df['unique_region'].unique()


# #### Access `ascending` region for patient B006

# In[4]:


unique_regions = ["B006_Ascending"]
meta_df = meta_df.loc[meta_df['unique_region'].isin(unique_regions)]
meta_df


# ### Create AnnData object
# The dataset can be split into the `cell-expression matrix` (indexed from columns 1 to 47) and rest of the metadata (indexed from columns 47 onwards) this also including the position of cells under `XCorr` and `YCorr`.

# In[5]:


adata = AnnData(X = np.array(meta_df.iloc[:, :47]), obs = meta_df.iloc[:, 47:])
adata.var_names_make_unique()
# Just to save RAM
del meta_df
gc.collect()


# ### Having a feel for the `adata` object
# Add the position array under `adata.obsm['coord_xy']`

# In[6]:


# Stack coordinate array
coord_keys = ('Xcorr', 'Ycorr', 'coord_xy')
adata.obsm[coord_keys[2]] = np.vstack((adata.obs[coord_keys[0]].values,adata.obs[coord_keys[1]].values)).T

# See the adata.obs for this sample
adata.obs


# In[7]:


adata


# In[8]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
cmap = plt.get_cmap('tab20')
sc.pl.scatter(adata, x='Xcorr', y='Ycorr', color='Community',  color_map=cmap, title=f"Tissue segments in region {unique_regions}", size = 5)#show = False, size = 5)
plt.show()


# ## Running BANKSY
# As a demonstration to identify the communities, we run BANKSY using the following (default) parameters for `domain segmentation`:
# - $k_{geom} = 15 $
# -  $\lambda = 0.8$
# - m = 1 (first order azimuthal transform)
# </br>
# 
# From, the BANKSY embeddings, we then
# - Run PCA with 20 PCs
# - Perform Leiden clustering with a resolution parameter of 2.0
# <br>

# In[9]:


from banksy.main import median_dist_to_nearest_neighbour
from banksy.initialize_banksy import initialize_banksy
from banksy.embed_banksy import generate_banksy_matrix
from banksy.main import concatenate_all

k_geom = 15  # only for fixed type
max_m = 1  # azumithal transform up to kth order
nbr_weight_decay = "scaled_gaussian"  # can also be "reciprocal", "uniform" or "ranked"
resolutions = None  # clustering resolution for leiden algorithm
max_labels = 8 # Number of clusters for tissue segmentation
pca_dims = [20]  # Dimensionality in which PCA reduces to
lambda_list = [0.8]


# ### Initalize the weighted neighbourhood graphs for BANKSY 

# In[10]:


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

# In[11]:


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

# In[12]:


from banksy_utils.umap_pca import pca_umap

pca_umap(banksy_dict,
         pca_dims = pca_dims,
         add_umap = True
         )


# ## Run Leiden Parition and plot the results

# In[13]:


from banksy.cluster_methods import run_Leiden_partition

banksy_df, max_num_labels = run_Leiden_partition(
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
    banksy_df,
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
banksy_df.to_csv(os.path.join(file_path, f"CODEX-{unique_regions}_BANKSY.csv"))


# 

# ## Perform nonspatial clustering (for comparsion)

# In[14]:


# Add nonspatial clustering
nonspatial_dict = {"nonspatial" : {0.0: {"adata": concatenate_all([adata.X], 0, adata=adata), } } }

pca_umap(nonspatial_dict, pca_dims = pca_dims, add_umap = True )

from banksy.cluster_methods import run_Leiden_partition

nonspatial_df, max_num_labels = run_Leiden_partition(
    nonspatial_dict,
    resolutions,
    num_nn = 50,
    num_iterations = -1,
    partition_seed = 1234,
    match_labels = True,
    max_labels = max_labels,
)

from banksy.plot_banksy import plot_results

c_map =  'tab20' # specify color map

plot_results(
    nonspatial_df,
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


# ## Compare similarity between BANKSY clusters and CODEX communities

# In[15]:


from sklearn.metrics import adjusted_rand_score as ari, adjusted_mutual_info_score as ami
from sklearn.metrics import matthews_corrcoef as mcc
# See the visualize the communities that we want to detect
adata.obs['Community']


# ### Access clustering results from BANKSY in domain segmentation mode using `banksy_df.labels[{label_index}]`

# In[16]:


# We can access the clusters from banksy using the following command
banksy_spatial_clusters = banksy_df.labels[banksy_df.index[0]]
banksy_spatial_clusters.dense


# ### Access clustering results from nonspatial clustering in a similar way

# In[17]:


# We can access the clusters from banksy using the following command
nonspatial_clusters = nonspatial_df.labels[nonspatial_df.index[0]]
nonspatial_clusters.dense


# ### Compare the similarity between clusters from BANKSY vs nonspatial clusters and the annotated communities

# In[18]:


def calculate_metrics(cluster_labels, annotated_labels):
    # A custom function to calcualte all metrics
    ari_score  = ari(cluster_labels, annotated_labels)
    ami_score =   ami(cluster_labels, annotated_labels)

    if isinstance(annotated_labels.dtype, pd.CategoricalDtype):
        print("Converting annotations to required 'int' type for computing MCC")
        annotated_labels = annotated_labels.cat.codes

    mcc_score =  mcc(cluster_labels,annotated_labels )
    return ari_score, ami_score, mcc_score

nonspatial_ari, nonspatial_ami, nonspatial_mcc = calculate_metrics(nonspatial_clusters.dense, adata.obs['Community'])


# ### Calculate the similarity between BANKSY labels and annotated communities

# In[19]:


banksy_ari, banksy_ami, banksy_mcc = calculate_metrics(banksy_spatial_clusters.dense, adata.obs['Community'])


# ### Plot bar chart comparing their similarities

# In[20]:


def bar_plot(metrics, methods):
    ''' Custom function to generate bar chart comparing metrices of labels produced by different methods'''
    fig, ax = plt.subplots(figsize=(8,8),layout='constrained')
    x = np.arange(len(methods))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    # method is banksy, nonspatial,
    # metric is ari, ami, mcc
    for method, metric  in metrics.items():
        offset = width * multiplier
        print(metric)
        rects = ax.bar(x + offset, metric, width, label=method)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Metrices', fontsize=18)
    ax.set_title("Similarity between BANKSY labels and annotated communities", fontsize=20)
    ax.set_xticks(x + width, methods, fontsize=16)
    ax.legend(loc='upper left', fontsize=18)
    fig.show()

### Plot the similarity between BANKSY labels and annotated communities
methods = ('Non-spatial labels', 'BANKSY labels')
metrics = {
    'Adjusted Rand Index' : (nonspatial_ari, banksy_ari),
    'Adjusted Mutual Information': (nonspatial_ami, banksy_ami),
    'Matthew Correlation Coefficient': (nonspatial_mcc,banksy_mcc ),
}
bar_plot(metrics, methods)


# In[ ]:




