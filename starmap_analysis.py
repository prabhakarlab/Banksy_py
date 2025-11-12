#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore") 
import os, time
import pandas as pd
import numpy as np
import random
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, issparse

from banksy.initialize_banksy import initialize_banksy
from banksy.run_banksy import run_banksy_multiparam
from banksy_utils.color_lists import spagcn_color

start = time.perf_counter_ns()
random_seed = 1234
cluster_algorithm = 'leiden'
np.random.seed(random_seed)
random.seed(random_seed)


# ## Input-Ouput (IO) Options
# 1. Loading '.h5ad' file
# 2. Saving output images and '.csv' files in 'output_folder'

# In[2]:


from banksy_utils.load_data import load_adata

'''Main tunable variation in running the BANKSY algorithm'''

'''Input File'''
file_path = os.path.join("data", "starmap")
adata_filename = "starmap_BY3_1k.h5ad"

gcm_filename = ""
locations_filename = ""

load_adata_directly = True

'''Saving output figures'''
save_fig = False

output_folder = os.path.join(os.getcwd(), 'data', 'starmap', 'tmp_png', f'{cluster_algorithm}', f'seed{random_seed}')

# Colour map
c_map = 'tab20'

coord_keys = ('x', 'y', 'spatial')
num_clusters = 7
sample = 'starmap'

raw_y, raw_x, adata = load_adata(file_path,
                                 load_adata_directly,
                                 adata_filename,
                                 gcm_filename,
                                 locations_filename,
                                 coord_keys)

adata.var_names_make_unique()


# ## Initial Round of filtering by cell-count and gene counts (Skipped)
# 1. Define 'min_count' and 'max_count' for filtering genes by cell-count
# 2. 'MT-filter' for filtering MT-genes
# 3. 'n_top_genes' for selecting the top n HVGs or SVGs
# 
# Note that: For STARmap, we skip this step as the data matrix loaded has already been preprocessed using the ipython script provided by the authors

# ### Load Manual annotations from datafile

# In[3]:


'''Load manual annotations'''
adata = adata[adata.obs["cluster_name"].notnull()]

annotations =  pd.read_csv(os.path.join(file_path, "Starmap_BY3_1k_meta_annotated_18oct22.csv"))

manual_labels =  "smoothed_manual" # Ground truth annotations in pd.DataFrame
annotation_key = 'manual_annotations' # Key to access annotations in adata.obs[annotation_keys]

print(annotations.loc[:,manual_labels ])
adata.obs[annotation_key] = annotations.loc[:,manual_labels ].values
adata.obs[annotation_key] = adata.obs[annotation_key].astype('category')
print(adata.obs[annotation_key])

# Add spatial coordinates to '.obsm' attribute
adata.obsm[coord_keys[2]] = pd.concat([adata.obs[coord_keys[0]], adata.obs[coord_keys[1]]], axis=1).to_numpy()


# # Specifying parameters for BANKSY

# In[4]:


resolutions = [0.8] # clustering resolution for Leiden clustering

pca_dims = [20] # number of dimensions to keep after PCA

lambda_list = [.8] # lambda

k_geom = 15 #spatial neighbours

max_m = 1 # use AGF

nbr_weight_decay = "scaled_gaussian" # can also be "reciprocal", "uniform" or "ranked"


# # Initialize Banksy Object

# In[5]:


banksy_dict = initialize_banksy(
    adata,
    coord_keys,
    k_geom,
    nbr_weight_decay=nbr_weight_decay,
    max_m=max_m,
    plt_edge_hist=True,
    plt_nbr_weights=True,
    plt_agf_angles=False,
    plt_theta=False,
)


# # Run BANKSY using defined parameters

# In[6]:


results_df = run_banksy_multiparam(
    adata,
    banksy_dict,
    lambda_list,
    resolutions,
    color_list = spagcn_color,
    max_m = max_m,
    filepath = output_folder,
    key = coord_keys,
    pca_dims = pca_dims,
    annotation_key = annotation_key,
    max_labels = num_clusters,
    cluster_algorithm = cluster_algorithm,
    match_labels = False,
    savefig = False,
    add_nonspatial = False,
    variance_balance = False,
)


# In[7]:


results_df


# In[8]:


from banksy_utils.refine_clusters import refine_clusters

results_df = refine_clusters(adata,
                             results_df,
                             coord_keys=coord_keys,
                             color_list=spagcn_color,
                             savefig=True,
                             output_folder=output_folder,
                             refine_method='once',
                             annotation_key=annotation_key,
                             num_neigh=6)

run_time = (time.perf_counter_ns() - start) * 1e-9
print(f"BANKSY runtime = {round(run_time / 60, 3)} mins")
results_df.to_csv(os.path.join(output_folder, 'results.csv'))


# #Modify banksy parameters

# #Refined parameters 

# In[9]:


resolutions = [.9] # clustering resolution for Leiden clustering

pca_dims = [20] # number of dimensions to keep after PCA

lambda_list = [.8] # lambda

k_geom = 8 #spatial neighbours

max_m = 1 # use AGF

nbr_weight_decay = "scaled_gaussian" # can also be "reciprocal", "uniform" or "ranked"


# In[10]:


banksy_dict_modi = initialize_banksy(
    adata,
    coord_keys,
    k_geom,
    nbr_weight_decay=nbr_weight_decay,
    max_m=max_m,
    plt_edge_hist=True,
    plt_nbr_weights=True,
    plt_agf_angles=False,
    plt_theta=False,
)


# In[11]:


results_df_modi = run_banksy_multiparam(
    adata,
    banksy_dict_modi,
    lambda_list,
    resolutions,
    color_list = spagcn_color,
    max_m = max_m,
    filepath = output_folder,
    key = coord_keys,
    pca_dims = pca_dims,
    annotation_key = annotation_key,
    max_labels = num_clusters,
    cluster_algorithm = cluster_algorithm,
    match_labels = False,
    savefig = False,
    add_nonspatial = False,
    variance_balance = False,
)


# In[12]:


results_df_modi


# In[13]:


results_df_modi = refine_clusters(adata,
                             results_df_modi,
                             coord_keys=coord_keys,
                             color_list=spagcn_color,
                             savefig=True,
                             output_folder=output_folder,
                             refine_method='once',
                             annotation_key=annotation_key,
                             num_neigh=6)

run_time = (time.perf_counter_ns() - start) * 1e-9
print(f"BANKSY runtime = {round(run_time / 60, 3)} mins")
results_df.to_csv(os.path.join(output_folder, 'results.csv'))


# In[ ]:




