import os, gc
import numpy as np
import anndata
from banksy_utils.slideseq_ref_data import markergenes_dict
import scanpy as sc
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score

def pad_clusters(clust2annotation: dict, 
                 original_clusters: list, 
                 pad_name: str = "other"):
    """
    Annotate clusters (in form of a dict) by assigning them annotations
    using clust2annotation dict. Modifies clust2annotation dict in place.

    converts to str if given integers as original clusters.
    
    For any clusters not defined in clust2annotation:
    - assign all with pad_name if str input given
    - keep original name if pad_name is None
    """
    for cluster in original_clusters:
        cluster = str(cluster)
        if cluster not in clust2annotation:
            if pad_name is not None:
                clust2annotation[cluster] = pad_name
            else:
                clust2annotation[cluster] = cluster

def refine_cell_types(adata_spatial,
                      adata_nonspatial,
                      cluster2annotation_refine: dict):
    '''
    Applies the pad_clusters and return the refined clusters 

    Returns refined adata_spatial, adata_nonspatial
    '''
    for adata in [adata_nonspatial, adata_spatial]:

        pad_clusters(cluster2annotation_refine, adata.obs["cell type"], pad_name=None)
        #print(f"New cluster dictionary: {cluster2annotation_refine}")

        adata.obs["cell type refined"] = adata.obs["cell type"].map(
            cluster2annotation_refine
        ).astype('category')
        
    # remove other category
    adata_spatial_filtered = adata_spatial[adata_spatial.obs["cell type refined"] != "other",:]
    adata_nonspatial_filtered = adata_nonspatial[adata_nonspatial.obs["cell type refined"] != "other",:]

    return adata_spatial_filtered, adata_nonspatial_filtered

def create_metagene_df(adata_allgenes: anndata.AnnData,
                       coord_keys: list,
                       markergenes_dict: dict = markergenes_dict):
    '''
    Note that this function works for Slideseq dataset by default
    creates a metagene dataframe from the markergenes dataset from dropviz.org

    can enter your own markergenes dictionary as an optional argument markergenes_dict
    in the form {"cell type": [list of genes...],...}
    '''
    print("Generating Metagene data...\n")

    keys = [coord_keys[0], coord_keys[1]]
    metagene_df = adata_allgenes.obs.loc[:,keys].copy()

    for layer in markergenes_dict:

        markers = [marker for marker in markergenes_dict[layer] 
                if marker in adata_allgenes.var.index]

        print(f"{len(markers)} DE markers for {layer}")

        if markers < markergenes_dict[layer]:
            print(f"{len(markergenes_dict[layer])-len(markers)} "
                f"scRNA-seq DE genes in {layer} absent/filtered from slideseq dataset")

        layer_slice = adata_allgenes[:, markers]

        metagene = np.array(np.mean(layer_slice.X, axis = 1))
        
        metagene_df[layer] = metagene

    return metagene_df

def get_DEgenes(adata, cell_type, top_n=20, verbose=True):
    DE_genes = sc.get.rank_genes_groups_df(adata, group=cell_type)["names"][:top_n].tolist()
    if verbose: print(f"Top {top_n} DE genes for {cell_type} : {DE_genes}\n")
    return DE_genes

def get_metagene_difference(adata, DE_genes1, DE_genes2, m=0):
    """
    Compute the differences in metagene and neighbour metagene
    espression for a particular cell type
    must have run scanpy's rank_genes_groups first
    """
    
    def get_metagene(DE_genes):

        genes_slice = adata[:, DE_genes]
        genes_slice_nbr = adata[:, [gene + f"_nbr_{m:1d}" for gene in DE_genes]]

        metagene = np.mean(genes_slice.X, axis = 1)
        metagene_nbr = np.mean(genes_slice_nbr.X, axis = 1)
        
        return metagene, metagene_nbr
    
    metagene1, metagene_nbr1 = get_metagene(DE_genes1)
    metagene2, metagene_nbr2 = get_metagene(DE_genes2)
    
    return metagene1-metagene2, metagene_nbr1-metagene_nbr2


def create_spatial_nonspatial_adata(results_df: pd.DataFrame,
                                    pca_dims,
                                    lambda_list, 
                                    resolutions,
                                    cluster2annotation_spatial,
                                    cluster2annotation_nonspatial):
    """
    Creates spatial and nonspatial anndata object from results_df
    """
    params_name = f"scaled_gaussian_pc{pca_dims[0]:2d}_nc{lambda_list[0]:0.2f}_r{resolutions[0]:0.2f}"
    print(params_name)
    adata_spatial = results_df.loc[params_name, "adata"]
    label_name = f"labels_{params_name}"
    adata_spatial.obs['cell type'] = adata_spatial.obs[label_name].map(
        cluster2annotation_spatial
    ).astype('category')

    params_name = f"nonspatial_pc{pca_dims[0]:2d}_nc0.00_r{resolutions[0]:0.2f}"
    print(params_name)
    adata_nonspatial = results_df.loc[params_name, "adata"]
    label_name = f"labels_{params_name}"
    adata_nonspatial.obs['cell type'] = adata_nonspatial.obs[label_name].map(
        cluster2annotation_nonspatial
    ).astype('category')

    return adata_spatial, adata_nonspatial

def calculate_ari(adata, manual:str, predicted:str):
    return adjusted_rand_score(
        adata.obs[manual].cat.codes, 
        adata.obs[predicted].cat.codes
    )