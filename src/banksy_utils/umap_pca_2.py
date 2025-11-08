"""
This script contains the function to carry out dimensionality reduction on the banksy matrices via PCA and UMAP

This has been modified to allow pca_umap to be run directly on single anndata objects instead of on the dictionary.
Use function pca_umap for dictionary format and pca_umap_adata for adata format.

needs to be tested further.

Yifei May 2023
modified by Nigel 7 Nov 2024
"""
from sklearn.decomposition import PCA
import umap
import anndata
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, issparse
from banksy_utils.pca import plot_remaining_variance
from typing import List, Union
import numpy as np

def pca_umap(banksy_dict: dict,
             pca_dims: List[int] = [20,],
             plt_remaining_var: bool = True,
             add_umap: bool = False,
             **kwargs) -> None:
    '''
    Applies pca_umap to all adata objects in the banksy_dict

    Args:
        banksy_dict (dict): The processing dictionary containing info about the banksy matrices
    
    Optional Arg:
        pca_dims (List of integers): A list of integers which the PCA will reduce to
    
    Variable Args (kwargs):
        figsize (tuple of integers): A tuple for adjusting figure size

    Returns: Plot of remaining variance 
    '''
    
    options = {
        'figsize': (8,3)
    }
    print(f'Current decay types: {list(banksy_dict.keys())}')

    for nbr_weight_decay in banksy_dict:
        
        for lambda_param in banksy_dict[nbr_weight_decay]:
            
            if isinstance(lambda_param, str):
                continue # skip weights matrices

            print(
                f"\nReducing dims of dataset in (Index = {nbr_weight_decay}, lambda = {lambda_param})\n"
                + "=" * 50 + "\n"
            )
            
            # Retrieve anndata object
            # -----------------------
            
            adata_temp = banksy_dict[nbr_weight_decay][lambda_param]["adata"]
            
            param_str = f"(decay type = {nbr_weight_decay}, lambda = {lambda_param})"
            
            pca_umap_adata(adata_temp, pca_dims, plt_remaining_var, add_umap, options=options, param_str=param_str)

def pca_umap_adata(adata: anndata.AnnData,
                   pca_dims: List[int] = [20,],
                   plt_remaining_var: bool = True,
                   add_umap: bool = False,
                   options: dict = {'figsize': (8,3)},
                   param_str: str = "",
                   **kwargs) -> None:
    '''
    PCA_UMAP first applies dimensionality reduction via PCA,
    then applies UMAP to cluster the groups

    Args:
        banksy_dict (dict): The processing dictionary containing info about the banksy matrices
    
    Optional Arg:
        pca_dims (List of integers): A list of integers which the PCA will reduce to
    
    Variable Args (kwargs):
        figsize (tuple of integers): A tuple for adjusting figure size

    Returns: Plot of remaining variance 
    '''
    
    X = adata.X.todense() if issparse(adata.X) else adata.X
    
    X[np.isnan(X)] = 0

    # Reduce dimensions by PCA and then UMAP
    # --------------------------------------
                     
    if isinstance(pca_dims, int):
        pca_dims = [pca_dims,]
      
    for pca_dim in pca_dims:

        if isinstance(pca_dim, int):
            # We manually specify the number of PCs
            print(f"Setting the total number of PC = {pca_dim}")
            pca = PCA(n_components=pca_dim)
            reduced = pca.fit_transform(X)
        
        elif isinstance(pca_dim, float):
            # Otherwise, we specify the number of cumulative variance that each PC should represent
            print(f"Setting the total cumulative variance of PCs = {pca_dim}")
            pca = PCA(n_components=pca_dim)
            reduced = pca.fit_transform(X)

        #print(f'Noise Variance of PCA: {pca.noise_variance_}')
        #print(f'Variance of each PCs: {pca.explained_variance_ratio_}')
        print(
            f"Original shape of matrix: {X.shape}"
            f"\nReduced shape of matrix: {reduced.shape}\n" + "-" * 60 
            + f"\nmin_value = {reduced.min()}, mean = {reduced.mean()}, max = {reduced.max()}\n"
        )

        adata.obsm[f"reduced_pc_{pca_dim} {param_str}"] = reduced
        
        # Plot variance contribution for each component (elbow plot)
        # ----------------------------------------------------------
        if plt_remaining_var:
            plot_remaining_variance(
                pca, 
                figsize=options["figsize"], 
                title=f"remaining variance {param_str}"
            )

        # UMAP
        # ----

        if add_umap:
            print(f'Conducting UMAP and adding embeddings to adata.obsm["reduced_pc_{pca_dim}_umap"]')
            reducer = umap.UMAP(transform_seed = 42)
            umap_embedding = reducer.fit_transform(reduced)
            print(f"UMAP embedding\n" + "-" * 60  + f"\nshape: {umap_embedding.shape}\n\n")

            # save in AnnData object[obsm] attribute
            # ------------------
            adata.obsm[f"reduced_pc_{pca_dim}_umap {param_str}"] = umap_embedding                    
            print(adata.obsm)
