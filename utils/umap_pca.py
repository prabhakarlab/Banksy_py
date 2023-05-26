"""
Create BANKSY Matrix from annadata

Yifei May 2023
"""
from sklearn.decomposition import PCA
import umap
import anndata
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, issparse
from utils.pca import plot_remaining_variance
from typing import List

def pca_umap(processing_dict: dict,
                  pca_dims: List[int] = [20,],
                  plt_remaining_var: bool = True,
                  **kwargs) -> None:
    '''
    PCA_UMAP first applies dimensionality reduction via PCA,
    then applies UMAP to cluster the groups

    Args:
        processing_dict (dict): The processing dictionary containing info about the banksy matrices
    
    Optional Arg:
        pca_dims (List of integers): A list of integers which the PCA will reduce to
    
    Variable Args (kwargs):
        figsize (tuple of integers): A tuple for adjusting figure size

    Returns: Plot of remaining variance 
    '''
    options = {
        'figsize': (8,3)
    }
    print(f'Current decay types: {list(processing_dict.keys())}')

    for nbr_weight_decay in processing_dict:
        
        for lambda_param in processing_dict[nbr_weight_decay]:
            
            if isinstance(lambda_param, str):
                continue # skip weights matrices

            print(
                f"\nReducing dims of dataset in (Index = {nbr_weight_decay}, lambda = {lambda_param})\n"
                + "=" * 50 + "\n"
            )
            
            # Retrieve anndata object
            # -----------------------
            
            adata_temp = processing_dict[nbr_weight_decay][lambda_param]["adata"]
            
            X = adata_temp.X.todense() if issparse(adata_temp.X) else adata_temp.X
            
            # Reduce dimensions by PCA and then UMAP
            # --------------------------------------
                
            for pca_dim in pca_dims:

                pca = PCA(n_components=pca_dim)
                reduced = pca.fit_transform(X)
                
                print(
                    f"Original shape of matrix: {X.shape}"
                    f"\nReduced shape of matrix: {reduced.shape}\n" + "-" * 60 
                    + f"\nmin_value = {reduced.min()}, max = {reduced.max()}\n"
                )
                
                # Plot variance contribution for each component (elbow plot)
                # ----------------------------------------------------------
                if plt_remaining_var:
                    plot_remaining_variance(
                        pca, 
                        figsize=options["figsize"], 
                        title=f"decay type = {nbr_weight_decay}, lambda = {lambda_param}"
                    )

                # UMAP
                # ----

                reducer = umap.UMAP(transform_seed = 42)
                umap_embedding = reducer.fit_transform(reduced)
                print(f"UMAP embedding\n" + "-" * 60  + "\nshape: {umap_embedding.shape}\n\n")

                # save in dictionary
                # ------------------

                adata_temp.obsm[f"reduced_pc_{pca_dim}"] = reduced
                adata_temp.obsm[f"reduced_pc_{pca_dim}_umap"] = umap_embedding
                
                print(adata_temp.obsm)

    