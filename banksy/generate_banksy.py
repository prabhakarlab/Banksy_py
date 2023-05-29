"""
Create BANKSY Matrix from annadata

Yifei May 2023
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, issparse

import anndata

import gc
from datetime import datetime
from banksy.main import concatenate_all

def create_banksy(adata: anndata.AnnData,
                  processing_dict: dict,
                  lambda_list: list,
                  max_m: str,
                  plot_std: bool = False,
                  save_matrix: bool = False,
                  save_folder: str = './data'):
    
    '''Creates the banksy_matrix
    Returns the banksy matrix'''

    # Print time of ran
    time_str = datetime.now().strftime("%b-%d-%Y-%H-%M")
    print(f'Runtime {time_str}')

    # Print gene_list
    gene_list = adata.var.index
    print(f"\n{len(gene_list)} genes to be analysed:\nGene List:\n{gene_list}\n")

    for nbr_weight_decay in processing_dict:
        
        # First create neighbour matrices
        nbr_matrices = create_nbr_matrix(adata, processing_dict, nbr_weight_decay, max_m)

        # Create matrix list
        mat_list, concatenated = create_mat_list(adata, nbr_matrices,max_m)
        processing_dict[nbr_weight_decay]["norm_counts_concatenated"] = concatenated

        print(f"\nCell by gene matrix has shape {adata.shape}\n")

        for lambda_param in lambda_list:
            gc.collect()
            # Create BANKSY matrix by concatenating all 
            banksy_matrix = concatenate_all(mat_list, lambda_param, adata)
            
            print(f"Shape of BANKSY matrix: {banksy_matrix.shape}\n"
                f"type of banksy_matrix: {type(banksy_matrix)}\n",)
            
            # plot standard deviations per gene / nbr gene
            if plot_std:
                print("Warning, plotting standard deviation per gene may cause Jupyter to crash \ndue to the expensive conversion from sparse to dense matrix")
                st_dev_pergene = convert2dense(banksy_matrix.X).std(axis=0)
                plot_std_per_gene(st_dev_pergene, lambda_param)

            # save as a new AnnData object
            # ----------------------------
            
            processing_dict[nbr_weight_decay][lambda_param]= {"adata": banksy_matrix,}        

            # save the banksy matrix as a csv if needed
            # -------------------------------
            if save_matrix:
                try:
                    file_name= f"adata_{nbr_weight_decay}_l{lambda_param}_{time_str}.csv"

                    if not os.path.exists(save_folder):
                        print(f"Making save-folder at {save_folder}")
                        os.makedirs(save_folder)

                    banksy_matrix.write(filename=file_name)
                    print(f'Wrote Banksy_file: {file_name} at {save_folder}')
                
                except PermissionError:
                    print("\nWARNING: Permission denied to save file. Not saving adata.\n")

    
    return processing_dict, banksy_matrix

def create_nbr_matrix(adata,
                      processing_dict: dict,
                      nbr_weight_decay: str,
                      max_m: int):

    '''Computes the neighbour averaged feature matrices'''
    # Create neighbour matrices
    print(f"Decay Type: {nbr_weight_decay}")
    print(f"Weights Object: {processing_dict[nbr_weight_decay]}")

    X_dense = convert2dense(adata.X)
    nbr_matrices = {}
    nbr_mat_0 = processing_dict[nbr_weight_decay]['weights'][0] @ X_dense
    nbr_matrices[0] = nbr_mat_0
    

    for m in range(1, max_m + 1):

        weights = processing_dict[nbr_weight_decay]["weights"][m]  
        weights_abs = weights.copy()

        weights_abs.data = np.absolute(weights_abs.data)
        nbr_avgs = weights_abs @ X_dense
        nbr_mat = np.zeros(adata.X.shape,)

        for n in range(weights.indptr.shape[0]-1):
            ind_temp = weights.indices[weights.indptr[n]:weights.indptr[n+1]]
            weight_temp = weights.data[weights.indptr[n]:weights.indptr[n+1]]
            zerod = X_dense[ind_temp,:] - nbr_avgs[n,:]
            nbr_mat[n,:] = np.absolute( np.expand_dims(weight_temp, axis=0) @ zerod )
        
        nbr_matrices[m] = nbr_mat

    return nbr_matrices     

def create_mat_list(
        adata,
        nbr_matrices,
        max_m: int):
    
    '''Auxiliary function to create a list of neighbouring matrices'''
    
    mat_list = [adata.X]
    for m in range(max_m+1):
        mat_list.append(nbr_matrices[m])

    # Also save the concatenated un-weighted data for visualization later
    if issparse(adata.X):
        concatenated = sparse.hstack(mat_list,)
    else:
        concatenated = np.concatenate(mat_list, axis=1,)

    return mat_list, concatenated

def plot_std_per_gene(st_dev_pergene, lambda_param, **kwargs):
    '''Plots the standard deviation per gene'''
    options = {'figsize':(8,2), 
               'width':1, 
               'color':'slateblue'}
    options.update(kwargs)
    
    fig_title = f"Standard deviations for neighbourhood contribution = {lambda_param}"
    fig, ax = plt.subplots(figsize=options['figsize'])
    
    ax.bar(np.arange(len(st_dev_pergene)), st_dev_pergene, 
            width=options['width'], 
            color = options['color'], 
            linewidth=0)
    
    ax.set_title(fig_title)


def convert2dense(X):
    '''Auxiliary function to convert sparse matrix to dense'''
    return X.todense() if issparse(X) else X
