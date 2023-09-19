"""
Create BANKSY Matrix from annadata

Yifei May 2023
"""
import os, time, gc
import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, issparse

import anndata
from copy import deepcopy
from datetime import datetime
from banksy.main import concatenate_all
from sklearn.decomposition import PCA
from typing import Tuple


def generate_banksy_matrix(adata: anndata.AnnData,
                           banksy_dict: dict,
                           lambda_list: list,
                           max_m: int,
                           plot_std: bool = False,
                           save_matrix: bool = False,
                           save_folder: str = './data',
                           variance_balance: bool = False,
                           verbose: bool = True) -> Tuple[dict, np.ndarray]:
    '''Creates the banksy matrices with the set hyperparameters given.
    Stores the computed banksy matrices in the banksy_dict object
    Returns the *last* banksy matrix that was computed.'''

    # Print time of ran
    time_str = datetime.now().strftime("%b-%d-%Y-%H-%M")
    # Print gene_list
    gene_list = adata.var.index
    if verbose:
        print(f'Runtime {time_str}')
        print(f"\n{len(gene_list)} genes to be analysed:\nGene List:\n{gene_list}\n")

    for nbr_weight_decay in banksy_dict:

        # First create neighbour matrices
        nbr_matrices = create_nbr_matrix(adata, banksy_dict, nbr_weight_decay, max_m, variance_balance)

        # Create matrix list
        mat_list, concatenated = create_mat_list(adata, nbr_matrices, max_m)

        # save un-nomalized (i.e. no lambda factor) version of BANKSY matrix
        banksy_dict[nbr_weight_decay]["norm_counts_concatenated"] = concatenated

        if verbose:
            print(f"\nCell by gene matrix has shape {adata.shape}\n")

        for lambda_param in lambda_list:
            gc.collect()
            # Create BANKSY matrix by concatenating all 
            banksy_matrix = concatenate_all(mat_list, lambda_param, adata)

            if verbose:
                print(f"Shape of BANKSY matrix: {banksy_matrix.shape}\nType of banksy_matrix: {type(banksy_matrix)}\n")

            # plot standard deviations per gene / nbr gene
            if plot_std:
                st_dev_pergene = convert2dense(banksy_matrix.X).std(axis=0)
                plot_std_per_gene(st_dev_pergene, lambda_param)

            # save as a new AnnData object
            # ----------------------------

            banksy_dict[nbr_weight_decay][lambda_param] = {"adata": banksy_matrix, }

            # save the banksy matrix as a csv if needed
            # -------------------------------
            if save_matrix:
                try:
                    file_name = f"adata_{nbr_weight_decay}_l{lambda_param}_{time_str}.csv"

                    if not os.path.exists(save_folder):
                        print(f"Making save-folder at {save_folder}")
                        os.makedirs(save_folder)

                    banksy_matrix.write(filename=file_name)
                    print(f'Wrote Banksy_file: {file_name} at {save_folder}')

                except PermissionError:
                    print("\nWARNING: Permission denied to save file. Unable to save adata.\n")

    return banksy_dict, banksy_matrix


def create_nbr_matrix(adata,
                      banksy_dict: dict,
                      nbr_weight_decay: str,
                      max_m: int,
                      variance_balance: bool = False,
                      verbose: bool = True):
    '''Computes the neighbour averaged feature matrices'''
    # Create neighbour matrices
    start_time = time.perf_counter()

    n_components = 20
    X_dense = convert2dense(adata.X)

    if verbose:
        # Show attributes of current cell-gene matrix
        print(f"Decay Type: {nbr_weight_decay}")
        print(f"Weights Object: {banksy_dict[nbr_weight_decay]}")
        # print(f'\nMean and Std of Cells matrix {round(X_dense.mean(),2)} | {round(X_dense.std(),2)}')

    if variance_balance:
        ## We need to balance the variance of the cell's expression
        ## With the variance of the neighbour expression matrix
        pca_own = PCA(n_components=n_components)
        # Zscore across the whole row (axis = 0)

        pca_own.fit_transform(np.nan_to_num(sp.stats.zscore(X_dense, axis=0), nan=0))
        sum_of_cell_pca_var = np.sum(np.square(pca_own.singular_values_))

        if verbose:
            print(f'\nSize of Own | Shape: {X_dense.shape}\nTop 3 entries of Cell Mat:')
            print(X_dense[:3, :3])
            print(f'\nVariance of top {n_components} PCs of cells\'own expression: {round(sum_of_cell_pca_var, 2)}')

    # m == 0, the Nbr matrix, before AGF
    nbr_matrices = {}
    nbr_mat_0 = banksy_dict[nbr_weight_decay]['weights'][0] @ X_dense
    nbr_matrices[0] = nbr_mat_0

    if verbose:
        print(f'\nNbr matrix | Mean: {round(nbr_mat_0.mean(), 2)} | Std: {round(nbr_mat_0.std(), 2)}')
        print(f'Size of Nbr | Shape: {nbr_mat_0.shape}\nTop 3 entries of Nbr Mat:\n')
        print(nbr_mat_0[:3, :3])

    # for m >= 1, AGF is implemented
    for m in range(1, max_m + 1):

        weights = banksy_dict[nbr_weight_decay]["weights"][m]
        weights_abs = weights.copy()

        weights_abs.data = np.absolute(weights_abs.data)
        nbr_avgs = weights_abs @ X_dense
        nbr_mat = np.zeros(adata.X.shape, )

        for n in range(weights.indptr.shape[0] - 1):
            ind_temp = weights.indices[weights.indptr[n]:weights.indptr[n + 1]]
            weight_temp = weights.data[weights.indptr[n]:weights.indptr[n + 1]]
            zerod = X_dense[ind_temp, :] - nbr_avgs[n, :]
            nbr_mat[n, :] = np.absolute(np.expand_dims(weight_temp, axis=0) @ zerod)

        nbr_matrices[m] = nbr_mat
        gc.collect()

        if verbose:
            print(f'\nAGF matrix | Mean: {round(nbr_mat.mean(), 2)} | Std: {round(nbr_mat.std(), 2)}')
            print(f'Size of AGF mat (m = {m}) | Shape: {nbr_mat.shape}')
            print(f'Top entries of AGF:\n{nbr_mat[:3, :3]}')

    if variance_balance:

        for m in range(0, max_m + 1):
            # Balance variance for (neighbourhood / AGF) matrix
            pca_nbr = PCA(n_components=n_components)
            # Zscore across the whole row
            nbr_mat = deepcopy(nbr_matrices[m])
            print(f'\nBalancing (Nbr / AGF) matrix (m = {m}), Variance of Nbr_mat = {round(nbr_mat.var(), 3)}')
            pca_nbr.fit_transform(np.nan_to_num(sp.stats.zscore(nbr_mat, axis=0), nan=0))

            sum_of_nbr_pca_var = np.sum(np.square(pca_nbr.singular_values_))
            balance_factor = np.sqrt(sum_of_cell_pca_var / sum_of_nbr_pca_var)
            nbr_matrices[m] = balance_factor * nbr_mat

            if verbose:
                print(f'Performed Variance Balancing')
                print(f'Variance of top {n_components} PCs of nbrs\' expression {round(sum_of_nbr_pca_var, 2)}')
                print(f'Multiplied nbrs\' expression by balance factor: {round(balance_factor, 2)}')
                print(f'After, Variance of Nbr_mat = {round(nbr_matrices[m].var(), 2)}')

    gc.collect()
    print(f'Ran \'Create BANKSY Matrix\' in {round((time.perf_counter() - start_time) / 60, 2)} mins')
    return nbr_matrices


def create_mat_list(
        adata,
        nbr_matrices,
        max_m: int
):
    '''Auxiliary function to create a list of neighbouring matrices.
    Combines original expression matrix with neighbour matrices (mean and AGF)
    into a list of matrices.
    Also concatenates the matrices into an (unscaled) single matrix.'''

    mat_list = [adata.X]
    for m in range(max_m + 1):
        mat_list.append(nbr_matrices[m])

    # Also save the concatenated un-weighted data for visualization later
    if issparse(adata.X):
        concatenated = sparse.hstack(mat_list, )
    else:
        concatenated = np.concatenate(mat_list, axis=1, )

    return mat_list, concatenated


def plot_std_per_gene(st_dev_pergene, lambda_param, **kwargs):
    '''Plots the standard deviation per gene'''
    options = {'figsize': (8, 2),
               'width': 1,
               'color': 'slateblue'}
    options.update(kwargs)

    fig_title = f"Standard deviations for neighbourhood contribution = {lambda_param}"
    fig, ax = plt.subplots(figsize=options['figsize'])

    ax.bar(np.arange(len(st_dev_pergene)), st_dev_pergene,
           width=options['width'],
           color=options['color'],
           linewidth=0)

    ax.set_title(fig_title)


def convert2dense(X):
    '''Auxiliary function to convert sparse matrix to dense'''
    if isinstance(X, np.ndarray):
        return X
    if issparse(X) or isinstance(X, anndata._core.views.ArrayView):
        X = X.toarray()
    else:
        raise TypeError("sparse format not recognised")

    print('Check if X contains only finite (non-NAN) values')
    assert np.isfinite(X).all()

    return X
