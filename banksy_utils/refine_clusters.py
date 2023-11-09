import anndata
import numpy as np
import pandas as pd

from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp

from banksy.labels import Label
from banksy_utils.color_lists import spagcn_color
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import copy
import os, time, gc


def refine_clusters(adata: anndata.AnnData,
                    results_df: pd.DataFrame,
                    coord_keys: tuple,
                    color_list: list = spagcn_color,
                    savefig: bool = False,
                    output_folder: str = "",
                    refine_method: str = "once",
                    refine_iterations: int = 1,
                    annotation_key: str = "manual_annotations",
                    num_neigh: int = 6,
                    verbose: bool = False
                    ) -> pd.DataFrame:
    '''
    Function to refine predicted labels based on nearest neighbours

    Args:
        adata (AnnData): Original anndata object

        results_df (pd.DataFrame): DataFrame object containing the results from BANKSY

    Optional Args: 
        color_list (list); default = spagcn : List in which colors are used to plot the figures. 

        refine_method ("auto" | "once" | "iter_num" ): 
                To refine clusters once only or iteratively refine multiple times
                If "auto" is specified, the refinement procedure completes iteratively 
                    until only 0.5% of the nodes are changed.

                If "iter_num" is specified, specify the 'refine_iterations' parameter

        num_neigh (int) : Number of nearest-neighbours the refinement is conducted over

        savefig (bool): To save figure in folder or not.

        annotation_key (str): The key in whicb the ground truth annotations are accessed under 
                adata.obs[annotation_key],

                If no ground truth is present, then set annotation_key = None

    '''

    start_time = time.perf_counter()

    if annotation_key:
        annotations = adata.obs[annotation_key].cat.codes.tolist()
    else:
        annotations = None

    refined_ari_results, total_entropy_dict = {}, {}

    for params_name in results_df.index:
        adata_temp = results_df.loc[params_name, 'adata']
        raw_labels = results_df.loc[params_name, 'labels']
        num_clusters = results_df.loc[params_name, 'num_labels']
        lambda_p = results_df.loc[params_name, "lambda_param"]

        if annotation_key:
            ari_temp = results_df.loc[params_name, "ari"]

        if isinstance(raw_labels, Label):
            raw_labels = raw_labels.dense

        if coord_keys[2] not in adata_temp.obsm_keys():
            adata_temp.obsm[coord_keys[2]] = pd.concat([adata.obs[coord_keys[0]], adata.obs[coord_keys[1]]],
                                                       axis=1).to_numpy()

        # Refine Clusters
        refined_list = raw_labels[:]

        if refine_method.__eq__("auto"):
            total_entropy, num_iter = 1.0, 1
            # We keep on refining until only 1.0% of the nodes are swapped
            if verbose:
                print(f'Auto Refine')
            while ((total_entropy * 100 > 5) and (num_iter < 20)):
                refined_list, refined_ari, total_entropy = refine_once(adata_temp,
                                                                       refined_list,
                                                                       annotations,
                                                                       coord_keys,
                                                                       num_neigh)
                if verbose:
                    print(f'Refined iteration: {num_iter} | Total Entropy: {round(total_entropy, 2)}')
                num_iter += 1

        elif refine_method.__eq__("once"):
            print(f'Refine only once')
            refined_list, refined_ari, total_entropy = refine_once(adata_temp,
                                                                   refined_list,
                                                                   annotations,
                                                                   coord_keys,
                                                                   num_neigh)
            print(f'Refined once | Total Entropy: {round(total_entropy, 2)}')

        elif refine_method.__eq__("num_iter"):
            # Refine a specifed number of iterations  
            print(f'Refine {refine_iterations} times')
            for i in range(refine_iterations):
                refined_list, refined_ari, total_entropy = refine_once(adata_temp,
                                                                       refined_list,
                                                                       annotations,
                                                                       coord_keys,
                                                                       num_neigh)

                if verbose:
                    print(f'Refined iteration: {i} | Total Entropy: {round(total_entropy, 2)}')

        else:
            print("No valid refine type specified! {\"auto\", \"once\", \"num_iter\"}")
            return results_df

        if annotation_key:
            refined_ari_results[params_name] = refined_ari
            raw_ari = f'\nARI = {round(ari_temp, 2)}'
            refined_ari = f'\nRefined ARI = {round(refined_ari, 2)}'

        total_entropy_dict[params_name] = total_entropy / np.log(num_clusters)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
        # fig = plt.figure(figsize=main_figsize, constrained_layout=True)
        grid = fig.add_gridspec()
        raw_clusters, refined_clusters = [], []

        for i in raw_labels:
            raw_clusters.append(color_list[i])

        if annotation_key:
            title1 = f'λ = {round(lambda_p, 2)}' + raw_ari
            title2 = f'λ = {round(lambda_p, 2)}' + refined_ari
        else:
            title1 = f'λ = {round(lambda_p, 2)}'
            title2 = f'λ = {round(lambda_p, 2)}'

        subplot_sc(ax1,
                   adata_temp,
                   coord_keys,
                   raw_clusters,
                   num_clusters,
                   title1)

        for i in refined_list:
            refined_clusters.append(color_list[i])

        subplot_sc(ax2,
                   adata_temp,
                   coord_keys,
                   refined_clusters,
                   num_clusters,
                   title2)

        if savefig:
            print(f'{params_name} saved refined plot at {output_folder}')
            save_name = os.path.join(output_folder, str(params_name) + '.png')
            fig.savefig(save_name, format='png', dpi=108)

    if annotation_key:
        results_df["refined_ari"] = refined_ari_results

    results_df["total_Entropy_normalized"] = total_entropy_dict
    gc.collect()
    print(f'Time taken for refinement = {round((time.perf_counter() - start_time) / 60, 2)} min')
    return results_df


def subplot_sc(ax: plt.Axes,
               adata_temp: anndata.AnnData,
               coord_keys: tuple,
               clusters_colormap: list,
               num_clusters: int,
               title: str) -> None:
    ax.scatter(adata_temp.obs[coord_keys[0]],
               adata_temp.obs[coord_keys[1]],
               c=clusters_colormap,
               vmin=0, vmax=num_clusters - 1,
               s=50, alpha=1.0)

    ax.set_aspect('equal', 'datalim')
    ax.set_title(title, fontsize=20, fontweight="bold", )
    ax.axes.invert_yaxis()

    # Turn of ticks and frame
    ax.set(frame_on=False)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())


def refine_once(adata_temp: anndata.AnnData,
                raw_labels: list,
                truth_labels: list,
                coord_keys: tuple,
                num_neigh: int = 6,
                threshold_percentile: int = None):
    '''Refined clusters with respect to their mean neighbourhood labels,
    if more than 50% of a node's neighbours is of the same label, 
    we swap this node's label with theirs'''

    refined_list = copy.deepcopy(raw_labels)
    nbrs = NearestNeighbors(algorithm='ball_tree').fit(adata_temp.obsm[coord_keys[2]])
    csr_mat = nbrs.kneighbors_graph(n_neighbors=num_neigh, mode='distance')

    if threshold_percentile:
        # if a 'threshold_percentile' is set,
        # cut off edges that are longer than the threshold_percentile
        nnz_inds = csr_mat.nonzero()
        threshold = np.quantile(csr_mat.data, threshold_percentile)
        keep = np.where(csr_mat.data < threshold)[0]
        n_keep = len(keep)
        csr_mat = sp.csr_matrix((np.ones(n_keep), (nnz_inds[0][keep], nnz_inds[1][keep])), shape=csr_mat.shape)

    indptr = csr_mat.indptr
    num_nodes = len(indptr) - 1

    num_nodes_swapped = 0
    total_entropy = 0

    for i in range(num_nodes):
        nbrs = csr_mat[i]
        nbrs_len = len(nbrs.indices)

        if nbrs_len == 0:
            # If node has no neighbours, we skip refinement process
            continue

        current_label = raw_labels[i]  # Node's current label
        nbr_labels = [current_label]

        for n in nbrs.indices:
            # Constructs list of the neighbourhood's labels
            nbr_labels.append(raw_labels[n])

        # Count the number of unqiue labels
        unique, counts = np.unique(nbr_labels, return_counts=True)
        max_counts = np.max(counts)
        res = unique[np.argmax(counts)]  # Majority of neighourhood expression

        # Threshold count for refinement process
        threshold_count = (nbrs_len // 2) + 1
        if (res != current_label) and (max_counts >= threshold_count):
            num_nodes_swapped += 1
            refined_list[i] = res

        # Calculate the (local) neighbourhood's Entropy
        local_entropy = entropy(counts)
        total_entropy += local_entropy

    refined_ari = adjusted_rand_score(refined_list, truth_labels) if truth_labels else -1

    swapped_ratio = num_nodes_swapped / num_nodes
    total_entropy /= num_nodes

    print(f'\nNumber of nodes swapped {num_nodes_swapped} | ratio: {round(swapped_ratio, 3)}')
    print(f'Total Entropy: {round(total_entropy, 2)}')

    return refined_list, refined_ari, total_entropy


def entropy(counts: list) -> float:
    '''Takes a list of labels and outputs the cross Entropy'''
    total_count = np.sum(counts)
    p = counts / total_count
    return np.sum(-p * np.log(p))
