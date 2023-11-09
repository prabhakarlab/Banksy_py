"""
High-level function for generating all relevant plots for each BANKSY/non-spatial result

refactored from ipython notebook in earlier version of code.

-Yifei/Nigel 21 Aug 23
"""

import os, gc
import matplotlib.pyplot as plt
import scanpy as sc

from typing import Union, List, Tuple
import numpy as np
import pandas as pd
from banksy_utils.plotting import plot_2d_embeddings, plot_labels_seperately
from banksy.labels import plot_connections, Label
import anndata

import matplotlib.ticker as ticker
from scipy.sparse import csc_matrix, csr_matrix


def plot_results(
        results_df: pd.DataFrame,
        weights_graph: Union[csc_matrix, csr_matrix],
        c_map: str,
        match_labels: bool,
        coord_keys: Tuple[str],
        max_num_labels: int = 20,
        save_fig: bool = False,
        save_fullfig: bool = False,
        save_seperate_fig: bool = False,
        save_path: str = None,
        plot_dot_plot: bool = False,
        plot_heat_map: bool = False,
        n_genes: int = 5,  # number of genes to use for heatmap/dotplot
        color_list: List[str] = [],
        dataset_name: str = "slideseq_mousecerebellum",
        main_figsize: Tuple[float, float] = (15, 9),
        **kwargs
) -> None:
    '''
    Plot and visualize the results of BankSY 

    Args:
        results_df (pd.DataFrame): DataFrame containing all the results

        weight_graph: weight_graph object in a dictionary

        max_num_labels (int): Maximum number of labels

        match_labels (bool): If the match labels options was previously indicated

        file_path: str, file_path to save plot 

    Optional args (kwargs):

        save_all_h5ad: to save a copy of the temporary anndata object as .h5ad format

        cmap_name : Color map settings for plotting banksy

        file_path (str): default file path is 'data/slide_seq/v1'
    
    Returns:
        The main figure for visualization using banksy
    '''

    options = {
        'save_all_h5ad': False,
        'group_num': 20
    }
    options.update(kwargs)

    for params_name in results_df.index:

        label_index = 'relabeled' if match_labels else 'labels'

        labels = results_df.loc[params_name, label_index]
        adata_temp = results_df.loc[params_name, "adata"]
        num_pcs = results_df.loc[params_name, 'num_pcs']

        pc_temp = adata_temp.obsm[f"reduced_pc_{num_pcs}"]
        umap_temp = adata_temp.obsm[f"reduced_pc_{num_pcs}_umap"]

        label_name = f"labels_{params_name}"
        adata_temp.obs[label_name] = np.char.mod('%d', labels.dense)
        adata_temp.obs[label_name] = adata_temp.obs[label_name].astype('category')

        if options['save_all_h5ad']:
            # If we want to save the anndata object as .h5ad format
            save_path = os.path.join(save_path, f"{dataset_name}_{params_name}.h5ad")
            adata_temp.write(save_path)

        adata_temp.obsm[coord_keys[2]] = np.vstack(
            (adata_temp.obs[coord_keys[0]].values,
             adata_temp.obs[coord_keys[1]].values)
        ).T

        # Plot Main figure
        # ----------------

        fig, grid = _initialize_main_figure(main_figsize=main_figsize)

        # Auxiliary funtion to plot labels
        _plot_labels(adata_temp,
                     key=coord_keys[2],
                     labels=labels,
                     cmap=c_map,
                     color_list=color_list,
                     max_num_labels=max_num_labels,
                     params_name=params_name,
                     fig=fig,
                     grid=grid)

        if save_fig:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            print(f"Saving figure at {os.path.join(save_path, f'{dataset_name}_{params_name}_spatialmap.png')}")
            fig.savefig(os.path.join(save_path, f'{dataset_name}_{params_name}_spatialmap.png'))

        # Seperate Location plots
        # -----------------------

        plot_labels_seperately(
            labels, adata_temp.obsm[coord_keys[2]],
            embeddings=umap_temp,
            cmap_name=c_map,
            colour_list=color_list,
            default_colour="tab:red",
            plots_per_row=3,
            spot_size=1,
            subplot_size=(5, 5),
            flip_axes=False,
            verbose=False,
            save_fig=save_seperate_fig,
            save_path=os.path.join(save_path, f'{dataset_name}_{params_name}_clusters_seperate.png'),
        )

        # Plot UMAP (again but with labels)
        # ---------------------------------

        ax_umap = fig.add_subplot(grid[0, -2:])
        plot_2d_embeddings(umap_temp, labels.dense,
                           method_str="UMAP",
                           space_str="",
                           xlabel="UMAP 1", ylabel="UMAP 2",
                           ax=ax_umap,
                           cmap_name=c_map,
                           plot_cmap=False,
                           )

        # Plot 1st 2 dimensions of PCA
        # ----------------------------

        dim_sets = (pc_temp[:, :2], pc_temp[:, 1:3])
        dims1 = (0, 1,)
        dims2 = (1, 2,)
        axes = [fig.add_subplot(grid[1, 2 + axnum]) for axnum in range(2)]

        for dim_set, dim1, dim2, ax in zip(dim_sets, dims1, dims2, axes):
            plot_2d_embeddings(dim_set, labels.dense,
                               method_str=f"PCA {dim1 + 1} / {dim2 + 1}",
                               space_str="",
                               xlabel=f"PCA {dim1 + 1}", ylabel=f"PCA {dim2 + 1}",
                               ax=ax,
                               cmap_name=c_map,
                               plot_cmap=False,
                               title_fontsize=9)

        # Plot connectivity between labels
        # --------------------------------

        ax_connections = fig.add_subplot(grid[-1, -2:])

        plot_connections(
            labels,
            weights_graph,
            ax_connections,
            zero_self_connections=True,
            title_str="Connections between label",
            colormap_name=c_map,
        )

        # (optional) Use scanpy functions to plot heatmaps or dotplots
        # ------------------------------------------------------------

        if plot_dot_plot or plot_heat_map:
            num_groups = options['group_num']
            groups_subset = [
                str(n) for n in range(num_groups) if str(n) in adata_temp.obs[label_name].cat.categories
            ]
            print(f"plotting groups: {groups_subset}")

        if plot_heat_map:
            sc.pl.rank_genes_groups_heatmap(adata_temp, n_genes=n_genes,
                                            groups=groups_subset,
                                            vmin=-3, vmax=3, cmap='bwr',
                                            swap_axes=True)

        if plot_dot_plot:
            sc.pl.rank_genes_groups_dotplot(adata_temp, n_genes=n_genes,
                                            groups=groups_subset, )

        if save_fullfig:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            print(f"Saving full-figure at {os.path.join(save_path, f'{dataset_name}_{params_name}_full_figure.png')}")
            fig.savefig(os.path.join(save_path, f'{dataset_name}_{params_name}_full_figure.png'))


def _initialize_main_figure(main_figsize=(15, 9),
                            width_ratios=(2, 0.1, 0.5, 0.5),
                            height_ratios=(1, 0.3, 1),
                            ) -> Tuple[plt.Figure, plt.grid]:
    '''Auxiliary Function to initialize main figure and associated grids'''

    fig = plt.figure(figsize=main_figsize, constrained_layout=True)

    grid = fig.add_gridspec(ncols=4, nrows=3,
                            width_ratios=width_ratios,
                            height_ratios=height_ratios)

    return fig, grid


def _plot_labels(adata_temp: anndata.AnnData,
                 key: str,
                 labels: Label,
                 cmap: str,
                 color_list: List[str],
                 max_num_labels: int,
                 params_name: str,
                 fig: plt.Figure,
                 grid: plt.grid):
    '''
    Plots the spatial map of cluster labels (each with different colour) in the main figure
    '''

    ax_locs = fig.add_subplot(grid[:, 0])

    # We assign color of points to color_list if they are specified
    if color_list:
        c = [color_list[lbl] for lbl in labels.dense]
    else:
        c = labels.dense

    scatterplot = ax_locs.scatter(adata_temp.obsm[key][:, 0],
                                  adata_temp.obsm[key][:, 1],
                                  c=c,
                                  cmap=cmap,
                                  vmin=0, vmax=max_num_labels - 1,
                                  s=3, alpha=1.0)

    ax_locs.set_aspect('equal', 'datalim')
    ax_locs.set_title(f'BANKSY Labels ({params_name})', fontsize=20, fontweight="bold", )
    # ax_locs.set_ylim(ax_locs.get_ylim()[::-1])

    ax_cbar = fig.add_subplot(grid[:, 1])
    cbar = fig.colorbar(
        scatterplot,
        boundaries=np.arange(max_num_labels + 1) - 0.5,
        cax=ax_cbar,
    )
    cbar.set_ticks(labels.ids)
    cbar.set_ticklabels(labels.ids)
    # Turn of ticks and frame
    ax_locs.set(frame_on=False)
    ax_locs.xaxis.set_major_locator(ticker.NullLocator())
    ax_locs.yaxis.set_major_locator(ticker.NullLocator())
