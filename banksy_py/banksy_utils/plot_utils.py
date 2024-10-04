#
# Higher level plotting functions (i.e. generating multiple plots)
# lower level plot functions are in banksy_utils.plotting
# Refactored/edited by Yifei
#

import numpy as np
from numpy.random import randint
import pandas as pd

import os
import gc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from anndata import AnnData
import anndata
from banksy_utils.plotting import plot_edge_histogram, plot_graph_weights
from banksy_utils.cluster_utils import pad_clusters, get_DEgenes, get_metagene_difference

from banksy_utils.slideseq_ref_data import dropviz_dict, markergenes_dict

import scanpy as sc

from scipy.stats import pearsonr, pointbiserialr
from itertools import cycle
from typing import Union, Tuple, List
import scipy.sparse as sparse


def plot_qc_hist(
        adata: anndata.AnnData,
        total_counts_cutoff: int,
        n_genes_high_cutoff: int,
        n_genes_low_cutoff: int,
        **kwargs):
    '''
    Plots a 1-by-4 figure of histograms for
    1: total_counts (unfiltered), 2: total_counts (filtered), 3: n_genes (unfiltered), 4: n_genes (filtered)

    Args:
        Anndata: AnnData Object containing 'total_counts' and 'n_genes_by_counts' 
        total_counts_cutoff: Lower threshold for total counts, if 
        n_genes_high_thrsehold: Upper threshold of gene count

    Optional Args:
        figsize; default = (15,4): size of the figures
        bins1-4; default = 'auto': the number of bins for plotting the histograms (e.g., bin1=80 sets the 80 bins for the first histogram)
    '''
    # Edit bin_size_for_plts as a option, remove bins1,2
    options = {'figsize': (15, 4),
               'bin_options': ['auto', 'auto', 'auto', 'auto']}

    options.update(kwargs)
    bin_sizes = options['bin_options']

    fig, axs = plt.subplots(1, 4, figsize=(15, 4))

    sns.histplot(adata.obs["total_counts"], kde=False, bins=bin_sizes[0], ax=axs[0]).set(
        title='Total Counts (unbounded)')

    sns.histplot(adata.obs["total_counts"][adata.obs["total_counts"] < total_counts_cutoff],
                 kde=False, bins=bin_sizes[1], ax=axs[1]).set(title=f'Total Counts (bounded)')

    sns.histplot(adata.obs["n_genes_by_counts"], kde=False, bins=bin_sizes[2], ax=axs[2]).set(
        title='n-genes by counts (unbounded)')

    sns.histplot(adata.obs["n_genes_by_counts"][
                     (adata.obs["n_genes_by_counts"] < n_genes_high_cutoff) & (
                             adata.obs["n_genes_by_counts"] > n_genes_low_cutoff)
                     ], kde=False, bins=bin_sizes[3], ax=axs[3]).set(title='n-genes by counts (bounded)')

    fig.tight_layout()
    gc.collect()


def plot_cell_positions(adata: anndata.AnnData,
                        raw_x: pd.Series,
                        raw_y: pd.Series,
                        coord_keys: Tuple[str],
                        **kwargs) -> None:
    """
    Plots the position of cells in the dataset

    Args: 
        adata - Anndata containing information about the cell
        raw_x - Positions of the cells
        raw_y

    Optional Args:
        add_cricle: adding a circle patch for AFT data

    Returns scatter plot of the cell positions 
    """
    options = {
        'fig_size': (8, 8),
        's': 0.2,
        'c1': 'red',
        'c2': 'slateblue',
        'label1': 'Original cells',
        'label2': 'Remaining cells',
        'add_circle': False,
        'puck_center': (3330, 3180),
        'puck_radius': 2550
    }

    options.update(kwargs)
    print(options)

    fig, ax = plt.subplots(figsize=options['fig_size'])
    ax.scatter(raw_x, raw_y, s=options['s'],
               c=options['c1'], label=options['label1'])
    ax.scatter(adata.obs[coord_keys[0]], adata.obs[coord_keys[1]],
               s=options['s'], c=options['c2'], label=options['label2'])
    ax.set_title("Cell positions")
    if options['add_circle']:
        circle = plt.Circle(
            options['puck_center'], options['puck_radius'], color='g', fill=False)
        ax.add_patch(circle)

    ax.axis("equal")
    gc.collect()


def plot_edge_histograms(distance_graph: sparse.csr_matrix,
                         weights_graph: sparse.csr_matrix,
                         decay_type: str,
                         m: int,
                         **kwargs) -> None:
    '''An utility function to plot subplots of each histograms
    using the plot_edge_histogram function for each distance and weights'''

    options = {'rows': 1, 'cols': 2, 'figsize': (10, 5)}
    options.update(kwargs)

    # Plots the histograms of distance and weights each m
    fig, ax = plt.subplots(
        nrows=options['rows'], ncols=options['cols'], figsize=options['figsize'])
    plot_edge_histogram(distance_graph, ax[0], title="distances between cells")
    plot_edge_histogram(weights_graph, ax[1], title="weights between cells")
    fig.suptitle(f"nbr decay type = {decay_type}, m = {m}")
    fig.tight_layout()


def plot_weights(
        adata: anndata.AnnData,
        processing_dict: dict,
        nbr_weight_decay: str,
        max_m: int,
        fig_title: str,
        coord_keys: Tuple[str],
        theta_graph: sparse.csr_matrix = None,
        **kwargs):
    '''Plots the weighted graphs using the plotting function
    Optional Args: theta_graph, max_weight, markersize, figsize
    '''

    options = {'max_weight': 0.5, 'markersize': 1.5, 'figsize': (15, 8)}
    options.update(kwargs)

    for m in range(max_m + 1):
        # Plots the weighted graph for each iterations of m
        plot_graph_weights(
            adata.obsm[coord_keys[2]],
            processing_dict[nbr_weight_decay]["weights"][m],
            theta_graph=theta_graph,
            max_weight=options["max_weight"],
            markersize=options["markersize"],
            figsize=options["figsize"],
            title=f"{fig_title}, m = {m}"
        )

        ax = plt.gca()
        ax.axis("equal")


def plot_theta_graph(adata: anndata.AnnData,
                     theta_graph: sparse.csr_matrix,
                     coord_keys: Tuple[str],
                     **kwargs):
    '''Shows the theta graph of the neigbourhood of a random cell'''

    options = {'alpha': 0.4, 'figsize': (6, 6)}
    options.update(kwargs)

    key = coord_keys[2]
    for cell in randint(1, adata.shape[0], 1):
        fig, ax = plt.subplots(figsize=options['figsize'])
        ax.plot(adata.obsm[key][cell, 0], adata.obsm[key]
        [cell, 1], "bo", alpha=options['alpha'])
        ptr_start = theta_graph.indptr[cell]
        ptr_end = theta_graph.indptr[cell + 1]
        for nbr, angle in zip(theta_graph.indices[ptr_start:ptr_end], theta_graph.data[ptr_start:ptr_end]):
            ax.plot(adata.obsm[key][nbr, 0], adata.obsm[key]
            [nbr, 1], "ro", alpha=options['alpha'])
            ax.annotate(f"{angle / np.pi * 180:0.1f}",
                        (adata.obsm[key][nbr, 0], adata.obsm[key][nbr, 1]))
            ax.axis("equal")

        ax.set_title(f"Azimuthal Angles in a cell's neighborhood")

    gc.collect()


def plot_DE_genes(adata_spatial: anndata.AnnData,
                  adata_nonspatial: anndata.AnnData,
                  plot_heatmap: bool = False,
                  plot_dot_plot: bool = False,
                  ) -> None:
    '''
    Function to plot the Differentially-Expressed (DE) genes per group

    Returns adata_nonspatial, adata_spatial subsequent plotting if needed
    '''

    for adata in [adata_spatial, adata_nonspatial]:

        # Plot Differentially Expressed (DE) genes per group
        # --------------------------------------------------
        sc.tl.rank_genes_groups(
            adata, groupby='cell type', method='wilcoxon', verbose=True
        )
        sc.tl.dendrogram(adata, groupby='cell type')

        if plot_heatmap:
            sc.pl.rank_genes_groups_heatmap(adata, n_genes=6,
                                            vmin=-3, vmax=3, cmap='bwr',
                                            swap_axes=True)
        if plot_dot_plot:
            sc.pl.rank_genes_groups_dotplot(adata, n_genes=5)

    gc.collect()


def _adata_filter_self(adata: anndata.AnnData,
                       is_nbr_colname: str = "is_nbr",
                       run_rank_genes_groups: bool = True,
                       cell_type_colname: str = "cell type refined", ) -> anndata.AnnData:
    """
    Filter AnnData object to remove all neighbour features (mean/AGF)
    leaving only self-expresssion features
    Assumes AnnData object has .var column of type bool that defines if each feature is self or neighbour
    """

    adata_self = adata[:, adata.var[is_nbr_colname] == False].copy()

    if run_rank_genes_groups:
        sc.tl.rank_genes_groups(
            adata_self, groupby=cell_type_colname, method='wilcoxon', verbose=True
        )
        sc.tl.dendrogram(adata_self, groupby=cell_type_colname)

    return adata_self


def plot_DE_genes_refined(adata_spatial: anndata.AnnData,
                          plot_heatmap: bool = True,
                          plot_dotplot: bool = False,
                          n_genes: int = 5,
                          cell_type_colname: str = "cell type refined",
                          save_fig: bool = False,
                          file_path: str = "",
                          save_name: str = "",
                          **kwargs) -> anndata.AnnData:
    '''
    Function to plot Differentially Expressed (DE) genes 
    by self-expression for the refined anndata

    Args:
        adata_spatial: this should be the BANKSY matrix

    Optional args:
        save_fig (default = False)
        kwargs: optional specifications for saving figure
    '''

    options = {
        'fig_size': (12, 5),
        'save_format': 'eps',
        'dpi': 1200
    }
    options.update(kwargs)

    # filter AnnData object for only self-expresssion features
    adata_self = _adata_filter_self(adata_spatial,
                                    run_rank_genes_groups=True,
                                    cell_type_colname=cell_type_colname, )

    if plot_heatmap:

        if save_fig:
            save_path = os.path.join(file_path, save_name + options['save_format'])
        else:
            save_path = None

        sc.pl.rank_genes_groups_heatmap(adata_self, n_genes=n_genes,
                                        vmin=-3, vmax=3, cmap='bwr', swap_axes=True,
                                        save=save_path)

    if plot_dotplot:
        fig2, ax = plt.subplots(figsize=options['fig_size'])

        sc.pl.rank_genes_groups_dotplot(adata_self, n_genes=n_genes, groups=None, ax=ax)

        if save_fig:
            fig2.savefig(os.path.join(file_path, save_name + "dotplot." + options['save_format']),
                         format=options['save_format'], dpi=options['dpi'])

    return adata_self


def plot_connection_grid(
        weights: sparse.csr_matrix,
        adata_nonspatial: anndata.AnnData,
        adata_spatial: anndata.AnnData,
        cell_types: list,  # list of clusters/cell types to compare
        cell_types_colname: str = "cell type refined",
        cmap: str = "viridis",
        save_fig: bool = False,
        file_path: str = "",
        save_name: str = "",
        **kwargs) -> None:
    # Changed weights to be argument instead of processing_dict;
    # weights = processing_dict["scaled_gaussian"]["weights"][0].copy()
    '''
    Plots connection grid showing intermingling extent between spatially
    distinct layers
    '''

    options = {
        'fig_size': (12, 5),
        'save_format': 'eps',
        'dpi': 1200
    }
    options.update(kwargs)

    print(
        f"All refined cell types: {adata_spatial.obs[cell_types_colname].cat.categories}")

    fig, axs = plt.subplots(nrows=1, ncols=2,
                            figsize=(len(cell_types) * 2.4 + 3, len(cell_types) * 1.2))

    connection_grids = []

    max_nondiag = 0.0

    for adata in [adata_nonspatial, adata_spatial]:

        one_hots = []
        for cell_type in cell_types:
            one_hots.append(
                np.array(adata.obs[cell_types_colname] == cell_type))

        # one hot labels
        # --------------
        one_hot_array = np.stack(one_hots, axis=0)
        # print(one_hot_array.shape, one_hot_array.dtype)

        # adjacency matrix
        # ----------------
        # use the m=0 weights matrix (default 15 nearest nbrs)
        weights.data = np.ones_like(weights.data)
        print(weights.data, weights.shape)

        # Unnormalized summed connections
        # -------------------------------
        connection_grid = one_hot_array @ weights @ one_hot_array.T

        # Normalize
        # ---------
        connection_grid /= np.diag(connection_grid)[:, np.newaxis]

        connection_grids.append(connection_grid)

        # find max non-diagonal by zeroing out diagonal values (all diagonals = 1)
        connections_zerod = connection_grid.copy()
        np.fill_diagonal(connections_zerod, 0)
        max_nondiag = max(max_nondiag, np.amax(connections_zerod))

    for n, connection_grid in enumerate(connection_grids):
        heat_map = sns.heatmap(connection_grid,
                               linewidth=1,
                               vmax=1,
                               cmap=cmap, annot=True, fmt="0.2f",
                               yticklabels=cell_types, xticklabels=cell_types,
                               square=True, ax=axs[n])
    gc.collect()
    fig.tight_layout()

    if save_fig:
        save_location = os.path.join(file_path, save_name)
        fig.savefig(save_location, format=options['save_format'], dpi=options['dpi'])


def plot_clusters(adata_spatial: anndata.AnnData,
                  adata_nonspatial: anndata.AnnData,
                  colors_dict: dict,
                  clusters_colname: str = "cell type",
                  save_fig: bool = False,
                  file_path: str = "",
                  save_name: str = "",
                  **kwargs) -> None:
    '''
    Plot spatial maps of clusters for spatial and nonspatial, side by side
    (or any 2 clustering outputs you want to compare)
    '''

    options = {
        'fig_size': (8, 6),
        'save_format': 'eps',
        'dpi': 1200,
        'key': 'coord_xy',
        'title': ['Spatial', "Nonspatial"]
    }
    options.update(kwargs)

    coords_key = options['key']

    fig, axs = plt.subplots(
        nrows=1, ncols=2, figsize=options['fig_size'], constrained_layout=True
    )

    for n, adata in enumerate([adata_spatial, adata_nonspatial]):

        ax_loc = axs[n]

        # umap_name = f"reduced_pc_{pca_dims[0]:2d}_umap"

        for layer in adata.obs[clusters_colname].cat.categories:

            if layer in colors_dict:
                c = colors_dict[layer]
                locations = adata.obsm[coords_key][adata.obs[clusters_colname] == layer, :]
                # umap_embedding = adata.obsm[umap_name][adata.obs[clusters_colname] == layer, :]
                loc_plot = ax_loc.scatter(
                    locations[:, 0], locations[:, 1], c=c, s=.2, alpha=1.0
                )

        ax_loc.set_aspect('equal', 'datalim')
        ax_loc.axis("off")
        ax_loc.set_title(options['title'][n])

    if save_fig:
        save_location = os.path.join(file_path, save_name)
        fig.savefig(save_location, format=options['save_format'], dpi=options['dpi'])


def _plot_embedding(embedding: str,  # key for obsm entry in an AnnData object
                    ax: plt.Axes,
                    adata: AnnData,
                    cluster_subset: list,
                    spot_size: float,
                    colours_dict: dict,
                    cluster_colname: str = "cell type",
                    ) -> None:
    """
    Plot an embedding in spatial or expression umap axes

    The embedding should be stored as an obsm (i.e. multidimensional observation data array)
    entry in the AnnData object provided as adata.
    Only first 2 dimenstions of the embedding will be plotted

    plots clusters not in 'cluster_subset' in gray
    """

    # plot other clusters in gray first
    # (to make sure spots are below those of desired clusters in case of overlap)

    for cluster in adata.obs[cluster_colname].cat.categories:

        if cluster not in cluster_subset:
            c = "whitesmoke"
            locations = adata.obsm[embedding][adata.obs[cluster_colname] == cluster, :]
            scatterplot = ax.scatter(locations[:, 0], locations[:, 1], c=c,
                                     s=spot_size, alpha=1.0)

    # then plot the clusters in 'cluster subset'

    for cluster in adata.obs[cluster_colname].cat.categories:

        if cluster in cluster_subset:
            c = colours_dict[cluster]
            locations = adata.obsm[embedding][adata.obs[cluster_colname] == cluster, :]
            scatterplot = ax.scatter(locations[:, 0], locations[:, 1], c=c,
                                     s=spot_size, alpha=1.0)


def _plot_continuous(x, y, c, ax,
                     spot_size: float = 0.3,
                     vmax: float = None,
                     cmap: str = "Blues",
                     title: str = None,
                     plot_cbar: bool = True,
                     ) -> None:
    """
    Plot continous-valued location plot (e.g. markergene, RCTD weights)
    """

    scatterplot = ax.scatter(x, y,
                             c=c, s=spot_size,
                             alpha=1.0, cmap=cmap,
                             vmax=vmax, vmin=0,
                             )

    ax.set_aspect('equal', 'datalim')

    if title is not None:
        ax.set_title(title, fontsize=8, fontweight="bold", )
    # ax.set_ylim(ax_locs.get_ylim()[::-1])
    ax.axis("off")

    if plot_cbar:
        cax = ax.inset_axes([0.8, 0.0, 0.16, 0.04])
        # cax.axis("off")
        ax.figure.colorbar(scatterplot, cax=cax, orientation='horizontal')


def plot_markergene_sets(metagene_df: pd.DataFrame,
                         rctd_coord: pd.DataFrame,
                         rctd_weights: pd.DataFrame,
                         coord_keys: list,  # keys for x and y columns
                         reference_dict: dict = dropviz_dict,  # mapping celltype names to ref names
                         markergenes_dict: dict = markergenes_dict,  # lists of markergenes for each celltype
                         plot_cbar: bool = False,
                         save_fig: bool = False,
                         file_path: str = "",
                         save_name: str = "",
                         **kwargs) -> None:
    '''
    Plots all the marker gene sets for scRNA-seq clusters 

    defaults to labels and DE genes from dropviz as a reference.
    '''
    options = {
        'subplot_size': (3, 3),
        'save_format': 'png',
        'dpi': 1200
    }
    options.update(kwargs)

    subplot_size = options['subplot_size']
    num_cols = 2
    num_rows = len(reference_dict) // num_cols + 1
    figsize_x = (subplot_size[0] * num_cols) * 2.1
    figsize_y = subplot_size[1] * num_rows

    fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=True)
    grid = fig.add_gridspec(ncols=num_cols * 2, nrows=num_rows)

    for n, layer in enumerate(reference_dict):

        # plot the RCTD weights

        ax = fig.add_subplot(grid[n // num_cols, 2 * (n % num_cols)])

        clust_num = reference_dict[layer] - 1
        weights = np.array(rctd_weights[:, clust_num]).flatten()

        _plot_continuous(rctd_coord["x"], rctd_coord["y"], weights,
                         ax, vmax=1, title=layer + "\nRCTD weights",
                         spot_size=0.2, plot_cbar=plot_cbar,
                         )

        if layer in markergenes_dict:

            # if available, plot the meta-gene weights for the marker genes

            ax = fig.add_subplot(grid[n // num_cols, 2 * (n % num_cols) + 1])

            c = metagene_df[layer]
            if isinstance(c, pd.Series) and c.isnull().any():
                c.fillna(0, inplace=True)

            _plot_continuous(metagene_df[coord_keys[0]], metagene_df[coord_keys[1]], c,
                             ax, vmax=1, title=layer + "\nmarker genes",
                             spot_size=0.2, plot_cbar=plot_cbar,
                             )
    if save_fig:
        save_location = os.path.join(file_path, save_name)
        fig.savefig(save_location, format=options['save_format'], dpi=options['dpi'])


def plot_cluster_subset(cluster_subset: list,
                        colours_dict: dict,
                        adata: anndata.AnnData,
                        ax: plt.Axes,
                        title: str = None,
                        umap_name: str = "umap",
                        spot_size: float = 0.4,
                        plot_umap: bool = False,
                        ) -> None:
    """
    Plot a subset of clusters using specified colours from colours_dict
    """

    # plot spatial locations
    _plot_embedding("coord_xy", ax, adata, cluster_subset, spot_size, colours_dict)
    ax.set_aspect('equal', 'datalim')

    if title is not None:
        ax.set_title(title, fontsize=20, fontweight="bold", )
    # ax.set_ylim(ax_locs.get_ylim()[::-1])

    ax.axis("off")

    # plot  UMAP as inset
    if plot_umap:
        axins = ax.inset_axes([0.9, 0.0, 0.25, 0.25])
        _plot_embedding(umap_name, axins, adata, cluster_subset, spot_size, colours_dict)
        axins.set_aspect('equal', 'datalim')
        # axins.axis("off")
        axins.set_ylabel("UMAP2", fontsize=8, fontweight="bold", )
        axins.set_xlabel("UMAP1", fontsize=8, fontweight="bold", )
        axins.spines['right'].set_visible(False)
        axins.spines['top'].set_visible(False)
        axins.xaxis.set_ticks([])
        axins.yaxis.set_ticks([])


def plot_weights_comparsion(adata_spatial: anndata.AnnData,
                            adata_nonspatial: anndata.AnnData,
                            adata_allgenes: anndata.AnnData,  # unfiltered Anndata object to plot marker gene expr
                            cluster_keys: List[list],  # keys for spatial, nonspatial, RCTD, markergenes
                            colours_dict: dict,
                            rctd_coord: pd.DataFrame = None,
                            rctd_weights: pd.DataFrame = None,
                            reference_dict: dict = dropviz_dict,  # mapping celltype names to ref names
                            markergenes_dict: dict = markergenes_dict,  # lists of markergenes for each celltype
                            save_fig: bool = False,
                            file_path: Union[str, None] = None,
                            save_png_name: Union[str, None] = None,
                            save_eps_name: Union[str, None] = None,
                            **kwargs) -> None:
    '''
    Plots the comparsion between spatial, nonspatial and RCTD weights and marker gene expression

    '''
    options = {
        'subplot_size': (3, 3),
        'save_eps_name': "subplots_layers_sseqcerebellum.eps",
        'dpi': 1200
    }
    options.update(kwargs)

    subplot_size = options['subplot_size']
    num_cols = len(cluster_keys)

    num_rows = 4
    if rctd_coord is None:
        num_rows -= 1
    if reference_dict is None:
        num_rows -= 1

    figsize_x = (subplot_size[0] * num_cols) * 1.1
    figsize_y = subplot_size[1] * num_rows

    print(f"\nGenerating {num_rows} by {num_cols} "
          f"grid of plots (figsize = {figsize_y}, {figsize_x})\n")

    fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=True)
    grid = fig.add_gridspec(ncols=num_cols, nrows=num_rows)

    for col, cluster_key in enumerate(cluster_keys):

        # BANKSY
        # ------

        ax = fig.add_subplot(grid[0, col])
        plot_cluster_subset(cluster_key[0], colours_dict, adata_spatial,
                            ax, spot_size=0.2, plot_umap=False)

        # non-spatial
        # -----------

        ax = fig.add_subplot(grid[1, col])
        plot_cluster_subset(cluster_key[1], colours_dict, adata_nonspatial,
                            ax, spot_size=0.2, plot_umap=False)

        # Plot scRNAseq metagene (from top DE genes)
        # ------------------------------------------

        if rctd_coord is not None and rctd_weights is not None and reference_dict is not None:

            ax = fig.add_subplot(grid[2, col])

            weights = np.zeros((len(rctd_coord),))

            for dv_clust in cluster_key[2]:
                clust_num = reference_dict[dv_clust] - 1
                weights += np.array(rctd_weights[:, clust_num]).flatten()

            _plot_continuous(rctd_coord.loc[:, "x"], rctd_coord["y"],
                             weights, ax, vmax=1, plot_cbar=False)

        # Plot scRNAseq metagene (from top DE genes)
        # ------------------------------------------

        if markergenes_dict is not None and adata_allgenes is not None:

            ax = fig.add_subplot(grid[-1, col])

            markers = []

            for sc_clust in cluster_key[-1]:

                # get a list of marker genes from the reference that are also in the dataset (unfiltered AnnData)
                markers_temp = [m for m in markergenes_dict[sc_clust] if m in adata_allgenes.var.index]

                if markers_temp < markergenes_dict[sc_clust]:
                    print(f"{len(markergenes_dict[sc_clust]) - len(markers_temp)} "
                          f"scRNA-seq DE genes in {sc_clust} absent/filtered from slideseq dataset")

                markers += markers_temp

            print(f"{len(markers)} scRNAseq DE markers for clusters {cluster_key[-1]}")

            # sum marker genes to obtain a metagene summarizing marker gene expression
            markers_slice = adata_allgenes[:, markers]
            metagene = np.mean(markers_slice.X, axis=1)

            _plot_continuous(adata_spatial.obs["xcoord"], adata_spatial.obs["ycoord"],
                             metagene, ax, vmax=1, plot_cbar=False)

    if save_fig:
        if save_png_name:
            fig.savefig(os.path.join(file_path, save_png_name), format='png', dpi=options['dpi'])
        if save_eps_name:
            fig.savefig(os.path.join(file_path, save_eps_name), format='eps', dpi=options['dpi'])


def compare_weights(adata_spatial: anndata.AnnData,
                    adata_nonspatial: anndata.AnnData,
                    metagene_df: anndata.AnnData,
                    rctd_coord: pd.DataFrame,
                    rctd_weights: pd.DataFrame,
                    cell_types: list,  # named cell-types derived from unsupervised (BANKSY/nonspatial) clustering
                    reference: list,  # names of cell-types in the reference data
                    cell_type_colname: str = "cell type refined",
                    reference_dict: dict = dropviz_dict,  # mapping of refernece cell-type names to rctd_weights column
                    save_fig: bool = False,
                    **kwargs) -> None:
    '''
    Plots a heatmap of correlations between
     (1) cell types assigned by unsupervised clsutering (BANKSY or non-spatial or other)
     (2) cell type composition estimates by reference-guided deconvolution (e.g. RCTD)
         or expression of marker gene(s) from a reference dataset
    Note that (1) is discrete and (2) is continous. Hence we measure the
    point-biserial correlation (equivalent to pearson correlation for continuous values)
    '''
    options = {
        'file_path': "data/slide_seq/v1",
        'dpi': 1200
    }
    options.update(kwargs)

    # join the dataframes to get the intersection of cells from RCTD and unsupervised clustering
    df_intersect = rctd_coord.join(adata_spatial.obs, how="inner")
    print(f"Number of cells in both RCTD and adata: {len(df_intersect)}")

    rctd_df = rctd_coord.copy()

    for cell_type in reference_dict:
        # column in weights matrix this cell type corresponds to
        col_num = reference_dict[cell_type] - 1
        rctd_df[cell_type] = rctd_weights[:, col_num]

    rctd_df = rctd_df.loc[df_intersect.index, :]

    metagene_df_intersect = metagene_df.loc[df_intersect.index, :]

    adata_spatial_rctd = adata_spatial[df_intersect.index, :]
    adata_nonspatial_rctd = adata_nonspatial[df_intersect.index, :]

    for comparison_df, s in [(rctd_df, "RCTD"), (metagene_df_intersect, "Metagene")]:

        print(f"Comparing to {s}")

        for adata, s2 in [(adata_nonspatial_rctd, "non-spatial"), (adata_spatial_rctd, "spatial")]:

            correlation_grid = np.zeros((len(cell_types), len(reference)), )
            pval_grid = np.zeros_like(correlation_grid)

            for col, ref in enumerate(reference):
                for row, compare in enumerate(cell_types):
                    mask = (adata.obs[cell_type_colname] == compare).astype(float).values

                    # compute point biserial correlation (r value and p-value)
                    r, pval = pointbiserialr(mask, comparison_df.loc[:, ref].values)
                    # print(f"Pearson correlation of {compare} to {ref} : r = {r}, p-value = {pval:0.3f}")

                    correlation_grid[row, col] = r
                    pval_grid[row, col] = pval

            fig, axs = plt.subplots(nrows=1, ncols=2,
                                    figsize=(len(reference) * 3.5 + 3, len(cell_types) * 0.9))

            # heatmap of r values
            heat_map = sns.heatmap(correlation_grid, linewidth=1,
                                   vmin=-.9, vmax=.9, cmap="bwr", annot=True, fmt="0.2f",
                                   yticklabels=cell_types, xticklabels=reference,
                                   square=True, ax=axs[0])

            # heatmap of p-values
            pvals = sns.heatmap(pval_grid, linewidth=1,
                                vmin=0, vmax=1,  # cmap = "bwr",
                                annot=True, fmt="0.1e",
                                yticklabels=cell_types, xticklabels=reference,
                                square=True, ax=axs[1])
            fig.suptitle(f"comparing {s2} clustering to {s}")
            fig.tight_layout()

            if save_fig:
                fig.savefig(os.path.join(options['file_path'], f"corr_to_{s}_{s2}.eps"),
                            format='eps', dpi=options['dpi'])


def plot_self_vs_nbr(adata_spatial: anndata.AnnData,
                     adata_self: anndata.AnnData,
                     metagene: anndata._core.views.ArrayView,
                     metagene_nbr: anndata._core.views.ArrayView,
                     metagene_nbr_1: anndata._core.views.ArrayView,
                     cell_type: str,
                     cell_type2: str,
                     mask_1: str,
                     mask_2: str,
                     lambda_param: int,
                     top_n: int = 20,
                     cell_type_colname: str = "cell type refined"
                     ) -> None:
    '''
    2D plot of metagene expression (self as x-axis and neighbour as y-axis)
    for 2 cell types using seabourn 'jointplot'
    '''

    dfs = []  # list of dataframes for each of the 2 clusters

    for mask, colour, cluster in [(mask_1, "red", cell_type), (mask_2, "blue", cell_type2)]:
        self_expr = np.sqrt(1 - lambda_param) * metagene[mask]
        nbr_expr = np.sqrt(lambda_param) * metagene_nbr[mask]
        nbr_expr_2 = np.sqrt(lambda_param) * metagene_nbr_1[mask]

        #     metagene_plot = ax.scatter(self_expr, nbr_expr, c=colour, s=.5, alpha=.2)
        dfs.append(
            pd.DataFrame(data={'self': self_expr, 'nbr': nbr_expr, "cluster": cluster})
        )

    df_combined = pd.concat(dfs)
    # df_combined["cluster"] = df_combined["cluster"].astype("category")
    # display(df_combined.dtypes)

    sns.jointplot(data=df_combined, x="self", y="nbr", hue="cluster", kind="kde",
                  joint_kws={"thresh": 0.03, "levels": 10, "bw_adjust": 1.5, "fill": False})


def plot_self_vs_nbr_metagene_all(
        adata_spatial: anndata.AnnData,
        adata_self: anndata.AnnData,
        reference_cluster,  # all clusters will be plotted against this one
        lambda_param: int,
        cell_type_colname: str = "cell type refined",
        max_m: int = 0,  # m=0 is neighbour average, m=1 is AGF
        top_n: int = 20,  # number of top genes to sum
        savefig: bool = False, ) -> None:
    '''
    2D plot of metagene expression (self as x-axis and neighbour as y-axis)
    for all cell types vs a reference cell type
    done with matplotlib, without using seabourn 'jointplot'
    '''

    if adata_self is None:
        adata_self = _adata_filter_self(adata_spatial, run_rank_genes_groups=True,
                                        cell_type_colname=cell_type_colname, )

    for m in range(max_m + 1):

        # Plot all clusters vs one cluster
        # --------------------------------

        cell_types = [cluster for cluster in adata_self.obs[cell_type_colname].cat.categories
                      if cluster != reference_cluster]

        print(f"Comparing <{reference_cluster}>  with:  {cell_types}\n")

        DE_genes1 = get_DEgenes(adata_self, reference_cluster, top_n=top_n)
        mask_1 = adata_spatial.obs[cell_type_colname] == reference_cluster

        subplot_size = (5, 5)
        num_cols = 2
        num_rows = len(cell_types) // num_cols + 1
        print(num_rows, num_cols)
        figsize_x = (subplot_size[0] * num_cols) * 1.1
        figsize_y = subplot_size[1] * num_rows

        fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=True)
        grid = fig.add_gridspec(ncols=num_cols, nrows=num_rows)

        for n, cell_type in enumerate(cell_types):

            subgrid = grid[n // num_cols, n % num_cols].subgridspec(5, 5)
            ax = fig.add_subplot(subgrid[1:, :-1])

            DE_genes2 = get_DEgenes(adata_self, cell_type, top_n=top_n, verbose=False)
            metagene, metagene_nbr = get_metagene_difference(
                adata_spatial, DE_genes1, DE_genes2, m=m
            )

            mask_2 = adata_spatial.obs[cell_type_colname] == cell_type

            dfs = []

            min_x, max_x = 0.0, 0.0

            for mask, colour, cluster in [(mask_1, "red", reference_cluster), (mask_2, "blue", cell_type), ]:
                self_expr = np.sqrt(1 - lambda_param) * metagene[mask]
                nbr_expr = np.sqrt(lambda_param) * metagene_nbr[mask]

                # plot scatterplot of all cells in 2D expression coordinates
                metagene_plot = ax.scatter(
                    self_expr, nbr_expr, c=colour, s=.6, alpha=0.1, label=cluster
                )

                dfs.append(
                    pd.DataFrame(data={'self': self_expr, 'nbr': nbr_expr, "cluster": cluster})
                )

                min_x = min(np.amin(self_expr), min_x)
                max_x = max(np.amax(self_expr), max_x)

            df_combined = pd.concat(dfs)

            # plot 2D density plot (this is the main plot)
            # --------------------------------------------

            sns.kdeplot(
                data=df_combined, x="self", y="nbr", hue="cluster", palette=["red", "blue"],
                alpha=0.5,  # common_norm = False, #common_grid=True,
                bw_adjust=1.2, thresh=0.05, levels=12,
                # multiple="stack",
                ax=ax,
            )

            # ax.set_xlim([min_x, max_x])
            # ax.set_ylim([min_x, max_x])
            ax.set_xlabel("self")
            ax.set_ylabel("neighbour")
            # ax.set_title(f"{cell_type_2} vs\n{cell_type}", fontsize=8)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # plot 1D density plot on top of main plot
            # ----------------------------------------

            ax_top = fig.add_subplot(subgrid[:1, :-1], sharex=ax)
            sns.kdeplot(data=df_combined, x="self", hue="cluster", palette=["red", "blue"],
                        alpha=0.5, common_norm=False,  # linewidth = 0.2, #common_grid=True,
                        bw_adjust=1.2, legend=False,
                        # multiple="stack",
                        ax=ax_top)
            # ax_top.legend().set_visible(False)
            ax_top.axis("off")
            xmin, xmax = ax_top.get_xaxis().get_view_interval()
            ymin, _ = ax_top.get_yaxis().get_view_interval()
            ax_top.add_artist(
                plt.Line2D((xmin, xmax), (ymin, ymin), color='gray', linewidth=1, )
            )

            # plot 1D density plot on right of main plot
            # ------------------------------------------

            ax_right = fig.add_subplot(subgrid[1:, -1:], sharey=ax)
            sns.kdeplot(data=df_combined, y="nbr", hue="cluster", palette=["red", "blue"],
                        alpha=0.5, common_norm=False,  # linewidth = 0.2, #common_grid=True,
                        bw_adjust=1.2, legend=False,
                        # multiple="stack",
                        ax=ax_right)
            # ax_right.legend().set_visible(False)
            ax_right.axis("off")
            xmin, _ = ax_right.get_xaxis().get_view_interval()
            ymin, ymax = ax_right.get_yaxis().get_view_interval()
            ax_right.add_artist(
                plt.Line2D((xmin, xmin), (ymin, ymax), color='gray', linewidth=1, )
            )

        fig.tight_layout()

        if savefig:
            fig.savefig(f"metagene_nbr_plot_supp_m{m}.png", format='png', dpi=1200)
