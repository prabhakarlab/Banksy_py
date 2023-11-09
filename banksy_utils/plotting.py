"""
Functions for plotting

Nigel 3 dec 2020

Modified 16 May 2023
"""

from typing import Union, Tuple, Collection

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import seaborn as sns

from scipy.sparse import csr_matrix

import warnings

from banksy_utils.time_utils import timer
from banksy.labels import Label


@timer
def plot_edge_histogram(graph: csr_matrix,
                        ax: mpl.axes.Axes,
                        title: str = "edge weights",
                        bins: int = 100):
    """
    plot a histogram of the edge-weights a graph
    """
    counts, bins, patches = ax.hist(graph.data, bins=bins)

    median_dist = np.median(graph.data)
    mode_dist = bins[np.argmax(counts)]
    ax.axvline(median_dist, color="r", alpha=0.8)
    ax.axvline(mode_dist, color="g", alpha=0.8)
    ax.set_title("Histogram of " + title)

    print(f"\nEdge weights ({title}): "
          f"median = {median_dist}, mode = {mode_dist}\n")

    return median_dist, mode_dist


@timer
def plot_2d_embeddings(embedding: np.ndarray,
                       labels: Union[np.ndarray, list],
                       method_str: str = "2D projection",
                       space_str: str = "of some space",
                       xlabel: str = "Dimension 1",
                       ylabel: str = "Dimension 2",
                       ax: mpl.axes.Axes = None,
                       plot_cmap: bool = True,
                       cmap_name: str = "Spectral",
                       figsize: tuple = (8, 8),
                       title_fontsize: int = 12,
                       **kwargs,
                       ) -> None:
    """
    plot embeddings in 2D, with coloured labels for each point

    :param embedding: the embedding matrix (only 1st 2 columns are used)
    :param labels: integer labels for each point
    """
    num_labels = len(np.unique(labels))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax
        fig = ax.get_figure()

    scatterplot = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=labels,
        cmap=cmap_name,
        s=0.2, alpha=0.5,
        **kwargs,
    )

    # ax.set_aspect("equal", "datalim")
    # ax.set_aspect("datalim")
    ax.set_title(f"{method_str} {space_str}",
                 fontsize=title_fontsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    print(f"number of labels: {num_labels}")

    if plot_cmap:
        fig.colorbar(
            scatterplot,
            boundaries=np.arange(num_labels + 1) - 0.5
        ).set_ticks(np.arange(num_labels))


@timer
def plot_graph_weights(locations,
                       graph,
                       theta_graph=None,  # azimuthal angles
                       max_weight=1,
                       markersize=1,
                       figsize=(8, 8),
                       title: str = None,
                       flip_yaxis: bool = False,
                       ax=None,
                       ) -> None:
    """
    Visualize weights in a spatial graph, 
    heavier weights represented by thicker lines
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    edges, weights, theta = [], [], []

    if theta_graph is not None:
        assert isinstance(theta_graph, csr_matrix)
        assert theta_graph.shape[0] == graph.shape[0]

    for start_node_idx in range(graph.shape[0]):

        ptr_start = graph.indptr[start_node_idx]
        ptr_end = graph.indptr[start_node_idx + 1]

        for ptr in range(ptr_start, ptr_end):
            end_node_idx = graph.indices[ptr]

            # append xs and ys as columns of a numpy array
            edges.append(locations[[start_node_idx, end_node_idx], :])
            weights.append(graph.data[ptr])
            if theta_graph is not None:
                theta.append(theta_graph.data[ptr])

    print(f"Maximum weight: {np.amax(np.array(weights))}\n")
    weights /= np.amax(np.array(weights))

    if theta_graph is not None:
        norm = mpl.colors.Normalize(vmin=min(theta), vmax=max(theta), clip=True)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap="bwr")
        c = [mapper.to_rgba(t) for t in theta]
    else:
        c = "C0"

    line_segments = LineCollection(
        edges, linewidths=weights * max_weight, linestyle='solid', colors=c, alpha=0.7)
    ax.add_collection(line_segments)

    ax.scatter(locations[:, 0], locations[:, 1], s=markersize, c="gray", alpha=.6, )

    if flip_yaxis:  # 0 on top, typical of image data
        ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_aspect('equal', 'datalim')

    if title is not None:
        ax.set_title(title)


def plot_continuous(x, y, c, ax,
                    spot_size: int = 0.3,
                    vmax: float = None,
                    cmap: str = "Blues",
                    title: str = None,
                    plot_cbar: bool = True,
                    ) -> None:
    """
    Plot continous-valued location plot (e.g. markergene, RCTD weights)
    """
    scatterplot = ax.scatter(x, y, c=c,
                             s=spot_size, alpha=1.0,
                             cmap=cmap,
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


def plot_cluster_subset(cluster_subset: list,
                        colours_dict: dict,
                        adata,
                        ax,
                        title: str = None,
                        umap_name: str = "umap",
                        spot_size: float = 0.4,
                        plot_umap: bool = False,
                        ):
    """
    Plot a subset of clusters using specified colours from colours_dict
    """

    def plot_embedding(embedding, ax):

        for cluster in adata.obs["cell type"].cat.categories:

            if cluster not in cluster_subset:
                c = "whitesmoke"
                locations = adata.obsm[embedding][adata.obs["cell type"] == cluster, :]
                scatterplot = ax.scatter(locations[:, 0], locations[:, 1], c=c,
                                         s=spot_size, alpha=1.0)

        for cluster in adata.obs["cell type"].cat.categories:

            if cluster in cluster_subset:
                c = colours_dict[cluster]
                locations = adata.obsm[embedding][adata.obs["cell type"] == cluster, :]
                scatterplot = ax.scatter(locations[:, 0], locations[:, 1], c=c,
                                         s=spot_size, alpha=1.0)

    # plot locations
    plot_embedding("coord_xy", ax)
    ax.set_aspect('equal', 'datalim')
    if title is not None:
        ax.set_title(title, fontsize=20, fontweight="bold", )
    # ax.set_ylim(ax_locs.get_ylim()[::-1])
    ax.axis("off")

    # plot  UMAP as inset
    if plot_umap:
        axins = ax.inset_axes([0.9, 0.0, 0.25, 0.25])
        plot_embedding(umap_name, axins)
        axins.set_aspect('equal', 'datalim')
        # axins.axis("off")
        axins.set_ylabel("UMAP2", fontsize=8, fontweight="bold", )
        axins.set_xlabel("UMAP1", fontsize=8, fontweight="bold", )
        axins.spines['right'].set_visible(False)
        axins.spines['top'].set_visible(False)
        axins.xaxis.set_ticks([])
        axins.yaxis.set_ticks([])


@timer
def plot_genes(genes: list,
               df: pd.DataFrame,
               results_df: pd.DataFrame = None,  # only used for plotting spatialDE length-scales
               x_colname: str = "X",
               y_colname: str = "  Y",
               plots_per_row: int = 4,
               colormap: str = "Blues",
               vmin: float = None,
               vmax: float = None,
               take_log: bool = False,
               main_title: str = None,
               verbose: bool = True,
               ):
    """
    Plot the count distributions of a list of genes
    """

    num_genes = len(genes)
    num_rows = len(genes) // plots_per_row + 1

    print(
        f"Number of genes: {num_genes}\n"
        f"Plots per row: {plots_per_row}\n"
        f"Number of rows: {num_rows}"
    )

    fig = plt.figure(figsize=(6 * plots_per_row, 4 * num_rows))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    df_columns = df.columns

    sns.set_style("white")

    for gene_index, gene in enumerate(genes):

        if gene not in df_columns:
            warnings.warn(f"Gene {gene} not found in dataframe!")
            continue

        ax = fig.add_subplot(num_rows, plots_per_row, gene_index + 1)
        x = df[x_colname].values
        y = df[y_colname].values
        counts = df[gene].values

        if take_log:
            counts = np.log10(counts + 1)

        ax.scatter(
            x, y, s=10, c=counts,
            cmap=colormap,
            vmin=vmin, vmax=vmax,
            alpha=0.9,
        )

        df.plot.scatter(
            x=x_colname, y=y_colname, s=10, c=gene,
            colormap=colormap,
            vmin=vmin, vmax=vmax, alpha=0.9,
            ax=ax
        )

        if results_df is not None:

            # Get the set of p-values (1 for each length)
            # -------------------------------------------

            gene_results = results_df.loc[results_df["g"] == gene, ["l", "pval"]]
            gene_results.sort_values(by="l", inplace=True)

            pvals = -1 * np.log10(gene_results["pval"].values)
            # pvals = gene_results["pval"].values
            # normalize to the highest p-value
            pvals /= np.max(pvals)
            num_boxes = len(pvals)

            y_lower, y_upper = ax.get_ylim()
            x_lower, x_upper = ax.get_xlim()
            y_span = np.abs(y_upper - y_lower)
            x_span = np.abs(x_upper - x_lower)
            y_step = y_span / num_boxes
            x_step = x_span / 20

            if verbose:
                print(f"{gene_results}\n"
                      f"P-values:\n{pvals}"
                      f"\n y lower limit : {y_lower}, y upper limit : {y_upper}"
                      f"\n x lower limit : {x_lower}, x upper limit : {x_upper}")

            cmap = plt.cm.get_cmap('viridis')

            for box_num in range(num_boxes):
                rect = Rectangle((x_upper, y_lower + y_step * box_num), x_step, y_step,
                                 linewidth=1, facecolor=cmap(pvals[box_num]), edgecolor="gray",
                                 clip_on=False, )
                ax.add_patch(rect)

        # Title
        # -----

        if "blank" in gene or "Blank" in gene:
            title_colour = "r"
        else:
            title_colour = "b"

        ax.set_title(gene,
                     fontdict={'fontsize': 20,
                               'fontweight': 'bold',
                               'color': title_colour, }
                     )
        # plt.gca().invert_yaxis()
        ax.axis('equal')

    if main_title is not None:
        fig.suptitle(main_title, fontsize=25,
                     # fontproperties={'fontsize': 25,
                     #                 'fontweight': 'bold',
                     #                 }
                     )

    plt.show()


def plot_labels_seperately(label: Label,
                           locations: np.ndarray,
                           embeddings: np.ndarray = None,
                           embedding_str: str = "UMAP",
                           plots_per_row: int = 3,
                           cmap_name: str = "Spectral",
                           colour_list: list = None,
                           max_id: int = None,
                           default_colour: str = "tab:red",
                           background_colour: str = "whitesmoke",
                           subplot_size: Tuple[float, float] = (5, 5),
                           spot_size: float = 0.2,
                           alpha: float = 1,
                           flip_axes: bool = False,
                           show_axes: bool = False,
                           save_fig: bool = False,
                           save_path: str = None,
                           verbose: bool = False,
                           ) -> None:
    """
    Plot each label's locations individually in subplots
    (nonlabel locations in gray)
    If embeddings is not None, UMAP or other embedding will be plotted.

    label: Label object
    locations: x/y coordinates
    embeddings: embedding coordinates (2D) e.g. UMAP or t-SNE
    embedding_str: type of embedding
    cmap_name: name of standard colourmap or reference to custom colourmap
    max_id: highest cluster ID if not present in current clusters (e.g. when matching labels)
    default_colour: colour for clusters if no colourmap given
    background_colour: colour for spots not in cluster
    subplot_size: size of individual subplot
    flip_axes: flip y axis
    show_axes: show x and y axis borders
    """

    num_rows = label.num_labels // plots_per_row + 1
    figsize_x = (subplot_size[0] * plots_per_row) * 1.1
    if embeddings is not None:
        figsize_x *= 2
    figsize_y = subplot_size[1] * num_rows

    if verbose:
        print(f"\nGenerating {num_rows} by {plots_per_row} "
              f"grid of plots (figsize = {figsize_y}, {figsize_x})\n"
              f"Plotting labels: {label.ids}, "
              f"max ID: {max(max_id, label.max_id)}\n")

    fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=True)

    if embeddings is not None:
        grid = fig.add_gridspec(ncols=2 * plots_per_row, nrows=num_rows)
    else:
        grid = fig.add_gridspec(ncols=plots_per_row, nrows=num_rows)

    if cmap_name is not None:
        cmap = mpl.cm.get_cmap(cmap_name)

    if max_id is None:
        max_id = label.max_id

    # Plot each label
    # ---------------

    for n, label_id in enumerate(label.ids):

        grid_y = n // plots_per_row
        grid_x = n % plots_per_row

        if verbose:
            print(f"Plotting grid position: {grid_y}, {grid_x}")

        if embeddings is not None:

            ax = fig.add_subplot(grid[grid_y, 2 * grid_x])
            ax_umap = fig.add_subplot(grid[grid_y, 2 * grid_x + 1])

        else:

            ax = fig.add_subplot(grid[grid_y, grid_x])

        onehot = label.get_onehot()
        label_mask = np.squeeze(onehot[n, :].toarray().astype(bool))

        if cmap_name is None:
            if colour_list is not None:
                assert label_id < len(colour_list)
                c = colour_list[label_id]
            else:
                c = default_colour
        else:
            # matplotlib says to add an additional axis (1 x 3 shape)
            # print(f"{label_id + 0.5}/{max_id + 1}={(label_id + 0.5) / (max_id + 1)}")
            c = np.expand_dims(cmap((label_id + 0.5) / (max_id + 1)), axis=0)

        other_spots = ax.scatter(
            locations[~label_mask, 0], locations[~label_mask, 1],
            c=background_colour, s=spot_size, alpha=alpha,
        )

        cluster_spots = ax.scatter(
            locations[label_mask, 0], locations[label_mask, 1],
            c=c, s=spot_size, alpha=alpha,
        )

        if flip_axes:
            ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_aspect('equal', 'datalim')
        ax.set_title(f'{label_id}', fontsize=4 * subplot_size[0], fontweight="bold")

        # Plot UMAP
        # ---------

        if embeddings is not None:
            umap_other = ax_umap.scatter(
                embeddings[~label_mask, 0], embeddings[~label_mask, 1],
                c=background_colour, s=spot_size, alpha=alpha,
            )

            umap_cluster = ax_umap.scatter(
                embeddings[label_mask, 0], embeddings[label_mask, 1],
                c=c, s=spot_size, alpha=alpha,
            )

        # ax_umap.set_aspect('equal', 'datalim')
        ax_umap.set_title(f'{label_id} {embedding_str}',
                          fontsize=4 * subplot_size[0], fontweight="bold")
        ax_umap.set_xlabel(embedding_str + "1")
        ax_umap.set_ylabel(embedding_str + "2")

        if not show_axes:
            ax.axis("off")
            ax_umap.axis("off")

    if save_fig and isinstance(save_path, str):
        fig.savefig(save_path)


def plot_label_subset(labels_to_plot: Collection,
                      label: Label,
                      locations: np.ndarray,
                      embeddings: np.ndarray,
                      cmap_name: str = "Spectral",
                      max_id: int = None,
                      figsize: Tuple[float, float] = (10, 5),
                      verbose: bool = True,
                      save_fig: bool = False,
                      save_path: str = None,
                      ) -> None:
    """
    Plot a subset of labels by colour
    Remaining labels will be shown in gray
    :param labels_to_plot: subset of labels to plot
    :param label: Label object
    :param locations: spatial coordinates
    :param embeddings: UMAP/PCA
    :param max_id: max label id (if None, use max id of Label object)
    """

    cmap = mpl.cm.get_cmap(cmap_name)

    if max_id is None:
        max_id = label.max_id

    labels_to_plot = [label_id for label_id in labels_to_plot
                      if label_id in label.ids]

    if len(labels_to_plot) == 0:
        warnings.warn("labels provided not found in Label set")
        return

    if verbose:
        print(f"plotting labels: {labels_to_plot}")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    all_labels_mask = np.zeros((label.num_samples,), dtype=bool)

    # Plot each label by colour
    # -------------------------

    for label_id in labels_to_plot:
        n = list(label.ids).index(label_id)
        print(f"\nPlotting label {label_id} (row {n})")

        label_mask = np.squeeze(label.onehot[n, :].toarray().astype(bool))

        all_labels_mask = all_labels_mask | label_mask

        c = np.expand_dims(cmap(label_id / max_id), axis=0)

        ax[0].scatter(
            locations[label_mask, 0],
            locations[label_mask, 1],
            c=c, s=0.8, alpha=1,
        )

        ax[1].scatter(
            embeddings[label_mask, 0],
            embeddings[label_mask, 1],
            c=c, s=0.8, alpha=1,
        )

    # Plot remaining locations in Gray
    # --------------------------------

    ax[0].scatter(
        locations[~all_labels_mask, 0],
        locations[~all_labels_mask, 1],
        c="whitesmoke", s=0.8, alpha=1,
    )

    ax[1].scatter(
        embeddings[~all_labels_mask, 0],
        embeddings[~all_labels_mask, 1],
        c="whitesmoke", s=0.8, alpha=1,
    )

    if save_fig and isinstance(save_path, str):
        fig.savefig(save_path)


if __name__ == "__main__":
    print("test")
