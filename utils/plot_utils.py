import numpy as np
from numpy.random import randint
import pandas as pd

import os, gc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from anndata import AnnData
import anndata
from utils.plotting import plot_edge_histogram, plot_graph_weights
from utils.util_data import *
from utils.cluster_utils import pad_clusters, get_DEgenes, get_metagene_difference
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
    ### Edit bin_size_for_plts as a option, remove bins1,2
    options = {'figsize': (15,4),
               'bin_options': ['auto', 'auto', 'auto', 'auto']}
    
    options.update(kwargs)
    bin_sizes = options['bin_options']

    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    sns.histplot(adata.obs["total_counts"], kde=False, bins=bin_sizes[0], ax=axs[0]).set(title='Total Counts (unbounded)')
    sns.histplot(adata.obs["total_counts"][adata.obs["total_counts"] < total_counts_cutoff], kde=False, bins=bin_sizes[1], ax=axs[1]).set(title=f'Total Counts (bounded)')
    sns.histplot(adata.obs["n_genes_by_counts"], kde=False, bins=bin_sizes[2], ax=axs[2]).set(title='n-genes by counts (unbounded)')
    sns.histplot(adata.obs["n_genes_by_counts"][
        (adata.obs["n_genes_by_counts"] <  n_genes_high_cutoff) & (adata.obs["n_genes_by_counts"] >  n_genes_low_cutoff)
    ], kde=False, bins=bin_sizes[3], ax=axs[3]).set(title='n-genes by counts (bounded)')
    fig.tight_layout()
    gc.collect()


def  plot_cell_positions(adata: anndata.AnnData,
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
        'fig_size': (8,8),
        's':0.2,
        'c1':  'red',
        'c2':'slateblue',
        'label1':'Cells Removed',
        'label2': 'Cells Filtered',
        'add_circle': False,
        'puck_center': (3330,3180),
        'puck_radius' : 2550
    }

    options.update(kwargs)
    print(options)

    fig, ax = plt.subplots(figsize = options['fig_size'])
    ax.scatter(raw_x, raw_y, s = options['s'], c = options['c1'], label= options['label1'])
    ax.scatter(adata.obs[coord_keys[0]], adata.obs[coord_keys[1]], s = options['s'], c = options['c2'], label= options['label2'])
    ax.set_title("Cell positions")
    if options['add_circle']:
        circle = plt.Circle(options['puck_center'],options['puck_radius'], color='g', fill=False)
        ax.add_patch(circle)

    ax.axis("equal")
    gc.collect()


def plot_edge_histograms(distance_graph: sparse.csr_matrix,
                         weights_graph: sparse.csr_matrix,
                         decay_type: str,
                         m: int,
                         **kwargs):
    '''An utility function to plot subplots of each histograms
    using the plot_edge_histogram function for each distance and weights'''

    options = {'rows': 1, 'cols':2, 'figsize': (10,5)}
    options.update(kwargs)

    # Plots the histograms of distance and weights each m
    fig, ax = plt.subplots(nrows=options['rows'], ncols=options['cols'], figsize= options['figsize'])
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
    options = {'max_weight': 0.5, 'markersize':1.5, 'figsize': (15,8)}
    options.update(kwargs)
    
    for m in range(max_m+1):

        # Plots the weighted graph for each iterations of m
        plot_graph_weights(
                adata.obsm[coord_keys[2]], 
                processing_dict[nbr_weight_decay]["weights"][m],
                theta_graph=theta_graph,
                max_weight=options["max_weight"],
                markersize=options["markersize"],
                figsize=options["figsize"],
                title = f"{fig_title}, m = {m}"
            )

        ax = plt.gca()
        ax.axis("equal")


def plot_theta_graph(adata: anndata.AnnData,
                     theta_graph: sparse.csr_matrix,
                     coord_keys: Tuple[str],
                     **kwargs):
    
    '''Shows the theta graph of the neigbour hood of a random cell'''

    options = {'alpha': 0.4, 'figsize': (6,6)}
    options.update(kwargs)

    key = coord_keys[2]
    for cell in randint(1, adata.shape[0], 1):
        fig, ax = plt.subplots(figsize=options['figsize'])
        ax.plot(adata.obsm[key][cell,0], adata.obsm[key][cell,1], "bo", alpha=options['alpha'])
        ptr_start = theta_graph.indptr[cell]
        ptr_end = theta_graph.indptr[cell+1]
        for nbr, angle in zip(theta_graph.indices[ptr_start:ptr_end],theta_graph.data[ptr_start:ptr_end]):
            ax.plot(adata.obsm[key][nbr,0], adata.obsm[key][nbr,1], "ro", alpha = options['alpha'])
            ax.annotate(f"{angle/np.pi*180:0.1f}",(adata.obsm[key][nbr,0], adata.obsm[key][nbr,1]))
            ax.axis("equal")

    gc.collect()

def plot_DE_genes(adata: anndata.AnnData,
                  adata_spatial: anndata.AnnData,
                  adata_nonspatial: anndata.AnnData,
                  plot_heatmap = False,
                  plot_dot_plot = False
                  ) -> None:
    '''
    Function to plot the Differentially-Expressed (DE) genes per group
    
    Returns adata_nonspatial, adata_spatial subsequent plotting if needed
    '''

    for adata in [adata_spatial, adata_nonspatial]:
            
        # Plot Differentially Expressed (DE) genes per group
        # --------------------------------------------------
        sc.tl.rank_genes_groups(adata, groupby='cell type', method='wilcoxon', verbose=True)
        sc.tl.dendrogram(adata, groupby='cell type')

        if plot_heatmap:
            sc.pl.rank_genes_groups_heatmap(adata, n_genes=6, 
                                            vmin=-3, vmax=3, cmap='bwr', 
                                            swap_axes=True)
        if plot_dot_plot:
            sc.pl.rank_genes_groups_dotplot(adata, n_genes=5)

    gc.collect()


def plot_DE_genes_refined(adata_spatial_filtered: anndata.AnnData,
                          save_fig: bool = False,
                          file_path: str = "",
                          save_name: str = "",
                          **kwargs):
    '''
    Function to plot DE genes for the refined anndata

    Args:
        adata_spatial_filtered

    Optional args:
        save_fig (default = False)
        kwargs: optional specifications for saving figure
    '''
    options = {
        'fig_size': (12,5),
        'save_format':'eps',
        'dpi': 1200
    }
    options.update(kwargs)

    adata_self = adata_spatial_filtered[:,adata_spatial_filtered.var["is_nbr"]==False].copy()

    sc.tl.rank_genes_groups(adata_self, groupby='cell type refined', method='wilcoxon', verbose=True)
    sc.tl.dendrogram(adata_self, groupby='cell type refined')

    sc.pl.rank_genes_groups_heatmap(adata_self, n_genes=5, 
                                    vmin=-3, vmax=3, cmap='bwr', 
                                    swap_axes=True)

    fig, ax = plt.subplots(figsize=options['fig_size'])

    sc.pl.rank_genes_groups_dotplot(adata_self, n_genes=4, 
                                    groups = None,
                                    ax=ax)
    
    if save_fig:
        save_location = os.path.join(file_path, save_name)
        fig.savefig(save_location, format=options['save_format'], dpi=options['dpi'])

    return adata_self

def plot_connnection_grid(
        processing_dict: dict,
        adata_nonspatial: anndata.AnnData,
        adata_spatial: anndata.AnnData,
        cell_types: list,
        save_fig: bool = False,
        file_path: str  = "",
        save_name: str = "",
        **kwargs):
    
    #### Add weights as an arguement;
    # weights = processing_dict["scaled_gaussian"]["weights"][0].copy() 
    '''
    Plots connection grid for anndata
    '''
    options ={
        'fig_size': (12,5),
        'save_format':'eps',
        'dpi': 1200
    }
    options.update(kwargs)

    print(f"All refined cell types: {adata_spatial.obs['cell type refined'].cat.categories}")

    fig, axs = plt.subplots(nrows = 1, ncols = 2, 
                            figsize=(len(cell_types)*2.4 + 3, len(cell_types)*1.2))

    connection_grids = []

    max_nondiag = 0.0

    for adata in [adata_nonspatial, adata_spatial]:
        
        one_hots = []
        for cell_type in cell_types:
            one_hots.append(np.array(adata.obs["cell type refined"]==cell_type))
        
        # one hot labels
        # --------------
        one_hot_array = np.stack(one_hots, axis = 0)
        #print(one_hot_array.shape, one_hot_array.dtype)
        
        # adjacency matrix
        # ----------------
        # use the m=0 weights matrix (default 15 nearest nbrs)
        weights = processing_dict["scaled_gaussian"]["weights"][0].copy()
        weights.data = np.ones_like(weights.data)
        print(weights.data, weights.shape)
        
        # Unnormalized summed connections
        # -------------------------------
        connection_grid = one_hot_array @ weights @ one_hot_array.T
        print(connection_grid)
        
        # Normalize
        # ---------
        connection_grid /= np.diag(connection_grid)[:,np.newaxis]
        print(connection_grid)

        connection_grids.append(connection_grid)
        
        # find max non-diagonal by zeroing out diagonal values (all diagonals = 1)
        connections_zerod = connection_grid.copy()
        np.fill_diagonal(connections_zerod, 0)
        max_nondiag = max(max_nondiag, np.amax(connections_zerod))

    for n, connection_grid in enumerate(connection_grids):
        
        heat_map = sns.heatmap(connection_grid, 
                            linewidth = 1, 
                            vmax = 1,
                            cmap = "viridis", annot = True, fmt="0.2f",
                            yticklabels=cell_types, xticklabels=cell_types, 
                            square=True, ax=axs[n])
    gc.collect()  
    fig.tight_layout()

    if save_fig:
        save_location = os.path.join(file_path, save_name)
        fig.savefig(save_location, format=options['save_format'], dpi=options['dpi'])


def plot_clusters(adata_spatial: anndata.AnnData,
                  adata_nonspatial: anndata.AnnData,
                  pca_dims: list,
                  color_dict: dict = colours_dict,
                  save_fig: bool = False,
                  file_path: str = "",
                  save_name: str = "",
                  **kwargs):
    '''
    Plots clusters for spatial and nonspatial graphs

    Options (kwargs): 
        'fig_size': (8,6),
        'save_format':'eps',
        'dpi': 1200
    '''
    options ={
        'fig_size': (8,6),
        'save_format':'eps',
        'dpi': 1200,
        'key': 'coord_xy'
    }

    options.update(kwargs)
    key = options['key']

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=options['fig_size'], constrained_layout=True)

    for n, adata in enumerate([adata_spatial, adata_nonspatial]):
        
        ax_loc = axs[n]
        
        umap_name = f"reduced_pc_{pca_dims[0]:2d}_umap"
        
        for layer in adata.obs["cell type"].cat.categories:
                    
            if layer in colours_dict:
                c = colours_dict[layer]
                locations = adata.obsm[key][adata.obs["cell type"]==layer,:]
                umap_embedding = adata.obsm[umap_name][adata.obs["cell type"]==layer,:]
                loc_plot = ax_loc.scatter(locations[:, 0], locations[:, 1], c=c, s=.2, alpha=1.0)
        
        ax_loc.set_aspect('equal', 'datalim')
        ax_loc.axis("off")
       
    if save_fig:
        save_location = os.path.join(file_path, save_name)
        fig.savefig(save_location, format=options['save_format'], dpi=options['dpi'])


def plot_continous_for_clusters(
        rctd_coord: pd.DataFrame, 
        rctd_weights: pd.DataFrame) -> None:
    
    '''Generate continuous plot for weights in each cluster'''

    for clust_num in range(rctd_weights.shape[1]):
        fig, ax = plt.subplots()
        weights = np.array(rctd_weights[:,clust_num]).flatten()
        plot_continuous(rctd_coord.loc[:,"y"], rctd_coord["x"], weights, ax, vmax = 1, title = str(clust_num))


def plot_markergene_sets(metagene_df: pd.DataFrame,
                        rctd_coord: pd.DataFrame, 
                        rctd_weights: pd.DataFrame,
                        coord_keys,
                        save_fig: bool = False,
                        file_path: str = "",
                        save_name: str = "",
                        **kwargs) -> None:
    '''
    Plots all the marker gene sets for scRNA-seq clusters 
    
    '''
    options = {
        'subplot_size': (3,3),
        'save_format':'png',
        'dpi': 1200
    }
    options.update(kwargs)

    subplot_size = options['subplot_size']
    num_cols = 2
    num_rows = len(dropviz_dict) // num_cols + 1
    figsize_x = (subplot_size[0] * num_cols) * 2.1
    figsize_y = subplot_size[1] * num_rows

    fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=True)
    grid = fig.add_gridspec(ncols=num_cols*2, nrows=num_rows)

    for n, layer in enumerate(dropviz_dict):
        
        ax = fig.add_subplot(grid[n // num_cols, 2*(n % num_cols)])
        
        clust_num = dropviz_dict[layer] - 1
        weights = np.array(rctd_weights[:,clust_num]).flatten()
        
        plot_continuous(rctd_coord["x"], rctd_coord["y"], weights, 
                        ax, vmax = 1, title = layer+"\nRCTD weights",
                        spot_size = 0.2,
                    )
        
        if layer in markergenes_dict:

            ax = fig.add_subplot(grid[n // num_cols, 2*(n % num_cols) + 1])
            
            c = metagene_df[layer]
            if isinstance(c, pd.Series) and c.isnull().any():
                c.fillna(0,inplace=True)

            plot_continuous(metagene_df[coord_keys[0]], metagene_df[coord_keys[1]], c, 
                            ax, vmax = 1, title = layer+"\nmarker genes",
                            spot_size = 0.2,
                        )
    if save_fig:
        save_location = os.path.join(file_path, save_name)
        fig.savefig(save_location, format=options['save_format'], dpi=options['dpi'])



def plot_continuous(x, y, c, ax,
                    spot_size:int = 0.3,
                    vmax = None,
                    cmap = "Blues",
                    title = None,
                    plot_cbar:bool = True,
                    ):
    """
    Plot continous-valued location plot (e.g. markergene, RCTD weights)
    """
    ### This is to prevent any NaN values the color argument for the scatter plot, which will throw an error
    

    scatterplot = ax.scatter(x, y, c=c,
                             s=spot_size, alpha=1.0, cmap = "Blues", 
                             vmax=vmax, vmin=0,
                            )

    ax.set_aspect('equal', 'datalim')
    if title is not None:
        ax.set_title(title, fontsize=8, fontweight="bold",)
    #ax.set_ylim(ax_locs.get_ylim()[::-1])
    ax.axis("off")
    
    if plot_cbar:
        cax = ax.inset_axes([0.8, 0.0, 0.16, 0.04])
        #cax.axis("off")
        ax.figure.colorbar(scatterplot, cax=cax, orientation='horizontal')


def plot_cluster_subset(cluster_subset:list,
                        colours_dict: dict,
                        adata: anndata.AnnData,
                        ax: plt.Axes,
                        title:str=None,
                        umap_name:str="umap",
                        spot_size:float = 0.4,
                        plot_umap:bool=False,
                       ):
    """
    Plot a subset of clusters using specified colours from colours_dict
    """

    # plot locations
    plot_embedding("coord_xy", ax, adata, cluster_subset, spot_size, colours_dict)
    ax.set_aspect('equal', 'datalim')
    if title is not None:
        ax.set_title(title, fontsize=20, fontweight="bold",)
    #ax.set_ylim(ax_locs.get_ylim()[::-1])
    ax.axis("off")
    
    # plot  UMAP as inset
    if plot_umap:
        axins = ax.inset_axes([0.9, 0.0, 0.25, 0.25])
        plot_embedding(umap_name, axins, adata, cluster_subset, spot_size)
        axins.set_aspect('equal', 'datalim')
        #axins.axis("off")
        axins.set_ylabel("UMAP2", fontsize=8, fontweight="bold",)
        axins.set_xlabel("UMAP1", fontsize=8, fontweight="bold",)
        axins.spines['right'].set_visible(False)
        axins.spines['top'].set_visible(False)
        axins.xaxis.set_ticks([])
        axins.yaxis.set_ticks([])

def plot_embedding(embedding, ax,
                   adata,
                   cluster_subset,
                   spot_size,
                   colours_dict):

    for cluster in adata.obs["cell type"].cat.categories:

        if cluster not in cluster_subset:
            
            c = "whitesmoke"
            locations = adata.obsm[embedding][adata.obs["cell type"]==cluster,:]
            scatterplot = ax.scatter(locations[:, 0], locations[:, 1], c=c, 
                                        s=spot_size, alpha=1.0)
    
    for cluster in adata.obs["cell type"].cat.categories:
        
        if cluster in cluster_subset:
            
            c = colours_dict[cluster]
            locations = adata.obsm[embedding][adata.obs["cell type"]==cluster,:]
            scatterplot = ax.scatter(locations[:, 0], locations[:, 1], c=c, 
                                        s=spot_size, alpha=1.0)

def plot_weights_comparsion(adata_spatial: anndata.AnnData,
                            adata_nonspatial: anndata.AnnData,
                            adata_allgenes: anndata.AnnData,
                            layers: list,
                            rctd_coord: pd.DataFrame,
                            rctd_weights: pd.DataFrame,
                            save_fig = False,
                            file_path: Union[str, None] = None,
                            save_png_name: Union[str, None] = None,
                            save_eps_name: Union[str, None] = None,
                            **kwargs) -> None:
    '''
    Plots the comparsion between spatial, nonspatial and RCTD weights
    
    '''
    options = {
        'subplot_size': (3,3),
        'save_eps_name':  "subplots_layers_sseqcerebellum.eps",
        'dpi': 1200
    }
    options.update(kwargs)
        
    subplot_size = options['subplot_size']
    num_cols = len(layers)
    num_rows = 4
    figsize_x = (subplot_size[0] * num_cols) * 1.1
    figsize_y = subplot_size[1] * num_rows

    print(f"\nGenerating {num_rows} by {num_cols} "
        f"grid of plots (figsize = {figsize_y}, {figsize_x})\n")

    fig = plt.figure(figsize=(figsize_x, figsize_y),
                    constrained_layout=True)
    grid = fig.add_gridspec(ncols=num_cols, nrows=num_rows)

    for col, layer in enumerate(layers):
        
        # BANKSY
        # ------
        
        ax = fig.add_subplot(grid[0, col])
        plot_cluster_subset(layer[0], colours_dict_fine, adata_spatial, 
                            ax, umap_name="reduced_pc_16_umap", 
                            spot_size=0.2, plot_umap=False)  
        
        # non-spatial
        # -----------
        
        ax = fig.add_subplot(grid[1, col])
        plot_cluster_subset(layer[1], colours_dict_fine, adata_nonspatial, 
                            ax, umap_name="reduced_pc_16_umap", 
                            spot_size=0.2, plot_umap=False)
        
        # Plot scRNAseq metagene (from top DE genes)
        # ------------------------------------------
        
        
        ax = fig.add_subplot(grid[2, col])
        
        weights = np.zeros((len(rctd_coord),))
        
        for dv_clust in layer[2]:
            
            clust_num = dropviz_dict[dv_clust] - 1
            weights += np.array(rctd_weights[:,clust_num]).flatten()
            
        plot_continuous(rctd_coord.loc[:,"x"], rctd_coord["y"], weights, 
                        ax, vmax = 1, plot_cbar = False)
            
        
        # Plot scRNAseq metagene (from top DE genes)
        # ------------------------------------------
        
        ax = fig.add_subplot(grid[3, col])

        markers = []
        
        for sc_clust in layer[3]:
            
            markers_temp = [m for m in markergenes_dict[sc_clust] if m in adata_allgenes.var.index]
            
            if markers_temp < markergenes_dict[sc_clust]:
                print(f"{len(markergenes_dict[sc_clust])-len(markers_temp)} "
                    f"scRNA-seq DE genes in {sc_clust} absent/filtered from slideseq dataset")
                
            markers += markers_temp
        
        print(f"{len(markers)} scRNAseq DE markers for clusters {layer[2]}")
        
        layer_slice = adata_allgenes[:, markers]
        metagene = np.mean(layer_slice.X, axis = 1)
        
        plot_continuous(adata_spatial.obs["xcoord"], adata_spatial.obs["ycoord"], metagene, 
                        ax, vmax = 1, plot_cbar = False)
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
                    save_fig = False,
                    **kwargs) -> None:
    '''
    Plots the comparsion between spatial, nonspatial and RCTD weights
    
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
    for cell_type in dropviz_dict:
        col_num = dropviz_dict[cell_type] - 1 # column in weights matrix this cell type corresponds to
        rctd_df[cell_type] = rctd_weights[:,col_num]

    # combine Purkinje and Bergmann weights

    for df in [rctd_df,metagene_df]:
        df["Purkinje/Bergmann"] = df["PurkinjeNeuron_Pcp2"] + df["BergmannGlia_Gpr37l1"]

    rctd_df = rctd_df.loc[df_intersect.index,:]

    metagene_df_intersect = metagene_df.loc[df_intersect.index,:]

    adata_spatial_rctd = adata_spatial[df_intersect.index,:]
    adata_nonspatial_rctd = adata_nonspatial[df_intersect.index,:]

    comparsion_list = [(rctd_df,"RCTD"), (metagene_df_intersect,"Metagene")]
    
    for comparison_df, s in comparsion_list:

        print(f"Comparing to {s}")

        for adata, s2 in [(adata_nonspatial_rctd,"non-spatial"), (adata_spatial_rctd,"spatial")]:

            correlation_grid = np.zeros((len(cell_types),len(reference)),)
            pval_grid = np.zeros_like(correlation_grid)

            for col, ref in enumerate(reference):
                for row, compare in enumerate(cell_types):

                    mask = (adata.obs["cell type refined"] == compare).astype(float).values

                    r, pval = pointbiserialr(mask, comparison_df.loc[:,ref].values)
                    #print(f"Pearson correlation of {compare} to {ref} : r = {r}, p-value = {pval:0.3f}")

                    correlation_grid[row,col] = r
                    pval_grid[row,col] = pval

            fig, axs = plt.subplots(nrows = 1, ncols = 2, 
                                    figsize=(len(reference)*2.4 + 3, len(cell_types)*0.9))
            heat_map = sns.heatmap(correlation_grid, linewidth = 1 , 
                                vmin = -.9, vmax = .9, cmap = "bwr", annot = True, fmt="0.2f",
                                yticklabels=cell_types, xticklabels=reference, 
                                square=True, ax=axs[0])
            pvals = sns.heatmap(pval_grid, linewidth = 1 , 
                                vmin = 0, vmax = 1, #cmap = "bwr", 
                                annot = True, fmt="0.1e",
                                yticklabels=cell_types, xticklabels=reference, 
                                square=True, ax = axs[1])
            fig.suptitle(f"comparing {s2} clustering to {s}")
            fig.tight_layout()
            
            if save_fig:
                fig.savefig(os.path.join(options['file_path'], f"corr_to_{s}_v2_{s2}.eps"), format='eps', dpi=options['dpi'])


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
                            top_n=20) -> None:
    '''
    Plots the comparsion between spatial, nonspatial and RCTD weights
    
    '''

    dfs = []

    for mask, colour, cluster in [(mask_1,"red",cell_type),(mask_2,"blue",cell_type2)]:
        
        self_expr = np.sqrt(1-lambda_param) * metagene[mask]
        nbr_expr = np.sqrt(lambda_param) * metagene_nbr[mask]
        nbr_expr_2 = np.sqrt(lambda_param) * metagene_nbr_1[mask]
        
    #     metagene_plot = ax.scatter(self_expr, nbr_expr, c=colour, s=.5, alpha=.2)
        dfs.append(pd.DataFrame(data={'self': self_expr, 'nbr': nbr_expr, "cluster":cluster}))

    df_combined = pd.concat(dfs)
    # df_combined["cluster"] = df_combined["cluster"].astype("category")
    # display(df_combined.dtypes)
    
    sns.jointplot(data=df_combined, x="self", y="nbr", hue="cluster", kind="kde",
                joint_kws = {"thresh":0.03,"levels":10, "bw_adjust":1.5, "fill":False})

    # Plot all clusters vs one cluster
    # --------------------------------

    cell_type = "Granular Layer"
    cell_types = [cluster for cluster in adata_self.obs["cell type refined"].cat.categories 
                if cluster != cell_type
                ]
    print(f"Comparing <{cell_type}> with:\n{cell_types}")

    DE_genes1 = get_DEgenes(adata_self, cell_type, top_n = top_n)

    mask_1 = adata_spatial.obs["cell type refined"] == cell_type

    subplot_size = (5,5)
    num_cols = 2
    num_rows = len(cell_types) // num_cols + 1
    print(num_rows,num_cols)
    figsize_x = (subplot_size[0] * num_cols) * 1.1
    figsize_y = subplot_size[1] * num_rows

    fig2 = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=True)
    grid = fig2.add_gridspec(ncols=num_cols, nrows=num_rows)


def plot_self_vs_nbr_metagene_all(
                            adata_spatial: anndata.AnnData,
                            adata_self: anndata.AnnData,
                            metagene,
                            metagene_nbr,
                            cell_type,
                            cell_type2,
                            mask_1,
                            mask_2,
                            lambda_param,
                            max_m: int,
                            top_n=20,
                            savefig: bool = False) -> None:
        
        for m in range(max_m+1):
            
            # Plot all clusters vs one cluster
            # --------------------------------

            cell_type = "Granular Layer"
            cell_types = [cluster for cluster in adata_self.obs["cell type refined"].cat.categories 
                        if cluster != cell_type
                        ]
            print(f"Comparing <{cell_type}> with:\n{cell_types}")

            DE_genes1 = get_DEgenes(adata_self, cell_type, top_n = top_n)

            # DE_genes = sc.get.rank_genes_groups_df(adata_self, group=cell_type)["names"][:top_n].tolist()
            # print(f"Top {top_n} DE genes for {cell_type} : {DE_genes}\n")

            # genes_slice = adata_spatial[:, DE_genes]
            # genes_slice_nbr = adata_spatial[:, [gene + "_nbr" for gene in DE_genes]]

            # metagene = np.mean(genes_slice.X, axis = 1)
            # metagene_nbr = np.mean(genes_slice_nbr.X, axis = 1)

            mask_1 = adata_spatial.obs["cell type refined"] == cell_type

            subplot_size = (5,5)
            num_cols = 2
            num_rows = len(cell_types) // num_cols + 1
            print(num_rows,num_cols)
            figsize_x = (subplot_size[0] * num_cols) * 1.1
            figsize_y = subplot_size[1] * num_rows

            fig2 = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=True)
            grid = fig2.add_gridspec(ncols=num_cols, nrows=num_rows)


            for n, cell_type2 in enumerate(cell_types):

                #ax = fig2.add_subplot(grid[n//num_cols, n%num_cols])
                subgrid = grid[n//num_cols, n%num_cols].subgridspec(5, 5)
                ax = fig2.add_subplot(subgrid[1:,:-1])

                DE_genes2 = get_DEgenes(adata_self, cell_type2, top_n = top_n, verbose=False)
                metagene, metagene_nbr = get_metagene_difference(adata_spatial, DE_genes1, DE_genes2, m=m)

                mask_2 = adata_spatial.obs["cell type refined"] == cell_type2

                dfs = [] #used for seaborn plot

                min_x, max_x = 0.0, 0.0

                for mask,colour,cluster in [(mask_1,"red",cell_type),(mask_2,"blue",cell_type2),]:

                    self_expr = np.sqrt(1-lambda_param) * metagene[mask]
                    nbr_expr = np.sqrt(lambda_param) * metagene_nbr[mask]
                    metagene_plot = ax.scatter(self_expr, nbr_expr, c=colour, s=.6, alpha=0.1, label=cluster)

                    dfs.append(pd.DataFrame(data={'self': self_expr, 'nbr': nbr_expr, "cluster":cluster}))

                    min_x = min(np.amin(self_expr),min_x)
                    max_x = max(np.amax(self_expr),max_x)

                df_combined = pd.concat(dfs)

                sns.kdeplot(
                    data=df_combined, x="self", y="nbr", hue="cluster", palette = ["red","blue"], 
                    alpha=0.5, #common_norm = False, #common_grid=True,
                    bw_adjust=1.2, thresh=0.05, levels = 12,
                    #multiple="stack", 
                    ax=ax,
                )

                #ax.set_xlim([min_x, max_x])
                #ax.set_ylim([min_x, max_x])
                ax.set_xlabel("self")
                ax.set_ylabel("neighbour")
                #ax.set_title(f"{cell_type_2} vs\n{cell_type}", fontsize=8)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                ax_top = fig2.add_subplot(subgrid[:1,:-1], sharex=ax)
                sns.kdeplot(data=df_combined, x="self", hue="cluster", palette = ["red","blue"], 
                            alpha=0.5, common_norm = False, #linewidth = 0.2, #common_grid=True,
                            bw_adjust=1.2, legend = False,
                            #multiple="stack", 
                            ax=ax_top)
                #ax_top.legend().set_visible(False)
                ax_top.axis("off")
                xmin, xmax = ax_top.get_xaxis().get_view_interval()
                ymin, _ = ax_top.get_yaxis().get_view_interval()
                ax_top.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='gray', linewidth=1,))

                ax_right = fig2.add_subplot(subgrid[1:,-1:], sharey=ax)
                sns.kdeplot(data=df_combined, y="nbr", hue="cluster", palette = ["red","blue"], 
                            alpha=0.5, common_norm = False,  #linewidth = 0.2, #common_grid=True,
                            bw_adjust=1.2, legend = False,
                            #multiple="stack", 
                            ax=ax_right)
                #ax_right.legend().set_visible(False)
                ax_right.axis("off")
                xmin, _ = ax_right.get_xaxis().get_view_interval()
                ymin, ymax = ax_right.get_yaxis().get_view_interval()
                ax_right.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='gray', linewidth=1,))


            fig2.tight_layout()
            if savefig:
                fig2.savefig(f"metagene_nbr_plot_supp_m{m}.png", format='png', dpi=1200)