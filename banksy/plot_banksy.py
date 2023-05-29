import os, gc
import matplotlib.pyplot as plt
import scanpy as sc

from typing import Union, List
import numpy as np
import pandas as pd
from utils.plotting import plot_2d_embeddings, plot_labels_seperately, plot_label_subset
from banksy.labels import plot_connections
import anndata
from typing import Tuple
import matplotlib as mpl
import matplotlib.gridspec as gridspec

def plot_results(
        results_df: pd.DataFrame,
        weights_graph: dict,
        c_map: Union[int, float],
        max_num_labels: int,
        match_labels: bool,
        filepath: str,
        coord_keys: Tuple[str],
        plot_dot_plot: bool = False,
        plot_heat_map: bool = False,
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
        use_sc_plot: plot scanpy 

        save_all_h5ad: to save a copy of the temporary anndata object as .h5ad format

        cmap_name : Color map settings for plotting banksy

        file_path (str): default file path is 'data/slide_seq/v1'
    
    Returns:
        The main figure for visualization using banksy
    '''
    ## Put filepath and color map as input variables
    options = {
        'use_sc_plot' : False,
        'save_all_h5ad': True,
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
            save_path = os.path.join(filepath, f"slideseq_mousecerebellum_{params_name}.h5ad")
            adata_temp.write(save_path)

        adata_temp.obsm[coord_keys[2]] = np.vstack(
            (adata_temp.obs[coord_keys[0]].values,
             adata_temp.obs[coord_keys[1]].values)
        ).T

        fig, grid = plot_main_figure(adata_temp,
                                     key = coord_keys[2],  
                                     use_sc_plot = options["use_sc_plot"], 
                                     label_name = label_name)
        
        ### Auxiliary funtion to plot labels
        plot_labels(adata_temp,
                coord_keys[2],
                labels,
                c_map,
                max_num_labels,
                params_name,
                fig, 
                grid)

         # Seperate Location plots
        # -----------------------
        
        plot_labels_seperately(
            labels, adata_temp.obsm[coord_keys[2]],
            embeddings = umap_temp,
            cmap_name = c_map,
            default_colour="tab:red",
            plots_per_row = 3,
            spot_size = 1,
            subplot_size = (5, 5),
            flip_axes = False,
            verbose = False,
        )
            

        # Plot UMAP (again but with labels)
        # ---------------------------------

        ax_umap = fig.add_subplot(grid[0, -2:])
        plot_2d_embeddings(umap_temp, labels.dense,
                        method_str="UMAP",
                        space_str="",
                        xlabel="UMAP 1", ylabel="UMAP 2",
                        ax = ax_umap,
                        cmap_name = c_map,
                        plot_cmap = False,
                        )
        
        # Plot 1st 2 dimensions of PCA
        # ----------------------------

        dim_sets = (pc_temp[:, :2], pc_temp[:, 1:3])
        dims1 = (0, 1,)
        dims2 = (1, 2,)
        axes = [fig.add_subplot(grid[1, 2+axnum]) for axnum in range(2)]

        for dim_set, dim1, dim2, ax in zip(dim_sets, dims1, dims2, axes):
            plot_2d_embeddings(dim_set, labels.dense,
                            method_str=f"PCA {dim1 + 1} / {dim2 + 1}",
                            space_str="",
                            xlabel=f"PCA {dim1 + 1}", ylabel=f"PCA {dim2 + 1}",
                            ax = ax,
                            cmap_name = c_map,
                            plot_cmap = False,
                            title_fontsize = 9)
            
        # Plot connectivity between labels
        # --------------------------------

        ax_connections = fig.add_subplot(grid[-1, -2:])

        plot_connections(
            labels, weights_graph,
            ax_connections,
            zero_self_connections=True,
            title_str="connections between label",
            colormap_name = c_map,
        )
        

        ### Add these options to plot or not, 
        ### plot_heatmap = False, plot_dotplot = False
        if plot_dot_plot or plot_heat_map:
            num_groups = options['group_num']
            groups_subset = [str(n) for n in range(num_groups) if str(n) in adata_temp.obs[label_name].cat.categories]
            print(f"plotting groups: {groups_subset}")
        if plot_heat_map:
            sc.pl.rank_genes_groups_heatmap(adata_temp, n_genes=6, 
                                            groups=groups_subset, 
                                            vmin=-3, vmax=3, cmap='bwr', 
                                            swap_axes=True)
        if plot_dot_plot:
            sc.pl.rank_genes_groups_dotplot(adata_temp, n_genes=5, 
                                            groups=groups_subset, 
                                        )

def plot_main_figure(adata_temp,
                    key,
                    use_sc_plot: bool = False,
                    label_name: str or None = None):
    '''Auxiliary Function to plot main figure'''
    # Default figure options
    main_figsize = (15,9)
    width_ratios = [2, 0.1, 0.5, 0.5]
    height_ratios = [1, 0.3, 1]

    sc_plot_size = 5

    if use_sc_plot:
        # If we want to generate a sc plot
        sc.pl.embedding(adata_temp,
                        basis = key,
                        color = label_name,
                        size = sc_plot_size)
    
    fig = plt.figure(figsize=main_figsize, constrained_layout=True)

    grid = fig.add_gridspec(ncols=4, nrows=3, width_ratios=width_ratios, height_ratios=height_ratios)

    return fig, grid

def plot_labels(adata_temp,
                key,
                labels,
                cmap,
                max_num_labels,
                params_name,
                fig, 
                grid):
        '''
        Plots the labels for the main figure
        '''

        ax_locs = fig.add_subplot(grid[:, 0])

        scatterplot = ax_locs.scatter(adata_temp.obsm[key][:, 0], 
                                    adata_temp.obsm[key][:, 1],
                                    c=labels.dense, 
                                    cmap=cmap,
                                    vmin = 0, vmax=max_num_labels-1,
                                    s=1.5, alpha=1.0)
        
        ax_locs.set_aspect('equal', 'datalim')
        ax_locs.set_title(f'BANKSY Labels ({params_name})', 
                        fontsize=20, 
                        fontweight="bold",
                        )
        #ax_locs.set_ylim(ax_locs.get_ylim()[::-1])

        ax_cbar = fig.add_subplot(grid[:, 1])
        cbar = fig.colorbar(
            scatterplot,
            boundaries=np.arange(max_num_labels + 1) - 0.5,
            cax = ax_cbar,
        )
        cbar.set_ticks(labels.ids)
        cbar.set_ticklabels(labels.ids)