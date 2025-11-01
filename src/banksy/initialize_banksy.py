"""
initialize the graphs neccesary to run BANKSY for a given number of neighbours,
and vidualize graphs/histograms for QC purposes.

TODO: allow for parameter sweep across multiple k_geom settings

Yifei Aug 2023
"""

from typing import Tuple

import anndata
import numpy as np
from banksy_utils.plot_utils import plot_edge_histograms, plot_weights, plot_theta_graph
from banksy.main import generate_spatial_weights_fixed_nbrs
from sklearn.neighbors import NearestNeighbors

def initialize_banksy(adata: anndata.AnnData,
                      coord_keys: Tuple[str],
                      k_geom: int = 15,
                      nbr_weight_decay: str = 'scaled_gaussian',
                      max_m: int = 1,
                      plt_edge_hist: bool = True,
                      plt_nbr_weights: bool = True,
                      plt_agf_angles: bool = False,
                      plt_theta: bool = True
                      ) -> dict:
    '''Main Function that initializes the BANKSY Object as a dictionary
    
    Input Args:
        adata (AnnData): AnnData object containing the data matrix

        k_geom (int) : The number of neighbours in which the edges,
        weights and theta graph are constructed. 

        nbr_weight_decay (str): Type of neighbourhood decay function, can be 'scaled_gaussian' or 'reciprocal'

        max_m (int): Maximum order of azimuthal gabor filter, we recommend a default of 1

    
    Optional Args:
        plt_edge (bool): Visualize the edge histogram*

        plt_weight (bool): Visualize the weights graph

        plt_agf_weights (bool): Visualize the AGF weights

        plt_theta (bool): Visualize angles around random cell
    '''

    banksy_dict = {}  # Dictionary containing all the information across different parameter settings during runtime

    weights_graphs = {}  # Sub-dictionary containing weighted graph

    # Find nearest nearest neighbours object. 
    nbrs = NearestNeighbors(algorithm='ball_tree').fit(adata.obsm[coord_keys[2]])
    
    # Optional: print median distance for diagnostics

    distances, _ = nbrs.kneighbors(n_neighbors=1)
    print(f"\nMedian distance to closest cell = {np.median(distances):.4f}\n")
    
    for m in range(max_m + 1):

        weights_graph, distance_graph, theta_graph = generate_spatial_weights_fixed_nbrs(
            adata.obsm[coord_keys[2]],
            m=m,
            k_geom=k_geom,
            decay_type=nbr_weight_decay,
            nbr_object=nbrs,
            verbose=False,
            max_radius=None
        )

        weights_graphs[m] = weights_graph

        if plt_edge_hist:
            print(f'----- Plotting Edge Histograms for m = {m} -----')
            plot_edge_histograms(
                distance_graph=distance_graph,
                weights_graph=weights_graph,
                decay_type=nbr_weight_decay,
                m=m
            )

    banksy_dict[nbr_weight_decay] = {"weights": weights_graphs}

    if plt_nbr_weights:
        # Plot weights graph (Optional)
        print(f'----- Plotting Weights Graph -----')
        plot_weights(
            adata,
            banksy_dict,
            nbr_weight_decay,
            max_m,
            fig_title=f'Decay Type {nbr_weight_decay}',
            coord_keys=coord_keys
        )

    if plt_agf_angles:
        print(f'----- Plotting Azimuthal Angles -----')
        plot_weights(
            adata,
            banksy_dict,
            nbr_weight_decay,
            max_m,
            fig_title=f'Azimuthal Angles',
            coord_keys=coord_keys,
            theta_graph=theta_graph
        )

    if plt_theta:
        # plot angles of neigbhours around a random cell
        print(f'----- Plotting theta Graph -----')
        plot_theta_graph(
            adata,
            theta_graph,
            coord_keys
        )

    return banksy_dict
