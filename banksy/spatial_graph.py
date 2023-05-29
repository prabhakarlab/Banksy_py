import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Union, Tuple, List

from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors


from banksy.csr_operations import remove_greater_than, row_normalize


#
# spatial graph generation
# ========================
#

def generate_spatial_distance_graph(locations: np.ndarray,
                                    nbr_object: NearestNeighbors = None,
                                    num_neighbours: int = None,
                                    radius: Union[float, int] = None,
                                    ) -> csr_matrix:
    """
    generate a spatial graph with neighbours within a given radius
    """

    num_locations = locations.shape[0]

    if nbr_object is None:
        # set up neighbour object
        nbrs = NearestNeighbors(algorithm='ball_tree').fit(locations)
    else:  # use provided sklearn NN object
        nbrs = nbr_object

    if num_neighbours is None:
        # no limit to number of neighbours
        return nbrs.radius_neighbors_graph(radius=radius,
                                           mode="distance")

    else:
        assert isinstance(num_neighbours, int), (
            f"number of neighbours {num_neighbours} is not an integer"
        )

        graph_out = nbrs.kneighbors_graph(n_neighbors=num_neighbours,
                                          mode="distance")

        if radius is not None:
            assert isinstance(radius, (float, int)), (
                f"Radius {radius} is not an integer or float"
            )

            graph_out = remove_greater_than(graph_out, radius,
                                            copy=False, verbose=False)

        return graph_out


def theta_from_spatial_graph(locations: np.ndarray,
                             spatial_graph: csr_matrix,
                             ):
    """
    get azimuthal angles from spatial graph and coordinates
    (assumed dim 1: x, dim 2: y, dim 3: z...)

    returns CSR matrix with theta (azimuthal angles) as .data
    """

    theta_data = np.zeros_like(spatial_graph.data, dtype=np.float32)

    for n in range(spatial_graph.indptr.shape[0] - 1):
        ptr_start, ptr_end = spatial_graph.indptr[n], spatial_graph.indptr[n + 1]
        nbr_indices = spatial_graph.indices[ptr_start:ptr_end]

        self_coord = locations[[n], :]
        nbr_coord = locations[nbr_indices, :]
        relative_coord = nbr_coord - self_coord

        theta_data[ptr_start:ptr_end] = np.arctan2(
            relative_coord[:, 1], relative_coord[:, 0])

    theta_graph = spatial_graph.copy()
    theta_graph.data = theta_data

    return theta_graph


def generate_spatial_weights_fixed_nbrs(locations: np.ndarray,
                                        m: int = 0,  # azimuthal transform order
                                        num_neighbours: int = 10,
                                        decay_type: str = "reciprocal",
                                        nbr_object: NearestNeighbors = None,
                                        verbose: bool = True,
                                        ) -> Tuple[csr_matrix, csr_matrix, Union[csr_matrix, None]]:
    """
    generate a graph (csr format) where edge weights decay with distance
    """

    distance_graph = generate_spatial_distance_graph(
        locations,
        nbr_object=nbr_object,
        num_neighbours=num_neighbours * (m + 1),
        radius=None,
    )

    if m > 0:
        theta_graph = theta_from_spatial_graph(locations, distance_graph)
    else:
        theta_graph = None

    graph_out = distance_graph.copy()

    # compute weights from nbr distances (r)
    # --------------------------------------

    if decay_type == "uniform":

        graph_out.data = np.ones_like(graph_out.data)

    elif decay_type == "reciprocal":

        graph_out.data = 1 / graph_out.data

    elif decay_type == "reciprocal_squared":

        graph_out.data = 1 / (graph_out.data ** 2)

    elif decay_type == "scaled_gaussian":

        indptr, data = graph_out.indptr, graph_out.data

        for n in range(len(indptr) - 1):

            start_ptr, end_ptr = indptr[n], indptr[n + 1]
            if end_ptr >= start_ptr:
                # row entries correspond to a cell's neighbours
                nbrs = data[start_ptr:end_ptr]
                median_r = np.median(nbrs)
                weights = np.exp(-(nbrs / median_r) ** 2)
                data[start_ptr:end_ptr] = weights

    elif decay_type == "ranked":

        linear_weights = np.exp(
            -1 * (np.arange(1, num_neighbours + 1) * 1.5 / num_neighbours) ** 2
        )

        indptr, data = graph_out.indptr, graph_out.data

        for n in range(len(indptr) - 1):

            start_ptr, end_ptr = indptr[n], indptr[n + 1]

            if end_ptr >= start_ptr:
                # row entries correspond to a cell's neighbours
                nbrs = data[start_ptr:end_ptr]

                # assign the weights in ranked order
                weights = np.empty_like(linear_weights)
                weights[np.argsort(nbrs)] = linear_weights

                data[start_ptr:end_ptr] = weights

    else:
        raise ValueError(
            f"Weights decay type <{decay_type}> not recognised.\n"
            f"Should be 'uniform', 'reciprocal', 'reciprocal_squared', "
            f"'scaled_gaussian' or 'ranked'."
        )

    # make nbr weights sum to 1
    graph_out = row_normalize(graph_out, verbose=verbose)

    # multiply by Azimuthal Fourier Transform
    if m > 0:
        graph_out.data = graph_out.data * np.exp(1j * m * theta_graph.data)

    return graph_out, distance_graph, theta_graph