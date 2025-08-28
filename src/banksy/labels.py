"""
Label Object
and functions dealing with labels
"""

import copy
import numpy as np

from typing import List, Tuple, Union

from scipy.sparse import csr_matrix, issparse
from scipy.optimize import linear_sum_assignment

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import StrMethodFormatter
from matplotlib.collections import PolyCollection

from banksy_utils.time_utils import timer
from banksy.csr_operations import row_normalize


class Label(object):

    def __init__(self,
                 labels_dense: Union[np.ndarray, list],
                 verbose: bool = True,
                 ) -> None:

        self.verbose = verbose

        # Check type, dimensions, ensure all elements non-negative
        # --------------------------------------------------------

        if isinstance(labels_dense, list):
            labels_dense = np.asarray(labels_dense, dtype=np.int32)
        elif isinstance(labels_dense, np.ndarray):
            pass
        else:
            raise TypeError(
                f"Labels provided are of type {type(labels_dense)}. "
                f"Should be list or 1-dimensional numpy ndarray.\n"
            )

        assert labels_dense.ndim == 1, (
            f"Label array has {labels_dense.ndim} dimensions, "
            f"should be 1-dimensional."
        )
        assert np.issubdtype(labels_dense.dtype, np.integer), (
            f"Label array data type is {labels_dense.dtype}, "
            f"should be integer."
        )
        assert np.amin(labels_dense) >= 0, (
            f"Some of the labels have negative values.\n"
            f"All labels must be 0 or positive integers.\n"
        )

        # Initialize attributes
        # ---------------------

        # dense array of labels, same as provided
        self.dense = labels_dense
        # total number of data-points with label (e.g. number of cells)
        self.num_samples = len(labels_dense)

        # number of instances of each integer up to maximum label id
        self.bins = np.bincount(self.dense)
        # unique labels (all non-negative integers)
        self.ids = np.nonzero(self.bins)[0]
        # counts per label (same order as self.ids)
        self.counts = self.bins[self.ids]
        # highest integer id for a label
        self.max_id = np.amax(self.ids)
        # total number of labels
        self.num_labels = len(self.ids)

        self.onehot = None
        self.normalized_onehot = None

    def __repr__(self) -> str:
        return (
            f"{self.num_labels} labels, {self.num_samples} samples, "
            f"ids: {self.ids}, counts: {self.counts}"
        )

    def __str__(self) -> str:
        return (
            f"Label object:\n"
            f"Number of labels: {self.num_labels}, "
            f"number of samples: {self.num_samples}\n"
            f"ids: {self.ids}, counts: {self.counts},\n"
        )

    def get_onehot(self) -> csr_matrix:
        """
        return one-hot sparse array of labels.
        If not already computed, generate the sparse array from dense label array
        """
        if self.onehot is None:
            self.onehot = self.generate_onehot(verbose=False)

        return self.onehot

    def get_normalized_onehot(self) -> csr_matrix:
        """
        return normalized one-hot sparse array of labels.
        """
        if self.normalized_onehot is None:
            self.normalized_onehot = self.generate_normalized_onehot(
                verbose=False
            )

        return self.normalized_onehot

    def generate_normalized_onehot(self,
                                   verbose: bool = False,
                                   ) -> csr_matrix:
        """
        generate a normalized onehot matrix where each row is
        normalized by the count of that label
        e.g. a row [0 1 1 0 0] will be converted to [0 0.5 0.5 0 0]
        """
        return row_normalize(self.get_onehot().astype(np.float64),
                             copy=True,
                             verbose=verbose)

    def generate_onehot(self,
                        verbose: bool = False,
                        ) -> csr_matrix:
        """
        convert an array of labels to a
        num_labels x num_samples sparse one-hot matrix

        Labels MUST be integers starting from 0,
        but can have gaps in between e.g. [0,1,5,9]
        """

        # initialize the fields of the CSR
        # ---------------------------------

        indptr = np.zeros((self.num_labels + 1,), dtype=np.int32)
        indices = np.zeros((self.num_samples,), dtype=np.int32)
        data = np.ones_like(indices, dtype=np.int32)

        if verbose:
            print(f"\n--- {self.num_labels} labels, "
                  f"{self.num_samples} samples ---\n"
                  f"initalized {indptr.shape} index ptr: {indptr}\n"
                  f"initalized {indices.shape} indices: {indices}\n"
                  f"initalized {data.shape} data: {data}\n")

        # update index pointer and indices row by row
        # -------------------------------------------

        for n, label in enumerate(self.ids):

            label_indices = np.nonzero(self.dense == label)[0]
            label_count = len(label_indices)

            previous_ptr = indptr[n]
            current_ptr = previous_ptr + label_count
            indptr[n + 1] = current_ptr

            if verbose:
                print(f"indices for label {label}: {label_indices}\n"
                      f"previous pointer: {previous_ptr}, "
                      f"current pointer: {current_ptr}\n")

            if current_ptr > previous_ptr:
                indices[previous_ptr:current_ptr] = label_indices

        return csr_matrix((data, indices, indptr),
                          shape=(self.num_labels, self.num_samples))


#
#                  Relabeling Functions
# =====================================================
# Functions for swapping around or changing label ids
#
#

def _rand_binary_array(array_length, num_onbits):
    array = np.zeros(array_length, dtype=np.int32)
    array[:num_onbits] = 1
    np.random.shuffle(array)
    return array


@timer
def expand_labels(label: Label,
                  max_label_id: int,
                  sort_labels: bool = False,
                  verbose: bool = True,
                  ) -> Label:
    """
    Spread out label IDs such that they
    range evenly from 0 to max_label_id
    e.g. [0 1 2] -> [0 5 10]

    Useful if you need to be consistent with other
    label sets with many more label IDs.
    This spreads the out along the colour spectrum/map
    so that the colours are not too similar to each other.

    use sort_labels if the list of ids are not already sorted
    (which it should usually be)
    """
    if verbose:
        print(f"Expanding labels with ids: {label.ids} "
              f"so that ids range from 0 to {max_label_id}\n")

    if sort_labels:
        ids = np.sort(copy.copy(label.ids))
    else:
        ids = copy.copy(label.ids)

    # make sure smallest label-id is zero
    ids_zeroed = ids - np.amin(label.ids)

    num_extra_labels = max_label_id - np.amax(ids_zeroed)

    multiple, remainder = np.divmod(num_extra_labels, label.num_labels - 1)

    # insert regular spaces between each id
    inserted = np.arange(label.num_labels) * multiple

    # insert remaining spaces so that max label id equals given max_id
    extra = _rand_binary_array(label.num_labels - 1, remainder)

    expanded_ids = ids_zeroed + inserted
    expanded_ids[1:] += np.cumsum(extra)  # only add to 2nd label and above

    if verbose:
        print(f"Label ids zerod: {ids_zeroed}.\n"
              f"{multiple} to be inserted between each id: {inserted}\n"
              f"{remainder} extra rows to be randomly inserted: {extra}\n"
              f"New ids: {expanded_ids}")

    expanded_dense = (expanded_ids @ label.get_onehot()).astype(np.int32)

    return Label(expanded_dense, verbose=label.verbose)


@timer
def match_labels(labels_1: Label,
                 labels_2: Label,
                 extra_labels_assignment: str = "random",
                 verbose: bool = True,
                 ) -> Label:
    """
    Match second set of labels to first, returning a new Label object
    Uses scipy's version of the hungarian algroithm (linear_sum_assigment)
    """

    max_id = max(labels_1.max_id, labels_2.max_id)
    num_extra_labels = labels_2.num_labels - labels_1.num_labels

    if verbose:
        print(f"Matching {labels_2.num_labels} labels "
              f"against {labels_1.num_labels} labels.\n"
              f"highest label ID in both is {max_id}.\n")

    onehot_1, onehot_2 = labels_1.get_onehot(), labels_2.get_onehot()

    cost_matrix = (onehot_1 @ onehot_2.T).toarray()

    labels_match_1, labels_match_2 = linear_sum_assignment(cost_matrix,
                                                           maximize=True)

    if verbose:
        print("\nMatches:\n", list(zip(labels_match_1, labels_match_2)))

    # temporary list keeping track of which labels are still available for use
    available_labels = list(range(max_id + 1))
    # list to be filled with new label ids
    relabeled_ids = -1 * np.ones((labels_2.num_labels,), dtype=np.int32)

    # reassign labels
    # ---------------

    for index_1, index_2 in zip(labels_match_1, labels_match_2):
        label_1 = labels_1.ids[index_1]
        label_2 = labels_2.ids[index_2]
        if verbose:
            print(f"Assigning first set's {label_1} to "
                  f"second set's {label_2}.\n"
                  f"labels_left: {available_labels}")
        relabeled_ids[index_2] = label_1
        available_labels.remove(label_1)

    # Assign remaining labels (if 2nd has more labels than 1st)
    # ---------------------------------------------------------

    if num_extra_labels > 0:

        unmatched_indices = np.nonzero(relabeled_ids == -1)[0]

        assert num_extra_labels == len(unmatched_indices), (
            f"number of unmatched label IDs {len(unmatched_indices)} "
            f"does not match mumber of "
            f"extra labels in second set {num_extra_labels}.\n"
        )

        if extra_labels_assignment == "random":
            relabeled_ids[unmatched_indices] = np.random.choice(available_labels,
                                                                size=num_extra_labels,
                                                                replace=False, )

        elif extra_labels_assignment == "greedy":

            def _insert_label(array: np.ndarray,
                              max_length: int,
                              added_labels: list = [],
                              ) -> Tuple[np.ndarray, int, list]:
                """
                insert a label in the middle of the largest interval
                assumes array is alreay sorted!
                """
                if len(array) >= max_length:
                    return array, max_length, added_labels
                else:
                    intervals = array[1:] - array[:-1]
                    max_interval_index = np.argmax(intervals)
                    increment = intervals[max_interval_index] // 2
                    label_to_add = array[max_interval_index] + increment
                    inserted_array = np.insert(
                        array,
                        max_interval_index + 1,
                        label_to_add,
                    )
                    added_labels.append(label_to_add)
                    return _insert_label(inserted_array, max_length, added_labels)

            sorted_matched = np.sort(relabeled_ids[relabeled_ids != -1])

            if verbose:
                print(f"already matched ids (sorted): {sorted_matched}")

            _,_,added_labels = _insert_label(sorted_matched, labels_2.num_labels)

            relabeled_ids[unmatched_indices] = np.random.choice(added_labels,
                                                                size=num_extra_labels,
                                                                replace=False, )

        elif extra_labels_assignment == "optimized":
            raise NotImplementedError(
                f"haven't figured out how to do this yet.\n"
            )

        else:
            raise ValueError(
                f"Extra labels assignment method not recognised, "
                f"should be random, greedy or optimized.\n"
            )

        if verbose:
            print(f"\nRelabeled labels: {relabeled_ids}\n")

    # generate re-labeled dense array
    # -------------------------------
    #
    # reordered (matmul) sparse_labels -> dense relabelled array
    #   e.g.
    #   [ 1 3 2 ] @ [ 0 ... 1 0  -> [ 2 ... 1 3 ]
    #                 0 ... 0 1
    #                 1 ... 0 0]
    #

    relabeled_dense = (relabeled_ids @ onehot_2).astype(np.int32)

    return Label(relabeled_dense, verbose=labels_2.verbose)


def match_label_series(label_list: List[Label],
                       least_labels_first: bool = True,
                       extra_labels_assignment: str = "greedy",
                       verbose: bool = True,
                       ) -> Tuple[List[Label], int]:
    """
    Match a list of labels to each other, one after another
    in order of increasing (if least_labels_first is true)
    or decreasing (least_labels_first set to false)
    number of label ids.
    Returns the relabeled list in original order
    """
    num_label_list = [label.num_labels for label in label_list]
    max_num_labels = max(num_label_list)
    sort_indices = np.argsort(num_label_list)

    print(f"\nMaximum number of labels = {max_num_labels}\n"
          f"Indices of sorted list: {sort_indices}\n")

    ordered_relabels = []

    if least_labels_first:
        ordered_relabels.append(
            expand_labels(label_list[sort_indices[0]],
                          max_num_labels - 1)
        )
        if verbose:
            print(f"First label, expanded label ids: "
                  f"{ordered_relabels[0]}")
    else:
        # argsort is in asceding order, reverse it
        sort_indices = sort_indices[:, :, -1]
        # already has max number of labels, no need to expand
        ordered_relabels.append(
            label_list[sort_indices[0]]
        )

    for index in sort_indices[1:]:

        current_label = label_list[index]
        previous_label = ordered_relabels[-1]

        if verbose:
            print(f"\nRelabeling:\n{current_label}\n"
                  f"with reference to\n{previous_label}\n"
                  + "-" * 70 + "\n")

        relabeled = match_labels(
            previous_label, current_label,
            extra_labels_assignment=extra_labels_assignment,
            verbose=verbose,
        )

        ordered_relabels.append(relabeled)

    sort_indices_list = list(sort_indices)
    original_order_relabels = [ordered_relabels[sort_indices_list.index(n)]
                               for n in range(len(label_list))]

    return original_order_relabels, max_num_labels


#
#
#                           Label connections
# =========================================================================
#
# Functions to determine connections strengths between different labels
# Must be given:
#       (1) A Label object
#       (2) weights matrix representing the connections between each
#           sample (e.g. the spatial graph)
#
#

def interlabel_connections(label: Label,
                           weights_matrix: Union[csr_matrix, np.ndarray],
                           ) -> np.ndarray:
    """
    compute connections strengths between labels,
    normalized by number of each label
    Requires a weights_matrix of shape (num_samples x num_samples)
    reprensenting the spatial graph between each sample
    """
    assert weights_matrix.ndim == 2, (
        f"weights matrix has {weights_matrix.ndim} dimensions, should be 2."
    )

    assert weights_matrix.shape[0] == weights_matrix.shape[1] == label.num_samples, (
        f"weights matrix dimenisons do not match number of samples"
    )

    normalized_onehot = label.generate_normalized_onehot(verbose=False)

    print(
        f"\nmatrix multiplying labels x weights x labels-transpose "
        f"({normalized_onehot.shape} "
        f"x {weights_matrix.shape} "
        f"x {normalized_onehot.T.shape})\n"
    )

    connections = normalized_onehot @ weights_matrix @ normalized_onehot.T

    if issparse(connections):
        connections = connections.toarray()

    return connections


def plot_connections(label: Label,
                     weights_matrix: Union[csr_matrix, np.ndarray],
                     ax: mpl.axes.Axes,
                     zero_self_connections: bool = True,
                     normalize_by_self_connections: bool = False,
                     shapes_style: bool = True,
                     max_scale: float = 0.46,
                     colormap_name: str = "Spectral",
                     title_str="connection strengths between types",
                     title_fontsize: float = 12,
                     label_fontsize: float = 12,
                     verbose: bool = True,
                     ) -> None:
    """
    plot the connections between labels
    given as a num_label by num_label matrix of connection strengths

    :param ax: axes to plot on
    :param shapes_style: use shapes or heatmap
    :param max_scale: only used for shape, gives maximum size of square
    """

    fig = ax.get_figure()

    connections = interlabel_connections(label, weights_matrix)

    if zero_self_connections:
        np.fill_diagonal(connections, 0)
    elif normalize_by_self_connections:
        connections /= connections.diagonal()[:, np.newaxis]

    connections_max = np.amax(connections)

    # set label colours
    # -----------------

    cmap = mpl.cm.get_cmap(colormap_name)
    id_colours = {id: cmap(id / label.max_id) for id in label.ids}

    if shapes_style:

        # Coloured triangles with size representing connection strength
        # -------------------------------------------------------------

        left_triangle = np.array((
            (-1., 1.),
            # (1., 1.),
            (1., -1.),
            (-1., -1.)
        ))

        right_triangle = np.array((
            (-1., 1.),
            (1., 1.),
            (1., -1.),
            # (-1., -1.)
        ))

        polygon_list = []
        colour_list = []

        ax.set_ylim(- 0.55, label.num_labels - 0.45)
        ax.set_xlim(- 0.55, label.num_labels - 0.45)
        # ax.invert_yaxis()

        for label_1 in range(connections.shape[0]):
            for label_2 in range(connections.shape[1]):

                if label_1 <= label_2:

                    for triangle in [left_triangle, right_triangle]:
                        center = np.array((label_1, label_2))[np.newaxis, :]
                        scale_factor = connections[label_1, label_2] / connections_max
                        offsets = triangle * max_scale * scale_factor
                        polygon_list.append(center + offsets)

                    colour_list += (id_colours[label.ids[label_2]],
                                    id_colours[label.ids[label_1]])

        collection = PolyCollection(polygon_list,
                                    facecolors=colour_list,
                                    edgecolors="face",
                                    linewidths=0)

        ax.add_collection(collection)

        ax.tick_params(labelbottom=False, labeltop=True)
        ax.xaxis.set_tick_params(pad=-2)

    else:

        # Heatmap of connection strengths
        # -------------------------------

        heatmap = ax.imshow(connections,
                            cmap='viridis',
                            interpolation="nearest")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        fig.colorbar(heatmap, cax=cax)
        cax.tick_params(axis='both', which='major', labelsize=6, rotation=-45)

        # change formatting if values too small
        if connections_max < 0.001:
            cax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1e}'))

    # Set tick labels positions and colour
    # ------------------------------------

    ax.set_aspect('equal')

    ax.set_xticks(np.arange(label.num_labels), )
    ax.set_xticklabels(label.ids, fontsize=label_fontsize,
                       fontweight="bold", rotation=0)

    ax.set_yticks(np.arange(label.num_labels), )
    ax.set_yticklabels(label.ids, fontsize=label_fontsize,
                       fontweight="bold")

    for ticklabels in [ax.get_xticklabels(), ax.get_yticklabels()]:
        for n, id in enumerate(label.ids):
            ticklabels[n].set_color(id_colours[id])

    # title
    # -----

    ax.set_title(title_str, fontsize=title_fontsize, fontweight="bold")


#
#                                        Test Script
# ==========================================================================================
#
#

if __name__ == "__main__":
    array1 = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    array2 = np.array([2, 2, 2, 1, 1, 1, 1, 5, 5, 5, 3])

    labels_1 = Label(array1, verbose=False)
    labels_2 = Label(array2, verbose=False)
    matched_labels_2 = match_labels(labels_1, labels_2, verbose=False)
    expanded = expand_labels(labels_1, 11)

    names = ["First", "second", "matched", "expanded first"]
    labels_list = [labels_1, labels_2, matched_labels_2, expanded]
    for name, label in zip(names, labels_list):
        print(f"\n {name}\n" + "-" * 50)
        print(label, "\n", label.dense,
              f"\nonehot:\n{label.get_onehot().toarray()}\n")

    print(f"\nOne-hot normalized:\n"
          f"{labels_2.get_normalized_onehot().toarray()}")

    weights_matrix = csr_matrix(np.random.random_sample((11, 11)))
    # print(weights_matrix)

    connections = interlabel_connections(expanded, weights_matrix)

    print(f"\nConnections Matrix:\n{connections}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    plot_connections(expanded, weights_matrix,
                     ax,
                     zero_self_connections=False)

    plt.show()
