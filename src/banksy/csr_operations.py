"""
Functions that operate on CSR matrices

Nigel 3 dec 2020
"""
import copy
import numpy as np
from scipy.sparse import csr_matrix

from typing import Union

from banksy_utils.time_utils import timer


@timer
def remove_greater_than(graph: csr_matrix,
                        threshold: float,
                        copy: bool = False,
                        verbose: bool = True):
    """
    Remove values greater than a threshold from a CSR matrix
    """
    if copy:
        graph = graph.copy()

    greater_indices = np.where(graph.data > threshold)[0]

    if verbose:
        print(f"CSR data field:\n{graph.data}\n"
              f"compressed indices of values > threshold:\n{greater_indices}\n")

    # delete the entries in data and index fields
    # -------------------------------------------

    graph.data = np.delete(graph.data, greater_indices)
    graph.indices = np.delete(graph.indices, greater_indices)

    # update the index pointer
    # ------------------------

    hist, _ = np.histogram(greater_indices, bins=graph.indptr)
    cum_hist = np.cumsum(hist)
    graph.indptr[1:] -= cum_hist

    if verbose:
        print(f"\nCumulative histogram:\n{cum_hist}\n"
              f"\n___ New CSR ___\n"
              f"pointers:\n{graph.indptr}\n"
              f"indices:\n{graph.indices}\n"
              f"data:\n{graph.data}\n")

    return graph


def filter_by_rank_and_threshold(graph: csr_matrix,
                                 max_rank: int = 3,
                                 threshold: Union[float, int] = 5,
                                 copy: bool = True,
                                 verbose: bool = True):
    """
    Filter a csr matrix by removing elements in each ROW
    which are either:
        (1) Among the LARGEST elemtns by rank (up to max_rank)
        (2) HIGHER than or equal to a threshold value
    """

    indptr, indices, data = graph.indptr, graph.indices, graph.data

    # initialize mask (default is to keep all entries)
    keep_mask = np.ones_like(data, dtype=bool)
    # need to update the index pointer since some entries will be deleted
    new_indptr = np.zeros_like(indptr)

    current_ptr = 0  # keep track of the modified index pointer for each row

    for row_num in range(graph.shape[0]):

        start_ptr, end_ptr = indptr[row_num], indptr[row_num + 1]
        num_row_elements = end_ptr - start_ptr

        if num_row_elements > 0:  # make sure row is not empty

            if num_row_elements <= max_rank:
                # keep the whole row, since the worst ranked element
                # is still within the max_rank
                current_ptr += num_row_elements
            else:
                row_data = data[start_ptr:end_ptr]

                max_rank_value = row_data[np.argsort(row_data)[-max_rank]]
                row_mask = (row_data >= max_rank_value) | (row_data >= threshold)

                keep_mask[start_ptr:end_ptr] = row_mask

                current_ptr += sum(row_mask)

                # NOTE: we use the max-ranked value here so that if
                # multiple elements have the max-ranked value
                # all of them will be selected.
                # If we just picked the highest ranked elements up to max_rank
                # we will only get one of these elements

                if verbose:
                    print(f"Value of maximum ranked element in "
                          f"{row_data} is {max_rank_value}. "
                          f"keep mask for this row is: {row_mask}")

        new_indptr[row_num + 1] = current_ptr

    if copy:
        new_graph = csr_matrix((data[keep_mask],
                                indices[keep_mask],
                                new_indptr),
                               shape=graph.shape)
    else:
        graph.data = data[keep_mask]
        graph.indices = indices[keep_mask]
        graph.indptr = new_indptr
        new_graph = graph

    return new_graph


@timer
def row_normalize(graph: csr_matrix,
                  copy: bool = False,
                  verbose: bool = True):
    """
    Normalize a compressed sparse row (CSR) matrix by row
    """
    if copy:
        graph = graph.copy()

    data = graph.data

    for start_ptr, end_ptr in zip(graph.indptr[:-1], graph.indptr[1:]):

        row_sum = data[start_ptr:end_ptr].sum()

        if row_sum != 0:
            data[start_ptr:end_ptr] /= row_sum

        if verbose:
            print(f"normalized sum from ptr {start_ptr} to {end_ptr} "
                  f"({end_ptr - start_ptr} entries)",
                  np.sum(graph.data[start_ptr:end_ptr]))

    return graph


def insert_empty_rows(matrix: csr_matrix,
                      num_rows: int,
                      inplace: bool = False,
                      verbose: bool = True,
                      ) -> csr_matrix:
    """
    insert num_rows empty rows  between every row of the given matrix
    """
    assert num_rows >= 1, (
        f"Number of rows to insert was {num_rows}, must be at least 1"
    )

    if inplace:
        padded_matrix = matrix
    else:
        padded_matrix = copy.copy(matrix)

    if verbose: print(f"\noriginal length of indexptr: {padded_matrix.indptr}\n"
                      f"Shape of original matrix: {padded_matrix.shape}\n")

    padded_matrix.indptr = np.repeat(padded_matrix.indptr, num_rows + 1)[num_rows:-num_rows]
    padded_matrix.shape = (len(padded_matrix.indptr), padded_matrix.shape[1])

    if verbose: print(f"modified length of indexptr: {padded_matrix.indptr}\n"
                      f"Shape of modified matrix: {padded_matrix.shape}\n")

    return padded_matrix


def elements_per_row(matrix: csr_matrix):
    """
    find the number of nonzero elements at each row of the CSR matrix
    """
    indptr = matrix.indptr
    return indptr[1:] - indptr[:-1]


def find_nonempty_rows(matrix: csr_matrix,
                       ) -> np.ndarray:
    """
    find indices of non-empty rows of a CSR matrix
    """
    return np.nonzero(elements_per_row(matrix))[0]


def find_empty_rows(matrix: csr_matrix,
                    ) -> np.ndarray:
    """
    find indices of empty rows of a CSR matrix
    """
    return np.nonzero(elements_per_row(matrix) == 0)[0]


@timer
def onehot_labels_toarray(onehot_labels: np.ndarray,
                          verbose: bool = True,
                          ) -> np.ndarray:
    """
    Convert one-hot labels to dense array of labels

     reordered (matmul) sparse_labels -> dense relabelled array
       e.g.
       [ 1 3 2 ] @ [ 0 ... 1 0  -> [ 2 ... 1 3 ]
                     0 ... 0 1
                     1 ... 0 0]

    Onehot labels must have shape (num_labels x num_samples)
    and all nonzero entries should be 1
    """
    print(np.arange(onehot_labels.shape[0]))
    print(onehot_labels.shape)

    dense_array = (np.arange(onehot_labels.shape[0]) @ onehot_labels).astype(np.int32)

    if verbose:
        print(f"Dense array of labels: {dense_array}")

    return dense_array


@timer
def labels_to_onehot(labels: np.ndarray,
                     verbose: bool = True,
                     normalize_by_numlabels: bool = True,
                     ):
    """
    convert an array of labels to a 
    num_labels x num_samples sparse one-hot matrix

    Labels MUST be integers starting from 0,
    but can have gaps in between e.g. [0,1,5,9]
    """

    assert np.issubdtype(labels.dtype, np.integer), (
        f"labels provided are of type {labels.dtype}. "
        f"Should be integer.\n"
    )

    assert np.amin(labels) >= 0, (
        f"Some of the labels have negative values.\n"
        f"All labels must be 0 or positive integers.\n"
    )

    labels_list = np.unique(labels)
    max_label = np.amax(labels_list)
    # Note: onehot matrix will have max_label + 1 rows because 0 is a label
    num_labels = len(labels_list)  # this is the number of used labels
    num_samples = len(labels)
    num_per_label = []

    # initialize the fields for the CSR
    # ---------------------------------

    indptr = np.zeros((max_label + 2,), dtype=np.int32)
    indices = np.zeros((num_samples,), dtype=np.int32)
    data = np.ones_like(indices, dtype=np.float64)

    if verbose:
        print(f"\n--- {num_labels} labels, {num_samples} samples ---\n"
              f"Maximum label is {max_label}"
              f"initalized {indptr.shape} index ptr: {indptr}\n"
              f"initalized {indices.shape} indices: {indices}\n"
              f"initalized {data.shape} data: {data}\n")

    # update index pointer and indices row by row
    # -------------------------------------------

    for label in range(max_label + 1):

        label_indices = np.nonzero(labels == label)[0]
        num_current_label = len(label_indices)
        num_per_label.append(num_current_label)

        previous_ptr = indptr[label]
        current_ptr = previous_ptr + num_current_label
        indptr[label + 1] = current_ptr

        if verbose:
            print(f"indices for label {label}: {label_indices}\n"
                  f"previous pointer: {previous_ptr}, "
                  f"current pointer: {current_ptr}\n")

        if current_ptr >= previous_ptr:
            indices[previous_ptr:current_ptr] = label_indices

        if normalize_by_numlabels:
            data[previous_ptr:current_ptr] /= num_current_label

    onehot_matrix = csr_matrix((data, indices, indptr),
                               shape=(max_label + 1, num_samples))

    # assert len(num_per_label) == (max_label + 1) , (
    #     "error computing numbers of each label"
    # )

    return onehot_matrix, labels_list, num_per_label
