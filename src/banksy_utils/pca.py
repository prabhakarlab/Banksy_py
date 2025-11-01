"""
Function to compute noise-equivalent singular value 
for selection of number of PCs, as described in (Moffit et. al. 2018)

Also, functions to plot variance contributions and singular values

Nigel 4 mar 2021
"""

import numpy as np
from sklearn.decomposition import PCA

from banksy_utils.time_utils import timer

from typing import Tuple, List, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@timer
def noise_equiv_singular_value(data: np.ndarray,
                               num_permutations: int = 50,
                               average_type: str = "mean",
                               verbose: bool = True,
                               ) -> Tuple[float, np.ndarray]:
    """
    get the noise-equivalent maximum singular value of a data matrix
    Each column will be seperately randomly permuted and singular values computed
    """
    assert data.ndim == 2, f"Data has {data.ndim} dimensions, should have 2"

    n_rows, ncols = data.shape

    # set up random number generator for permuting columns
    rng = np.random.default_rng()

    # shuffle columns (n times) and compute highest singular value
    # ------------------------------------------------------------

    all_singular_values = np.zeros((num_permutations,))

    for n in range(num_permutations):

        temp = data.copy()

        # permute each row seperately
        for col in range(ncols):
            temp[:, col] = rng.permutation(temp[:, col])

        all_singular_values[n] = PCA(n_components=1).fit(temp).singular_values_[0]

    if verbose:
        print(f"List of all permuted top "
              f"singular values:\n{all_singular_values}\n")

    # Average the singular values
    # ---------------------------

    if average_type == "mean":
        return np.mean(all_singular_values), all_singular_values

    elif average_type == "median":
        return np.median(all_singular_values), all_singular_values

    else:
        raise ValueError(
            "Average type not recognised. Should be 'mean' or 'median'."
        )


def plot_singular_values(pca: PCA,
                         noise_highest_sv: Union[float, np.ndarray] = None,
                         title: str = None,
                         figsize: Tuple[int, int] = (6, 3),
                         ax=None,
                         ) -> None:
    """
    Plot variance contribution for each component (elbow plot)
    :param pca: PCA object from scikit-learn (Must already be fit to data)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    singular_values = pca.singular_values_
    num_pcs = len(singular_values)
    pcs = np.arange(1, num_pcs + 1)

    ax.plot(pcs, singular_values, color='royalblue', marker='o')

    if noise_highest_sv is not None:

        if isinstance(noise_highest_sv, float):

            ax.hlines(y=noise_highest_sv,
                      xmin=0, xmax=num_pcs + 1,
                      linewidth=1, color='r')

        elif isinstance(noise_highest_sv, np.ndarray):

            mean_sv = np.mean(noise_highest_sv)
            std_sv = np.std(noise_highest_sv)

            ax.hlines(y=mean_sv,
                      xmin=0, xmax=num_pcs + 1,
                      linewidth=std_sv * 2, color='r', alpha=0.2)
            ax.hlines(y=mean_sv,
                      xmin=0, xmax=num_pcs + 1,
                      linewidth=0.2, color='firebrick')

    ax.set_xticks(pcs)
    ax.set_xticklabels(pcs)
    ax.set_xlabel("number of components")
    ax.set_ylabel("singular values")
    if title is not None:
        ax.set_title(title)


def plot_remaining_variance(pca: PCA,
                            title: str = None,
                            figsize: Tuple[int, int] = (6, 3),
                            ax=None,
                            ) -> None:
    """
    Plot variance contribution for each component (elbow plot)
    :param pca: PCA object from scikit-learn (Must already be fit to data)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    remaining_variance = 1 - np.cumsum(pca.explained_variance_ratio_)
    components = np.arange(1, len(remaining_variance) + 1)
    ax.plot(components, remaining_variance,
            color='forestgreen', marker='o')

    ax.set_xticks(components)
    ax.set_xticklabels(components)
    ax.set_xlabel("number of components")
    ax.set_ylabel("remaining variance")
    if title is not None:
        ax.set_title(title)
