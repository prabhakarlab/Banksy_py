import gc
import os
from typing import List, Union, Tuple

import anndata
from matplotlib import pyplot as plt, ticker as ticker

from banksy.embed_banksy import generate_banksy_matrix
from banksy.labels import Label
from banksy.main import concatenate_all
from banksy.cluster_methods import run_Leiden_partition, run_mclust_partition
from banksy_utils.umap_pca import pca_umap


def run_banksy_multiparam(adata: anndata.AnnData,
                          banksy_dict: dict,
                          lambda_list: List[int],
                          resolutions: List[int],
                          color_list: Union[List, str],
                          max_m: int,
                          filepath: str,
                          key: Tuple[str],
                          match_labels: bool = False,
                          pca_dims: List[int] = [20, ],
                          savefig: bool = True,
                          annotation_key: str = "cluster_name",
                          max_labels: int = None,
                          variance_balance: bool = False,
                          cluster_algorithm: str = 'leiden',
                          partition_seed: int = 1234,
                          add_nonspatial: bool = True,
                          **kwargs):
    options = {
        'save_all_h5ad': True,
        'save_name': 'slideseq_mousecerebellum_',
        'no_annotation': True,
        's': 50,
        'a': 1.0
    }

    options.update(kwargs)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    banksy_dict, banksy_matrix = generate_banksy_matrix(adata,
                                                        banksy_dict,
                                                        lambda_list,
                                                        max_m,
                                                        variance_balance=variance_balance)

    # Add nonspatial banksy matrix
    if add_nonspatial:
        banksy_dict["nonspatial"] = {0.0: {"adata": concatenate_all([adata.X], 0, adata=adata), }}

    pca_umap(banksy_dict,
             pca_dims=pca_dims,
             plt_remaining_var=False)

    if annotation_key:
        annotations = Label(adata.obs[annotation_key].cat.codes.tolist())
    else:
        annotations = None

    # Clustering algorithm
    if cluster_algorithm == 'leiden':
        print(f'Conducting clustering with Leiden Parition')
        results_df, max_num_labels = run_Leiden_partition(
            banksy_dict=banksy_dict,
            resolutions=resolutions,
            num_nn=50,
            num_iterations=-1,
            partition_seed=partition_seed,
            match_labels=match_labels,
            annotations=annotations,
            max_labels=max_labels
        )

    elif cluster_algorithm == 'mclust':
        print(f'Conducting clustering with mcluster algorithm')
        try:
            import rpy2
        except ModuleNotFoundError:
            print(f'Package rpy2 not installed, try pip install')

        match_labels = False
        results_df, max_num_labels = run_mclust_partition(
            banksy_dict=banksy_dict,
            partition_seed=partition_seed,
            match_labels=match_labels,
            annotations=annotations,
            num_labels=max_labels,
        )

    for params_name in results_df.index:
        gc.collect()

        adata_temp = results_df.loc[params_name, 'adata']
        raw_labels = results_df.loc[params_name, 'labels']
        num_clusters = results_df.loc[params_name, 'num_labels']
        lambda_p = results_df.loc[params_name, "lambda_param"]

        if annotation_key:
            ari_temp = results_df.loc[params_name, "ari"]
            ari_label = f'\nari = {round(ari_temp, 2)}'
        else:
            ari_label = ""

        raw_clusters = []

        if isinstance(raw_labels, Label):
            raw_labels = raw_labels.dense
        for i in raw_labels:
            raw_clusters.append(color_list[i])

        fig = plt.figure(figsize=(8, 8), constrained_layout=True)
        ax = fig.gca()
        title = f'λ = {round(lambda_p, 2)}' + ari_label

        print(f'Anndata {adata_temp.obsm}')
        subplot_sc(ax,
                   adata_temp,
                   key,
                   raw_clusters,
                   num_clusters,
                   title)

        if savefig:
            save_name = os.path.join(filepath, f'{params_name}.png')
            fig.savefig(save_name, format='png', dpi=108)

    return results_df


def subplot_sc(ax: plt.Axes,
               adata_temp: anndata.AnnData,
               coord_keys: Tuple[str],
               clusters_colormap: List,
               num_clusters: int,
               title: str) -> None:

    ax.scatter(adata_temp.obs[coord_keys[0]].values,
               adata_temp.obs[coord_keys[1]].values,
               c=clusters_colormap,
               vmin=0, vmax=num_clusters - 1,
               s=50000 / adata_temp.n_obs, alpha=1.0)

    ax.set_aspect('equal', 'datalim')
    ax.set_title(title, fontsize=20, fontweight="bold", )
    ax.axes.invert_yaxis()

    # Turn of ticks and frame
    ax.set(frame_on=False)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
