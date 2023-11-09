import re, gc, time
import pandas as pd
from IPython.display import display
from banksy.main import LeidenPartition
from banksy.labels import Label, match_label_series
import anndata
import leidenalg
from sklearn.metrics.cluster import adjusted_rand_score

import sys
import warnings
try:
    import rpy2
except ModuleNotFoundError:
    warn_str = ("\nNo rpy2 installed. BANKSY will run, but mclust will not work.\n"+
                "Note: you can still use the default leiden option for clustering.\n"+
                "Install rpy2 and R in your conda environment if you want to use mclust for clustering.\n")
    warnings.warn(warn_str)

def run_mclust_partition(
        banksy_dict: dict,
        partition_seed: int = 1234,
        annotations = None,
        num_labels: int = None,
        **kwargs 
        ) -> dict: 
        '''
        Main driver function that runs the banksy computation; 
        1. Create neighbours graph in BANKSY Space
        2. Convert tos shared nearest neighbours
        3. Convert to igraph format for Leignalg package
        4. Leiden clustering is conducted using Leiden algorithm.

        Args:
            banksy_dict (dict): The processing dictionary containing:
            |__ nbr weight decay
            |__ lambda_param
                |__ anndata  

            resolutions: Resolution of the partition
                
            num_nn (int), default = 50: Number of nearest neighbours

            num_iterations (int), default = -1: 

            partition_seed (int): seed for partitioning (Leiden) algorithm
            
            match_labels (bool); default = True: Determines if labels are kept consistent across different hyperparameter settings

        Optional args (kwargs):
            Other parameters to the Leiden Partition

            shared_nn_max_rank (int), default = 3

            shared_nn_min_shared_nbrs (int), default = 5
        
        Returns:
            results_df (pd.DataFrame): A pandas dataframe containing the results of the partition
        '''

        if "rpy2" not in sys.modules:
            raise ModuleNotFoundError(
                "No rpy2 installed. Install rpy2 and R in your conda environment if you want to use mclust for clustering step")

        options = {
            'no_annotation': True,
        }
        options.update(kwargs)

        results = {}

        for nbr_weight_decay in banksy_dict:
            
            print(f"Decay type: {nbr_weight_decay}")

            for lambda_param in banksy_dict[nbr_weight_decay]:
                if not isinstance(lambda_param, float):
                    continue # skip other dictionary keys except lambda parameters

                print(f"Neighbourhood Contribution (Lambda Parameter): {lambda_param}")
            
                adata_temp = banksy_dict[nbr_weight_decay][lambda_param]["adata"]

                pca_dims = get_pca_dims(adata_temp)
                print("PCA dims to analyse:", pca_dims)

                for pca_dim in pca_dims:

                    if isinstance(pca_dim, str):
                        continue # skip full concatenated matrices
                    
                    print("\n" + "=" * 100 + 
                        f"\nSetting up partitioner for (nbr decay = {nbr_weight_decay}), "
                        f"Neighbourhood contribution = {lambda_param}, "
                        f"PCA dimensions = {pca_dim})\n" + "=" * 100 + "\n")
                    
                    used_obsm = f"reduced_pc_{pca_dim}"

                    print(f"\nFitting M-cluster algorithm with {num_labels} label\n" + "-" * 30 + "\n")

                    # Main partition via the m-cluster
                    # ---------------------------------------------------------------

                    print(f'Clustering PC_Embedding in obsm: {used_obsm}\n of type: {type(adata_temp.obsm[used_obsm])}')

                    adata = mclust_R(adata_temp, 
                                        num_labels, 
                                        model_names='EEE', 
                                        used_obsm=used_obsm, 
                                        random_seed=partition_seed)


                    # store results in dictionary
                    # ---------------------------------------------------------------

                    param_str = f"{nbr_weight_decay}_pc{pca_dim}_nc{lambda_param:0.2f}_labels{num_labels:0.2f}_mclust"
                    if (options['no_annotation']) and (not annotations):
                        print("No annotated labels")
                        results[param_str] = {
                            "decay": nbr_weight_decay,
                            "lambda_param": lambda_param,
                            "num_pcs":pca_dim,
                            "num_labels":num_labels,
                            "labels":adata.obs['mclust'],
                            "adata": banksy_dict[nbr_weight_decay][lambda_param]["adata"]
                        }

                    else:
                        print("Computing ARI for annotated labels")
                        results[param_str] = {
                            "decay":nbr_weight_decay,
                            "lambda_param": lambda_param,
                            "num_pcs": pca_dim,
                            "num_labels": num_labels,
                            "labels": adata.obs['mclust'],
                            "adata": banksy_dict[nbr_weight_decay][lambda_param]["adata"],
                            "ari" : adjusted_rand_score(annotations.dense, adata.obs['mclust']),
                        }
                
                        adata_temp.obs[param_str] = adata.obs['mclust']
                        print(f"Label Dense {adata.obs['mclust']}")
        
        results_df, max_num_labels = convert2df(results, match_labels = False,  resolution = False)
        return results_df, max_num_labels



def mclust_R(adata: anndata.AnnData,
              num_cluster: int,
              model_names: str ='EEE',
              used_obsm: str ='emb_pca',
              random_seed: int =1234):
        """\
        Clustering using the mclust algorithm.
        The parameters are the same as those in the R package mclust.

        This function is modified from STAGATE's original implementation of mclust
        see https://github.com/zhanglabtools/STAGATE/blob/main/STAGATE/utils.py
        """
        import numpy as np
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri as np2ri

        np.random.seed(random_seed)
        robjects.r.library("mclust")
        rpy2.robjects.numpy2ri.activate()
        r_random_seed = robjects.r['set.seed']
        r_random_seed(random_seed)
        rmclust = robjects.r['Mclust']
        
        embedding_np_array  = adata.obsm[used_obsm]
        print(f'obsm {used_obsm} : of type {type(embedding_np_array)} | shape {embedding_np_array.shape}')
        print(f'Array element is finite: {np.isfinite(embedding_np_array).all()}')

        r_embedding_vec = np2ri.numpy2rpy(embedding_np_array)

        res = rmclust(r_embedding_vec, num_cluster, model_names)
        mclust_res = np.array(res[-2]).astype('int')
        adata.obs['mclust'] = mclust_res.tolist()

        print(adata.obs['mclust'])
        return adata


def run_Leiden_partition(
        banksy_dict: dict,
        resolutions: list,          
        num_nn: int = 50,
        num_iterations: int = -1,
        partition_seed: int = 1234,
        match_labels: bool = True,
        annotations = None,
        max_labels: int = None,
        **kwargs) -> dict:
    '''
    Main driver function that runs Leiden partition across the banksy matrices stored in banksy_dict.
    See the original leiden package: https://leidenalg.readthedocs.io/en/stable/intro.html

    Args:
        banksy_dict (dict): The processing dictionary containing:
        |__ nbr weight decay
          |__ lambda_param
            |__ anndata  

        resolutions: Resolution of the partition
            
        num_nn (int), default = 50: Number of nearest neighbours

        num_iterations (int), default = -1: 

        partition_seed (int): seed for partitioning (Leiden) algorithm
        
        match_labels (bool); default = True: Determines if labels are kept consistent across different hyperparameter settings

    Optional args (kwargs):
        Other parameters to the Leiden Partition

        shared_nn_max_rank (int), default = 3

        shared_nn_min_shared_nbrs (int), default = 5
    
    Returns:
        results_df (pd.DataFrame): A pandas dataframe containing the results of the partition
    '''
    options = {
        'shared_nn_max_rank ': 3,
        'shared_nn_min_shared_nbrs' : 5,
        'verbose': True
    }
    options.update(kwargs)

    results = {}

    for nbr_weight_decay in banksy_dict:
        
        print(f"Decay type: {nbr_weight_decay}")

        for lambda_param in banksy_dict[nbr_weight_decay]:
            if not isinstance(lambda_param, float):
                continue # skip other dictionary keys except lambda parameters

            print(f"Neighbourhood Contribution (Lambda Parameter): {lambda_param}")
        
            adata_temp = banksy_dict[nbr_weight_decay][lambda_param]["adata"]

            pca_dims = get_pca_dims(adata_temp)
            print("PCA dims to analyse:", pca_dims)

            for pca_dim in pca_dims:
                
                if isinstance(pca_dim, str):
                    continue # skip full concatenated matrices
                
                print("\n" + "=" * 100 + 
                    f"\nSetting up partitioner for (nbr decay = {nbr_weight_decay}), "
                    f"Neighbourhood contribution = {lambda_param}, "
                    f"PCA dimensions = {pca_dim})\n" + "=" * 100 + "\n")
                
                banksy_reduced = adata_temp.obsm[f"reduced_pc_{pca_dim}"]
              
                partitioner = LeidenPartition(
                    banksy_reduced,
                    num_nn = num_nn,
                    nns_have_weights = True,
                    compute_shared_nn = True,
                    filter_shared_nn = True,
                    shared_nn_max_rank = options['shared_nn_max_rank '],
                    shared_nn_min_shared_nbrs = options['shared_nn_min_shared_nbrs'],
                    verbose = options['verbose'], 
                )

                if max_labels and not resolutions:
                    print("No resolutions indicated, trying to search for a suitable resolution from labels")
                    print(f"Finding resolution that matches {max_labels}")
                    resolutions = find_resolutions(
                        partitioner,
                        max_labels,
                        num_iterations,
                        partition_seed
                    )

                for resolution in resolutions:

                    print(f"\nResolution: {resolution}\n" + "-" * 30 + "\n")

                    # Main partition via the Leiden Algorithm
                    # ---------------------------------------------------------------

                    label, modularity = partitioner.partition(
                        resolution = resolution,
                        partition_metric = leidenalg.RBConfigurationVertexPartition,
                        n_iterations = num_iterations,
                        seed = partition_seed,
                    )

                    # store results in dictionary
                    # ---------------------------------------------------------------

                    param_str = f"{nbr_weight_decay}_pc{pca_dim}_nc{lambda_param:0.2f}_r{resolution:0.2f}"
                    if not annotations:
                        print("No annotated labels")
                        results[param_str] = {
                            "decay": nbr_weight_decay,
                            "lambda_param": lambda_param,
                            "num_pcs":pca_dim,
                            "resolution":resolution,
                            "num_labels": label.num_labels,
                            "labels": label,
                            "adata": banksy_dict[nbr_weight_decay][lambda_param]["adata"]
                        }

                    else:
                        print("Computing ARI for annotated labels")
                        results[param_str] = {
                            "decay": nbr_weight_decay,
                            "lambda_param": lambda_param,
                            "num_pcs": pca_dim,
                            "resolution": resolution,
                            "num_labels": label.num_labels,
                            "labels": label,
                            "adata": banksy_dict[nbr_weight_decay][lambda_param]["adata"],
                            "ari" : adjusted_rand_score(annotations.dense, label.dense),
                            # For determining ari of specific domains
                            #"ari_domain_straight" : adjusted_rand_score(adata_temp .obs["domain"].cat.codes.values, label.dense),
                            #"ari_domain_smooth" : adjusted_rand_score(adata_temp.obs["smoothed_manual"].cat.codes.values, label.dense),
                            #"ari_domain_smooth2" : adjusted_rand_score(adata_temp .obs["smoothed_manual2"].cat.codes.values, label.dense),
                        }
                
                        adata_temp.obs[param_str] = label.dense
                        
                        print(f"Label Dense {label.dense}")

    
    results_df, max_num_labels = convert2df(results, match_labels)

    return results_df, max_num_labels

def get_pca_dims(adata_temp):
    '''
    Auxiliary function to gets PCA dimensions from the anndata object
    '''
    pca_dims = []

    for key in adata_temp.obsm_keys():

        print(key, "\n")
        match = re.search(r"reduced_pc_([0-9]+)$", key)
        if match:
            pca_dims.append(int(match.group(1)))
            continue
        
        # else try to search for a float 
        match = re.search(r"reduced_pc_(0\.[0-9]+)$", key)
        if match:
            pca_dims.append(float(match.group(1)))
        
    return pca_dims


def convert2df(results: dict,
               match_labels: bool,
               resolution: bool = True) -> pd.DataFrame:
    '''
    Auxiliary function to convert results dictionary to pandas dataframe

    Also matchs labels if required
    '''
    results_df = pd.DataFrame.from_dict(results, orient = 'index')
    
    # sort by indices
    if resolution:
        sort_id = ["decay", "lambda_param", "num_pcs", "resolution"]
    else:
        sort_id = ["decay", "lambda_param", "num_pcs"]

    results_df.sort_values(by=sort_id, ascending=True, inplace=True)

    print("\nAfter sorting Dataframe")
    print(f"Shape of dataframe: {results_df.shape}")

    # find the maximum number of labels across all parameter sets
    # -----------------------------------------------------------
    ## If match labels are required
    if match_labels:
            results_df["relabeled"], max_num_labels = match_label_series(
            results_df["labels"],
            extra_labels_assignment="greedy",
            verbose=False
            )
            
            print('\nMatched Labels')
    
    max_num_labels = results_df["num_labels"].max()
    display(results_df)
    gc.collect()
    return results_df, max_num_labels

def find_resolutions(partitioner,
                    max_labels,
                    num_iterations,
                    partition_seed,
                    starting_resolution = 0.6,
                    step = 0.05,
                    max_iter = 25) -> int:
    
    '''Auxiliary function to find the resolution that matches the specified labels
    Returns a list containing a single resolution value that matches the number of clusters'''
    resolution = starting_resolution 
    label, _ = partitioner.partition(
                        resolution = resolution,
                        partition_metric = leidenalg.RBConfigurationVertexPartition,
                        n_iterations = num_iterations,
                        seed = partition_seed,
    )

    num_labels = label.num_labels
    # perform a simple linear start for labels
    iter = 1
    while (num_labels != max_labels) and (iter < max_iter):
        if num_labels < max_labels:
            # We have less labels than specified
            # increase resolution to increase the number of labels
            resolution += step
            label, _ = partitioner.partition(
                    resolution = resolution,
                    partition_metric = leidenalg.RBConfigurationVertexPartition,
                    n_iterations = num_iterations,
                    seed = partition_seed)
            num_labels = label.num_labels

        else:
            # There are now more labels then 
            resolution -= step
            label, _ = partitioner.partition(
                    resolution = resolution,
                    partition_metric = leidenalg.RBConfigurationVertexPartition,
                    n_iterations = num_iterations,
                    seed = partition_seed)
            num_labels = label.num_labels
        
        iter +=1
    
    print(f"Resolution found: {resolution} for labels: {max_labels}")
    return [resolution,]
