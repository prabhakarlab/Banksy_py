import re, gc
import pandas as pd
from IPython.display import display
from banksy.main import LeidenPartition
from banksy.labels import Label, match_label_series
import leidenalg
from sklearn.metrics.cluster import adjusted_rand_score

def run_partition(
        processing_dict: dict,
        resolutions: list,          
        num_nn: int = 50,
        num_iterations: int = -1,
        partition_seed: int = 1234,
        match_labels: bool = True,
        annotations = None,
        **kwargs) -> dict:
    '''
    Main driver function that runs banksy; 
    1. Create neighbours graph in BANKSY Space
    2. Convert tos shared nearest neighbours
    3. Convert to igraph format for Leignalg package
    4. Leiden clustering is conducted using Leiden algorithm.

    Args:
        processing_dict (dict): The processing dictionary containing:
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
        'no_annotation': True,
        'shared_nn_max_rank ': 3,
        'shared_nn_min_shared_nbrs' : 5,
        'verbose': True
    }
    options.update(kwargs)

    results = {}

    for nbr_weight_decay in processing_dict:
        
        print(f"Decay type: {nbr_weight_decay}")

        for lambda_param in processing_dict[nbr_weight_decay]:
            if not isinstance(lambda_param, float):
                continue # skip other dictionary keys except lambda parameters

            print(f"Neighbourhood Contribution (Lambda Parameter): {lambda_param}")
        
            adata_temp = processing_dict[nbr_weight_decay][lambda_param]["adata"]

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
                    if options['no_annotation']:
                        results[param_str] = {
                            "decay": nbr_weight_decay,
                            "lambda_param": lambda_param,
                            "num_pcs":pca_dim,
                            "resolution":resolution,
                            "num_labels":label.num_labels,
                            "labels":label,
                            "adata": processing_dict[nbr_weight_decay][lambda_param]["adata"]
                        }

                    else:
                        results[param_str] = {
                            "decay":nbr_weight_decay,
                            "lambda_param": lambda_param,
                            "num_pcs":pca_dim,
                            "resolution":resolution,
                            "num_labels":label.num_labels,
                            "labels":label,
                            "adata": processing_dict[nbr_weight_decay][lambda_param]["adata"],
                            "ari" : adjusted_rand_score(annotations.dense, label.dense),
                            "ari_domain_straight" : adjusted_rand_score(adata_temp .obs["domain"].cat.codes.values, label.dense),
                            "ari_domain_smooth" : adjusted_rand_score(adata_temp.obs["smoothed_manual"].cat.codes.values, label.dense),
                            "ari_domain_smooth2" : adjusted_rand_score(adata_temp .obs["smoothed_manual2"].cat.codes.values, label.dense),
                        }
                
                        adata_temp.obs[param_str] = label.dense

    
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
                
    return pca_dims


def convert2df(results: dict,
               match_labels: bool) -> pd.DataFrame:
    '''
    Auxiliary function to convert results dictionary to pandas dataframe

    Also matchs labels if required
    '''
    results_df = pd.DataFrame.from_dict(results, orient = 'index')
    
    # sort by indices
    sort_id = ["decay", "lambda_param", "num_pcs", "resolution"]
    results_df.sort_values(by=sort_id, ascending=True, inplace=True)

    print("\nAfter sorting Dataframe")
    print(f"Shape of dataframe: {results_df.shape}")
    # find the maximum number of labels across all parameter sets
    # -----------------------------------------------------------

    max_num_labels = results_df["num_labels"].max()
    print(f"Maximum number of labels: {max_num_labels}\n")

    ## If match labels are required
    if match_labels:
            results_df["relabeled"], max_num_labels = match_label_series(
            results_df["labels"],
            extra_labels_assignment="greedy",
            verbose=False)
            print('\nMatched Labels')
    
    display(results_df)
    gc.collect()
    return results_df, max_num_labels