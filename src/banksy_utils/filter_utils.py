import time, gc
import scanpy as sc
import anndata
from typing import Tuple
import numpy as np
from copy import deepcopy as dcp

sc.settings.verbosity = 1 # Hide information verbose

def print_max_min(adata: anndata.AnnData) -> None:
    '''Displays the max and min of anndata dataset'''
    print("Displaying max and min of Dataset")
    print(f"Max: {adata.X.max()}, Min: {adata.X.min()}\n")


def normalize_total(adata: anndata.AnnData) -> anndata.AnnData:
    '''Normalizes the dataset inplace'''
    
    print("--- Max-Min before normalization -----")
    print_max_min(adata)

    sc.pp.normalize_total(adata, inplace = True)
    print("--- Max-Min after normalization -----")
    print_max_min(adata)
    return adata


def filter_cells(adata: anndata.AnnData,
                 min_count: int,
                 max_count: int,
                 MT_filter: int,
                 gene_filter: int) -> anndata.AnnData:
    
    
    '''
    Inital filter of cells by defining thresholds for min_count, max_count.
    Cells with MT and gene counts below the `MT_filter` and `gene_filter` wil be filtered out
    '''
    
    print(f"Cells before filtering: {adata.n_obs}\n")
    sc.pp.filter_cells(adata, min_counts=min_count)
    sc.pp.filter_cells(adata, max_counts=max_count)
    print(f"Cells after count filter: {adata.n_obs}\n")

    adata = adata[adata.obs["pct_counts_mt"] < MT_filter]
    print(f"cells after MT filter: {adata.n_obs}\n")

    sc.pp.filter_genes(adata, min_cells=gene_filter)
    print(f"Genes after minimum cells per gene filter: {adata.n_vars}\n")
    return adata


def feature_selection(
        adata: anndata.AnnData,
        sample: str,
        coord_keys: Tuple[str],
        hvgs: int = 2000,
        svgs: int = 0,
        load_preprocesssed: bool = True,
        path_to_hvgs: str = None,
        save_genes: bool = False,
        show_figs: bool = False,
        svg_knn: int = 36) -> anndata.AnnData:


        '''
        Parameters:
            adata (anndata): AnnData object containing cell-by-gene

            coord_keys: a tuple of strings to access the spatial coordinates of the anndata object

            hvgs (int): Number of highly variable genes to select

            svgs (int): Number of spatially variable genes to select


        '''
        try:
            from scgft.tl import get_svgs, get_hvgs
        except ModuleNotFoundError:
            print("Error scgft module is not in directory")
            print('Try settings get `svgs = 0`')
            return adata
        
        start_time = time.perf_counter()
    
        if svgs and hvgs:
            print(f'Before filtering: {adata}')
            hvg_genes = get_hvgs(dcp(adata),
                        top_hvgs = hvgs,
                        flavor='seurat')
                        
            svg_genes = get_svgs(dcp(adata),
                        top_svgs = svgs,
                        num_neighbours = svg_knn,
                        spatial_key = coord_keys[2],
                        plot_svg=show_figs)
                        
            adata = adata[:, adata.var['genename'].isin(hvg_genes) | adata.var['genename'].isin(svg_genes)]
            print(f"Number of filtered dataset in SVGs {adata.var['genename'].isin(svg_genes).value_counts()}")
            print(f"Number of filtered dataset in HVGs {adata.var['genename'].isin(hvg_genes).value_counts()}")
            
        elif svgs:
            try:
                from scgft.tl import get_svgs
            except ModuleNotFoundError:
                print("Error scgft module is not in directory to filter by SVGs")
                print('Try settings get `svgs = 0`')
                return adata
            
            svg_genes = get_svgs(dcp(adata),
                                top_svgs = svgs,
                                num_neighbours = svg_knn,
                                spatial_key=coord_keys[2],
                                plot_svg=show_figs)
            
            print(len(svg_genes))
            adata = adata[:, adata.var['genename'].isin(svg_genes)]
            print(adata.var['genename'].isin(svg_genes))

        ### Filter HVGs by the top 1000
        elif hvgs:
            if load_preprocesssed:
                print(f'reading hvg-genes from {path_to_hvgs}')
                hvg_genes = np.loadtxt(path_to_hvgs, delimiter=",", dtype=str)

                adata = adata[:, adata.var['gene_ids'].isin(hvg_genes)]

            else:
                print('Using Scanpy\'s get HVG')
                hvg_genes = get_hvgs(dcp(adata),
                            top_hvgs = hvgs,
                            flavor ='seurat_v3')
                        
                adata = adata[:, adata.var['genename'].isin(hvg_genes)]
        
        print(f"Ran filter in {round(time.perf_counter()-start_time,2)} s")
        print(f'No of SVGs: {svgs}, No of HVGs: {hvgs}')

        if save_genes:
            # Saves list of genes as csv file
            gene_ids =  adata.var['gene_ids'].to_numpy()
            title = f'Filtered_genes_for_sample_{sample}.csv'
            np.savetxt(title, 
                gene_ids,
                delimiter =", ", 
                fmt ='% s')

        gc.collect()

        return adata


def filter_hvg(adata: anndata.AnnData,
               n_top_genes: int,
               flavor: str = "seurat"):
    
    '''Creates a copy of the original annadata object
    Applies log-transformation, 
    then filter by highly-varaiable genes

    Input: 
        anndata: The anndata object 
        n_top_genes: The top highly variable genes to be filtered by 
        flavor = "seurat" or "seurat_v3: 
            if flavor is seurat, a log-transform is applied
            otherwise if flavor is seurat_v3, we do not apply the log-transform
    
    Returns the transformed and filtered dataset
    '''
    ### TO BE EDITED
    if flavor.__eq__("seurat"):
        # If we want to apply log-transform
        adata_log1p = adata.copy()
        sc.pp.log1p(adata_log1p)
        
        print("--- Normalized and log-transformed data -----")
        print_max_min(adata_log1p)

        sc.pp.highly_variable_genes(adata_log1p, 
                                flavor=flavor, 
                                n_top_genes=n_top_genes)
        
        hvg_filter = adata_log1p.var["highly_variable"]
    
    elif flavor.__eq__("seurat_v3"):
        # using the raw count data without log-transform
        print("--- Normalized data -----")
        sc.pp.highly_variable_genes(adata, 
                                flavor=flavor, 
                                n_top_genes=n_top_genes)
        
        hvg_filter = adata.var["highly_variable"]
    
    else:
        print(f"Flavour: {flavor} is not recognized, please use either \'seurat\' or \'seurat_v3\'")
        return

    # save a copy of the unfiltered
    adata_allgenes = adata.copy()
    adata = adata[:, hvg_filter]

    print(f"Displaying dataset after filtering by HVG")
    print(adata)
    return adata, adata_allgenes