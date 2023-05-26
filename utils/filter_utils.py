from IPython.display import display
import scanpy as sc
import anndata
sc.settings.verbosity = 1 # Hide information verbose

def print_max_min(anndata: anndata.AnnData) -> None:
    '''Displays the max and min of anndata dataset'''
    print("Displaying max and min of Dataset")
    print(f"Max: {anndata.X.max()}, Min: {anndata.X.min()}\n")


def normalize_total(anndata: anndata.AnnData) -> anndata.AnnData:
    '''Normalizes the dataset inplace'''
    
    print("--- Max-Min before normalization -----")
    print_max_min(anndata)

    sc.pp.normalize_total(anndata, inplace = True)
    print("--- Max-Min after normalization -----")
    print_max_min(anndata)
    return anndata


def filter_hvg(annadata: anndata.AnnData,
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
        adata_log1p = annadata.copy()
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
        sc.pp.highly_variable_genes(annadata, 
                                flavor=flavor, 
                                n_top_genes=n_top_genes)
        
        hvg_filter = annadata.var["highly_variable"]

    # save a copy of the unfiltered
    adata_allgenes = annadata.copy()
    annadata = annadata[:, hvg_filter]

    print(f"Displaying dataset after filtering by HVG")
    display(annadata)
    return annadata, adata_allgenes


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