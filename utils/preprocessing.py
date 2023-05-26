"""
Utility Function to preprocess annadata 

Input: filepath, adata_filename

Returns: Annadata object

Yifei 15 May 2023
"""

from anndata import AnnData
import anndata
import scanpy as sc

def preprocess_data(adata: anndata.AnnData):
    '''
    Preprocess Anndata object by removing duplicate names,
    creating a ["MT"] series for MT
    Also calculates the QC metrics and putting them in place in the adata object

    Input Args:
      adata: (Anndata object)

    Return: 
        pre-processed Annadata object
    '''
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # Calulates QC metrics and put them in place to the adata object
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    return adata
