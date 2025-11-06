"""
Utility Function to load dataset as anndata object

Input: filepath, adata_filename

Returns: Annadata object

Yifei 15 May 2023
"""
from IPython.display import display

import os
import numpy as np
import pandas as pd
import scipy.sparse as sparse

from anndata import AnnData
import anndata
from typing import Tuple 

def load_adata(filepath: str,
               load_adata_directly: bool,
                adata_filename: str,
                gcm_filename: str = "", 
                locations_filename: str = "",
                coord_keys: Tuple[str] = ("xcoord", "ycoord", "xy_coord")
                ) -> anndata.AnnData:
    
    ### Add extra Option to load from adata or use existing Adata
    ### Add description to explain the function

    '''Utility Function to load dataset as anndata object,
        We assume that all files are in the same folder (filepath)

        Input Args: 
            - filepath (str): Path to folder of the file

            - load_adata_directly (bool): Whether to load annadata file directly, 
                If set to False, will attempt to convert gcm_file and locations_file as anndata object

            - adata_filename (str): File name of the designated Anndata object

        If current annadata is not present and we want to convert raw data to annadata
        
        Optional Args (For converting raw data to Anndata Object if it is not present):  
            - gcm_filename (str): .csv file containing the genome information of the cells
            - locations_filename (str): .csv file containing the x-y coordinates of the cells
            - coord_keys: A tuple of keys to index the x,y coordinates of the anndata object

        Returns: Loaded Annadata and the raw coordinates (raw_x, raw_y) as pd.Series'''
    if load_adata_directly:
            print(f"Looking for {os.path.join(filepath, adata_filename)}")
            if os.path.isfile(os.path.join(filepath, adata_filename)):
                
                # read existing anndata if present in folder
                # ------------------------------------------
                print("Attemping to read Annadata directly")
                adata = anndata.read_h5ad(os.path.join(filepath, adata_filename))
                print("Anndata file successfully loaded!")

            else:
                print(f"No such files {adata_filename} in {filepath}, please check the directory path and file names")
                print(f"Alternatively, try to convert raw files to anndata if by setting \'load_adata_directly = False\'")
    else:
        # If 'load_data_directly is set to false, try to read raw text files and convert to anndata
        try:
            gcm_df = pd.read_csv(os.path.join(filepath, gcm_filename), index_col = 0)
            locations_df = pd.read_csv(os.path.join(filepath, locations_filename), index_col = 0)
            print(f'GCM data successfully read as {gcm_df}\n Location data successfuly read as {locations_df}')
        except:
            Exception("Error occured when reading csv files, check the if files are permissible to read")

        
        sparse_X = sparse.csc_matrix(gcm_df.values.T)
        
        adata = AnnData(X = sparse_X, 
                        obs = locations_df, 
                        var = pd.DataFrame(index = gcm_df.index))
        
        adata.write(os.path.join(filepath, adata_filename))

        adata.obs[coord_keys[0]] = locations_df.loc[:, coord_keys[0]]
        adata.obs[coord_keys[1]] = locations_df.loc[:, coord_keys[1]]
        print("Anndata file successfully written!")
    
    ### Added keys,
    try:
        print(f'Attempting to concatenate spatial x-y under adata.obsm[{coord_keys[2]}]')
        x_coord, y_coord, xy_coord = coord_keys[0], coord_keys[1], coord_keys[2]
        raw_y, raw_x = adata.obs[y_coord], adata.obs[x_coord]
        adata.obsm[xy_coord] = np.vstack((adata.obs[x_coord].values,adata.obs[y_coord].values)).T
        print('Concatenation success!')
    except:
        print(f"Error in concatenating the matrices under adata.obsm[{coord_keys[2]}]\n raw_x, raw_y will return None")
        raw_y, raw_x = None, None

    return raw_y, raw_x, adata

def display_adata(adata: anndata.AnnData) -> None:
    '''
    Print summary / metadata of annadata object
    '''
    print("Displaying adata Object and their attributes")
    print("Adata attributes and dimensions:")
    display(adata)
    print(f"Matrix sparsity: {adata.X.nnz} filled elements "
      f"({adata.X.nnz/adata.X.shape[0]/adata.X.shape[1]:0.2f}) "
      f"out of {adata.X.shape[0]*adata.X.shape[1]}\n"
      f"max: {np.amax(adata.X.data)}, min: {np.amin(adata.X.data)}")

    # y and x locations / coordinates
    # -------------------------------
    print("\nDisplaying observations (adata.obs)")
    display(adata.obs)

    print("Displaying variables (adata.var)")
    display(adata.var)
