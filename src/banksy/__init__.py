"""
BANKSY: Spatial Clustering Algorithm that Unifies Cell-Typing and Tissue Domain Segmentation

This package provides tools for spatial transcriptomics analysis by incorporating
neighborhood information into cell clustering.
"""

__version__ = "1.0.0"
__author__ = "Nigel Chou, Yifei Yue, Vipul Singhal"

# Don't import everything eagerly to avoid circular imports
# Users should import from submodules:
# from banksy.initialize_banksy import initialize_banksy
# from banksy.run_banksy import run_banksy_multiparam
# from banksy.labels import Label
# from banksy.cluster_methods import run_Leiden_partition, run_mclust_partition

__all__ = ["__version__"]
