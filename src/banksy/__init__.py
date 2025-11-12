"""
BANKSY: Spatial Clustering Algorithm that Unifies Cell-Typing and Tissue Domain Segmentation

Building Aggregates with a Neighborhood Kernel and Spatial Yardstick (BANKSY) is a method
for clustering spatial transcriptomic data by augmenting the transcriptomic profile of each
cell with an average of the transcriptomes of its spatial neighbors.

For more details, see: https://github.com/prabhakarlab/Banksy_py
"""

__version__ = "1.3.4"
__author__ = "Nigel Chou, Yifei Yue, Vipul Singhal"

# Users should import from submodules:
# from banksy.initialize_banksy import initialize_banksy
# from banksy.run_banksy import run_banksy_multiparam
# from banksy.embed_banksy import generate_banksy_matrix

__all__ = ["__version__"]
