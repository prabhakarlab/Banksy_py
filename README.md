# <b> BANKSY </b>

## <b> Neighbourhood Informed Clustering for Cell Type and Tissue Domain Annotation </b>

### *Vipul Singhal\*, Nigel Chou\*, Joseph Lee, Jinyue Liu, Wan Kee Chock, Li Lin, YunChing Chang, Erica Teo, Hwee Kuan Lee, Kok Hao Chen<sup>#</sup> and Shyam Prabhakar<sup>#</sup>*

## <b> Overview  </b>

This repository contains the code base and examples for `BANKSY: A Spatial Clustering Algorithm that Unifes Cell Typing and Tissue Domain Segmentation.` BANKSY is a method for clustering spatial transcriptomic data by augmenting the transcriptomic profile of each cell with an average of the transcriptomes of its spatial neighbors. 

By incorporating neighborhood information for clustering, BANKSY is able to:

1. **improve cell-type assignment** in noisy data

2. distinguish **subtly different cell-types** stratified by microenvironment

3. **identify spatial zones** sharing the same microenvironment

BANKSY is applicable to a wide variety of spatial technologies (e.g. 10x Visium, Slide-seq, MERFISH) and scales well to large datasets. For more details on use-cases and methods, see the preprint. 

In this Python version of BANKSY (compatible with Scanpy), we show how BANKSY can be used for task **1** (improving cell-type assignment) using Slide-seq and Slide-seq V2 mouse cerebellum datasets. This code base was used to generate figure 2 of the manuscript. The R version of BANKSY is available here (https://github.com/prabhakarlab/Banksy).

## <b> Prerequisites <a name="prereqs"></a>  </b>

### System requirements: <a name="sysreqs"></a>

Machine with 16 GB of RAM. (All datasets tested required less than 16 GB). No non-standard hardware is required.

### Software requirements: <a name="softreqs"></a>

This software requires the following packages and has been tested on the following versions:

2.	Python >= 3.9
3.	Scanpy >= 1.8.1, Anndata >= 0.7.1
4.	numpy >= 1.21
5.	scipy >= 1.6
6.	umap >= 0.5.1
7.	scikit-learn >= 0.24.2
8.	python-igraph >= 0.9

## <b> Getting Started <a name="getstart"></a> </b>

   
### Installation (recommended via Anaconda) <a name="install"></a>

1. We recommend users to download and install [Anaconda](https://www.anaconda.com/distribution/#download-section) if you have not done so.
2.	Create a new conda environment: [managing conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). 
3.	Use `conda install -c conda-forge scanpy python-igraph leidenalg` as described in the [scanpy documentation](https://scanpy.readthedocs.io/en/stable/installation.html). This will install scanpy as well as other neccesary dependencies. (Alternatively, users can install scanpy using pip via`pip install scanpy`).
4.	Download BANKSY Python package or clone this github repository using `git clone https://github.com/prabhakarlab/Banksy_py`
5.	Try out BANKSY by running the examples in the provide ipython notebooks: [`slideseqv1_analysis.ipynb`](./slideseqv1_analysis.ipynb) and/or [`slideseqv2_analysis.ipynb`](./slideseqv2_analysis.ipynb). More details and description of different steps in the process are provided within the notebooks.

 Note that due to the large size of the `slideseq_v2` dataset (~2.2 Gb of raw data), we recommend users to download the data from the [original source](https://singlecell.broadinstitute.org/single_cell/study/SCP948) and save it in the `data/slide_seq/v2` folder.



### <b> General Steps of the BANKSY algorithm </b>

1. **Preprocess** and **normalize** gene-cell matrix. This includes filtering out cells and genes by various criteria, and selecting the most highly variable genes or HVGs (for sequencing based spatial omics datasets e.g. 10X Visium and Slide-seq only). z-score by gene using `banksy.main.zscore` or `scanpy.pp.scale`.  Functions provided in the Scanpy package handle most of these steps. Parameters and filtering criterion may vary by spatial technology and dataset source.

2. Generate a **spatial graph** which defines spatial neighbour relationships using `banksy.main.generate_spatial_weights_fixed_nbrs`. This outputs a sparse adjacency matrix defining the graph. Visualize these with `utils.plotting.plot_graph_weights`.
   
   - `decay types`: `uniform` weights all neighbours equally, `reciprocal` weights neighbours by 1/r where r is the distance from neighbouring cell to the index cell. `ranked` ranks neighbouring cells by distance with farther cells having higher rank, then sets Gaussian decay by rank. Sum of neighbour weights are always normalized to 1 for each cell.
   - `generate_spatial_weights_fixed_radius` (not used in paper) generates a spatial graph where each cell is connected to all cells within a given radius. This leads to variable numbers of neighbours per cell.


<img src="./data/accessories/readme_pic1.png" alt="drawing" width="600"/>


3. Generate **neighbour expression matrix** (*n<sub>cells</sub>* by *n<sub>genes</sub>*) using spatial graph to average over spatial neighbours. The neighbour matrix can be computed simply by sparse matrix multiplication of the spatial graph's adjacency matrix with the gene-cell matrix.

4. Scale original expression matrix by *√(1 - λ)* and neighbour expression matrix by *√(λ)* and concatenate matrices to obtain **neighbour-augmented expression matrix** (*n<sub>cells</sub>* by *2n<sub>genes</sub>*) using `banksy.main.weighted_concatenate` with ```neighbourhood_contribution``` = λ. These operations are performed on the numerical data `adata.X`; use `banksy.main.bansky_matrix_to_adata` to recover Anndata object with the appropriate annotations.


<img src="./data/accessories/readme_pic2.png" alt="drawing" width="600"/>


*The following steps are identical to single cell RNA seq analysis:*

5. Use **PCA** to reduce expression matrix (either neighbour-augmented or original for comparison) to (*n<sub>cells</sub>* by *n<sub>PCA_dims</sub>*)
    
6. Find neighbours in expression space and **cluster** using graph-based clustering. Here we find expression-neighbours and perform Leiden clustering following the implemenation in Giotto.

### <b> Other useful tools in package </b>

* **`banksy.main.LeidenPartition`**   
Finds neighbours in expression space and performs Leiden clustering. Aims to replicate implementation from the Giotto package as of 2020 to align with R version of the code. Note that scanpy also has a [Leiden clustering implemenation](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.leiden.html) with a different procedure for defining expression neighbours that can be used as an alternative. BANKSY is compatible with any clustering algorithm that takes a feature-cell matrix as input.

* **`labels.Label`**   
Object for convenient computation with class labels. Converts labels to sparse one-hot vector for fast computation of connectivity across clusters with spatial graph. To obtain an array of integer labels in the usual format (e.g. [1, 1, 5, 2, ...]), use `Label.dense`.

## <b> Contributing </b>

Bug reports, questions, request for enhancements or other contributions can be raised at the [issue
page](https://github.com/prabhakarlab/Banksy_py/issues).

## <b> Authors <a name="authors"></a> </b>

* **Nigel Chou** (https://github.com/chousn)

* **Yifei Yue** (https://github.com/yifei-1021)

## <b> Acknowledgments <a name="ack"></a> </b>

* **Vipul Singhal** - *developed R version of BANKSY, compatible with seurat* - (https://github.com/vipulsinghal02)

* **Joseph Lee** - *developed R version of BANKSY, compatible with seurat* - (https://github.com/jleechung)

Refer to `requirements.txt` for the supported versions of different packages

### <b> Citations </b>
If you want to use or cite BANKSY, please refer to the following paper:

```
BANKSY: A Spatial Clustering Algorithm that Unifes Cell Typing and Tissue Domain Segmentation, (submitted).  
```
Article preprint can be accessed at the [bioRxiv repository](https://www.biorxiv.org/content/10.1101/2022.04.14.488259v1)

### <b> License </b>
This project is licensed under The GPLV3 license. See the [LICENSE.md](./LICENSE.md) file for details

### <b> Unlabelled Clustering using BANKSY with varying neighbourhood contributions (λ) </b>

<p align="center">
   <i> SlideSeq_v1 Dataset </i>
  <img src="./data/accessories/slideseq_v1_animate.gif" />
</p>

<p align="center">
   <i> SlideSeq_v2 dataset </i>
  <img src="./data/accessories/slideseq_v2_animate.gif" />
</p>
