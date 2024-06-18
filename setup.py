from setuptools import setup, find_packages

ld = """ # Banksy

This package wraps Banksy so it can be installed and used using PyPI.

All Banksy credit goes to the Banksy authors. Please check the original publication for more details: [Nature Article](https://www.nature.com/articles/s41588-024-01664-3).

If you use this package, please cite the original Banksy publication.

 """

setup(
    name='banksy_py',
    version='0.0.7',
    description='Spatial Domain analysis using Banksy.',
    long_description=ld,
    long_description_content_type='text/markdown',
    author = 'Nigel Chou;Yifei Yue',
    author_email = 'Nigel_Chou@gis.a-star.edu.sg',
    mantainer='albert.plaplanas@sanofi.com',
    mantainer_email='albert.plaplanas@sanofi.com',
    packages=find_packages(),
    url='https://github.com/AlbertPlaPlanas/Banksy_py',
    install_requires=[
        'anndata==0.8.0',
        'anyio>=4.0.0',
        'certifi',
        'h5py>=3.10.0',
        'igraph>=0.10.6',
        'importlib-metadata>=4.6',
        'joblib>=1.3.2',
        'jsonschema>=4.19.1',
        'jsonschema-specifications>=2023.7.1',
        # 'matplotlib<3.9.0',
        'matplotlib==3.6.2', # SO
        'matplotlib-inline>=0.1.6',
        # 'numpy>=1.24.4, <2.0',
        'numpy==1.22.4', # SO
        # 'pandas>=2.1.1',
        'pandas==1.5.2',
        'pickleshare>=0.7.5',
        # 'scanpy>=1.9.5',
        'scanpy==1.9.1', # SO
        #'scikit-learn>=1.3.1',
        'scikit-learn==1.2.0', # SO
        # 'scipy>=1.11.3',
        'scipy==1.10.0', # SO
        'seaborn>=0.13.0',
        'tornado>=5.1',
        'umap-learn>=0.5.4',
        'urllib3<4.0',
        'websocket-client<3.0',
    ],
    python_requires='>=3.8',
)

from setuptools import setup, find_packages

