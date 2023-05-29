'''
Yifei May 2023

Good old setup.py for users to install banksy with `pip install .`
To be tested.
'''

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')  # noqa

# get version
exec(open('banksy/version.py').read())

setup(
    name='BANKSY',
    version=__version__,  # noqa
    description='Source for BANKSY',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github', # TBC
    author="Viphul Singhal, Nigel Chou, Joseph Lee", # TBC
    author_email='vipulsinghal@gis.a-star.edu.sg', # TBC
    packages=find_packages(),
    python_requires='>=3.7, <4',
    install_requires=[ # TBC
        "numpy",
        "scipy",
        "pandas",
        "networkx",
        "matplotlib",
        "scikit-learn",
    ],
)