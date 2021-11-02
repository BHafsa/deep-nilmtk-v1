#!/usr/bin/env python
from setuptools import setup

setup(
    name='deep_nilmtk',
    version='0.0.1',    
    description='A pytorch-based DNN-NILM package.',
    url='https://github.com/shuds13/pyexample',
    author='Hafsa Bousbiat',
    author_email='eh_bousbiat@esi.dz',
    packages=['deep_nilmtk'],
    install_requires=[
        'mlflow>=1.15.0',
        'optuna>=2.2.0',
        'arviz>=0.11.2',
        'scikit-image>=0.17.2',
        'scikit-learn>=0.23.2',
        'scikit-plot>=0.3.7',
        'pytorch-lightning>=1.2.5',
        'torch>=1.8.0',
        'torchmetrics>=0.2.0',
        'torchvision>=0.9.0',
        'tqdm>=4.59.0',
        'traitlets>=4.1.0',
        'statsmodels>=0.12.2',
],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.6',
    ],
)



