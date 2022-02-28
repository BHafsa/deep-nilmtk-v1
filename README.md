

<div align="center">    
 
# Deep-NILMtk    

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)

| **`Documentation`**                                                                                           |
|---------------------------------------------------------------------------------------------------------------|
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://bhafsa.github.io/deep-nilmtk-v1/) |

</div>
Deep-NILMtk is an open source package designed specifically 
for deep models applied to solve NILM. It implements the general NILM 
pipeline independently of the deep 
learning backend. In its current version 
the toolkit considers two of the most popular 
deep learning pipelines. 
The training and testing phases are fully compatible with NILMtk.
Several MLOps tools are included that aim to support the 
evaluation and the validation of those models (e.g., cross-validation,
hyper-params optimization) case of NILM.



## What's included



### Automatic tracking of experiments

### Hyper-parameter optimisation

### Time-split cross validation

 Customisable pipeline



## How to use it?


Examples of publicly available NILM datasets can be found [here](https://drive.google.com/drive/folders/1IBuelHpdvPf_0KrSyNxeUSBdRFI5LS5j).

## How to install?

```bash
conda create --name deep-nilmtk
conda activate deep-nilmtk
# nilmtk can also be installed form source
conda install -c nilmtk nilmtk

conda install pip
pip install .
```

## Notes

The package requires the availability of a cuda driver installation and 
a compatible cudaDNN (cudadnn >= v8.1.0, cuda>=11.2).

## Authors

## Licence