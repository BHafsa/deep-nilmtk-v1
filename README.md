

<div align="center">    
 
# Deep-NILMtk    

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)]()


 **`Documentation`** |
------------------- |
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://bhafsa.github.io/deep-nilmtk-v1/) |
 


 
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
hyper-params optimization)in  the case of NILM.

## NILM Datasets

The NILMtk interafce is energy datasets with Hierarchical Data Format (HDF). The NIMLtk 
contains converters for several datasets. The current package is compatible with this same 
data format. 

We provide a [here](https://drive.google.com/drive/folders/1IBuelHpdvPf_0KrSyNxeUSBdRFI5LS5j)
a link  to four pre-converted datasets.



## How to install?

```bash
conda create --name deep-nilmtk
conda activate deep-nilmtk
# nilmtk can also be installed form source
conda install -c nilmtk nilmtk

conda install pip
pip install .
```

## Template starter project

A template project is offered in the following [link](https://github.com/BHafsa/DNN-NILM-template) leveraging on docker to offer a pre-configured execution environement and avoid any compatibility issues that could encounter the developers.

## Use the toolkit with colab

A **tutorial notebook** can be found [here](https://colab.research.google.com/drive/1ZVBB1Nd-b6LLDCK_FR8xiDrDjdJb5XcS?usp=sharing) leveraging resources available in Colab for training the NILM models. The tutorial was presented in [NILM22 workshop](http://nilmworkshop.org/2022/#program).


## What's new about Deep-NILMtk?

### I. Compatibility with several DL frameworks

Deep learning NILM models are implemented using the different standard deep learning frameworks. This aspect makes the 
comparability between models difficult. To overcome this shortcoming, Deep NILMtk decouples the NILM pipeline from
the deep learning framework. 
For now, the toolkit is compatible with both Tensorflow and PyTorch which are the most popular frameworks. 
However, the toolkit also takes into consideration the possibility of extending this with new frameworks 
by implementing a trainer respecting the following interface.

```python
class TrainerImplementor(metaclass=abc.ABCMeta):
    """
    Trainer Interface
    """
    @abc.abstractmethod
    def fit(self, model, dataset,
            chkpt_path,exp_name,results_path, logs_path,  version,
            batch_size=64, epochs=20, use_optuna=False, learning_rate=1e-4, optimizer='adam', patience_optim=5,
            train_idx=None, validation_idx=None):
        pass

    @abc.abstractmethod
    def get_dataset(self,  main, submain, seq_type,
                    in_size, out_size, point_position,
                    target_norm, quantiles= None,  loader= None, **kwargs):
        pass

    @abc.abstractmethod
    def train_test_split(self, dataset, train_idx, val_idx):
        pass

    @abc.abstractmethod
    def predict(self, model, mains):
        pass

    @abc.abstractmethod
    def load_model(self,model, path):
        pass
```

### II. Customisable NILM pipeline 

Due to design choices, NILM toolkits present sofar require implementing to pipeline form scratch 
despite the fact that many contribution share the majority of pipeline's components. To overcome this limitation, 
Deep-NILMtk reliefs the developers from this burden, the toolkit allows altering element from the
NILM pipeline  [notebook](link_here).

### III. Experiment templates

Comparability and benchmarking are important part of the research process. However, they remain as an obstacle 
in the NILM scholarship due to different experimental setups. Deep-nilmtk offer the option to define experiment 
design as templates that can become the base for benchmarking new models with existing models. A tutorial on how this 
can be implemented is available in the following [notebook](link_here).

### IV. ML tools

Machine learning is fast developing field and tools, that  help ML developers in their work, are being 
massively introduced. The deep learning NILM scholarship can benefit from these tools to help scholars 
efficiently achieve their goals with minimal coding. Deep-NILMtk contains three of the most popular tools 
used in the deep learning:

#### 1. Automatic tracking of experiments :

The toolkit offers automatic experiment tracking for all experiments
executed. This aspect allows digging deeper in the training process and 
gain more insights about the training and validation setups.
For generated artifacts, they are 
not automatically linked to the experiments. It is done only if 
the user explicitly specifies it using the corresponding parameter.

```python
# Model Definition    
#Experiment Definition
'model': NILMExperiment({
            # ... other params go here
            'log_artifacts':True})
```


#### 2. Hyper-parameter optimisation
Hyper parameters optimization is an important part in deep learning research 
where it is mandatory for scholars to optimise the parameters of their model. The offered 
toolkit therefore offers the option of performing such studies with minimal coding efforts 
to facilitate the implementation for scholars. It requires only the definition of a static 
function in the model class using and the explicit declaration of using Optuna, as 
illustrated in the following:

```python
# Model Definition
class nilm_model:
    ...
    @staticmethod
    def suggest_hparams(self, trial):
        # Returns a dictionary of suggested values for each parameter
        ...
        return  params_dictionnnay
    
#Experiment Definition
'model': NILMExperiment({
            # ... other params go here
            'use_optuna':True})
```


#### 3. Time-split cross validation

Building Deel Learning Models requires to split the data into training, validation and 
test sets. However, to better assess the robustness of the model using cross-validation 
is in some scenarios very common. To help easily perform such studies, the offered toolkit implements 
the sklearn.model_selection.
TimeSeriesSplit which is a suitable splitting strategy for timeseries. 
The use of cross-validation is triggered whenever a number of  ```kfolds > 1``` is specified, as illustrated in the 
following:

```python
#Experiment Definition using 3 folds cross-validation
'model': NILMExperiment({
            # ... other params go here
            'kfolds':3})
```



## Notes

The package requires the availability of a cuda driver installation and 
a compatible cudaDNN ```(cudadnn >= v8.1.0, cuda>=11.2)```.



## Copyright and license

Code released under the [MIT Licence](). 
