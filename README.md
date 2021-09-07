

<div align="center">    
 
# Deep-NILMtk 




<!--  
Conference   
-->   
</div>
 
## Description   

This repository contains deep baselines and three recent state-of-the-art models for the task of energy disaggregation implemented using NILMTK's Rapid Experimentation API and Pytorch lightning. 

The repository facilitates also the experiments management and model evaluation through the implementation of recent best practices in DL discipline. It performs automatic management of experiments with MLflow and allows automatic hyperparameters optimisation with Optuna.

## Why Deep-NILMtk  ?

Deep-NILMtk was originally developped to enable PyTorch community from interacting with NILMtk API and benifiting from its features. Then, it was further developed to include more recent models. 

It impelements a variety of features such as CV and hyper-param optimisation with minimal coding  to accelerate and facilitate prototyping and speed up the reseach process.


## How to run   
First, download and install dependencies  

```bash
# clone project   
git clone https://github.com/BHafsa/Deep-NILMtk.git
# install project   
cd Deep-NILMtk 
# Install Dependencies
conda env create -f DEEP-NILMtk-env.yml
conda activate DEEP-NILMtk-env
# Run experiments
cd experiments
python experiment-AL.py
```   


## Example of use

```python
'Seq2Seq': NILMExperiment({
          "model_name": 'Seq2Pointbaseline', # The network architecture
          'context_size': 481, # The sequence sength
          'seq_type' :"seq2point", # The data generation strategy
          'point_position':'median', # The point position in the target seuqence
          'feature_type':'mains', # The type of input features
          'max_nb_epochs':max_nb_epochs, # The numeb rof tarining iterations
      }),
```
## Example of Cross Validation

The use of cross-validation is automatically triggred if a number of folds bigger than 1 is specified. During testing time, an average over all predictions related to models of different folds is returned. 

```python
'Seq2Point':  NILMExperiment({
              # Model parameters
               "model_name": 'Seq2Pointbaseline', 
               'context_size': 481, 
               'seq_type' :"seq2point",
               'point_position':'median',
               'feature_type':'mains',
               # Number of folds
               'kfolds': kfolds, # The number of splits  during training
               # -----------------
               'max_nb_epochs':max_nb_epochs,
               })
```

## Example of Hyper-parameters Optimisation


```python
 NILMExperiment({
              # Model parameters
               "model_name": 'Seq2Pointbaseline', 
               'context_size': 481, 
               'seq_type' :"seq2point",
               'point_position':'median',
               'feature_type':'mains',
               # Optuna's parameters 
               'use_optuna':True,
               'n_trials': 30,
               # ---------------
               'max_nb_epochs':20,
               })

```

Here is an example of a set of hyperparameters to optimize using Optuna:

```python
@staticmethod
    def suggest_hparams(self, trial):
        '''
        Function returning list of params that will be suggested from optuna
    
        Parameters
        ----------
        trial : Optuna Trial.
    
        Returns
        -------
        dict: Dictionary of parameters with values suggested from optuna
    
        '''
        norm_ = trial.suggest_categorical('normalize', ['z-norm', 'lognorm'])
        window_length = trial.suggest_int('context_size', low=50, high=1800) 
        window_length += 1 if window_length % 2 == 0 else 0
        return {
            'input_norm': norm_,
            'target_norm':norm_,
            'context_size': window_length,
            'point_position':trial.suggest_categorical('point_position',['median','last_point'])
            }
```

## Experiments management
Below is an illustartion of mlflow UI that groups the experiments accroding to appliances, allow to insert notes nd compare between experiments related to one appliance. The UI can be viewed after finishing the experiments and obtaining the `mlruns` folder by running the ``` mlflow ui ``` command.

![MLflow](./figures/mlflow.png)

## Dependencies

> You'll need a the following python libraries run the code.

> [mlflow > 1.14.1](https://mlflow.org/docs/latest/index.html)

> [Pytorch > 1.7.0](https://pytorch.org/get-started/locally/)

> [Pytorch-lightning > 1.0.8 ](https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html)

> [Optuna 2.3.0]()

> ``pip install  torch pytorch-lightning optuna SciencePlots``


## Citation   
```
@article{Deep-NILMtk: Towards implementing DL best pratices in NILM,
  title={To},
  author={Bousbiat Hafsa, Anthony Faustine , Lukas Pereira, Christoph Klemenjak},
  journal={Location},
  year={2021}
}
```  

For any enquiries, please contact the main authors.






