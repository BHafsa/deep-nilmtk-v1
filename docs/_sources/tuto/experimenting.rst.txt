Experimenting with baselines
=======================================

Deep-NILMtk is an easy to use tool designed with full compatibility with NILMtk. As such, 
experimenting with the different models included in the tool is done using the NILMtk-API.

1. :Experiment Definition:

The experiment is defined as dictionnary in respect with NILMtk-API. The 
model specific hyper-parameter can be identified within the other parameters and they
will directly considered for building the model. The highlighted area illustrates an
example of experiment using the WAVENET model. 

.. code-block:: python
   :emphasize-lines: 12-19

    experiment = {
      'power': {'mains': ['active'],'appliance': ['active']},
      'sample_rate': 6,
      'appliances': [ 
            'fridge',
            'washing machine',
           	'dish washer',
          ],
      'artificial_aggregate': False,
      'DROP_ALL_NANS': True,
      'methods': {
              'WAVENILM': NILMExperiment({
                   "model_name": 'WAVENILM', 
                   'context_size': 481, 
                   'input_norm':'z-norm',
                   'target_norm':'z-norm',
                   'feature_type':'mains',
                   'max_nb_epochs':max_nb_epochs
                   }),
      },

      'train': {
        'datasets': {
         data: {
            'path': data_path,
            'buildings': {
                  1: {
                        'start_time': '2015-01-04',
                        'end_time': '2015-03-30'
              }
            }
         }
        }
      },
        'test': {
        'datasets': {
        
          data: {
            'path': data_path,
            'buildings': {
                  1: {
                        'start_time': '2015-04-16',
                        'end_time': '2015-05-15'
                    }
            }
          }
        },
            'metrics':['mae','nde','f1score', 'rmse']
        }

    }


2. :Setup calling:

For a smooth execution of the experiment, Deep_nilmtk incoporates a setup function that 
executes the experiments and generates teh corresponding artifacts directly linked to the 
related experiment in MLflow for an easy tracking of experiments and their findings.

.. code-block:: python

   from deep_nilmtk.utils import setup

   setup(experiment,  experiment_name = 'example_experiment', results_path='../results', mlflow_repo='../mlflow')

3. :Results checking:

After execution of the experiments, the results can be directly viewed using the 
Mlflow UI as follows:

.. code-block:: shell

    cd mlflow
    mlflow ui

