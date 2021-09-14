Hyper-parameters
=======================================

The hyper-parameter optimization is performed using Optuna. It can be done in
two steps as follows:

1. :Declare the hyper-parameter set:

Declare the set of parameter to be optimized inside the model through 
the definition of a static function return a dictionnary of values. The following Listing
provide an example of such function suggesting parameters for the window size, 
the normlization type, as well as the :

.. code-block:: python

    @staticmethod
    def suggest_hparams(self, trial):
        
        norm_ = trial.suggest_categorical('normalize', ['z-norm', 'lognorm'])
        window_length = trial.suggest_int('in_size', low=50, high=1800) 
        window_length += 1 if window_length % 2 == 0 else 0
        return {
            'input_norm': norm_,
            'output_norm':norm_,
            'in_size': window_length,
            }

2. :Specify the use of Optuna:

.. code-block:: python
   :emphasize-lines: 19

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
                   'max_nb_epochs':max_nb_epochs,
                   'use_optuna':True,
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

.. note::
   The use of optuna will generate several models and they will all have 
   saved as checkpoints in the corresponding folder splitted into different subfolders 
   labelled according to the trial ID.
