Cross Validation
=======================================

Cross validation is performed using sklearn.model_selection.TimeSeriesSplit which is 
a suitable splitting strategy for timeseries. The use of cross-validation is triggered 
whenever a number of klfolds > 1 is specified. 

.. note::
   Other paramaters of sklearn.model_selection.TimeSeriesSplit can also be specified 
   using same labels as the original function.

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
                   'kflods': 5,
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