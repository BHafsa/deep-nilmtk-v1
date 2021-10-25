Experiments Tracking
=======================================

The experiment tracking is performed automaticaly for all experiments
executed using deep_nilmtk. Neverthless, the generated artifcats are not automaticaly
linked in the experiments. It is done only if the user expliciltely specifies it using 
the corresponding parameter. 

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
                   'log_artificat':True,
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

