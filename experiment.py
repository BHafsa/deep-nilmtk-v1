from deep_nilmtk import setup, NILMExperiment


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
          'WAVENET': NILMExperiment({
               "model_name": 'WAVENILM', 
               'in_size': 481, 
               'input_norm':'z-norm',
               'target_norm':'z-norm',
               'feature_type':'mains',
               'seq_type': "seq2seq",
               'kernel_size':3,
               'max_nb_epochs':1,
               }), 
   },

   'train': {
     'datasets': {
      'ukdale': {
         'path': '../Deep-NILMtk/data/REFIT.h5',
         'buildings': {
               1: {
                     'start_time': '2015-01-04',
                     'end_time': '2015-01-06'
           }
         }
      }
     }
   },
     'test': {
     'datasets': {

       'ukdale': {
         'path': '../Deep-NILMtk/data/REFIT.h5',
         'buildings': {
               1: {
                     'start_time': '2015-04-16',
                     'end_time': '2015-04-18'
                 }
         }
       }
     },
         'metrics':['mae','nde','f1score', 'rmse']
     }

 }

setup(experiment,  experiment_name = 'example_experiment')