from deep_nilmtk import setup, NILMExperiment


max_nb_epochs = 8
sequence_length = 183
experiment = {
   'power': {'mains': ['active'],'appliance': ['active']},
   'sample_rate': 6,
   'threshold':{
     'kettle':500
   },
   'appliances': [
        #  'fridge',
        #  'washing machine',
        #  'dish washer',
        #  'microwave',
         'kettle',
       ],
   'artificial_aggregate': False,
   'DROP_ALL_NANS': True,
   'methods': {
      
        'Seq2Point': NILMExperiment({
               "model_name": 'Seq2Pointbaseline', 
               'input_norm':'z-norm',
               'target_norm':'z-norm',
               'in_size': sequence_length,
              #  'use_optuna':True,
              #  'n_trials':20,
               'feature_type':'mains',
               'seq_type':'seq2point', 
               'max_nb_epochs':max_nb_epochs
               }), 
   },

   'train': {
     'datasets': {
      'ukdale': {
         'path': '../../Data/ukdale.h5',
         'buildings': {
               1: {
                  'start_time': '2015-01-01',
                  'end_time': '2015-03-30'
           }
         }
      }
     }
   },
     'test': {
     'datasets': {

       'ukdale': {
         'path': '../../Data/ukdale.h5',
         'buildings': {
               1: {
                   'start_time': '2015-05-01',
                   'end_time':  '2015-05-30'
                 }
         }
       }
     },
        'metrics':['mae','nde','f1score', 'accuracy', 'precision'],
     }

 }

setup(experiment,  experiment_name = 'experiment_stage_1')
