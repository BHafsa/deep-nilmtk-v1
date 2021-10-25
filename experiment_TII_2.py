from deep_nilmtk import setup, NILMExperiment


max_nb_epochs = 30
sequence_length = 480
nfolds = 3


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
         'tempPool': NILMExperiment({
               "model_name": 'tempPool', 
               'experiment_label':'regression',
               'in_size': sequence_length, 
               'input_norm':'z-norm',
               'target_norm':'z-norm',
               'feature_type':'mains',
               'kfolds':3,
               'learning_rate':10e-7,
               'max_nb_epochs':max_nb_epochs,
               'task_type':'regression',
               'hidden_dim':64,
               'threshold_method':'at',
               'max_power':{
                      'kettle': 2600,
                  },
                  'threshold':{
                     'kettle': 30.,
                  },
                  'min_on':{
                    'kettle': 2,
                  },
                  'min_off':{
                      'kettle': 0,
                  },
               
               }), 
        'Seq2Point': NILMExperiment({
               "model_name": 'Seq2Pointbaseline', 
               'in_size': sequence_length, 
               'input_norm':'z-norm',
               'target_norm':'z-norm',
               'kfolds':3,
               'feature_type':'mains',
               'seq_type':'seq2point', 
                'point_position': 'midpoint',
               'max_nb_epochs':max_nb_epochs
               }),
        'Seq2Seq': NILMExperiment({
               "model_name": 'Seq2Seqbaseline', 
               'in_size': sequence_length, 
               'input_norm':'z-norm',
               'target_norm':'z-norm',
               'Kfolds':3,
               'feature_type':'mains',
               'seq_type':'seq2seq', 
               'max_nb_epochs':max_nb_epochs
               }),
        'Bert4NILM': NILMExperiment({
          "model_name": 'BERT4NILM', 
          'in_size': sequence_length, 
          'input_norm':'z-norm',
          'feature_type':'mains',
          'stride':10,
	        'kfolds':3,
          'max_nb_epochs':max_nb_epochs,
          'cutoff':{
              'aggregate': 6000,
              'kettle': 2500,
          },
          'threshold':{
             'kettle': 2200,
          },
          'min_on':{
            'kettle': 20,
          },
          'min_off':{
              'kettle': 0,
          },
        }),
       
        'UNETNiLMSeq2P':NILMExperiment({
                "model_name": 'UNETNiLMS2P', 
                'in_size': sequence_length,
                'feature_type':'mains', 
                'input_norm':'z-norm',
                'target_norm':'z-norm',
                'kfolds':3,
                'learning_rate':10e-4,
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
        'metrics':['mae','nde','f1score', 'accuracy', 'precision', 'recall'],
     }

 }

setup(experiment,  experiment_name = 'experiment_stage_2')
