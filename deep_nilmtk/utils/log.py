

import pickle

import warnings
warnings.filterwarnings("ignore")

import sys

import mlflow
import json

def start_logging(filename):
    """
    loggs all the output of the running into  file

    Args:
        filename (str): Name of the file where output will be logged

    Returns:
        file: 
    """
    f = open('../logs/experiment-{}.txt'.format(filename), 'w')
    sys.stdout = f
    return f

def stop_logging(f):
    f.close()
    sys.stdout = sys.__stdout_

def save_results(api_results_f1,time, experiment_name, path='../results'):
    error_df_f1 = api_results_f1.errors
    error_keys_df_f1 = api_results_f1.errors_keys
    # Save results in Pickle file.
    df_dict = {
        'error_keys': api_results_f1.errors_keys,
        'errors': api_results_f1.errors,
        'train_mains': api_results_f1.train_mains,
        'train_submeters': api_results_f1.train_submeters,
        'test_mains': api_results_f1.test_mains,
        'test_submeters': api_results_f1.test_submeters,
        'gt': api_results_f1.gt_overall,
        'predictions': api_results_f1.pred_overall,
        'execution_time':time,

    }
  
    pickle.dump(df_dict, open(f"{path}/{experiment_name}.p", "wb"))
  
    for metric, f1_errors in zip(error_keys_df_f1, error_df_f1):
        ff_errors = round(f1_errors, 3)
        ff_errors.to_csv(f'{path}/{experiment_name}_{metric}.csv', sep='\t')

def log_results(experiment, api_res):
    """
    This function logs the final results of the testing in the correspanding
    experiment for each disaggregator
    
    Args:
        experiment (dict): dict of the experiment in nilmtk format
        api_res (nilmtk-api reult): results of the execution as provided by nilmtk
    """
    for appliance in experiment['appliances']:  
        # Select the corresponding experiment and run ID
        mlflow.set_experiment(appliance)
        for disaggregator_name, disaggregator in experiment['methods'].items():
          with mlflow.start_run(disaggregator.run_id[appliance]):
            # Log test metrics for each appliance
            for metric in api_res.errors_keys:
                metric_value = api_res.errors[api_res.errors_keys.index(metric)][disaggregator_name][appliance]
                metric_label = experiment['test']['metrics'][api_res.errors_keys.index(metric)]
                mlflow.log_metric(f'test_{metric_label}', metric_value)
                mlflow.set_tag('mlflow.note.content', f'''

                Information about the data used for this experiment:
                Training: {json.dumps(experiment['train']['datasets'])}
                Testing: {json.dumps(experiment['test']['datasets'])} 

                ''')

