

import pickle

import warnings
warnings.filterwarnings("ignore")

import sys

import mlflow
import json
from ..disaggregate import NILMExperiment
def start_logging(filename, params):
    """Logs the output of the execution in the specified file

    :param filename: The name of the log file
    :type filename: str
    """
    f = open(f'{params["logs_path"]}/experiment-{filename}.txt' , 'w')
    sys.stdout = f
    return f

def stop_logging(f):
    """Stops logging the output in the file f

    :param f: Logs file
    :type f: file
    """
    f.close()
    sys.stdout = sys.__stdout_

def save_results(api_results_f1,time, experiment_name, path='../results'):
    """This function persists teh output of the predictions in a pickel file

    :param api_results_f1: Execution results as returned by the NILMtk-API
    :param time: execution time
    :param experiment_name: Name of the experiment
    :type experiment_name: str
    :param path: Path to the results folder, defaults to '../results'
    :type path: str, optional
    """
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

def log_results(experiment, api_res, multi_appliance= True):
    """This function logs the final results of the testing in the correspanding
    experiment for each disaggregator

    :param experiment: dict of the experiment in nilmtk format
    :type experiment: dict
    :param api_res: results of the execution as provided by nilmtk
    :type api_res: nilmtk-api result
    """
    
            
    for disaggregator_name, disaggregator in experiment['methods'].items():
        mlflow.set_experiment('Multi-appliance')
        for appliance in experiment['appliances']:
            if isinstance(disaggregator, NILMExperiment) and disaggregator.hparams['multi_appliance']:
                with mlflow.start_run(disaggregator.run_id['Multi-appliance']):
                  # Log test metrics for each appliance
                    for metric in api_res.errors_keys:
                        for appliance in experiment['appliances']: 
                            metric_value = api_res.errors[api_res.errors_keys.index(metric)][disaggregator_name][appliance]
                            metric_label = experiment['test']['metrics'][api_res.errors_keys.index(metric)]
                            mlflow.log_metric(f'test_{metric_label}_{appliance}', metric_value)
                    mlflow.set_tag('mlflow.note.content', f'''
            
                        Information about the data used for this experiment:
                        Training: {json.dumps(experiment['train']['datasets'])}
                        Testing: {json.dumps(experiment['test']['datasets'])} 
        
                      ''')
            else:
                mlflow.set_experiment(appliance)
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

