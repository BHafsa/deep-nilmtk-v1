
from .log import *
from nilmtk.api import API

from time import time
import mlflow


def setup(experiment, experiment_name, results_path= './output/results', mlflow_repo ='./output/mlruns/'):
    """Runs NILM experiments as defined with a NILMt-API

    :param experiment: Experiment definition
    :type experiment: dict
    :param experiment_name: The name of the current experiment
    :type experiment_name: str
    :param results_path: The path to the resulst folder, defaults to '../results'
    :type results_path: str, optional
    :param mlflow_repo: The path to the Mlflow folder, defaults to '../mlflow'
    :type mlflow_repo: str, optional
    """
    
    
    print(f'run {experiment_name}')
    mlflow.set_tracking_uri(mlflow_repo)
    start = time()
    api_res = API(experiment)  
    time_exec = round((time()-start)/60,1)
    ################### RESULTS ###################
    print('Experiment took: {} minutes'.format(time_exec))
    save_results(api_res, time_exec , experiment_name= experiment_name, path = results_path)
    log_results(experiment, api_res)
