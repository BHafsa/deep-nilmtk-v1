
from .log import *
from nilmtk.api import API

from time import time
import mlflow


def setup(experiment, experiment_name, results_path= '../results', mlflow_repo ='../mlflow'):
    """
    Runs NILM experiments as defined with a NILMt-API

    Args:
        experiment (Dict): Experiment definition
        experiment_name (string): The name of the current experiment
        results_path (str, optional): [description]. Defaults to '../results'.
    """
    f = start_logging(experiment_name)
    print(f'run {experiment_name}')
    mlflow.set_tracking_uri(mlflow_repo)
    start = time()
    api_res = API(experiment)  
    time_exec = round((time()-start)/60,1)
    ################### RESULTS ###################
    print('Experiment took: {} minutes'.format(time_exec))
    save_results(api_res, time_exec , experiment_name= experiment_name, path = results_path)
    log_results(experiment, api_res)
