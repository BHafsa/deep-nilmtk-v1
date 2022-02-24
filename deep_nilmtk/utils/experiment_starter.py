import logging

from .logger import *
from nilmtk.api import API

from time import time
import mlflow



def setup(experiment, experiment_name, results_path, mlflow_repo):
    """Runs NILM experiments as defined with a NILMt-API
    :param experiment: Experiment definition
    :type experiment: dict
    :param experiment_name: The name of the current experiment
    :type experiment_name: str
    :param results_path: The path to the results folder, defaults to '../results'
    :type results_path: str, optional
    :param mlflow_repo: The path to the Mlflow folder, defaults to '../mlflow'
    :type mlflow_repo: str, optional
    """
    print(f'run {experiment_name}')

    __resultpath__ = results_path
    if not os.path.exists(f'{results_path}'):
        os.path.exists(f'{results_path}')
    logging.info(f'The results of the current experiment are saved in {results_path}')
    mlflow.set_tracking_uri(mlflow_repo)
    logging.info(f'The MLflow logs are saved in  {mlflow_repo}')
    start = time()
    for method in experiment['methods']:
        experiment['methods'][method].result_path(results_path)
        experiment['methods'][method].hparams.update({
            'exp_name':experiment_name
        })
    api_res = API(experiment)
    time_exec = round((time() - start) / 60, 1)
    print('Experiment took: {} minutes'.format(time_exec))
    save_results(api_res, time_exec, experiment_name=experiment_name, path=results_path)
    log_results(experiment, api_res)
    return api_res
