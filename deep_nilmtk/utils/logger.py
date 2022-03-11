import os
import glob
from pytorch_lightning.loggers import TensorBoardLogger

import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import sys

import mlflow
import json


class DictLogger(TensorBoardLogger):
    """PyTorch Lightning `dict` logger."""

    # see https://github.com/PyTorchLightning/pytorch-lightning/blob/50881c0b31/pytorch_lightning/logging/base.py

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = []

    def log_metrics(self, metrics, step=None):
        """Logs the training metrics

        :param metrics: the values of the metrics
        :type metrics: dict
        :param step: the ID of the current epoch, defaults to None
        :type step: int, optional
        """
        super().log_metrics(metrics, step=step)
        self.metrics.append(metrics)


def get_latest_checkpoint(checkpoint_path):
    """Returns the latest checkpoint for the model

    :param checkpoint_path: The path to the checkpoints folder
    :type checkpoint_path: str
    :return: the latest checkpoint saved during training

    """

    checkpoint_path = str(checkpoint_path)
    list_of_files = glob.glob(checkpoint_path + '/*.ckpt')

    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
    else:
        latest_file = None
    return latest_file




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
    results = {

        appliance:{
            disaggregator_name:{
            experiment['test']['metrics'][api_res.errors_keys.index(metric)]: \
                api_res.errors[api_res.errors_keys.index(metric)][disaggregator_name][appliance] \
                for metric in api_res.errors_keys
            } for disaggregator_name, disaggregator in experiment['methods'].items()
        } for appliance in experiment['appliances']
    }

    for appliance in results:
        mlflow.set_experiment(appliance)
        for disaggregator_name, disaggregator in experiment['methods'].items():
            with mlflow.start_run(disaggregator.trainer.run_id[appliance]):
                mlflow.log_metrics(results[appliance][disaggregator_name])
                mlflow.set_tag('mlflow.note.content', f'''

                    Information about the data used for this experiment:
                    Training: {json.dumps(experiment['train']['datasets'])}
                    Testing: {json.dumps(experiment['test']['datasets'])}

                  ''')







































def results_dir(logs_path, results_path, figure_path):
    """
    Creates the results directory and its subdirectories
    :param logs_path: loggs path
    :param results_path: results path
    :param figure_path: figures path
    :return:
    """
    logs =    Path(logs_path) # logs/ log files containing the details of the execution
    results = Path(results_path) # results/ csv and pickle files recording the testing results
    figures = Path(figure_path) # figures/ folder to save figures

    logs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)