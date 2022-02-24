import copy
import logging

import optuna
import torch

from .cross_validator import CrossValidator
import mlflow
import numpy as np
from deep_nilmtk.config.models import __models__


class HparamsOptimiser:
    """
    Hyper-parameter opimisation class
    """
    def __init__(self, trainer_impl,  hparams):
        self.model = None
        self.dataset=None
        self.hparam = hparams
        self.model_class = __models__[self.hparam['backend']][self.hparam['model_name']]['model']
        self.trainer_impl = trainer_impl
        self.appliance_name = None

    def optimise(self, model, dataset, appliance_name):
        self.model = model
        self.dataset = dataset
        self.appliance_name = appliance_name
        # Use Optuna fot parameter optimisation of the model
        study = optuna.create_study(study_name=self.hparam['exp_name'], direction="minimize")
        if self.hparam['kfolds'] <= 1:
            study.optimize(self.objective, n_trials=self.hparam['n_trials'], callbacks=[self.save_best_model])
            # Load weights of the model
        else:
            study.optimize(self.objective_cv, n_trials=self.hparam['n_trials'], callbacks=[self.save_best_model])

        chkp_path = f'{self.hparam["results_path"]}/{self.hparam["checkpoints_path"]}/{appliance_name}/{self.hparam["model_name"]}/version_{self.hparam["version"]}/{study.user_attrs["trial_ID"]}'


        model = self.trainer_impl.load_model(model, chkp_path) if  self.hparam['kfolds'] <= 1 else {
            fold: self.trainer_impl.load_model(model, f'{chkp_path}/{fold+1}') for fold in range(self.hparam['kfolds'])
        }

        return model, study.user_attrs["best_run_id"]

    def save_best_model(self, study, trial):
        """Keeps track of the trial giving best results

        :param study: Optuna study
        :param trial: Optuna trial
        """
        if study.best_trial.number == trial.number:
            study.set_user_attr(key="trial_ID", value=trial.number)
            study.set_user_attr(key="best_run_id", value=trial.user_attrs["best_run_id"])

    def objective(self, trial):
        suggested_params = self.suggest_params(trial)
        # TODO: USE the hyper-params
        dataset = self.trainer_impl.get_dataset(self.dataset.original_inputs, self.dataset.original_targets,
                                                seq_type=self.hparam['seq_type'],
                                                target_norm=self.hparam['target_norm'],
                                                in_size=self.hparam['in_size'],
                                                out_size=self.hparam['out_size'],
                                                point_position=self.hparam['point_position'])
        model = self.model.__class__(self.hparam)
        # train the new model
        mlflow.set_experiment(self.appliance_name)
        with mlflow.start_run(run_name=self.hparam['model_name']):
            # Auto log all MLflow from lightening
            # Save the run ID to use in testing phase
            # Log parameters of current run

            version = self.hparam['version']
            self.hparam['version'] = f'{version}/{trial.number}'

            mlflow.log_params(self.hparam)
            model, loss = self.trainer_impl.fit(
                model, dataset,
                chkpt_path=f'{self.hparam["checkpoints_path"]}/{self.appliance_name}/{self.hparam["model_name"]}/version_{self.hparam["version"]}',
                exp_name=self.hparam['exp_name'],
                results_path=self.hparam['results_path'],
                logs_path=self.hparam['logs_path'],
                version=self.hparam['version'],
                batch_size=self.hparam['batch_size'],
                epochs=self.hparam['max_nb_epochs'],
                optimizer=self.hparam['optimizer'],
                learning_rate=self.hparam['learning_rate'],
                patience_optim=self.hparam['patience_optim'])
            self.hparam['version'] = version
            # saving the trained model
            trial.set_user_attr(key='best_run_id', value=mlflow.active_run().info.run_id)
            trial.set_user_attr(key="trial_ID", value=trial.number)

        return loss

    def suggest_params(self, trial):
        suggested_params_func = self.model_class.suggest_hparams
        if callable(suggested_params_func):
            suggested_params = self.model_class.suggest_hparams(trial)
            self.hparam.update(suggested_params)
        else:
            raise Exception(''' No params to optimise by optuna
                             A static function inside the NILM model should provide
                             a dictionnary of params suggested by optuna
                             see documentation for more details. ''')
        return suggested_params

    def objective_cv(self, trial):
        suggested_params = self.suggest_params(trial)
        logging.info(f'Training for the following parameters')
        # TODO: USE the hyper-params

        dataset = self.trainer_impl.get_dataset(self.dataset.original_inputs, self.dataset.original_targets,
                                                seq_type=self.hparam['seq_type'],
                                                target_norm=self.hparam['target_norm'],
                                                in_size=self.hparam['in_size'],
                                                out_size=self.hparam['out_size'],
                                                point_position=self.hparam['point_position'])

        model = self.model.__class__(self.hparam)
        # Use the cv to initialize train over the folds
        cv = CrossValidator(kfolds=self.hparam['kfolds'],
                            test_size=self.hparam['test_size'], gap=self.hparam['gap'])

        mlflow.set_experiment(self.appliance_name)
        version = self.hparam['version']
        self.hparam['version'] = f'{version}/{trial.number}'
        model, run_id, loss = cv.cross_validate(self.trainer_impl,
                                                dataset,
                                                model,
                                                self.appliance_name,
                                                self.hparam)
        self.hparam.update({
            'version': version
        })
        # saving the trained model
        trial.set_user_attr(key='best_run_id', value=run_id)
        trial.set_user_attr(key="trial_ID", value=trial.number)

        return loss
