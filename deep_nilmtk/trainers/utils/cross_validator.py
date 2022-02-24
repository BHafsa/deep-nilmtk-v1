import copy
import logging

from sklearn.model_selection import TimeSeriesSplit
import mlflow
import numpy as np

class CrossValidator:
    """
    Train models using cross validation
    """

    def __init__(self,  kfolds, test_size, gap):
        self.chk_points_repo = []
        self.mlflow_repo = []
        self.kfolds = kfolds
        self.test_size = test_size
        self.gap = gap

    def cross_validate(self,trainer_imp, dataset, model,appliance_name, hparams={}):
        """
        Cross validation on the
        :param main:
        :param target:
        :param model:
        :param nfolds:
        :return:
        """
        fold = TimeSeriesSplit(n_splits=self.kfolds, test_size=self.test_size,
                               gap=self.gap)
        models = {}
        fold_idx = 1
        best_losses = []
        for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(dataset.original_inputs)):
            new_model = model.__class__(hparams)
            fold_ds = trainer_imp.get_dataset(dataset.original_inputs[train_idx[0]:valid_idx[-1], :],
                                            dataset.original_targets[train_idx[0]:valid_idx[-1], :],
                                                seq_type=hparams['seq_type'],
                                                target_norm=hparams['target_norm'],
                                                in_size=hparams['in_size'],
                                                out_size=hparams['out_size'],
                                                point_position=hparams['point_position'])
            with mlflow.start_run():
                # Log parameters of current run
                mlflow.log_params(hparams)
                # Model Training
                model_pl, loss = trainer_imp.fit(
                    new_model, fold_ds,
                    chkpt_path=f'{hparams["checkpoints_path"]}/{appliance_name}/{hparams["model_name"]}/version_{hparams["version"]}/{fold_idx+1}',
                    exp_name=hparams['exp_name'],
                    results_path=hparams['results_path'],
                    logs_path=hparams['logs_path'],
                    version=hparams['version'],
                    batch_size=hparams['batch_size'],
                    epochs=hparams['max_nb_epochs'],
                    optimizer=hparams['optimizer'],
                    learning_rate=hparams['learning_rate'],
                    patience_optim=hparams['patience_optim'],
                    train_idx=train_idx, validation_idx=valid_idx
                )
                run_id= mlflow.active_run().info.run_id
                best_losses.append(loss)

            models[fold_idx] = model_pl
            logging.info(f"Finished training for fold {fold_idx}")
            fold_idx +=1

        logging.info(f"Finished training for the {self.kfolds}")
        return models, run_id, np.array(best_losses).mean()




