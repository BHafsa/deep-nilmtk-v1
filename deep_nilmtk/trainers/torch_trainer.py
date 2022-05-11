import numpy as np

from .trainer_implementor import TrainerImplementor
import torch
import logging
import pytorch_lightning as pl
from deep_nilmtk.utils.logger import DictLogger,  get_latest_checkpoint

from deep_nilmtk.models.pytorch import PlModel
from deep_nilmtk.data.loader.pytorch import GeneralDataLoader

import os
import mlflow

class TorchTrainer(TrainerImplementor):
    def __init__(self):
        self.batch_size = 64

    def log_init(self, chkpt_path, results_path, logs_path,  exp_name, version, patience_check=5):
        """
        Initialise the callbacks for the current experiment

        :param chkpt_path:  the path to save the checkpoint
        :param exp_name:  the name of the mlflow experiment
        :param version:  the version of the model
        :return:  checkpoint_callback, early_stop_callback, logger
        """
        logging.info(f"The checkpoints are logged in {chkpt_path} metric: val_loss , mode: min")
        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath=chkpt_path,
                                                                            monitor='val_loss',
                                                                            mode="min",
                                                                            save_top_k=1)

        early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                         min_delta=1e-4,
                                                         patience=patience_check,
                                                         mode="min")

        logger = DictLogger(f'{results_path}/{logs_path}',
                            name=exp_name,
                            version="single_appliance_experiment" + str(version) if version != '' else "single_appliance_experiment")

        return [checkpoint_callback, early_stop_callback], logger

    def fit(self, model, dataset,
            chkpt_path=None,exp_name=None,results_path=None, logs_path=None,  version=None,
            batch_size=64, epochs=20, use_optuna=False, learning_rate=1e-6, optimizer='adam', patience_optim=5,
            train_idx=None, validation_idx=None):
        # Load weights from the last checkpoint if any in the checkpoints path

        best_checkpoint = get_latest_checkpoint(f'{results_path}/{chkpt_path}')
        self.batch_size = batch_size

        pl_model = PlModel(model, optimizer=optimizer, learning_rate=learning_rate, patience_optim= learning_rate)

        callbacks_lst, logger = self.log_init(f'{results_path}/{chkpt_path}',results_path, logs_path, exp_name, version)
        logging.info(f'Training started for {epochs} epochs')

        if not os.path.exists(f'{results_path}/{chkpt_path}/'):
            os.makedirs(f'{results_path}/{chkpt_path}/')

        mlflow.pytorch.autolog()

        trainer = pl.Trainer(logger=logger,
                             max_epochs=epochs,
                             callbacks=callbacks_lst,
                             gpus=-1 if torch.cuda.is_available() else None,
                             resume_from_checkpoint=best_checkpoint if not use_optuna else None)
        dataset_train, dataset_validation = self.data_split(dataset , batch_size)
        # Fit the model using the train_loader, val_loader
        trainer.fit(pl_model, dataset_train, dataset_validation)

        val_losses = [metric['val_loss'] for metric in logger.metrics if len(logger.metrics)>1 and 'val_loss' in metric]

        return pl_model, np.min(val_losses) if len(val_losses)>0 else -1

    def get_dataset(self, main, submain=None, seq_type='seq2point',
                    in_size=99, out_size=1, point_position='mid_position',
                    target_norm='z-norm', quantiles= None,  loader= None, hparams=None):

        data = GeneralDataLoader(
            main, targets=submain,
            in_size=in_size,
            out_size=out_size,
            point_position=point_position,
            seq_type=seq_type,
            quantiles=quantiles,
            pad_at_begin=False
        ) if loader is None else \
            loader(main, submain, hparams)
        return data, data.params

    def data_split(self, data, batch_size, train_idx=None, val_idx=None):
        """
        Splits data to training and validation
        :param main:
        :param target:
        :param appliance_name:
        :return:
        """
        if train_idx is None or val_idx is None:
            train_idx = int(data.len * (1 - 0.15))
            val_idx = data.len - int(data.len * (1 - 0.15))
        train_data, val_data = torch.utils.data.random_split(data,
                                      [train_idx, val_idx],
                                      generator=torch.Generator().manual_seed(3407))
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size,
                                                 shuffle=False)
        return train_loader, val_loader

    def train_test_split(self, data, train_idx, val_idx, batch_size):
        return self.data_split(
            data,
            batch_size,
            train_idx,
            val_idx
        )

    def predict(self, model, mains, batch_size=64):

        test_loader = torch.utils.data.DataLoader(mains,
                                                   batch_size,
                                                   shuffle=False)

        network = model.model.eval()
        predictions = network.predict(model, test_loader)
        df = predictions['pred']

        return df

    def load_model(self, model, path):
        logging.warning(f'Loading Torch models from path :{path}')
        checkpoint = torch.load(get_latest_checkpoint(path))
        if not isinstance(model, PlModel):
            model = PlModel(model)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model



