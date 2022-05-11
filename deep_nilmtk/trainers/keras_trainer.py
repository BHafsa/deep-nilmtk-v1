from abc import ABC

import mlflow.keras
import tensorflow.keras.losses
import tensorflow.python.keras.models as Models

from .trainer_implementor import TrainerImplementor
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from deep_nilmtk.data.loader.tensorflow import GeneralDataLoader
import time
import numpy as np
import os


class KerasTrainer(TrainerImplementor):
    optimizers = {
        'adam': tf.keras.optimizers.Adam,
        'sgd': tf.keras.optimizers.SGD,
        'RMSprop': tf.keras.optimizers.RMSprop,
        'Adadelta': tf.keras.optimizers.Adadelta,
        'Adagrad': tf.keras.optimizers.Adagrad,
        'Adamax': tf.keras.optimizers.Adamax,
        'Nadam': tf.keras.optimizers.Nadam,
        'Ftrl': tf.keras.optimizers.Ftrl,

    }

    def __init__(self):
        self.len = 0
        self.in_size = None
        self.out_size = None

    def get_output_signature(self):
        pass

    def fit(self, model, dataset,
            chkpt_path=None, exp_name=None, results_path=None, logs_path=None, version=None,
            batch_size=64, epochs=20, use_optuna=False, learning_rate=1e-4, optimizer='adam', patience_optim=5,
            train_idx=None, validation_idx=None):

        if not os.path.exists(f'{results_path}/{chkpt_path}/'):
            os.makedirs(f'{results_path}/{chkpt_path}/')

        ds_series = dataset.dataset.batch(batch_size)

        opt = self.get_optimizer(optimizer, learning_rate, patience_optim)
        model.compile(loss='mse', optimizer=opt)

        train_data, val_data = self.train_test_split(ds_series, train_idx, validation_idx, batch_size)
        mlflow.keras.autolog()

        es = EarlyStopping(monitor='val_loss', min_delta=.01, verbose=1, patience=4)
        mc = ModelCheckpoint(f'{results_path}/{chkpt_path}/disaggregator.h5', monitor='val_loss',
                             save_weights_only=True, save_best_only=True, verbose=1)
        history = model.fit(train_data, validation_data=val_data, validation_freq=1,
                            batch_size=batch_size, epochs=epochs, callbacks=[es, mc], verbose=1)

        return model, history.history['val_loss'][-1]

    def get_optimizer(self, optimizer, learning_rate, patience_optim):
        opt = self.optimizers[optimizer](learning_rate=learning_rate)
        return opt

    def load_model(self, model, path):
        model.load_weights(f'{path}/disaggregator.h5')
        return model

    def get_dataset(self, main, submain=None, seq_type='seq2point',
                    in_size=99, out_size=1, point_position='mid_position',
                    target_norm='z-norm', quantiles=None, loader=None, hparams= None):
        self.len = len(main) - in_size
        params={}
        data = GeneralDataLoader(main, submain, seq_type, in_size, out_size, point_position) \
            if loader is None else \
            loader(main, submain, hparams)
        return data, params

    def train_test_split(self, dataset, train_idx=None, val_idx=None, batch_size=None):
        """
        splits the dataset into validation and training
        :param dataset:
        :param train_idx:
        :param val_idx:
        :return:
        """
        if train_idx is None or val_idx is None:
            train_idx = int(.90 * len(list(dataset)))
            val_idx = self.len - train_idx
        else:
            train_idx = (train_idx[-1]-train_idx[0]) // batch_size
            val_idx = (val_idx[-1]-val_idx[0]) // batch_size

        train_ds = dataset.take(train_idx)
        val_ds = dataset.skip(train_idx).take(val_idx)

        return train_ds, val_ds

    def predict(self, model, test_dataset, batch_size=64):
        """
        Generates the predictions for the test_dataset using the pre-trained
        model.

        :param model:
        :param test_dataset:
        :param batch_size:
        :return:
        """
        pred = []
        for batch in iter(test_dataset.dataset.batch(batch_size)):
            pred.append(model(batch).numpy())
        return np.concatenate(pred, axis=0)
