import torch.nn as nn
import torch.nn.functional as F

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import sys
import numpy as np


class Encoder(nn.Module):
    """
    Decoder block of the Temporal_pooling layer
    """

    def __init__(
            self,
            in_features=3,
            out_features=1,
            kernel_size=3,
            padding=1,
            stride=1,
            dropout=0.1,
    ):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.bn(F.relu(self.conv(x))))


class TemporalPooling(nn.Module):
    """
    Temporal Pooling mechanism that combines data with different scales.
    """

    def __init__(self, in_features=3, out_features=1, kernel_size=2,
                 dropout=0.1):
        super(TemporalPooling, self).__init__()
        self.kernel_size = kernel_size

        self.pool = nn.AvgPool1d(kernel_size=self.kernel_size,
                                 stride=self.kernel_size, padding=0)

        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1,

                              padding=0)

        self.bn = nn.BatchNorm1d(out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # TODO: verify that the inputs are in the right shape
        size = x.shape[2]
        x = self.pool(x)

        x = self.conv(x)

        x = self.bn(F.relu(x))
        x = F.interpolate(x, size=size,
                          mode='linear', align_corners=True)

        x = self.drop(x)
        return x


class Decoder(nn.Module):
    """
    Decoder block of the Temporal_pooling layer
    """

    def __init__(self, in_features=3, out_features=1, kernel_size=2, stride=2, padding=0, output_padding=0):
        super(Decoder, self).__init__()
        self.conv = nn.ConvTranspose1d(in_features, out_features,
                                       kernel_size=kernel_size, stride=stride, padding=padding,
                                       output_padding=output_padding,
                                       bias=False)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        return self.conv(x)


class PTPNet(nn.Module):
    """
    .. _ptp:
    Source: https://github.com/lmssdd/TPNILM
    Check the paper
    Non-Intrusive Load Disaggregation by Convolutional
    Neural Network and Multilabel Classification
    by Luca Massidda, Marino Marrocu and Simone Manca
    The hyperparameter dictionnary is expected to include the following parameters
    The hyperparameter dictionnary is expected to include the following parameters
    :param in_size: The input sequence length, defaults to 99
    :type in_size: int
    :param border: The delay between the input and out sequence, defaults to 30.
    :type border: int
    :param appliances: List of appliances
    :type appliances: list
    :param feature_type: The type of input features generated during pre-processing, defaults to 'main'.
    :type feature_type: str
    :param init_features: The number of features in the first encoder layer, defaults to 32.
    :type init_fetaure: int
    :param dropout: Dropout
    :type dropout: float
    :param target_norm: The type of normalization of the target data, defeaults to 'z-norm'.
    :type target_norm: str
    :param mean: The mean consumption of the target power, defaults to 0
    :type mean: float
    :param std: The STD consumption of the target power, defaults to 1
    :type std: float
    It can be used as follows:
    .. code-block::python
       'tempPool': NILMExperiment({
               "model_name": 'tempPool',
               'experiment_label':'regression',
               'in_size': 480,
               'input_norm':'z-norm',
               'target_norm':'z-norm',
               'feature_type':'mains',
               'max_nb_epochs':max_nb_epochs,
               'task_type':'regression',
               'hidden_dim':64,
               }),
    """

    def __init__(self, params):
        super(PTPNet, self).__init__()

        self.border = params["border"] if "border" in params else 30
        out_channels = len(params['appliances']) if 'appliances' in params else 1
        input_features = 4 if params['feature_type'] == 'combined' else 1
        features = params['init_features'] if 'init_features' in params else 32
        dropout = params['dropout'] if 'dropout' in params else 0.1
        self.seq_len = params['in_size'] if 'in_size' in params else 481

        output_len = 256

        p = 2
        k = 1

        self.encoder1 = Encoder(input_features, features, kernel_size=3,
                                padding=1, dropout=dropout)

        # (batch, input_len , 32)
        self.pool1 = nn.MaxPool1d(kernel_size=p, stride=p)

        self.encoder2 = Encoder(features * 1 ** k, features * 2 ** k,
                                kernel_size=3, padding=1, dropout=dropout)
        # (batch, input_len  / 2, 64)
        self.pool2 = nn.MaxPool1d(kernel_size=p, stride=p)
        self.encoder3 = Encoder(features * 2 ** k, features * 4 ** k,
                                kernel_size=3, padding=1, dropout=dropout)
        # (batch, [input_len - 12] / 4, 128)
        self.pool3 = nn.MaxPool1d(kernel_size=p, stride=p)

        self.encoder4 = Encoder(features * 4 ** k, features * 8 ** k,
                                kernel_size=3, padding=1, dropout=dropout)
        # (batch, [input_len - 30] / 8, 256)

        # Compute the output size of the encoder4 layer
        # (batch, S, 256)
        s = output_len / 8

        if int(s / 12) == 0:
            print(f"""
            Warning !!! the sequence length should be larger than {8 * 12}...
            Continuing with the current length could badly impact the performance :(
            """)

        self.tpool1 = TemporalPooling(features * 8 ** k, features * 2 ** k,
                                      kernel_size=int(s / 12) if int(s / 12) > 0 else 1,
                                      dropout=dropout)
        self.tpool2 = TemporalPooling(features * 8 ** k, features * 2 ** k,
                                      kernel_size=int(s / 6) if int(s / 6) > 0 else 1,
                                      dropout=dropout)

        self.tpool3 = TemporalPooling(features * 8 ** k, features * 2 ** k,
                                      kernel_size=int(s / 3) - int(s / 3) % 2 if int(s / 3) > 0 else 1,
                                      dropout=dropout)

        self.tpool4 = TemporalPooling(features * 8 ** k, features * 2 ** k,
                                      kernel_size=int(s / 2) - int(s / 2) % 2 if int(s / 2) > 0 else 1,
                                      dropout=dropout)

        padding = (((self.seq_len + self.border) // p ** 3) * p ** 3 - self.seq_len) // 2

        if self.seq_len % 2 == 0:
            self.decoder = Decoder(2 * features * 8 ** k, features * 1 ** k,
                                   kernel_size=p ** 3, stride=p ** 3, padding=padding)
        else:
            self.decoder = Decoder(2 * features * 8 ** k, features * 1 ** k,
                                   kernel_size=p ** 3, stride=p ** 3, padding=padding + 1, output_padding=1)

        self.activation = nn.Conv1d(features * 1 ** k, out_channels,
                                    kernel_size=1, padding=0)

        self.power = nn.Conv1d(features * 1 ** k, out_channels,
                               kernel_size=1, padding=0)

        self.pow_criterion = nn.MSELoss()
        self.act_criterion = nn.BCEWithLogitsLoss()

        self.pow_w = (params['task_type'] == 'regression')
        self.act_w = (params['task_type'] == 'classification')

        self.pow_loss_avg = 0.68
        self.act_loss_avg = 0.0045

        self.power_scale = params["power_scale"] if "power_scale" in params else 2000.0

    @staticmethod
    def suggest_hparams(self, trial):
        """
        Function returning list of params that will be suggested from optuna
        :param trial: Optuna Trial.
        :type trial: optuna.trial
        :return: Parameters with values suggested by optuna
        :rtype: dict
        """
        window_length = trial.suggest_int('in_size', low=50, high=1800)
        return {
            'in_size': window_length,
            "out_size": window_length,
        }

    def forward(self, x):
        """
        The step function of the model.
        :param x: A batch of the input features.
        :return: the power estimation, and the state estimation.
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        tp1 = self.tpool1(enc4)
        tp2 = self.tpool2(enc4)
        tp3 = self.tpool3(enc4)
        tp4 = self.tpool4(enc4)

        dec = self.decoder(torch.cat([enc4, tp1, tp2, tp3, tp4], dim=1))
        pw = self.power(dec)
        act = self.activation(F.relu(dec))

        return pw.permute(0, 2, 1), act.permute(0, 2, 1)

    def step(self, batch, sequence_type=None):
        """Disaggregates a batch of data
        :param batch: A batch of data.
        :type batch: Tensor
        :return: loss function as returned form the model and MAE as returned from the model.
        :rtype: tuple(float,float)
        """
        data, target_power, target_status = batch

        output_power, output_status = self(data)

        pow_loss = self.pow_criterion(output_power, target_power)
        act_loss = self.act_criterion(output_status, target_status)

        loss = (self.pow_w * pow_loss + self.act_w * act_loss)

        error = (output_power - target_power)
        mae = error.abs().data.mean()

        return loss, mae

    def predict(self, model, test_dataloader):
        """Generates predictions for the test data loader
        :param model: Pre-trained model
        :type model: nn.Module
        :param test_dataloader: The test data
        :type test_dataloader: dataLoader
        :return: Generated predictions
        :rtype: dict
        """
        net = model.model.eval()
        num_batches = len(test_dataloader)
        values = range(num_batches)

        s_hat = []
        p_hat = []
        x_true = []
        with tqdm(total=len(values), file=sys.stdout) as pbar:
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    x = batch
                    pw, sh = self(x)
                    sh = torch.sigmoid(sh)

                    s_hat.append(sh)
                    p_hat.append(pw)

                    # This is a strange normalisation method. Check how it works in an notebook
                    x_true.append(x[:, :,
                                  self.border:-self.border].detach().cpu().numpy().flatten())

                    del batch
                    pbar.set_description('processed %d' % (1 + batch_idx))
                    pbar.update(1)

                pbar.close()

        p_hat = torch.cat(p_hat, 0).float()
        s_hat = torch.cat(s_hat, 0).float()

        results = {
            # 'aggregates': torch.tensor(x_true),
            'pred': p_hat,  # this done to remove the data that was added
            # during padding, data at the end will automatically
            # be removed by the API
            'pred_states': s_hat
        }

        return results

