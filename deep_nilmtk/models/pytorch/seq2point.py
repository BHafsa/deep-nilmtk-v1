import logging

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import sys
from .layers import  create_conv1,  create_linear

import torch
import numpy as np

from collections import OrderedDict

# =============================================================================
# Sequence-to-Point models
# =============================================================================

class S2P(nn.Module):
    """
    This class is an abstract class  for Sequence to point models. By implementing this class,
    you can avoid to implement the predict and the forward functions.

    :param params: dictionnary of values relative to hyper-parameters.
    :type params: dict

    The dictionnay is expected to include the following keys:

    :param target_norm: the type of normlization of the target power, defaults to 'z-norm'.
    :type target_norm: str.
    :param mean: The mean consumption value of the target appliance, defaults to 0.
    :type mean: float.
    :param std: The std consumption value  the target power, defaults to 1
    :type std: float.
    :param min: The mininum consumption value of the target appliance, defaults to 0.
    :type min: float.
    :param max: The maximum consumption value  the target power, defaults to 1
    :type max: float.

    """



    def __init__(self, params):
        """Initialise the Seq-to-Point model

        :param params: dictionnary of values relative to hyper-parameters.
        :type params: dict
        """
        super(S2P, self).__init__()
        self.original_len = params['in_size']  if 'in_size' in params else 99
        self.target_norm = params['target_norm'] if 'target_norm' in params else 'z-norm'


    @staticmethod
    def suggest_hparams( trial):
        """
        Function returning list of params that will be suggested from optuna

        :param trial: Optuna Trial.
        :type trial: optuna.trial
        :return: Parameters with values suggested by optuna
        :rtype: dict
        """

        window_length = trial.suggest_int('in_size', low=99, high=560)
        window_length += 1 if window_length % 2 == 0 else 0
        return {
            'in_size': window_length,
            'outsize':1
        }

    def step(self, batch):
        """
        Disaggregates a batch of data

        :param batch: A batch of data
        :type batch: Tensor
        :return: loss function as returned form the model and the MAE as returned from the model.
        :rtype: tuple(float, float)
        """

        x, y  = batch

        out   = self(x)  # BxCxT

        error = (y - out)
        loss = F.mse_loss(out, y)
        mae = error.abs().data.mean()
        return  loss, mae



    def predict(self,  model, test_dataloader):
        """Generate prediction during testing for the test_dataLoader

        :param model: pre-trained model.
        :param test_dataloader: data loader for the testing period.
        :type test_dataloader: dataLoader
        :return: Disaggregated power consumption.
        :rtype: tensor
        """

        net = model.model.eval()
        num_batches = len(test_dataloader)
        values = range(num_batches)

        pred = []
        true = []


        with tqdm(total = len(values), file=sys.stdout) as pbar:
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    if len(batch)==2:
                        x, y  = batch
                        out = net(x)
                        pred.append(out)
                        true.append(y)
                    else:
                        x  = batch
                        out = net(x)
                        pred.append(out)

                    del  batch
                    pbar.set_description('processed %d' % (1 + batch_idx))
                    pbar.update(1)

                pbar.close()

        pred = torch.cat(pred, 0)

        if len(true)!=0:
            results = {"pred":pred, 'true':true}
        else:

            results = {"pred":pred}
        logging.warning(f'the max value predicted is {pred.max()}')
        return results


class Seq2Point(S2P):
    """
    .. _seqpoint:

    PyTorch implementation of the Seq-to-point
    NILM model as porposed in :
    https://dl.acm.org/doi/pdf/10.1145/3360322.3360844

    :param params: dictionnary of values relative to hyper-parameters.
    :type params: dict

    Besides the additional paramter from teh parent model, the params dictionnay is expected to include the following keys:

    :param feature_type: The number of input features, defaults to 1.
    :type feature_type: int
    :param appliances: A list of appliances.
    :type appliances: list of str
    :param pool_filter: The size of pooling filter, defaults to 50.
    :type pool_filter: int

    :param latent_size: The number of nodes in the last layer, defaults to 1024.
    :type latent_size: int

    """

    def __init__(self, params):

        super(Seq2Point, self).__init__(params)

        in_size=4 if params['feature_type']=="combined" else 1
        output_size = len(params['appliances']) if 'appliances' in params else 1
        pool_filter = params['pool_filter'] if 'pool_filter' in params else 50
        latent_size = params['latent_size'] if 'latent_size' in params else 1024

        self.pool_filter = pool_filter
        self.enc_net = nn.Sequential(create_conv1(in_size, 30, 10, bias=True, stride=1),
                                     nn.ReLU(),
                                     create_conv1(30, 40, 8, bias=True, stride=1),
                                     nn.ReLU(),
                                     create_conv1(40, 50, 6, bias=True, stride=1),
                                     nn.ReLU(),
                                     create_conv1(50, 50, 5, bias=True, stride=1),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     create_conv1(50, 50, 5, bias=True, stride=1),
                                     nn.ReLU(),
                                     nn.AdaptiveAvgPool1d(self.pool_filter),
                                     nn.Flatten())

        self.fc = nn.Sequential(create_linear(50*pool_filter, latent_size),

                                nn.Dropout(0.2),
                                nn.Linear(latent_size, output_size))


    def forward(self, x):
        if x.ndim!=3:
            x = torch.unsqueeze(x, 1)
        else:
            x = x.permute(0,2,1)
        x = self.enc_net(x)
        x = self.fc(x)

        if self.target_norm =='lognorm' :
            x = F.softplus(x)
        elif self.target_norm =='min-max':
            x= F.relu(x)

        return x

class RNN(S2P):
    """
        .. _rnn:
        PyTorch implementation of the RNN
        NILM model as porposed in :
        https://dl.acm.org/doi/pdf/10.1145/3360322.3360844

        :param params: dictionnary of values relative to hyper-parameters.
        :type params: dict

        Besides the additional paramter from teh parent model, the params dictionnay is expected to include the following keys:

        :param feature_type: The number of input features, defaults to 1.
        :type feature_type: int
        :param appliances: A list of appliances.
        :type appliances: list of str
    """
    def __init__(self, params):
        super(RNN, self).__init__(params)

        in_size=4 if params['feature_type']=="combined" else 1
        output_size = len(params['appliances']) if 'appliances' in params else 1

        self.model1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(
                in_channels = in_size,
                out_channels = 16,
                kernel_size = 4,
                stride = 1
            ))]))
        self.lstm1 = nn.LSTM(
            input_size = 16,
            hidden_size = 128,
            num_layers = 1,
            bidirectional = True
        )
        self.lstm2 = nn.LSTM(
            input_size = 256,
            hidden_size = 256,
            num_layers = 1,
            bidirectional = True
        )
        self.model2 = nn.Sequential(OrderedDict([

            ('Linear', nn.Linear(
                in_features = 2*256 , # The calculation according to input sequence !!!
                out_features= 128
            )),
            ('relu6', nn.Tanh()),
            ('Linear2', nn.Linear(
                in_features = 128,
                out_features = output_size
            )),
        ]))

    def forward(self,x ):
        y_pred = nn.functional.pad(x.permute(0,2,1), (1,2))

        y_pred =self.model1(y_pred).permute(2,0,1)
        y_pred, _ = self.lstm1(y_pred)
        y_pred, _ = self.lstm2(y_pred)
        y_pred = torch.mean(y_pred, dim=0).squeeze()
        y_pred = self.model2(y_pred)
        if self.target_norm =='lognorm' :
            x = F.softplus(x)
        elif self.target_norm =='min-max':
            x= F.relu(x)
        return y_pred

class WindowGRU(S2P):
    """
        .. _gru:
        PyTorch implementation of the Window-GRU
        NILM model as porposed in :
        https://dl.acm.org/doi/pdf/10.1145/3360322.3360844

        :param params: dictionnary of values relative to hyper-parameters.
        :type params: dict

        Besides the additional paramter from teh parent model, the params dictionnay is expected to include the following keys:

        :param feature_type: The number of input features, defaults to 1.
        :type feature_type: int
        :param appliances: A list of appliances.
        :type appliances: list of str
    """
    def __init__(self, params):
        super(WindowGRU, self).__init__(params)


        in_size=4 if params['feature_type']=="combined" else 1
        output_size = len(params['appliances']) if 'appliances' in params else 1

        self.model1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(
                in_channels = in_size,
                out_channels = 16,
                kernel_size = 4,
                stride = 1
            )),
            ('relu1', nn.ReLU())]))
        self.gru1 = nn.GRU(
            input_size = 16,
            hidden_size = 64,
            num_layers = 1,
            bidirectional = True
        )
        self.dropout = nn.Dropout(.5)
        self.gru2 = nn.GRU(
            input_size = 128,
            hidden_size = 128,
            num_layers = 1,
            bidirectional = True
        )
        self.model2 = nn.Sequential(OrderedDict([ ('relu3', nn.ReLU()),
                                                  ('dropout3', nn.Dropout(.5)),

                                                  ('Linear', nn.Linear(
                                                      in_features = 256 , # The calculation according to input sequence !!!
                                                      out_features= 128
                                                  )),
                                                  ('relu6', nn.ReLU()),
                                                  ('dropout3', nn.Dropout(.2)),
                                                  ('Linear2', nn.Linear(
                                                      in_features = 128,
                                                      out_features = output_size
                                                  )),
                                                  ]))

    def forward(self,x ):
        y_pred = nn.functional.pad(x.permute(0,2,1), (1,2))
        y_pred =self.model1(y_pred).permute(2,0,1)
        y_pred, _ = self.gru1(y_pred)
        y_pred =self.dropout(y_pred)
        y_pred, _ = self.gru2(y_pred)
        y_pred = torch.mean(y_pred, dim=0).squeeze()
        y_pred = self.model2(y_pred)
        if self.target_norm =='lognorm' :
            x = F.softplus(x)
        elif self.target_norm =='min-max':
            F.relu(x)

        return y_pred