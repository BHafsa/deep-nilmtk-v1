import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from tqdm import tqdm

from .layers import create_conv1, create_linear,  Up

import numpy as np


class UNETNILM(nn.Module):
    """
    .. _unet:

    UNET-NILM implementation
    The original paper can be found here: https://dl.acm.org/doi/abs/10.1145/3427771.3427859
    The hyperparameter dictionary is expected to include the following parameters

    :param appliances: List of appliances, defaults to 1
    :type appliances: list
    :param feature_type: The type of input features generated in the pre-processing, defaults  to 'main'
    :type feature_type: str
    :param n_channels: the number of output channels, defaults to 1
    :type n_channels: int
    :param pool_filter: Pooling filter, defaults to 8
    :type pool_filter: int
    :param latent_size: The latent size, defaults to 1024
    :type latent_size: int

    """

    def __init__(
            self,
            params):
        super().__init__()

        out_size = len(params['appliances']) if 'appliances' in params else 1

        if 'feature_type' in params:
            n_channels = 4 if params['feature_type'] == "combined" else 1
        else:
            n_channels = params['n_channels']

        pool_filter = params['pool_filter'] if 'pool_filter' in params else 8
        latent_size = params['latent_size'] if 'latent_size' in params else 1024
        self.target_norm = params['target_norm'] if 'target_norm' in params else 'z-norm'

        layers = [nn.Sequential(create_conv1(n_channels, 30,
                                             kernel_size=10,
                                             stride=2),
                                nn.ReLU())]

        layers.append(nn.Sequential(create_conv1(30, 30, 8, bias=True, stride=2),
                                    nn.ReLU()))
        layers.append(nn.Sequential(create_conv1(30, 40, 6, bias=True, stride=1),
                                    nn.ReLU()))
        layers.append(nn.Sequential(create_conv1(40, 50, 5, bias=True, stride=1),
                                    nn.ReLU(),
                                    nn.Dropout(0.2)))
        layers.append(nn.Sequential(create_conv1(50, 50, 5, bias=True, stride=1),
                                    nn.ReLU()))
        self.enc_layers = nn.ModuleList(layers)

        layers = []
        layers.append(Up(50, 40, 5, 1, nn.ReLU()))
        layers.append(Up(40, 30, 5, 1, nn.ReLU()))
        layers.append(Up(30, 30, 6, 1, nn.ReLU()))
        self.dec_layers = nn.ModuleList(layers)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(pool_filter),
                                nn.Flatten(),
                                create_linear(30 * pool_filter, latent_size),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(latent_size, out_size))


    def forward(self, x):
        if x.ndim != 3:
            x = torch.unsqueeze(x, 1)
        else:
            x = x.permute(0, 2, 1)
        xi = [self.enc_layers[0](x)]

        for layer in self.enc_layers[1:]:
            xi.append(layer(xi[-1]))

        for i, layer in enumerate(self.dec_layers):
            xi[-1] = layer(xi[-1], xi[-2 - i])

        out = self.fc(xi[-1])
        return out

    def step(self, batch):
        """Disaggregates a batch of data
        :param batch: A batch of data.
        :type batch: Tensor
        :return: loss function as returned form the model and MAE as returned from the model.
        :rtype: tuple(float,float)
        """

        x, y = batch
        out = self(x)  # BxCxT
        error = (y - out)

        loss = F.mse_loss(out, y)
        mae = error.abs().data.mean()
        return loss, mae



    def predict(self, model, test_dataloader):
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

        with tqdm(total=len(values), file=sys.stdout) as pbar:
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    if len(batch) == 2:
                        x, y = batch
                        out = net(x)
                        pred.append(out)
                        true.append(y)
                    else:
                        x = batch
                        out = net(x)
                        pred.append(out)
                    del batch
                    pbar.set_description('processed: %d' % (1 + batch_idx))
                    pbar.update(1)

                pbar.close()

        pred = torch.cat(pred, 0)



        if len(true) != 0:
            true = torch.cat(true, 0).expm1()
            results = {"pred": pred.detach().numpy(), 'true': true.detach().numpy()}
        else:
            results = {"pred": pred.detach().numpy()}
        return results


class UNETNILMSeq2Quantile(nn.Module):
    """UNET-NILM impelementation with quantile regression
    The orginal paper can be found here: https://dl.acm.org/doi/abs/10.1145/3427771.3427859
    The hyperparameter dictionnary is expected to include the following parameters

    :param appliances: List of appliances, defaults to 1
    :type appliances: list
    :param feature_type: The type of input features generated in the pre-processing, defaults  to 'main'
    :type feature_type: str
    :param n_channels: the number of output channels, defaults to 1
    :type n_channels: int
    :param pool_filter: Pooling filter, defaults to 8
    :type pool_filter: int
    :param latent_size: The latent size, defaults to 1024
    :type latent_size: int
    :param quantile: The quantiles to use during prediction, defaults to [0.1, 0.25, 0.5, 0.75, 0.9]
    :param quantile: list

    It can be used as follows:
    .. code-block:: python

        'UNETNiLMSeq2Q':NILMExperiment({
                        'model_name': 'UNETNiLMSeq2Quantile',
                        'in_size': 480,
                        'feature_type':'mains',
                        'input_norm':'z-norm',
                        'target_norm':'z-norm',
                        'kfolds':3,
                        'seq_type':'seq2quantile',
                        'max_nb_epochs':1 }),
    """

    def __init__(self, params):
        super().__init__()

        out_size = len(params['appliances'])
        seq_len = params['in_size'] if 'in_size' in params else 99
        seq_len += params['out_size'] if 'out_size' in params else 0
        self.seq_len = seq_len
        n_channels = 4 if params['feature_type'] == "combined" else 1
        quantiles = params['quantiles'] if 'quantiles' in params else [0.1, 0.25, 0.5, 0.75, 0.9]

        pool_filter = params['pool_filter'] if 'pool_filter' in params else 8
        latent_size = params['latent_size'] if 'latent_size' in params else 1024

        self.q = torch.tensor(quantiles)
        self.out_size = out_size
        self.unet = UNETNILM({
            'n_channels': n_channels,
            'out_size': out_size * seq_len // 2,
            'pool_filter': pool_filter,
            'latent_size': latent_size
        })


    def forward(self, x):
        # print(x.shape)
        out = self.unet(x).reshape(x.size(0), -1, self.out_size)
        return out

    def smooth_pinball_loss(self, y, q, tau, alpha=1e-2, kappa=1e3, margin=1e-2):
        """
        The implementation of the Pinball loss for NILM, original code can be found in :
        https://github.com/hatalis/smooth-pinball-neural-network/blob/master/pinball_loss.py
        Hatalis, Kostas, et al. "A Novel Smoothed Loss and Penalty Function
        for Noncrossing Composite Quantile Estimation via Deep Neural Networks." arXiv preprint (2019).
        """

        error = (y - q)
        q_loss = (tau * error + alpha * F.softplus(-error / alpha)).sum(0).mean()
        # calculate smooth cross-over penalty
        diff = q[1:, :, :] - q[:-1, :, :]
        penalty = kappa * torch.square(F.relu(margin - diff)).mean()
        loss = penalty + q_loss
        return loss

    def step(self, batch):
        """Disaggregates a batch of data
        :param batch: A batch of data.
        :type batch: Tensor
        :return: loss function as returned form the model and MAE as returned from the model.
        :rtype: tuple(float,float)
        """
        x, y = batch
        out = self(x)
        self.q = self.q.to(x.device)
        q_pred = torch.quantile(out, q=self.q, dim=1)

        y_q = y.permute(1, 0, 2)

        tau = self.q[:, None, None].expand_as(q_pred)
        loss = self.smooth_pinball_loss(y_q, q_pred, tau)
        mae = (y_q - q_pred).abs().data.sum(0).mean()
        return loss, mae

    def predict(self, model, test_dataloader):
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

        with tqdm(total=len(values), file=sys.stdout) as pbar:
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    if len(batch) == 2:
                        x, y = batch
                        out = net(x)
                        pred.append(out)
                        true.append(y)
                    else:
                        x = batch
                        out = net(x)
                        pred.append(out)
                    del batch
                    pbar.set_description('processed: %d' % (1 + batch_idx))
                    pbar.update(1)

                pbar.close()
        pred = torch.cat(pred, 0)

        q = self.q.to(x.device)
        q_pred = torch.quantile(pred, q=q, dim=1).permute(1, 0, 2)

        pred = q_pred[:, q_pred.size(1) // 2, :]

        results = {"pred": pred, "q_pred": q_pred, "pred_quantile": pred}
        return results