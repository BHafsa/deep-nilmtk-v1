import copy

import torch
import numpy as np

from deep_nilmtk.data.pre_process import pad_data
import logging
from deep_nilmtk.data.loader.utils import target_generator
class GeneralDataLoader(torch.utils.data.Dataset):
    """
    .. _generaldataset:

    This class implements the most two common NILM data generators:
    The seq-to-seq and seq-to-point. The data is generated correponding
    to the sequence type. For the seq-to-point models, the position of
    the point can also be specified according to the point_positon paramter.

    :param inputs: The aggregate power.
    :type inputs: np.array
    :param targets: The target appliance(s) power consumption, defaults to None
    :type targets: np.array, optional
    :param params: Hyper-parameter values, defaults to {}
    :type params: dict, optional

    The hyperparameter dictionnary is expected to include the following parameters

    :param in_size: The input sequence length, defaults to 99
    :type in_size: int
    :param out_size: The target sequence length, defaults to 0.
    :type out_size: int
    :param point_position: The position of the point for sequence, defaults to seq2quantile
    :type in_size: str
    :param quantiles: The list of quantiles to generate, defaults to [0.1, 0.25, 0.5, 0.75, 0.90]
    :type in_size: list of floats
    :param pad_at_begin: Specified how the padded values are inserted, defaults to False
    :type pad_at_begin: bool, optional
    """
    def __init__(self, inputs,  targets=None,  in_size = 99,
                 out_size = 1, point_position = 'last_position', seq_type='seq2point',
                 quantiles=[0.1, 0.25, 0.5, 0.75, 0.90], pad_at_begin=False, pad=False):

        self.in_size= in_size
        self.out_size= out_size
        self.point_position= point_position
        self.seq_type= seq_type
        self.original_inputs = copy.deepcopy(inputs)
        self.original_targets = copy.deepcopy(targets)
        #pad the sequence with zeros in the beginning and at the end
        logging.info (f"Inputs shape before padding {inputs.shape}, padding at the beginning is set to {pad_at_begin}")
        inputs  = pad_data(inputs, self.in_size, pad_at_begin)
        logging.info (f"Inputs shape after padding {inputs.shape}")
        if seq_type == 'seq2quantile':
            assert quantiles is not None
            self.q=torch.tensor(quantiles)
        self.inputs = torch.tensor(inputs).float()


        if targets is not None:
            print(targets)
            logging.info (f"Targets shape before padding {targets.shape}, the padding at the beginning is set to {pad_at_begin}")
            targets = pad_data(targets, self.in_size, pad_at_begin)
            logging.warning(f'the max value of target data is {targets.max()}')
            logging.info (f"Targets shape after padding {targets.shape}")
            self.targets = torch.tensor(targets).float()
            logging.info (f"inputs {self.inputs.shape}, targets {self.targets.shape}")
        else:
            self.targets = None
            logging.info(f"inputs {self.inputs.shape}")

        self.len = self.inputs.shape[0] - self.in_size
        self.indices = np.arange(self.inputs.shape[0])

        self.get_target_indice = target_generator(self.seq_type, self.in_size, self.out_size, self.point_position)


    def __len__(self):
        """
        Denotes the total number of samples
        """
        return self.len

    def get_sample(self, index):
        """
        Generate a sample of power sequence.

        :param index: The start index of the first sequence
        :type index: int
        :return: Aggregate power and target consumption during training and only aggreagte power during testing
        :rtype: np.array
        """
        in_indices = self.indices[index : index + self.in_size]
        inputs = self.inputs[sorted(in_indices)]
        if self.targets is not None:
            indice_target = self.get_target_indice(in_indices)

            targets = self.targets[indice_target]
            if self.seq_type =="seq2quantile":
                targets =torch.quantile(targets, q=self.q, dim=0)

            return inputs,  targets
        else:
            return inputs

    def __getitem__(self, index):
        return self.get_sample(index)

    def __copy__(self):

        return self