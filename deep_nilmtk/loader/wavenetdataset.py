import torch
import numpy as np
import random



def pad_data(data, context_size):
    """Performs data padding for both target and aggregate consumption

    :param data: The aggregate power
    :type data: np.array
    :param context_size: The length of teh sequence.
    :type context_size: int
    :return: The padded aggregate power.
    :rtype: np.array
    """
    sequence_length = context_size 
    units_to_pad = sequence_length // 2
    padding = (units_to_pad,units_to_pad)
    if data.ndim==1:
        new_mains = np.pad(data, padding,'constant',constant_values=(0,0))
        return new_mains
    else:
        new_mains = []
        for i in range(data.shape[-1]):
            new_mains.append(np.pad(data[:,i], padding,'constant',constant_values=(0,0)))
        return np.stack(new_mains).T

class WaveNetDataLoader(torch.utils.data.Dataset):
    """
    .. _wavenetdataset:

    This class is the dataLoader for the WaveNILM NILM model. The original code 
    can be found here: https://github.com/jiejiang-jojo/fast-seq2point/

    :param inputs: The aggregate power.
    :type inputs: np.array
    :param targets: The target appliance(s) power consumption, defaults to None
    :type targets: np.array, optional
    :param params: Hyper-parameter values, defaults to {}
    :type params: dict, optional

    The hyperparameter dictionnary is expected to include the following parameters

    :param in_size: The input sequence length, defaults to 99
    :type in_size: int
    :param kernel_size: The size of teh kernel, defaults to 3.
    :type kernel_size: int
    :param layers: The number of layers of the model, defaults to 6.
    :type layers: int

    .. note:: 
       This data loader generates target sequence with length L different from the input sequences
       L = (2 ** layers - 1) * (kernel_size - 1) + 1
    """
    def __init__(self, inputs,  targets=None,  params= {}):
        
        self.context_size= params['in_size'] if 'in_size' in params   else 99
        
        self.kernel_size = params['kernel_size'] if 'kernel_size' in params else 3 # has to be odd integer, since even integer may break dilated conv output size
        self.layers = params['layers'] if 'layers' in params else 6

        self.seq_len = (2 ** self.layers - 1) * (self.kernel_size - 1) + 1
        self.num_points = self.context_size + self.seq_len 

        #pad the sequence with zeros in the beginning and at the end
        inputs  = pad_data(inputs, self.num_points)
        
        self.inputs = torch.tensor(inputs).float()
        
        if targets is not None:
            targets = pad_data(targets, self.num_points)
            if  params['target_norm'] == 'z-norm':
                self.mean = np.mean(targets)
                self.std = np.std(targets)
                
                self.targets = torch.tensor( (targets - self.mean) / self.std ).float()
            else:
                self.targets = torch.tensor(targets).float().log1p()
                
            print(f"inputs {self.inputs.shape}, targets {self.targets.shape}")
        else:
            self.targets = None
            print(f"inputs {self.inputs.shape}")
        
        self.len = self.inputs.shape[0] - self.num_points
        self.indices = np.arange(self.inputs.shape[0])
        
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.len
    
    def get_sample(self, index):
        """
        Generate a sample of power sequence.

        :param index: The start index of the first sequence
        :type index: int
        :return: Aggregate power and target consumption during training and only aggreagte power during testing
        :rtype: np.array
        """
        indices = self.indices[index : index + self.num_points]
        inds_targs   = sorted(indices[self.seq_len // 2:self.num_points])
        
        
        inputs = self.inputs[sorted(indices)]
        if self.targets is not None:
            targets = self.targets[sorted(inds_targs)]
            return inputs,  targets
        else:
            return inputs
        
        
    
    def __getitem__(self, index):
        return self.get_sample(index)  