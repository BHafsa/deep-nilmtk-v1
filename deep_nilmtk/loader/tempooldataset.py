# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from ..preprocessing import  compute_status

def pad_data(data, units_to_pad, border =0 ):
    """Performs data padding for both target and aggregate consumption

    :param data: The aggregate power
    :type data: np.array
    :param units_to_pad: The numebr of values to add to input and output sequences.
    :type units_to_pad: tupe(int, int)
    :param border: The delay between input and output sequence, defaults to 0
    :type border: int, optional
    :return: The padded aggregate power.
    :rtype: np.array
    """
    
    padding = (border // 2, units_to_pad - border // 2)
    if data.ndim==1:
        new_mains = np.pad(data, padding,'constant',constant_values=(0,0))
        return new_mains
    else:
        new_mains = []
        for i in range(data.shape[-1]):
            new_mains.append(np.pad(data[:,i], padding,'constant',constant_values=(0,0)))
        return np.stack(new_mains).T

class TemPoolLoader(torch.utils.data.Dataset):
    """
    .. _ptpdataset:

    This class is the dataLoader for the temporal pooling NILM model. The original code 
    can be found here: https://github.com/UCA-Datalab/nilm-thresholding

    :param inputs: The aggregate power.
    :type inputs: np.array
    :param targets: The target appliance(s) power consumption, defaults to None
    :type targets: np.array, optional
    :param params: Hyper-parameter values, defaults to {}
    :type params: dict, optional

    The hyperparameter dictionnary is expected to include the following parameters

    :param in_size: The input sequence length, defaults to 99
    :type in_size: int
    :param border: The delay between the input and out sequence, defaults to 30.
    :type border: int
    """
    def __init__(
        self, inputs,  targets=None,  params= {}):

        self.length = params['in_size'] if 'in_size'  in params else 512
        self.border = params["border"] if "border" in params else 30
        
        self.train = params["train"] if "train" in params else False
        
        seq_len = self.length + self.border
        units_to_pad = ((inputs.shape[0] // seq_len ) + 1) * seq_len - inputs.shape[0] + 2  
        
        self.meter = pad_data(inputs, units_to_pad)
        
        self.appliance = targets
               
        
        if targets is not None:
           
            self.appliance = pad_data(targets, units_to_pad)
            self.status = compute_status(self.appliance, appliances_labels= params['appliances'], threshold_method = params['threshold_method'])
            if  params['target_norm'] == 'z-norm':
                self.mean = np.mean(targets)
                self.std = np.std(targets)
                
                self.appliance = torch.tensor( (self.appliance - self.mean) / self.std ).float()
            else:
                self.appliance = torch.tensor(self.appliance).float().log1p()
                
            print(f"inputs {self.meter.shape}, targets {self.appliance.shape}")
        else:
            self.appliance = None
            print(f"inputs {self.meter.shape}")    

        self.len = self.meter.shape[0] - self.length - self.border
        
        
        self.indices = np.arange(self.meter.shape[0]) 
        
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def __getitem__(self, index):
        input_indices = self.indices[index  : index + self.length + self.border ] 
        target_indices = self.indices[index + self.border//2 : index + self.length + self.border//2] 
        
        x = self.meter[input_indices ].astype("float32") 
        
      
        if self.appliance is not None:
            x = torch.tensor(x).reshape((x.shape[0],-1)).permute(1,0)
            y = self.appliance[target_indices] 
            s= self.status[target_indices]
            
            return x , y , torch.tensor(s)
        else:
            return torch.tensor(x).permute(1,0)

    def __len__(self):
        return self.len
