# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from ..preprocessing import  compute_status

def pad_data(data, units_to_pad, border =0, ):
    
    padding = (border // 2, units_to_pad - border // 2)
    if data.ndim==1:
        new_mains = np.pad(data, padding,'constant',constant_values=(0,0))
        return new_mains
    else:
        new_mains = []
        for i in range(data.shape[-1]):
            new_mains.append(np.pad(data[:,i], padding,'constant',constant_values=(0,0)))
        return np.stack(new_mains).T

class ptpLoader(torch.utils.data.Dataset):

    """

    This class is the dataLoader for the temporal pooling NILM model. The original code 
    can be found here: https://github.com/UCA-Datalab/nilm-thresholding

    .. _ptpdataset

    """
    
    def __init__(
        self, inputs,  targets=None,  params= {}):

        self.length = params['context_size'] if 'context_size'  in params else 512
        self.border = params["border"] if "border" in params else 30
        self.power_scale = params["power_scale"] if "power_scale" in params else 2000.0
        
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
