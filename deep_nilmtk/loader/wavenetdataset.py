import torch
import numpy as np
import random



def pad_data(data, context_size):
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
    

    .. _wavenetdataset
    """
    def __init__(self, inputs,  targets=None,  params= {}):
        
        self.context_size= params['context_size'] if 'context_size' in params   else 99
        
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