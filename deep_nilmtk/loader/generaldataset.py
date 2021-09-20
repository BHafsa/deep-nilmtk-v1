import torch
import numpy as np
import random



def pad_data(data, context_size, target_size, pad_at_begin= False):
    """
    Performs data padding for both target and aggregate consumption

    :param data: The aggregate power
    :type data: np.array
    :param context_size: The input sequence length
    :type context_size: int
    :param target_size: The target sequence length
    :type target_size: int
    :param pad_at_begin: Specified how the padded values are inserted, defaults to False
    :type pad_at_begin: bool, optional
    :return: The padded aggregate power.
    :rtype: np.array
    """
    sequence_length = context_size + target_size
    units_to_pad = sequence_length // 2 
    padding = (context_size,target_size) if pad_at_begin else (units_to_pad,units_to_pad+1)
    if data.ndim==1:
        new_mains = np.pad(data, padding,'constant',constant_values=(0,0))
        return new_mains
    else:
        new_mains = []
        for i in range(data.shape[-1]):
            new_mains.append(np.pad(data[:,i], padding,'constant',constant_values=(0,0)))
        return np.stack(new_mains).T

class generalDataLoader(torch.utils.data.Dataset):
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
    def __init__(self, inputs,  targets=None,  params= {}):
        
        self.context_size= params['in_size'] if 'in_size' in params   else 99
        target_size= params['out_size'] if 'out_size' in params   else 0
        self.point_position= params['point_position'] if 'point_position' in params   else "median"
        self.seq_type= params['seq_type'] if 'seq_type'  in params  else "seq2quantile"
        quantiles= params['quantiles'] if 'quantiles' in params   else [0.1, 0.25, 0.5, 0.75, 0.90]
        pad_at_begin = params['pad_at_begin'] if 'pad_at_begin'in params  else  False
        
        
        #pad the sequence with zeros in the beginning and at the end
        inputs  = pad_data(inputs, self.context_size, target_size, pad_at_begin)
        
        self.q=torch.tensor(quantiles)
        self.inputs = torch.tensor(inputs).float()
        
        self.num_points = self.context_size + target_size

        if targets is not None:
            targets = pad_data(targets, self.context_size, target_size, pad_at_begin)
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
        indices = self.indices[index : index + self.num_points]
        inds_context = sorted(indices[self.num_points//2:self.num_points])
        
        # inds_context = sorted(indices[:self.context_size])
        inds_targs   = sorted(indices[self.context_size:self.num_points])
        
        
        inputs = self.inputs[sorted(indices)]
        if self.targets is not None:
            if self.seq_type =="seq2point":
                if self.point_position == "last_point":
                    targets  = self.targets[inds_targs]
                    return inputs,  targets


                elif self.point_position == "median":
                    targets  =torch.median(self.targets[inds_context], dim=0).values
                    return inputs,  targets

                else:
                    print("This strategy is not implemented yet!! We kindly ask you to implement it and push the code for others :D ")
            
            elif self.seq_type =="seq2seq":
                targets = self.targets[sorted(indices)]
                
                return inputs,  targets
            
            elif self.seq_type =="seq2quantile":
                targets=torch.quantile(self.targets[indices], q=self.q, dim=0)
                print(f"""
                
                {targets.shape}
                
                """)
                return inputs,  targets
            
            else:
                targets  = self.targets[inds_targs]
                return inputs,  targets
        
            
        else:
            return inputs
        
        
    
    def __getitem__(self, index):
        return self.get_sample(index)  
    


