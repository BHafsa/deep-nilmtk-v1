import torch
import numpy as np
import random



def pad_data(data, context_size, target_size, pad_at_begin= False):
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

    This class implements the most two common NILM data generators:
    The seq-to-seq and seq-to-point. The data is generated correponding 
    to the sequence type. For the seq-to-point models, the position of 
    the point can also be specified according to the point_positon paramter. 

    .. _generaldataset

    """
    def __init__(self, inputs,  targets=None,  params= {}):
        
        self.context_size= params['context_size'] if 'context_size' in params   else 99
        target_size= params['target_size'] if 'target_size' in params   else 0
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
        'Denotes the total number of samples'
        return self.len
    
    def get_sample(self, index):
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
                return inputs,  targets
            
            else:
                targets  = self.targets[inds_targs]
                return inputs,  targets
        
            
        else:
            return inputs
        
        
    
    def __getitem__(self, index):
        return self.get_sample(index)  
    
    


def npsample_batch(x, y, size=None, sort=True):
    """Sample from numpy arrays along 2nd dim."""
    inds = np.random.choice(range(x.shape[1]), size=size, replace=False)
    if sort:
        inds.sort()
    return x[:, inds], y[:, inds]

def collate_fns(params, sample= False):
    """
    This function is responsible of generating 
    the dynamic sequences for each batch
    
    Args:
        max_num_context ([int]): number of context points
        max_num_extra_target ([int]): number of target points
        sample ([bool]): if true sample from the static length sequence to obtain smaller sequences
        sort (bool, optional): [description]. Defaults to True.
        context_in_target (bool, optional): [description]. Defaults to True.
    """
    
    max_num_context = params['context_size']
    max_num_extra_target = params['target_size']
  
    sort =True
    context_in_target = True
    
    def collate_fn(batch, sample=sample):

        # Collate
        
        if isinstance(batch[0], tuple):
            x = np.stack([x for x, y in batch], 0)
            y = np.stack([y for x, y in batch], 0)
            y = torch.from_numpy(y).float()
            y_context = y[:, :max_num_context]
            y_target_extra = y[:, max_num_context:]
        else:
            x = np.stack([x for x in batch], 0)
            
        x = torch.from_numpy(x).float()
        # Sample a subset of random size
        num_context = np.random.randint(100, max_num_context)
        num_extra_target = np.random.randint(20, max_num_extra_target)
         
        x_context = x[:, :max_num_context]
        
    
        x_target_extra = x[:, max_num_context:]
        
        
        if sample:

            # This is slightly differen't than normal, we are ensuring that our target point are in the future, to mimic deployment
            x_context, y_context = npsample_batch(
                x_context, y_context, size=num_context, sort=sort
            )

            x_target_extra, y_target_extra = npsample_batch(
                x_target_extra, y_target_extra, size=num_extra_target, sort=sort
            )
            
        
        if isinstance(batch[0], tuple):
            x_target = x_target_extra
            y_target = y_target_extra
            return x_context, y_context, x_target, y_target
        else:
            x_target = x_target_extra

            return x_context, x_target
            
    return collate_fn

