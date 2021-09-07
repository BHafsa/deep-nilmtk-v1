# -*- coding: utf-8 -*-
import torch
import numpy as np
import random

class BERTDataset(torch.utils.data.Dataset):
    """
    
    This class is dataLoader correponding to the BERT4NILM 
    model. The original code can be found here: https://github.com/Yueeeeeeee/BERT4NILM/
    .. _bertdataset:

    """
    def __init__(self, inputs, targets = None, params= {},main_cutoff = 6000):
        
        
        self.x = inputs 
        self.main_cutoff = main_cutoff
        
        self.threshold = [params['threshold'][params['appliances'][0]]] if 'threshold' in params else None
        
        self.cutoff = [params['cutoff'][params['appliances'][0]]] if 'cutoff' in params else None
        
        self.min_on = [params['min_on'][params['appliances'][0]]] if 'min_on' in params else None
        self.min_off = [params['min_off'][params['appliances'][0]]] if 'min_off' in params else None
        
        self.window_size = params['context_size'] if 'context_size' in params else 480
        self.stride = params['stride'] if 'stride' in params else 1
        self.mask_prob = params['mask_prob'] if 'mask_prob' in params else 0.25

        self.y =targets
        
        if self.y is not None:
            self.y [targets > self.cutoff] =  self.cutoff
           
            self.columns = self.y.shape[1] 
            self.status = self.compute_status(self.y)
            
        

        self.len = int(np.ceil((len(self.x) - self.window_size) / self.stride) + 1)
        

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        start_index = index * self.stride
        end_index = np.min(
            (len(self.x), index * self.stride + self.window_size))
        
        x = self.padding_seqs(self.x[start_index: end_index]).reshape(-1)
        
        if self.y is not None:
            # Training and validation Phase
            y = self.padding_seqs(self.y[start_index: end_index])
            status = self.padding_seqs(self.status[start_index: end_index])

            tokens = []
            labels = []
            on_offs = []
            for i in range(len(x)):
                prob = random.random()
                            
                if prob < self.mask_prob:
                    prob = random.random()
                    if prob < 0.8:
                        tokens.append(-1)
                    elif prob < 0.9:
                        tokens.append(np.random.normal())
                    else:
                        tokens.append(x[i])

                    labels.append(y[i])
                    on_offs.append(status[i])
                else:
                    tokens.append(x[i])
                    temp = np.array([-1] * self.columns)
                    labels.append(temp)
                    on_offs.append(temp)

            return torch.tensor(tokens), torch.tensor(labels), torch.tensor(on_offs)
        else :
            # Testing Phase
            tokens = []
            for i in range(len(x)):
                tokens.append(x[i])
            
            
            return torch.tensor(tokens) 

    def padding_seqs(self, in_array):
        if len(in_array) == self.window_size:
            return in_array
        try:
            out_array = np.zeros((self.window_size, in_array.shape[1]))
        except:
            out_array = np.zeros(self.window_size)

        length = len(in_array)
        out_array[:length] = in_array
        return out_array
    
    def compute_status(self, data):
        status = np.zeros(data.shape)
        if len(data.squeeze().shape) == 1:
            columns = 1
        else:
            columns = data.squeeze().shape[-1]

        if not self.threshold:
            self.threshold = [10 for i in range(columns)]
        
        if not self.min_on:
            self.min_on = [1 for i in range(columns)]
        if not self.min_off:
            self.min_off = [1 for i in range(columns)]
            
            
       

        for i in range(columns):
            initial_status = data[:, i] >= self.threshold[i]
            status_diff = np.diff(initial_status)
            
            events_idx = status_diff.nonzero()

            events_idx = np.array(events_idx).squeeze()
            events_idx += 1

            if initial_status[0]:
                events_idx = np.insert(events_idx, 0, 0)

            if initial_status[-1]:
                events_idx = np.insert(
                    events_idx, events_idx.size, initial_status.size)

            events_idx = events_idx.reshape((-1, 2))
            on_events = events_idx[:, 0].copy()
            off_events = events_idx[:, 1].copy()
            assert len(on_events) == len(off_events)

            if len(on_events) > 0:
                off_duration = on_events[1:] - off_events[:-1]
                off_duration = np.insert(off_duration, 0, 1000)
                
                on_events = on_events[off_duration > self.min_off[i]]
                off_events = off_events[np.roll(
                    off_duration, -1) > self.min_off[i]]

                on_duration = off_events - on_events
                on_events = on_events[on_duration >= self.min_on[i]]
                off_events = off_events[on_duration >= self.min_on[i]]
                assert len(on_events) == len(off_events)

            temp_status = data[:, i].copy()
            temp_status[:] = 0
            for on, off in zip(on_events, off_events):
                temp_status[on: off] = 1
            status[:, i] = temp_status

        return status

