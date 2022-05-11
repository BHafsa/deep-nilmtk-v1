# -*- coding: utf-8 -*-

import torch
import numpy as np
from deep_nilmtk.data.pre_process.features import compute_status
from deep_nilmtk.data.pre_process.normalize import normalize
from deep_nilmtk.data.pre_process import pad_data
import logging

class TempDataLoader(torch.utils.data.Dataset):
    """
    Temporal Pooling data loader
    """
    def __init__(self, inputs, targets=None, params={}):

        self.length = params['in_size'] if 'in_size' in params else 512
        self.border = params["border"] if "border" in params else 16
        self.power_scale = params["power_scale"] if "power_scale" in params else 2000.0

        self.train = params["train"] if "train" in params else False

        self.meter = pad_data(inputs.reshape(-1), self.length+2*self.border).reshape(-1)

        self.params = {}
        if targets is not None:
            self.appliance = pad_data(targets.values.reshape(-1), self.length+ 2*self.border).reshape(-1)
            self.status = compute_status(self.appliance, appliances_labels=params['appliances'],
                                         threshold_method=params['threshold_method'])
            self.params, self.appliance  = normalize(self.appliance.reshape(-1,1), params['target_norm'])
            logging.warning(f"inputs {self.meter.shape}, targets {self.appliance.shape}")
        else:
            self.appliance = None
            logging.warning(f"inputs {self.meter.shape}")

        self.len = len(self.meter) - (self.length + 2*self.border ) -1


    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def __getitem__(self, index):
        i = index
        x = self.meter[i : i + self.length + 2 * self.border].astype("float32")
        x = torch.tensor(x).reshape((x.shape[0], 1)).permute(1, 0)

        if self.appliance is not None:

            y = torch.tensor(np.expand_dims(self.appliance[i+self.border: i + self.length +self.border], 1)).reshape((-1, 1)).permute(1, 0)
            s =  torch.tensor(np.expand_dims(self.status[i+self.border: i + self.length +self.border], 1)).reshape((-1,1)).permute(1, 0)
            # print(i, self.meter.shape, x.shape, y.shape, s.shape , self.appliance.shape)
            return x, y, s
        else:
            # print(i, x.shape, self.appliance.shape)
            return x

