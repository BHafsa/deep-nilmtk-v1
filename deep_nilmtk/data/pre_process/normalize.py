import logging
import numpy as np



def z_norm(data,  params=None):
    """
    Normalizes the data using the z-normalization
    Works also with multi-dimensional data
    :param data: 2d array of power consumption
    :return: the mean power consumption, the std power consumption, and the normalized power
    """
    logging.warning(f'Data shape is {len(data)}')
    if len(data.shape) == 1:
        data = data.reshape(-1,1)
    mean = params['mean'] if params is not None else data.mean(axis=0).reshape(-1,data.shape[-1])
    std = params['std'] if params is not None else data.std(axis=0).reshape(-1,data.shape[-1])
    logging.info(f'Nomalizing the power using the mean={mean}, and std={std}')
    return {
        "mean":mean,
        "std":std
           }, (data -mean)/std

def min_max_norm(data, params=None):
    """
    Normalizes the data using min-max normalization
    :param data: 2d array of power consumption
    :return: the min consumption, the maximum consumption, and the normalized data
    """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    min_ = params['min'] if params is not None else  data.min(axis=0).reshape(-1,data.shape[-1])
    max_ = params['max'] if params is not None else  data.max(axis=0).reshape(-1,data.shape[-1])

    logging.info(f'Nomalising the power using the min={min_}, and max={max_}')
    return {
             'min':  min_,
             'max': max_
           }, (data-min_) / (max_-min_)

def log_norm(data, params=None):
    """
    Normalizes the data using log normalization
    :param data: 2d array of power consumption
    :return: the log-normalized data
    """
    assert len(data.shape) <= 2
    assert (data>=0).all() , 'Only positive numbers are allowed with log normalization. \
                                The activation function of teh last layer must be set accordingly'
    return {}, np.log(data + 1)

def normalize(data, type='z-norm', params=None):
    """
    Normalization of the data
    :param data: 2d array
    :param type: type of normalization
    :return:
    """
    logging.info(f'The sequences are being normalized  using the {type}')
    if params is not None:
        logging.warning(f"A predefined set of parameters is used :{params}")
    if type == 'z-norm':
        return z_norm(data, params)
    elif type =='min-max':
        logging.info(f'The sequences are being normalized using the min={min}, and max ={max}')
        return min_max_norm(data, params)
    elif type =='lognorm':
        return log_norm(data, params)
    else:
        logging.error('The type of normalization is not recognized. The problem is generated in the file deep_nilmtk.data.post_process.utils.py')
        raise Exception('The type of normalization is not recognized')