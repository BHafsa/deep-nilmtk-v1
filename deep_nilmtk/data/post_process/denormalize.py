import logging
import numpy as np


def z_denorm(data, mean, std):
    """
    Denormalize the data using that was normalized using z-norm
    :param data: 1d array of power consumption
    :param mean: the mean of the power consumption
    :param std: the std of the power consumption
    :return: denormalized power consumption
    """
    return data * std + mean


def min_max_denorm(data, min_, max_):
    """
    Denormalize the data using that was normalized using min-max normalization
    :param data: 1d array of power consumption
    :param min: the min of the power consumption
    :param max: the max of the power consumption
    :return: denormalized power consumption
    """
    return data * (max_ - min_) + min_


def log_denorm(data):
    """
    Denormalize the data using that was normalized using log normalization
    :param data: 1d array of power consumption
    :return: denormalized power consumption
    """

    return np.exp(data) - 1


def denormalize(data, type='z-norm', params=None):
    """
    A function the denormalize the
    :param params: a dictionnaty of parameters
    :param data: a 1D array of power consumption
    :param type: type of normalization
    :return: the denormalized power consumption
    """
    logging.info(f'The sequences are being denormalized  using the {type}')
    if type == 'z-norm':
        assert params is not None, f'Please specify the parameters for {type}'
        logging.warning(f"De-normalizing the target power using the mean={params['mean']}, and std={params['std']}")
        return z_denorm(data, params['mean'], params['std'])
    elif type == 'min-max':
        assert params is not None, f'Please specify the parameters for {type}'
        logging.info(f"De-Normalizing the target power using the min={params['min']}, and max ={params['max']}")
        return min_max_denorm(data, params['min'], params['max'])
    elif type == 'lognorm':
        return log_denorm(data)
    else:
        logging.error(
            'The type of normalization is not recognized. The problem is generated in the file '
            'deep_nilmtk.data.post_process.utils.py')
        raise Exception('The type of normalization is not recognized')
