import logging

import numpy as np
import pandas as pd

from .normalize import normalize
def pad_data(data, sequence_length, pad_at_begin= False):
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
    units_to_pad = 1 + sequence_length // 2
    padding = (sequence_length,) if pad_at_begin else (units_to_pad,units_to_pad+1)
    if data.ndim==1:
        new_mains = np.pad(data.reshape(-1), padding,'constant',constant_values=(0,0))
        return new_mains
    else:
        paddings = np.zeros((units_to_pad, data.shape[1]))
        new_mains = np.concatenate([paddings, data, paddings])
        return new_mains

def preprocess(mains,  norm_type, submains=None, params=None):
    """
    Preprocess the main data using normalization
    :param mains: the aggregate power
    :param submains: the power of appliances
    :return: pd.DataFrame, pd.DataFrame, dict
    """
    logging.warning("Data is preprocessed using default pre-preocessing function")
    logging.info(f"Number of data Sources is :{len(mains)} ")
    mains = np.concatenate(mains, axis=0)
    logging.info(f"Shape of data after concatenation is :{mains.shape} ")
    params, mains = normalize(mains, norm_type, params)
    if submains is not None:
        columns = [app_name for app_name, _ in submains]
        submains = pd.concat([
            pd.concat(targets) for _, targets in submains
        ], axis=1)
        submains.columns = columns
        logging.info(f'The target data contains the following appliances:{submains.columns} with shape {submains.shape}')
        return mains, pd.DataFrame(submains), params

    return mains, params