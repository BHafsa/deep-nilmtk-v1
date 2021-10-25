# -*- coding: utf-8 -*-

import numpy as np

from .threshold import *



def get_status(ser, thresholds):
    """[summary]

    :param ser: Target power consumption with shape = (num_series, series_len, num_meters)
    :type ser: np.array
    :param thresholds: Thresholds of target power with shape = (num_meters,)
    :type thresholds: np.array
    :return: An array (num_series, series_len, num_meters) with binary values indicating ON (1) and OFF (0) states.
    :rtype: np.array
    """
    
    # We don't want to modify the original series
    ser = ser.copy()

    ser_bin = np.zeros(ser.shape)
    num_app = ser.shape[-1]

    # Iterate through all the appliances
    for idx in range(num_app):
        if len(ser.shape) == 3:
            mask_on = ser[:, :, idx] > thresholds[idx]
            ser_bin[:, :, idx] = mask_on.astype(int)
        else:
            mask_on = ser[:, idx] > thresholds[idx]
            ser_bin[:, idx] = mask_on.astype(int)

    ser_bin = ser_bin.astype(int)

    return ser_bin

def get_status_means(ser, status):
    """Get means of ON/OFF status.

    :param ser: Power data
    :type ser: np.array
    :param status: The operational status of the target power
    :type status: np.array
    :return: Mean power consumption of each state
    :rtype: np.array
    """

    means = np.zeros((ser.shape[2], 2))

    # Compute the new mean of each cluster
    for idx in range(ser.shape[2]):
        # Flatten the series
        meter = ser[:, :, idx].flatten()
        mask_on = status[:, :, idx].flatten() > 0
        means[idx, 0] = meter[~mask_on].mean()
        means[idx, 1] = meter[mask_on].mean()

    return means


def _get_app_status_by_duration(y, threshold, min_off, min_on):
    """
    Calculates the operational status using operation duartions.

    :param y: Power consumption array with shape = (num_series, series_len) , num_series : Amount of time series - series_len : Length of each time series.
    :type y: np.array 
    :param threshold: threshold value.
    :type threshold: float
    :param min_off: minimum off duration of the appliance.
    :type min_off: int
    :param min_on: minimum on duration of the appliance.
    :type min_on: int
    :return: With binary values indicating ON (1) and OFF (0) states.
    :rtype: np.array
    """
    shape_original = y.shape
    y = y.flatten().copy()

    condition = y > threshold
    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx = d.nonzero()[0]

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    on_events = idx[:, 0].copy()
    off_events = idx[:, 1].copy()
    assert len(on_events) == len(off_events)

    if len(on_events) > 0:
        off_duration = on_events[1:] - off_events[:-1]
        off_duration = np.insert(off_duration, 0, 1000.)
        on_events = on_events[off_duration > min_off]
        off_events = off_events[np.roll(off_duration, -1) > min_off]
        assert len(on_events) == len(off_events)

        on_duration = off_events - on_events
        on_events = on_events[on_duration > min_on]
        off_events = off_events[on_duration > min_on]

    s = y.copy()
    s[:] = 0.

    for on, off in zip(on_events, off_events):
        s[on:off] = 1.

    s = np.reshape(s, shape_original)

    return s


def get_status_by_duration(ser, thresholds, min_off, min_on):
    """
    Calculates operational status of multiple meters using thresholds 

    :param ser: Power consumption shape = (num_series, series_len, num_meters) - num_series : Amount of time series- series_len : Length of each time series - num_meters : Meters contained in the array.
    :type ser: np.array
    :param thresholds: Thresholds of power consumption  shape = (num_meters,)
    :type thresholds: np.array
    :param min_off: Mimimum off duration with  shape = (num_meters,)
    :type min_off: np.array
    :param min_on: Mimimum on duration with  shape = (num_meters,)
    :type min_on: np.array
    :return: Operational status with binary values indicating ON (1) and OFF (0) states with shape (num_series, series_len, num_meters).
    :rtype: np.array
    """

    

    num_apps = ser.shape[-1]
    ser_bin = ser.copy()

    msg = f"Length of thresholds ({len(thresholds)})\n" \
          f"and number of appliances ({num_apps}) doesn't match\n"
    assert len(thresholds) == num_apps, msg

    msg = f"Length of thresholds ({len(thresholds)})\n" \
          f"and min_on ({len(min_on)}) doesn't match\n"
    assert len(thresholds) == len(min_on), msg

    msg = f"Length of thresholds ({len(thresholds)})\n" \
          f"and min_off ({len(min_off)}) doesn't match\n"
    assert len(thresholds) == len(min_off), msg

    for idx in range(num_apps):
        if ser.ndim == 3:
            y = ser[:, :, idx]
            ser_bin[:, :, idx] = _get_app_status_by_duration(y,
                                                             thresholds[idx],
                                                             min_off[idx],
                                                             min_on[idx])
        elif ser.ndim == 2:
            y = ser[:, idx]
            ser_bin[:, idx] = _get_app_status_by_duration(y,
                                                          thresholds[idx],
                                                          min_off[idx],
                                                          min_on[idx])

    return ser_bin




def compute_status(appliances,
                    # dates=None,
                    # period="1min",
                    # max_power=10000.0,
                    thresholds=None,
                    min_off=None,
                    min_on=None,
                    max_power =None,
                    threshold_std=True,
                    
                    return_means=False,
                    appliances_labels=[],
                    threshold_method='at'):
    """
    Calculates the operational status of appliances using the specified thresholding method

    :param appliances: Power consumption of target applainces
    :type appliances: np.array
    :param thresholds: Threhsold of each applaince, defaults to None
    :type thresholds: np.array, optional
    :param min_off: Minimum off duration, defaults to None
    :type min_off: np.array, optional
    :param min_on: Minimum on duration, defaults to None
    :type min_on: np.array, optional
    :param threshold_std: Decides about the use of STD to calcualte the thresholds, defaults to True
    :type threshold_std: bool, optional
    :param return_means: Specifiyies if the mean consumption of each status is required, defaults to False
    :type return_means: bool, optional
    :param appliances_labels: Labels of the considered appliances, defaults to []
    :type appliances_labels: list, optional
    :param threshold_method: The thresholding method to be used for status derivation, defaults to 'at'
    :type threshold_method: str, optional
    :return: Operational states with the thresholds used and the power consumption of each states
    :rtype: tuple
    """
    
    
    
    # Set the parameters according to given threshold method
    if threshold_method != "custom":
        (thresholds, min_off, min_on, threshold_std) = get_threshold_params(
            appliances_labels, threshold_method, minon=min_on, threshold=thresholds, minoff=min_off, max_power=max_power
        )
    
    arr_apps = np.expand_dims(appliances, axis=1)
    
    if (thresholds is None) or (min_on is None) or (min_off is None):
        thresholds, means = get_thresholds(
            arr_apps, use_std=threshold_std, return_mean=True
        )

        msg = "Number of thresholds doesn't match number of appliances"
        assert len(thresholds) == appliances.shape[1], msg

        status = get_status(arr_apps, thresholds)
    else:
        status = get_status_by_duration(arr_apps, thresholds, min_off, min_on)
        means = get_status_means(arr_apps, status)
    
    
    status = np.squeeze(status, 1)

    msg = "Number of records between appliance status and load doesn't " "match"
    assert status.shape[0] == appliances.shape[0], msg

    return_params = (status)
    if return_means:
        return_params += ((thresholds, means),)

    return return_params