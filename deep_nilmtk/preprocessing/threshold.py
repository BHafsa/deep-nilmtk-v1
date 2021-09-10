# -*- coding: utf-8 -*-
# =============================================================================
# Original code: https://github.com/UCA-Datalab/better_nilm/utils/threshold.py
# =============================================================================
import numpy as np
from math import ceil
from math import floor

import numpy as np
from sklearn.cluster import KMeans



# Power load thresholds (in watts) applied by AT thresholding
THRESHOLDS = {
    'dish washer': 10.,
    'fridge': 50.,
    'washing machine': 20.
}

# Time thresholds (in seconds) applied by AT thresholding
MIN_OFF = {
    'dish washer': 30,
    'fridge': 1,
    'washing machine': 3
}

MIN_ON = {
    'dish washer': 30,
    'fridge': 1,
    'washing machine': 30
}

MAX_POWER = {
    'dish washer': 2500,
    'fridge': 300,
    'washing machine': 2500
}


def get_threshold_params(appliances, threshold_method = 'at'):
    """Given the method name and list of appliances,
    this function results the necessary Args to use the method in
    ukdale_data.load_ukdale_meter

    :param appliances: List of aappliances
    :type appliances: list
    :param threshold_method: Thresholding method, defaults to 'at'
    :type threshold_method: str, optional
    :raises ValueError: Wrong thresholding method
    :raises ValueError: Missing parameters of an applaince
    :return: thresholds, min_off, min_on, threshold_std
    :rtype:tuple
    """

    if threshold_method == 'vs':
        # Variance-Sensitive threshold
        threshold_std = True
        thresholds = None
        min_off = None
        min_on = None
    elif threshold_method == 'mp':
        # Middle-Point threshold
        threshold_std = False
        thresholds = None
        min_off = None
        min_on = None
    elif threshold_method == 'at':
        # Activation-Time threshold
        threshold_std = False
        thresholds = []
        min_off = []
        min_on = []
        for label in appliances:
            
            if label not in THRESHOLDS.keys():
                msg = f"Appliance {label} has no AT info.\n" \
                      f"Available appliances: {', '.join(THRESHOLDS.keys())}"
                raise ValueError(msg)
            thresholds += [THRESHOLDS[label]]
            min_off += [MIN_OFF[label]]
            min_on += [MIN_ON[label]]
    else:
        raise ValueError(f"Method {threshold_method} doesnt exist\n"
                         f"Use one of the following: vs, mp, at")

    return thresholds, min_off, min_on, threshold_std



def _get_cluster_centroids(ser):
    """Returns ON and OFF cluster centroids' mean and std

    :param ser: An array with shape shape = (num_series, series_len, num_meters)
            - num_series : Amount of time series.
            - series_len : Length of each time series.
            - num_meters : Meters contained in the array.
    :type ser: np.array
    :return: mean and std consumption for each applaince
    :rtype: tuple
    """

    # We dont want to modify the original series
    ser = ser.copy()

    # Reshape in order to have one dimension per meter
    num_meters = ser.shape[2]

    # Initialize mean and std arrays
    mean = np.zeros((num_meters, 2))
    std = np.zeros((num_meters, 2))

    for idx in range(num_meters):
        # Take one meter record
        meter = ser[:, :, idx].flatten()
        meter = meter.reshape((len(meter), -1))
        kmeans = KMeans(n_clusters=2).fit(meter)

        # The mean of a cluster is the cluster centroid
        mean[idx, :] = kmeans.cluster_centers_.reshape(2)

        # Compute the standard deviation of the points in
        # each cluster
        labels = kmeans.labels_
        lab0 = meter[labels == 0]
        lab1 = meter[labels == 1]
        std[idx, 0] = lab0.std()
        std[idx, 1] = lab1.std()

    return mean, std

def get_thresholds(ser, use_std=True, return_mean=False):
    """Returns the estimated thresholds that splits ON and OFF appliances states.

    :param ser: An array with shape = (num_series, series_len, num_meters)
            - num_series : Amount of time series.
            - series_len : Length of each time series.
            - num_meters : Meters contained in the array.
    :type ser: np.array
    :param use_std: Consider the standard deviation of each cluster when computing the threshold. If not, the threshold is set in the middle point between    cluster centroids., defaults to True
    :type use_std: bool, optional
    :param return_mean: If True, return the means as second parameter., defaults to False
    :type return_mean: bool, optional
    :return: thresholds and mean consumption for each appliance
    :rtype: tuple 

    .. note:: The eman values are  only returned when return_mean is True (default False)
    """

    mean, std = _get_cluster_centroids(ser)

    # Sigma is a value between 0 and 1
    # sigma = the distance from OFF to ON at which we set the threshold
    if use_std:
        sigma = std[:, 0] / (std.sum(axis=1))
        sigma = np.nan_to_num(sigma)
    else:
        sigma = np.ones(mean.shape[0]) * .5

    # Add threshold
    threshold = mean[:, 0] + sigma * (mean[:, 1] - mean[:, 0])

    # Compute the new mean of each cluster
    for idx in range(mean.shape[0]):
        # Flatten the series
        meter = ser[:, :, idx].flatten()
        mask_on = meter >= threshold[idx]
        mean[idx, 0] = meter[~mask_on].mean()
        mean[idx, 1] = meter[mask_on].mean()

    if return_mean:
        return threshold, np.sort(mean)
    else:
        return threshold

