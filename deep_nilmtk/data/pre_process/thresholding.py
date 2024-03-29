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
    'washing machine': 20.,
    'kettle':20.,
    'microwave':20.,
    'television':20.
}

# Time thresholds (in seconds) applied by AT thresholding
MIN_OFF = {
    'dish washer': 30,
    'fridge': 1,
    'washing machine': 3,
    'kettle':1,
    'microwave':1,
    'television':30
}

MIN_ON = {
    'dish washer': 30,
    'fridge': 1,
    'washing machine': 30,
    'kettle':1,
    'microwave':1,
    'television':30
}

MAX_POWER = {
    'dish washer': 2500,
    'fridge': 300,
    'washing machine': 2500,
    'kettle': 2500,
    'microwave': 2500,
    'television': 1000
}


def get_threshold_params(appliances, threshold_method='at'):
    """
    Given the method name and list of appliances,
    this function results the necessary parameters to use the method in
    ukdale_data.load_ukdale_meter
    Parameters
    ----------
    threshold_method : str
    appliances : list
    Returns
    -------
    thresholds
    min_off
    min_on
    threshold_std
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
    """
    Returns ON and OFF cluster centroids' mean and std
    Parameters
    ----------
    ser : numpy.array
        shape = (num_series, series_len, num_meters)
        - num_series : Amount of time series.
        - series_len : Length of each time series.
        - num_meters : Meters contained in the array.
    Returns
    -------
    mean : numpy.array
        shape = (num_meters,)
    std : numpy.array
        shape = (num_meters,)
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
    """
    Returns the estimated thresholds that splits ON and OFF appliances states.
    Parameters
    ----------
    ser : numpy.array
        shape = (num_series, series_len, num_meters)
        - num_series : Amount of time series.
        - series_len : Length of each time series.
        - num_meters : Meters contained in the array.
    use_std : bool, default=True
        Consider the standard deviation of each cluster when computing the
        threshold. If not, the threshold is set in the middle point between
        cluster centroids.
    return_mean : bool, default=False
        If True, return the means as second parameter.
    Returns
    -------
    threshold : numpy.array
        shape = (num_meters,)
    mean : numpy.array
        shape = (num_meters,)
        Only returned when return_mean is True (default False)
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