from ..preprocessing import quantile_filter

def filter_prediction(data, w=10, q=50):
    """Filters the predictions

    param data: The input data power data.
    :type data: np.array
    :param sequence_length: The length of sequence, defaults to 10
    :type sequence_length: int, optional
    :param p: The percentile. Defaults to 50.
    :type p: int, optional
    :return: array of values for correponding percentile
    :rtype: np.array
    """
    return quantile_filter(data.squeeze(), w, q)