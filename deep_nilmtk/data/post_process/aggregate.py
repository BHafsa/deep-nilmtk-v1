import logging
import numpy as np


def aggregate_median( prediction, stride):
    """
    Aggregate the overleapping sequences using the mean

    Args:
        prediction (tensor[n_samples + window_size +1 1,window_size]): test predictions of the current model

    Returns:
        [type]: [description]
    """
    l = prediction.shape[1]
    n = prediction.shape[0] + l - 1
    sum_arr = np.zeros(n)
    o = len(sum_arr)
    for i in range(prediction.shape[0]):
        seq = []
        j =0
        while ((i-j)>=0 and j<l):
            seq.append(prediction[i-j,j])
            j+=1
        sum_arr[i] = np.median(np.array(seq))

        if i == prediction.shape[0] -1 :
            seq= []
            for j in range( l - 1):
                if j == l-2:
                    sum_arr [i+j+1] = prediction[prediction.shape[0]-1, prediction.shape[1]-1]
                else:
                    k = j + 1
                    seq =[]
                    while k<l  :
                        seq.append(prediction[i-k+1,k])
                        k+=1
                    sum_arr [i+j+1] = np.median(np.array(seq))

    return sum_arr

def aggregate_mean(prediction, stride=1):
    """Aggregate the overleapping sequences using the mean

    :param prediction: test predictions of the current model
    :type prediction: numpy/array
    :return: Aggregted sequence
    :rtype: numpy.array
    """
    print(prediction.shape)
    prediction = prediction.reshape(prediction.shape[0], -1,1)
    l = prediction.shape[1]
    s = stride
    n = (prediction.shape[0] - 1) * stride + l  # this is yo take into consideration the stride

    sum_arr = np.zeros((n, 1))
    counts_arr = np.zeros((n, 1))
    o = len(sum_arr)

    logging.info(f'The data contains {prediction.shape[0]} sequences of length {l}')
    logging.info(f'The final length of the data is {n}')

    for i in range(prediction.shape[0]):
        sum_arr[i * s:i * s + l, :] += prediction[i]
        counts_arr[i * s:i * s + l,:] += 1

    for i in range(len(sum_arr)):
        sum_arr[i] = sum_arr[i] / counts_arr[i]
    logging.info(f'Data shape: before aggregation  {prediction.shape}, after aggregation {sum_arr.shape}')
    return sum_arr



def aggregate_seq(data, type='mean', stride=1):
    """
    Aggregates the data after predictions are generated in the case of Seq2Seq models
    :param data: a 2d np array of data
    :param type: type of aggregation
    :return: 1d np array of the predictions
    """
    logging.info(f'The sequences are being aggregated using the {type}')
    if type == 'mean':
        return aggregate_mean(data, stride)
    elif type == 'median':
        return aggregate_median(data, stride)
    else:
        logging.error("The aggregation type is not recognized. The problem is generated in the file deep_nilmtk.data.post_process.aggregate.py")
        raise Exception('The sequence aggregation strategy is not recognized. Only two type of aggregation are possible (mean or median)')

