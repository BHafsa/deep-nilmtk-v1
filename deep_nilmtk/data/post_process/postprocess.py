import pandas as pd

from .denormalize import denormalize
from .aggregate import aggregate_mean

def remove_negatives(pred):
    pred[pred<0]=0
    return pred

def postprocess(predictions, type_target, params, aggregate=False, stride=1):
    """
    Post processing function for the predictions
    :param predictions: a 2d np array
    :param hparams: hparameters
    :return: processed energy consumption
    """

    processed_predictions = {
        app: remove_negatives(denormalize(aggregate_mean(predictions[app], stride), type=type_target, params=params[app])).reshape(-1) \
         if aggregate else remove_negatives(denormalize(predictions[app], type=type_target, params=params[app]).reshape(-1) ) for app in predictions
    }
    print(processed_predictions)
    return processed_predictions
