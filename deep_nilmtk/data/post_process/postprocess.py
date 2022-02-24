import pandas as pd

from .denormalize import denormalize
from .aggregate import aggregate_mean

def postprocess(predictions, type_target, params, aggregate=False):
    """
    Post processing function for the predictions
    :param predictions: a 2d np array
    :param hparams: hparameters
    :return: processed energy consumption
    """

    processed_predictions = {
        app: denormalize(aggregate_mean(predictions[app]), type=type_target, params=params[app]).reshape(-1) \
         if aggregate else denormalize(predictions[app], type=type_target, params=params[app]).reshape(-1)  for app in predictions
    }
    return processed_predictions
