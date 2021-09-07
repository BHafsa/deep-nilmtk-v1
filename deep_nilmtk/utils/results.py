from ..preprocessing import quantile_filter

def filter_prediction(data, w=10, q=50):
    return quantile_filter(data.squeeze(), w, q)