import numpy as np

def assertNumpyArraysEqual(arr1, arr2):
    if arr1.shape != arr2.shape:
        raise AssertionError("Shapes don't match")
    if not np.allclose(arr1.reshape(-1), arr2.reshape(-1)):
        raise AssertionError("Elements don't match!")