import unittest
from deep_nilmtk.data.loader.tensorflow import generate_sequences

__unittest = True

from deep_nilmtk.utils import assertNumpyArraysEqual
from sklearn.model_selection import TimeSeriesSplit

class TestGenralDataLoader(unittest.TestCase):

    def test_get_sample(self):
        in_size = 50
        out_size = 50

        pass

    def test_get_fold(self):
        fold = TimeSeriesSplit(n_splits=3)


        for