
import unittest
from deep_nilmtk.data.pre_process import *
__unittest = True

from deep_nilmtk.utils import assertNumpyArraysEqual
import numpy as np
from deep_nilmtk.data.pre_process import normalize

class TestPreprocess(unittest.TestCase):

    def test_pad_data(self):
        nb_points = 3600
        nb_app = 1
        data = np.random.randint(0,1220, nb_points * nb_app).reshape(-1,nb_app)
        padded_data = pad_data(data, 57)
        self.assertEqual(padded_data.shape[0], nb_points+56)
        self.assertEqual(padded_data.shape[1],nb_app)
        assertNumpyArraysEqual(padded_data[28:84,:], data[:56,:])