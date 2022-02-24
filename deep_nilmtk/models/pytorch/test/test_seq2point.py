import unittest
import  numpy as np

from deep_nilmtk.models.torch.seq2point import Seq2Point
import tensorflow as tf

from deep_nilmtk.utils.test import assertNumpyArraysEqual

class TestSeq2Point(unittest.TestCase):

    def test_froward(self):
        N = 2
        input_batch = np.random.random(64*125*1).reshape(64,125,1)
        model = Seq2Point(125,N).model
        output = model(input_batch)
        self.assertEqual(output.shape, (input_batch.shape[0], N))

