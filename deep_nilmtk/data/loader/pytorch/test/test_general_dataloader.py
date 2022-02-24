import unittest
from deep_nilmtk.data.loader.pytorch import GeneralDataLoader

__unittest = True

from deep_nilmtk.utils import assertNumpyArraysEqual
import numpy as np
from deep_nilmtk.data.pre_process import normalize

class TestGenralDataLoader(unittest.TestCase):

    def test_get_sample(self):
        inputs = np.random.randint(0, 1500, 3600).reshape(-1,1)
        targets = np.random.randint(0, 1500, 3600*2).reshape(-1,2)

        dataloader = GeneralDataLoader(
            inputs,
            targets,
            seq_type='seq2seq',
            in_size=56,
            out_size=56,
            target_norm='z-norm'
        )
        # the data is padded with 28,2 from the top and the bottom
        in_, out_ = dataloader.get_sample(29)
        self.assertEqual(in_.shape[0], out_.shape[0])
        self.assertEqual(out_.shape, (56,2))
        assertNumpyArraysEqual(out_.numpy(), targets[:56])
        #Testing the seq2point
        dataloader = GeneralDataLoader(
            inputs,
            targets,

            seq_type='seq2point',
            point_position='mid_position',
            in_size=56,
            out_size=1,
            target_norm='z-norm'

        )
        in_, out_ = dataloader.get_sample(29)
        assertNumpyArraysEqual(out_.numpy() , targets[56//2] )



