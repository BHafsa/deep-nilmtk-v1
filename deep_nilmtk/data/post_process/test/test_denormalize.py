import unittest

from deep_nilmtk.data.post_process.denormalize import *
from deep_nilmtk.data.pre_process.normalize import *
from deep_nilmtk.utils import assertNumpyArraysEqual

class TestDenormalize(unittest.TestCase):

    def test_z_norm(self):
        data = np.array([1,2,1,2,1,3,4,1]).reshape(-1,1)
        params, norm_data = normalize(data, type='z-norm')
        denorm_data = denormalize(norm_data, type='z-norm', params= params)
        assertNumpyArraysEqual(denorm_data, data)

    def test_min_max_norm(self):
        data = np.array([1,2,1,2,1,2,2,1]).reshape(-1,1)
        params, norm_data = normalize(data, type='min-max')

        denorm_data = denormalize(norm_data, type='min-max', params= params)
        assertNumpyArraysEqual(denorm_data, data)



    def test_log_norm(self):
        data = np.array([1,2,1,2,1,2,2,1])
        _, norm_data = normalize(data, type='lognorm')
        denorm_data = denormalize(norm_data, type='lognorm')
        assertNumpyArraysEqual(denorm_data, data)



    def test_other_norm(self):
        data = np.array([1,2,1,2,1,2,2,1])
        with self.assertRaises(Exception) as context:
            results = denormalize(data, type='other-norm')


if __name__ == '__main__':
    unittest.main()

