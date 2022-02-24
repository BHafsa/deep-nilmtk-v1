import unittest

from deep_nilmtk.data.post_process.denormalize import *
from deep_nilmtk.data.pre_process.normalize import *
from deep_nilmtk.utils import assertNumpyArraysEqual

class TestDenormalize(unittest.TestCase):

    def test_z_norm(self):
        data = np.array([1,2,1,2,1,3,4,1]).reshape(-1,2)
        mean, std, norm_data = normalize(data, type='z-norm')
        denorm_data = denormalize(norm_data, type='z-norm')
        assertNumpyArraysEqual(denorm_data, data)

    def test_min_max_norm(self):
        data = np.array([1,2,1,2,1,2,2,1]).reshape(-1,2)
        min_, max_, norm_data = normalize(data, type='min-max')
        denorm_data = denormalize(norm_data, type='min-max')
        assertNumpyArraysEqual(denorm_data, data)



    def test_log_norm(self):
        data = np.array([1,2,1,2,1,2,2,1])
        norm_data = normalize(data, type='lognorm')
        denorm_data = denormalize(norm_data, type='lognorm')
        assertNumpyArraysEqual(denorm_data, data)



    def test_other_norm(self):
        data = np.array([1,2,1,2,1,2,2,1])
        with self.assertRaises(Exception) as context:
            results = denormalize(data, type='other-norm')


if __name__ == '__main__':
    unittest.main()

