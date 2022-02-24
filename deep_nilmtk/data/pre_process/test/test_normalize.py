import unittest
from deep_nilmtk.data.pre_process.normalize import *

__unittest = True

from deep_nilmtk.utils import assertNumpyArraysEqual

class TestNormalize(unittest.TestCase):

    def test_z_norm(self):
        # test with data from only one appliance
        data = np.array([1,2,1,2,1,2,2,1]).reshape(-1,4)
        params, norm_data = normalize(data, type='z-norm')
        mean, std = params['mean'], params['std']
        assertNumpyArraysEqual(mean, data.mean(axis=0).reshape(-1,data.shape[-1]))
        assertNumpyArraysEqual(std, data.std(axis=0).reshape(-1,data.shape[-1]))
        self.assertEqual(data.shape, norm_data.shape)

        # test with data from only several appliances
        data = np.array([1,2,1,2,1,2,2,1]).reshape(-1,2)
        params, norm_data = normalize(data, type='z-norm')
        mean, std = params['mean'], params['std']
        assertNumpyArraysEqual(mean, data.mean(axis=0).reshape(-1,data.shape[-1]))
        assertNumpyArraysEqual(std, data.std(axis=0).reshape(-1,data.shape[-1]))
        self.assertEqual(data.shape, norm_data.shape)

    def test_min_max_norm(self):
        data = np.array([1,2,1,2,1,2,2,1]).reshape(-1,1)
        params,  norm_data = normalize(data, type='min-max')
        min_, max_ =  params['min'], params['max']
        assertNumpyArraysEqual(min_, data.min(axis=0).reshape(-1,data.shape[-1]))
        assertNumpyArraysEqual(max_, data.max(axis=0).reshape(-1,data.shape[-1]))
        self.assertEqual(data.shape, norm_data.shape)

        data = np.array([1,2,1,2,1,2,2,1]).reshape(-1,2)
        min_, max_, norm_data = normalize(data, type='min-max')
        assertNumpyArraysEqual(min_, data.min(axis=0).reshape(-1,data.shape[-1]))
        assertNumpyArraysEqual(max_, data.max(axis=0).reshape(-1,data.shape[-1]))
        self.assertEqual(data.shape, norm_data.shape)

    def test_log_norm(self):
        data = np.array([1,2,1,2,1,2,2,1]).reshape(-1,1)
        params, norm_data = normalize(data, type='lognorm')
        self.assertEqual(data.shape , norm_data.shape)

        # This case raises an exception because some values are
        # negative and the log can only work with positive values
        data = np.array([1,2,1,-4,-5,2,2,1])
        with self.assertRaises(Exception) as context:
            results = normalize(data, type='other-norm')

    def test_other_norm(self):
        data = np.array([1,2,1,2,1,2,2,1])
        with self.assertRaises(Exception) as context:
            results = normalize(data, type='other-norm')


if __name__ == '__main__':
    unittest.main()

