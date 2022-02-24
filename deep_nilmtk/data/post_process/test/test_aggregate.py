import unittest

from deep_nilmtk.data.post_process.aggregate import *


class TestAggregate(unittest.TestCase):

    def test_aggregate_mean(self):
        data = np.random.randint(0, 1200, 64 * 64).reshape(64, 64)
        agg_data = aggregate_seq(data, type='mean')
        self.assertEqual(agg_data.shape[0], data.shape[0] + data.shape[1] - 1)

    def test_aggregate_median(self):
        data = np.random.randint(0, 1200, 64 * 64).reshape(64, 64)
        agg_data = aggregate_seq(data, type='median')
        self.assertEqual(agg_data.shape[0], data.shape[0] + data.shape[1] - 1)

    def test_aggregate_others(self):
        data = np.random.randint(0, 1200, 64 * 64).reshape(64, 64)
        with self.assertRaises(Exception) as context:
            results = aggregate_seq(data, type='other-agg')


if __name__ == '__main__':
    unittest.main()
