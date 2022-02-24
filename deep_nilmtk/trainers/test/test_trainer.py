import unittest

from deep_nilmtk.trainers import Trainer, TorchTrainer

class TestTrainer(unittest.TestCase):

    def test_init(self):
        trainerImp=TorchTrainer({
            'test': 'Hello World'
        })
        trainer = Trainer(trainerImp)
        self.assertEqual(trainer.test(),  'Hello World')


