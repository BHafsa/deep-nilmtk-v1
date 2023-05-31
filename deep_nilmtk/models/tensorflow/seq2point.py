
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,Dropout,Flatten, Dense


class Seq2Point(tf.keras.Model):

    def __init__(self, hparams):
        super().__init__(name=hparams['model_name'] + str(hparams['version']))

        self.sequence_length = hparams['in_size']
        self.nb_appliances = len(hparams['appliances']) if hparams['multi_appliance'] else 1
        self.version = hparams['version']


        self.net = Sequential([
            Conv1D(30,10,activation="relu",input_shape=(self.sequence_length,1),strides=1),
            Conv1D(30, 8, activation='relu', strides=1),
            Conv1D(40, 6, activation='relu', strides=1),
            Conv1D(50, 5, activation='relu', strides=1),
            Dropout(.2),
            Conv1D(50, 5, activation='relu', strides=1),
            Dropout(.2),
            Flatten(),
            Dense(1024, activation='relu'),
            Dense(self.nb_appliances)
        ])
        self.build(input_shape = (None, self.sequence_length,1))

    def call(self, x, **kwargs):
        return self.net(x)

    @staticmethod
    def suggest_hparams(trial):
        """
        Function returning list of params that will be suggested from optuna

        :param trial: Optuna Trial.
        :type trial: optuna.trial
        :return: Parameters with values suggested by optuna
        :rtype: dict
        """

        window_length = trial.suggest_int('in_size', low=50, high=1800)
        window_length += 1 if window_length % 2 == 0 else 0
        return {
            # 'in_size': window_length,
            'point_position': trial.suggest_categorical('point_position', ['last_position', 'mid_position'])
        }









