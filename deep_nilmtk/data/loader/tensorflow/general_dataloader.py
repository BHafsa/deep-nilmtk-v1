import numpy as np

from deep_nilmtk.data.loader.utils import target_generator

from deep_nilmtk.data.pre_process import pad_data
import tensorflow as tf

class GeneralDataLoader:
    def __init__(self, main, target, type, in_size, out_size, point_position=None):
        self.original_inputs = main
        self.original_targets = target
        self.dataset = tf.data.Dataset.from_generator(lambda: generate_sequences(main, target, type, in_size, out_size,
                                          point_position),
                                          output_signature= (
                                              tf.TensorSpec(shape=[ in_size,None], dtype=tf.float32),
                                              tf.TensorSpec(shape=[ out_size, ], dtype=tf.float32)
                                          ) if target is not None else (tf.TensorSpec(shape=[ in_size,None], dtype=tf.float32))).shuffle(buffer_size=150)



def generate_sequences(main, target, type, in_size, out_size, point_position=None):

    target_indice = target_generator(type, in_size, out_size, point_position)
    main = pad_data(main, in_size)
    if target is not None:
        target = pad_data(target, in_size)
        for i in range(len(main) - in_size):
            yield main[i:i + in_size, :], target[target_indice(i + np.arange(in_size)), :]
    else:
        for i in range(len(main) - in_size):
            yield main[i:i + in_size, :]
