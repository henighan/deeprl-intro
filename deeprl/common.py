""" Common Things """
import tensorflow as tf


DEFAULT_KWARGS = {'epochs': 50, 'steps_per_epoch': 4000,
                  'env_name': None, 'hidden_sizes': (64,64),
                  'activation': tf.tanh}
