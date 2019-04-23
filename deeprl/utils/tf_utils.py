""" Tensorflow utility functions """
import tensorflow as tf
import numpy as np


def trainable_count(scope=''):
    """ returns the number of trainable parameters """
    return sum([np.prod(var.shape.as_list())
                for var in tf.trainable_variables()
                if scope in var.name])


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    """ Builds a multi-layer perceptron in Tensorflow.  """
    with tf.variable_scope('mlp'):
        for layer_no, hsz in enumerate(hidden_sizes[:-1]):
            with tf.variable_scope('layer_{}'.format(layer_no)):
                x = tf.layers.dense(x, units=hsz, activation=activation)
        with tf.variable_scope('output_layer'):
            x = tf.layers.dense(x, units=hidden_sizes[-1],
                                activation=output_activation)
    return x


def tfph(size, name=None):
    """ create tensorflow placeholder """
    shape = [None, size] if size else [None]
    return tf.placeholder(dtype=tf.float32, shape=shape, name=name)


def adam_opt(loss, learning_rate):
    """ Convenience function for optimizing a loss with Adam """
    return tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(loss)
