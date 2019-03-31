""" Tensorflow utility functions """
import tensorflow as tf


def trainable_count():
    """ returns the number of trainable parameters """
    return tf.reduce_sum([tf.reduce_prod(var.get_shape())
                          for var in tf.trainable_variables()])


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