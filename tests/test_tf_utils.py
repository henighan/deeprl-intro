""" Tests for tensorflow utils """
import tensorflow as tf
import numpy as np

from deeprl import tf_utils


class TestTfUtils(tf.test.TestCase):

    def test_trainable_count_smoke(self):
        """ smoke test trainable count """
        with self.cached_session() as sess:
            var = tf.get_variable('test', shape=[4, 5], trainable=True)
            ret = tf_utils.trainable_count()
            assert sess.run(ret) == 4*5

    def test_mlp_smoke(self):
        """ Smoke test mlp """
        batch_size = 3
        input_dim = 2
        output_dim = 4
        x = np.zeros(shape=[batch_size, input_dim], dtype=np.float32)
        with self.cached_session() as sess:
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
            ret = tf_utils.mlp(x_ph, hidden_sizes=(output_dim,))
            sess.run(tf.global_variables_initializer())
            n_trainable_variables = 2 # 1 kernel and 1 bias
            trainable_variables = tf.trainable_variables()
            self.assertEqual(len(trainable_variables), n_trainable_variables)
            ret_eval = sess.run(ret, feed_dict={x_ph: x})
            self.assertEqual(ret_eval.shape, (batch_size, output_dim))

    def test_mlp_multiple_layers(self):
        """ test mlp makes multiple layers, with weights of the correct
        shapes """
        batch_size = 3
        input_dim = 2
        hidden_sizes = (5, 4, 3)
        x = np.zeros(shape=[batch_size, input_dim], dtype=np.float32)
        with self.cached_session() as sess:
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
            ret = tf_utils.mlp(x_ph, hidden_sizes=hidden_sizes)
            sess.run(tf.global_variables_initializer())
            n_trainable_variables = 6 # 3 kernels and 3 bias
            ret_eval = sess.run(ret, feed_dict={x_ph: x})
            self.assertEqual(ret_eval.shape, (batch_size, 3))
            trainable_variables = sess.run(tf.trainable_variables())
            variable_shapes = [var.shape for var in trainable_variables]
            self.assertEqual(len(trainable_variables), n_trainable_variables)
            # kernels
            self.assertIn((2, 5), variable_shapes)
            self.assertIn((5, 4), variable_shapes)
            self.assertIn((4, 3), variable_shapes)
            # biases
            self.assertIn((5,), variable_shapes)
            self.assertIn((4,), variable_shapes)
            self.assertIn((3,), variable_shapes)
