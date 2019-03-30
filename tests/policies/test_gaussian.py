""" Tests for Gaussian Policy """
import tensorflow as tf
import numpy as np

from deeprl.policies import gaussian
from unittest.mock import Mock


class TestGaussian(tf.test.TestCase):
    """ Gaussian Policy Unittests """

    def test_make_log_std_smoke(self):
        """ smoke test make_log_std """
        dim = 12
        with self.cached_session() as sess:
            ret = gaussian.make_log_std_var(dim)
            n_trainable_variables = len(tf.trainable_variables())
            # ensure log_std is trainable
            self.assertEqual(n_trainable_variables, 1)
            sess.run(tf.global_variables_initializer())
            np.testing.assert_array_equal(ret.eval(), -0.5*np.ones([1, dim]))


    def test_sample_actions_smoke(self):
        """ smoke test sample_actions """
        batch_size = 5
        dim = 3
        mu = np.zeros(shape=[5, dim])
        log_std = np.zeros(shape=[1, dim])
        with self.cached_session() as sess:
            mu_ph = tf.placeholder(dtype=tf.float32, shape=[None, dim])
            log_std_ph = tf.placeholder(dtype=tf.float32, shape=[1, dim])
            ret = gaussian.sample_actions(mu_ph, log_std_ph)
            ret_eval = sess.run(ret, feed_dict={mu_ph: mu, log_std_ph: log_std})
            self.assertEqual(ret_eval.shape, (batch_size, dim))

    def test_mlp_gaussian_policy_smoke(self):
        """ smoke test of mlp_gaussian_policy """
        batch_size = 5
        obs_dim = 3
        act_dim = 2
        action_space = Mock()
        action_space.contains.return_value = True
        action_space.shape = (act_dim,)
        hidden_sizes = (4,)
        x = np.zeros(shape=[batch_size, obs_dim])
        a = np.zeros(shape=[batch_size, act_dim])
        with self.cached_session() as sess:
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None, obs_dim])
            a_ph = tf.placeholder(dtype=tf.float32, shape=[None, act_dim])
            ret = gaussian.mlp_gaussian_policy(
                x_ph, a_ph, hidden_sizes=hidden_sizes, activation=tf.tanh,
                output_activation=tf.tanh, action_space=action_space)
            sess.run(tf.global_variables_initializer())
            ret_pi, ret_logp, ret_logp_pi = sess.run(
                ret, feed_dict={x_ph: x, a_ph: a})
            self.assertEqual(ret_pi.shape, (batch_size, act_dim))
            self.assertEqual(ret_logp.shape, (batch_size,))
            self.assertEqual(ret_logp_pi.shape, (batch_size,))

    def test_gaussian_likelihood_log_std_2d(self):
        """ test for guassian likelihood when log_std is 2 dimensional """
        batch_size = 3
        dim = 2
        x = tf.constant(0, dtype=tf.float32, shape=[batch_size, dim])
        mu = tf.constant(0, dtype=tf.float32, shape=[batch_size, dim])
        log_std = tf.constant(1, dtype=tf.float32, shape=[batch_size, dim])
        ret = gaussian.gaussian_likelihood(x, mu, log_std)
        expected = (- dim - 0.5*dim*np.log(2*np.pi))*np.ones([batch_size])
        with self.cached_session():
            self.assertAllClose(expected, ret.eval())

    def test_gaussian_likelihood_log_std_1d(self):
        """ test for guassian likelihood when log_std is 1 dimensional """
        batch_size = 3
        dim = 2
        x = tf.constant(0, dtype=tf.float32, shape=[batch_size, dim])
        mu = tf.constant(0, dtype=tf.float32, shape=[batch_size, dim])
        log_std = tf.constant(1, dtype=tf.float32, shape=[dim])
        ret = gaussian.gaussian_likelihood(x, mu, log_std)
        expected = (- dim - 0.5*dim*np.log(2*np.pi))*np.ones([batch_size])
        with self.cached_session():
            self.assertAllClose(expected, ret.eval())

    def test_gaussian_likelihood_non_zero_difference(self):
        """ test for guassian likelihood when log_std is 1 dimensional """
        batch_size = 3
        dim = 2
        x = tf.constant(2, dtype=tf.float32, shape=[batch_size, dim])
        mu = tf.constant(0, dtype=tf.float32, shape=[batch_size, dim])
        log_std = tf.constant(1, dtype=tf.float32, shape=[dim])
        ret = gaussian.gaussian_likelihood(x, mu, log_std)
        expected_summation_term = -0.5*dim*(4/np.e**2 + 2)
        expected_scalar_term = -0.5*dim*np.log(2*np.pi)
        expected = (expected_summation_term
                    + expected_scalar_term)*np.ones([batch_size])
        with self.cached_session():
            self.assertAllClose(expected, ret.eval())
