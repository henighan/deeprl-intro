""" Tests for Categorical policy utils """
from unittest.mock import Mock

import tensorflow as tf
import numpy as np

from deeprl.policies import categorical
from deeprl import tf_utils


class TestCategorical(tf.test.TestCase):
    """ Categorical Policy Unittests """

    def setUp(self):
        super().setUp()
        self.n_cat = 5
        self.batch_size = 12
        self.logits_ph = tf_utils.tfph(5)
        self.logits = np.random.rand(self.batch_size, self.n_cat)

    def test_sample_actions(self):
        """ test sampling actions """
        sampled_actions = categorical.sample_actions(self.logits_ph)
        with self.cached_session() as sess:
            ret = sess.run(sampled_actions, feed_dict={
                self.logits_ph: self.logits})
        self.assertEqual(ret.shape, (self.batch_size,))
        self.assertLessEqual(max(ret), self.n_cat-1)
        self.assertGreaterEqual(min(ret), 0)

    def test_logits_to_log_probs(self):
        """ smoke tests for logits_to_log_probs """
        # we can subtract any constant from the logits and it doesnt change
        # the true probabilities. It DOES however increase numerical
        # stability :)
        max_subtracted_logits = self.logits - np.max(
            self.logits, axis=1, keepdims=True)
        unnormalized_probs = np.exp(max_subtracted_logits)
        probs = unnormalized_probs / np.sum(
            unnormalized_probs, axis=1, keepdims=True)
        log_probs = np.log(probs)
        with self.cached_session() as sess:
            ret = sess.run(categorical.logits_to_log_probs(self.logits_ph),
                           feed_dict={self.logits_ph: self.logits})
        self.assertEqual(ret.shape, (self.batch_size, self.n_cat))
        np.testing.assert_allclose(ret, log_probs)

    def test_log_prob_of_action(self):
        """ smoke test log_prob_of_action """
        log_probs_ph = tf_utils.tfph(self.n_cat)
        log_probs = np.random.rand(self.batch_size, self.n_cat)
        actions_ph = tf.placeholder(dtype=tf.int64, shape=[None])
        actions = np.random.randint(0, self.n_cat, self.batch_size, np.int64)
        with self.cached_session() as sess:
            ret = sess.run(categorical.log_prob_of_action(
                log_probs_ph, actions_ph), feed_dict={log_probs_ph: log_probs,
                                                      actions_ph: actions})
        for ind in range(self.batch_size):
            self.assertAlmostEqual(ret[ind], log_probs[ind, actions[ind]])

    def test_mlp_categorical_policy(self):
        """ smoke test mlp_categorical_policy """
        action_space = Mock()
        action_space.n = self.n_cat
        obs_ph = tf_utils.tfph(2)
        obs = np.random.rand(self.batch_size, 2)
        act_ph = tf.placeholder(dtype=tf.int64, shape=[None])
        act = np.random.randint(0, self.n_cat, self.batch_size, np.int64)
        with self.cached_session() as sess:
            ret_symbol = categorical.mlp_categorical_policy(
                obs_ph, act_ph, hidden_sizes=(4,), activation=tf.tanh,
                action_space=action_space)
            sess.run(tf.global_variables_initializer())
            ret = sess.run(ret_symbol, feed_dict={obs_ph: obs, act_ph: act})
        for val in ret:
            self.assertEqual(val.shape, (self.batch_size,))
