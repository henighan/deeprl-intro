""" Tests for Deterministic policy """
from unittest.mock import Mock

import tensorflow as tf
import numpy as np

from deeprl.policies import deterministic
from deeprl.utils import tf_utils


class TestDeterministic(tf.test.TestCase):
    """ Categorical Policy Unittests """

    def setUp(self):
        super().setUp()
        self.act_dim = 3
        self.batch_size = 12
        self.obs_dim = 5
        self.obs_ph = tf_utils.tfph(self.obs_dim)
        self.obs = np.random.randn(self.batch_size, self.obs_dim)

    def test_mlp_deterministic_policy(self):
        """ Smoke test mlp deterministic policy. Here, we just give a quick
        check that the output dimensions are right """
        action_space = Mock()
        action_space.shape = (self.act_dim,)
        pi = deterministic.mlp_deterministic_policy(
            self.obs_ph, hidden_sizes=(4,), activation=tf.tanh,
            action_space=action_space)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            ret = sess.run(pi, feed_dict={self.obs_ph: self.obs})
        assert ret.shape == (self.batch_size, self.act_dim)
