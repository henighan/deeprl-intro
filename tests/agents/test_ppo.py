""" tests for ppo agent """
import numpy as np
import tensorflow as tf
import gym

from deeprl.agents import PPO

from deeprl import tf_utils
from deeprl.tf_utils import tfph


class TestPPO(tf.test.TestCase):
    """ Tests for PPO Agent """

    def setUp(self):
        super().setUp()
        self.env = gym.make('MountainCarContinuous-v0')
        self.batch_size = 6
        self.ppo = PPO(hidden_sizes=(4,))

    def test_build_max_surrogate(self):
        """ smoke test build_max_surrogate """
        clip_ratio = 0.2
        adv_ph = tfph(None)
        adv = np.random.randn(self.batch_size)
        expected = (1 + clip_ratio*np.sign(adv))*adv
        with self.cached_session() as sess:
            ret = sess.run(self.ppo.build_max_surrogate(clip_ratio, adv_ph),
                           feed_dict={adv_ph: adv})
        self.assertEqual(ret.shape, (self.batch_size,))
        np.testing.assert_allclose(expected, ret)

    def test_build_clipped_surrogate(self):
        """ smoke test build_clipped_surrogate """
        sur_ph = tfph(None)
        max_sur_ph = tfph(None)
        sur = np.random.randn(self.batch_size)
        max_sur = np.random.randn(self.batch_size)
        expected = np.array([min(sur_i, max_sur_i) for (sur_i, max_sur_i)
                             in zip(sur, max_sur)])
        with self.cached_session() as sess:
            ret = sess.run(self.ppo.build_clipped_surrogate(
                    sur_ph, max_sur_ph),
                feed_dict = {sur_ph: sur, max_sur_ph: max_sur})
        np.testing.assert_allclose(ret, expected)

    def test_build_policy_loss(self):
        """ make sure that when we train the loss, logp of actions with high
        advantage go up, and vice versa. Also check that kl-div goes up as we
        update the policy """
        pass
