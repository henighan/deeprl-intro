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
        self.batch_size = 10
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
            ret = sess.run(
                self.ppo.build_clipped_surrogate(sur_ph, max_sur_ph),
                feed_dict={sur_ph: sur, max_sur_ph: max_sur})
        np.testing.assert_allclose(ret, expected)

    def test_build_policy_loss(self):
        """ make sure that when we train the loss, logp of actions with high
        advantage go up, and vice versa. Also check that kl-div goes up as we
        update the policy """
        learning_rate = 1e-3
        logp_old_ph = tfph(None)
        logp_old = np.log(np.random.rand(self.batch_size)).astype(np.float32)
        adv_ph = tfph(None)
        adv = np.random.randn(self.batch_size).astype(np.float32)
        logp = tf.get_variable('logp', dtype=tf.float32, trainable=True,
                               initializer=logp_old)
        placeholders = {'logp': logp_old_ph, 'adv': adv}
        pi_loss, pi_train_op = self.ppo.build_policy_loss(
            logp, placeholders, learning_rate)
        feed_dict = {logp_old_ph: logp_old, adv_ph: adv}
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            init_loss, init_kl = sess.run((pi_loss, self.ppo.kl_divergence),
                                          feed_dict=feed_dict)
            self.assertAlmostEqual(init_loss, -np.mean(adv), places=5)
            # since the new and old policies are the before training, kl
            # divergence should be zero
            self.assertTrue(all(init_kl == 0))
            sess.run(pi_train_op, feed_dict=feed_dict)
            after_loss, after_kl = sess.run((pi_loss, self.ppo.kl_divergence),
                                          feed_dict=feed_dict)
            # ensure the loss went down
            self.assertLess(after_loss, init_loss)
            delta_logp = sess.run(logp) - logp_old
            # ensure that logp goes up if adv > 0 and vice versa
            np.testing.assert_array_equal(np.sign(delta_logp),
                                          np.sign(adv))
            # ensure that kl_div went up
            print(after_kl)
            self.assertTrue(all(after_kl > 0))
            print(delta_logp)
            print(adv)
            self.assertTrue(False)
