""" tests for base agent """
from unittest import mock

import pytest
import numpy as np
import tensorflow as tf
import gym

from deeprl.agents.base import Base
from deeprl.utils import tf_utils


class TestBase(tf.test.TestCase):
    """ Tests for VPG Agent """

    def setUp(self):
        super().setUp()
        self.env = gym.make('MountainCarContinuous-v0')
        self.agent = Base(hidden_sizes=(4,))

    def test_create_placeholders_smoke(self):
        """ smoke test placeholder creation """
        self.agent.create_placeholders(self.env.observation_space,
                                       self.env.action_space)
        self.assertSetEqual(set(list(self.agent.placeholders.keys())),
                            {'obs', 'ret', 'adv', 'act'})

    def test_step(self):
        """ smoke test agent step """
        agent = Base(hidden_sizes=(4,))
        agent.create_placeholders(
            self.env.observation_space, self.env.action_space)
        obs = np.random.randn(self.env.observation_space.shape[0])
        agent.sess.run = mock.Mock(return_value={'val': [2.]})
        agent.log_each_step = {'val': 'VVals'}
        ret_to_buf, ret_to_log = agent.step(obs)
        assert ret_to_buf == {'val': 2}
        assert ret_to_log == {'VVals': 2}

    def test_delta_losses(self):
        """ smoke test delta losses """
        initial_losses = {'LossPi': 3.2, 'LossV': 12.3}
        new_losses = {'LossPi': 2.8, 'LossV': 10.3}
        ret_dlosses = self.agent.delta_losses(initial_losses, new_losses)
        assert set(list(ret_dlosses.keys())) == {'DeltaLossPi', 'DeltaLossV'}
        assert ret_dlosses['DeltaLossPi'] == pytest.approx(-0.4)
        assert ret_dlosses['DeltaLossV'] == pytest.approx(-2.0)

    def test_build_policy_loss_smoke(self):
        """ Make sure the loss goes down when training, and that training
        changes logp in the expected direction """
        batch_size = 4
        adv_ph = tf_utils.tfph(None)
        adv = np.ones(batch_size)
        logp = tf.get_variable(
            'adv', dtype=tf.float32, trainable=True,
            initializer=batch_size*[0.])
        loss, train_op = self.agent.build_policy_gradient_loss(
            logp, {'adv': adv_ph}, learning_rate=1e-3)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            old_loss = sess.run(loss, feed_dict={adv_ph: adv})
            sess.run(train_op, feed_dict={adv_ph: adv})
            new_loss = sess.run(loss, feed_dict={adv_ph: adv})
            new_logp = sess.run(logp)
        self.assertEqual(new_loss.shape, tuple())
        self.assertLess(new_loss, old_loss)
        self.assertTrue(all(new_logp > 0))

    def test_build_mse_loss_smoke(self):
        """ Make sure the loss goes down when training, and that training
        brings estimates closer to targets """
        batch_size = 4
        targets_ph = tf_utils.tfph(None)
        targets = np.ones(batch_size)
        estimates_var = tf.get_variable(
            'val', dtype=tf.float32, trainable=True,
            initializer=4*[0.])
        loss, train_op = self.agent.build_mse_loss(
            estimates_var, targets_ph, 1e-3)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            old_loss = sess.run(loss, feed_dict={targets_ph: targets})
            sess.run(train_op, feed_dict={targets_ph: targets})
            new_loss = sess.run(loss, feed_dict={targets_ph: targets})
            new_val = sess.run(estimates_var)
        self.assertEqual(new_loss.shape, tuple())
        self.assertLess(new_loss, old_loss)
        self.assertTrue(all(new_val > 0))

