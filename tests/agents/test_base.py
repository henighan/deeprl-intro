""" tests for base agent """
from unittest import mock

import pytest
import numpy as np
import tensorflow as tf
import gym

from deeprl.agents.base import Base


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
