""" Tests for ddpg agent """
from unittest import mock
import numpy as np
import tensorflow as tf
import gym

from deeprl.agents.ddpg import DDPG, TARGET, MAIN
from deeprl.utils import tf_utils


class TestDDPG(tf.test.TestCase):
    """ Tests for DDPG Agent """

    def setUp(self):
        super().setUp()
        self.env = gym.make('MountainCarContinuous-v0')
        self.agent = DDPG(hidden_sizes=(4,))
        self.obs_dim = 3
        self.act_dim = 1
        self.batch_size = 6
        self.obs_ph = tf_utils.tfph(self.obs_dim)
        self.act_ph = tf_utils.tfph(self.act_dim)
        self.placeholders = {'obs': self.obs_ph, 'act': self.act_ph,
                             'next_obs': self.obs_ph}
        self.obs = np.random.randn(self.batch_size, self.obs_dim)
        self.act = np.random.randn(self.batch_size, self.act_dim)

    def test_build_action_value_function(self):
        """ smoke test build action value function. The input feature vector
        should be a concatenation of the obs and act. Given this, make sure
        the kernel shapes make sense """
        ret = self.agent.build_action_value_function(
            self.obs_ph, self.act_ph, self.agent.hidden_sizes,
            self.agent.activation)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            trainable_variables = sess.run(tf.trainable_variables())
            assert len(trainable_variables) == 4 # 2 kernels and 2 biases
            variable_shapes = [var.shape for var in trainable_variables]
            # kernel shapes
            self.assertIn((self.act_dim + self.obs_dim, 4), variable_shapes)
            self.assertIn((4, 1), variable_shapes)
            # bias shapes
            self.assertIn((4,), variable_shapes)
            self.assertIn((1,), variable_shapes)
            ret_np = sess.run(ret, feed_dict={self.obs_ph: self.obs,
                                              self.act_ph: self.act})
        assert ret_np.shape == (self.batch_size,)

    def test_target_var_init(self):
        """ test target_var_init op, sets target and updated variables equal
        """
        with tf.variable_scope(TARGET):
            target_val = tf_utils.mlp(self.obs_ph, (4,), activation=tf.tanh)
        with tf.variable_scope(MAIN):
            updated_val = tf_utils.mlp(self.obs_ph, (4,), activation=tf.tanh)
        init_op = self.agent.target_var_init()
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            target_vars = tf_utils.var_list(TARGET)
            updated_vars = tf_utils.var_list(MAIN)
            target_nps, updated_nps = sess.run((target_vars, updated_vars))
            for targ, upd in zip(target_nps, updated_nps):
                assert targ.shape == upd.shape
                # the biases should actually be the same, all zeros
                if len(targ.shape) > 1:
                    assert not (targ == upd).all()
            # now set target and updated equal
            sess.run(init_op)
            # now make sure all target and updated parrameters are equal
            target_vars = tf_utils.var_list(TARGET)
            updated_vars = tf_utils.var_list(MAIN)
            target_nps, updated_nps = sess.run((target_vars, updated_vars))
            for targ, upd in zip(target_nps, updated_nps):
                assert targ.shape == upd.shape
                np.testing.assert_allclose(targ, upd)

    def test_target_update_op(self):
        """ test updating target parameters based on polyak """
        self.agent.polyak = 0.95
        with tf.variable_scope(TARGET):
            target_var = tf.get_variable('targ', dtype=tf.float64,
                                         initializer=np.array([0., 0.]))
        with tf.variable_scope(MAIN):
            updated_var = tf.get_variable('updated', dtype=tf.float64,
                                          initializer=np.array([1., 1.]))
        target_update_op = self.agent.build_target_update_op()
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(target_update_op)
            updated_targ = sess.run(target_var)
            np.testing.assert_allclose(updated_targ, np.array([0.05, 0.05]))

    def test_build_policy_and_qval(self):
        """ smoke test, make sure the number of parameters is right """
        pi, qval, qval_pi = self.agent.build_policy_and_qval(
            self.obs_ph, self.act_ph, self.env.action_space)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            pi_vars = tf_utils.var_list('pi')
            assert len(pi_vars) == 4 # 2 kernels and 2 biases
            qval_vars = tf_utils.var_list('qval')
            assert len(qval_vars) == 4 # 2 kernels and 2 biases

    def test_build_estimators(self):
        """ smoke test, make sure the number of parameters is right """
        estimators = self.agent.build_estimators(
            self.placeholders, None, self.env.action_space)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            pi_vars = tf_utils.var_list(MAIN + '/pi')
            assert len(pi_vars) == 4 # 2 kernels and 2 biases
            qval_vars = tf_utils.var_list(MAIN + '/qval')
            assert len(qval_vars) == 4 # 2 kernels and 2 biases
            pi_vars = tf_utils.var_list(TARGET + '/pi')
            assert len(pi_vars) == 4 # 2 kernels and 2 biases
            qval_vars = tf_utils.var_list(TARGET + '/qval')
            assert len(qval_vars) == 4 # 2 kernels and 2 biases

    def test_add_noise_and_clip_clipping(self):
        """ test the clipping functionality of add_noise_and_clip """
        act = np.array([3., -3.])
        act_noise = 0
        act_space = mock.Mock()
        act_space.high = np.ones_like(act)
        act_space.low = -1*act_space.high
        ret = self.agent.add_noise_and_clip(act, act_noise, act_space)
        np.testing.assert_allclose(ret, np.array([1.0, -1.0]))
