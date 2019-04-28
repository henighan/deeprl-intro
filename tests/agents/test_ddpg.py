""" Tests for ddpg agent """
from unittest import mock
import pytest
import numpy as np
import tensorflow as tf
import gym

from deeprl.agents.ddpg import DDPG, TARGET, MAIN, POLICY, QVAL
from deeprl.utils import tf_utils


class TestDDPG(tf.test.TestCase):
    """ Tests for DDPG Agent """

    def setUp(self):
        super().setUp()
        self.env = gym.make('MountainCarContinuous-v0')
        self.agent = DDPG(hidden_sizes=(4,))
        self.obs_dim = 2
        self.act_dim = 1
        self.batch_size = 6
        self.obs_ph = tf_utils.tfph(self.obs_dim, 'obs')
        self.act_ph = tf_utils.tfph(self.act_dim, 'act')
        self.is_term_ph = tf_utils.tfph(None, 'is_term')
        self.rew_ph = tf_utils.tfph(None, 'rew')
        self.placeholders = {'obs': self.obs_ph, 'act': self.act_ph,
                             'next_obs': self.obs_ph,
                             'is_term': self.is_term_ph, 'rew': self.rew_ph}
        self.obs = np.random.randn(self.batch_size, self.obs_dim)
        self.act = np.random.randn(self.batch_size, self.act_dim)
        self.is_term = np.random.randint(0, 2, self.batch_size).astype(
            np.float32)
        self.rew = np.random.randn(self.batch_size)
        self.feed_dict = {
            self.obs_ph: self.obs, self.act_ph: self.act,
            self.is_term_ph: self.is_term, self.rew_ph: self.rew}

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
        """ test target_var_init op, sets target and main variables equal
        """
        with tf.variable_scope(TARGET):
            target_val = tf_utils.mlp(self.obs_ph, (4,), activation=tf.tanh)
        with tf.variable_scope(MAIN):
            main_val = tf_utils.mlp(self.obs_ph, (4,), activation=tf.tanh)
        with self.agent.sess as sess:
            sess.run(tf.global_variables_initializer())
            target_vars = tf_utils.var_list(TARGET)
            main_vars = tf_utils.var_list(MAIN)
            target_nps, main_nps = sess.run((target_vars, main_vars))
            for targ, upd in zip(target_nps, main_nps):
                assert targ.shape == upd.shape
                # the biases should actually be the same, all zeros
                if len(targ.shape) > 1:
                    assert not (targ == upd).all()
            # now set target and main equal
            init_op = self.agent.target_var_init()
            # now make sure all target and main parrameters are equal
            target_vars = tf_utils.var_list(TARGET)
            main_vars = tf_utils.var_list(MAIN)
            target_nps, main_nps = sess.run((target_vars, main_vars))
            for targ, upd in zip(target_nps, main_nps):
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
            pi_vars = tf_utils.var_list(POLICY)
            assert len(pi_vars) == 4 # 2 kernels and 2 biases
            qval_vars = tf_utils.var_list(QVAL)
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

    def test_build_policy_loss(self):
        """ test that the loss function makes sense, that the qval goes
        up when training, and that no qval variables are trained """
        dim = 3
        qval_np = np.random.randn(dim)
        pi_np = np.random.randn(dim)
        input_np = qval_np + pi_np
        with tf.variable_scope(MAIN + '/' + QVAL):
            qval_var = tf.get_variable('main', initializer=qval_np)
        with tf.variable_scope(MAIN + '/' + POLICY):
            pi_var = tf.get_variable('target', initializer=pi_np)
        input_var = qval_var + pi_var
        expected_init_loss = -np.mean(input_np)
        loss, train_op = self.agent.build_policy_loss(input_var)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            init_loss, init_pi_var, init_qval_var, init_input = sess.run(
                (loss, pi_var, qval_var, input_var))
            # make sure the loss matches the same calc in numpy
            np.testing.assert_allclose(expected_init_loss, init_loss)
            sess.run(train_op)
            final_loss, final_pi_var, final_qval_var, final_input = sess.run(
                (loss, pi_var, qval_var, input_var))
            assert final_loss < init_loss # make sure the loss went down
            # make sure the qval variables were NOT changed
            np.testing.assert_allclose(init_qval_var, final_qval_var)
            # make sure the input_var went up
            # (corresponding to q(s, pi) going up)
            assert all(final_input > init_input)

    def test_build_qval_target(self):
        """ test building targets for qval loss """
        qval_pi_targ_ph = tf_utils.tfph(None, 'qval_pi_targ')
        qval_pi_targ_np = np.random.randn(self.batch_size)
        expected = self.rew \
            + (1 - self.is_term)*self.agent.gamma*qval_pi_targ_np
        target = self.agent.build_qval_target(
            qval_pi_targ_ph, self.placeholders)
        feed_dict = {qval_pi_targ_ph: qval_pi_targ_np, **self.feed_dict}
        with self.cached_session() as sess:
            ret = sess.run(target, feed_dict=feed_dict)
        np.testing.assert_allclose(expected, ret)

    def test_build_qval_loss(self):
        """ test that the loss function makes sense, that the
        qval gets closer to the targets when training, and that
        no policy (pi) or target variables are changed """
        target_np = np.random.randn(self.batch_size)
        with tf.variable_scope(TARGET + '/' + QVAL):
            target_var = tf.get_variable(
                'target', initializer=target_np)
        qval_np = np.random.randn(self.batch_size)
        pi_np = np.random.randn(self.batch_size)
        input_np = qval_np + pi_np
        loss_np = np.mean((input_np - target_np)**2)
        with tf.variable_scope(MAIN + '/' + QVAL):
            qval_var = tf.get_variable('qval', initializer=qval_np)
        with tf.variable_scope(MAIN + '/' + POLICY):
            pi_var = tf.get_variable('pi', initializer=pi_np)
        input_var = qval_var + pi_var
        loss, train_op = self.agent.build_qval_loss(input_var, target_var)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            init_loss, init_target, init_qval, init_pi = sess.run(
                (loss, target_var, qval_var, pi_var))
            assert init_loss == pytest.approx(loss_np)
            sess.run(train_op)
            final_loss, final_target, final_qval, final_pi = sess.run(
                (loss, target_var, qval_var, pi_var))
            # make sure the loss went down
            assert final_loss < init_loss
            # make sure target variables were not updated
            np.testing.assert_allclose(init_target, final_target)
            # make sure policy variables were not updated
            np.testing.assert_allclose(init_pi, final_pi)

    def test_step_start_steps(self):
        """ test step when still in initial exploring phase """
        self.agent.create_placeholders(
            self.env.observation_space, self.env.action_space)
        self.agent.act_space = self.env.action_space
        self.agent.act_space.sample = mock.Mock(return_value='act')
        self.agent.sess = mock.Mock()
        ret = self.agent.step(self.obs)
        assert ret == 'act'
        assert self.agent.n_training_steps == 1
        self.agent.act_space.sample.assert_called_once()


    def test_step_after_start_steps(self):
        """ test step after initial exploring phase """
        self.agent.create_placeholders(
            self.env.observation_space, self.env.action_space)
        self.agent.n_training_steps = int(1e6)
        self.agent.act_space = mock.Mock()
        act = np.random.randn(1, self.act_dim)
        self.agent.sess.run = mock.Mock(return_value=act)
        self.agent.add_noise_and_clip = mock.Mock(return_value='act')
        ret = self.agent.step(self.obs)
        assert ret == 'act'
        self.agent.act_space.sample.assert_not_called
        self.agent.add_noise_and_clip.assert_called_once()

    def test_step_testing_true(self):
        """ test step when in testing mode """
        self.agent.create_placeholders(
            self.env.observation_space, self.env.action_space)
        self.agent.act_space = mock.Mock()
        act = np.random.randn(1, self.act_dim)
        self.agent.sess.run = mock.Mock(return_value=act)
        self.agent.add_noise_and_clip = mock.Mock()
        ret = self.agent.step(self.obs, testing=True)
        np.testing.assert_allclose(ret, act[0])
        self.agent.act_space.sample.assert_not_called
        self.agent.add_noise_and_clip.assert_not_called()
        # DONT increment training_stps counter
        assert self.agent.n_training_steps == 0
