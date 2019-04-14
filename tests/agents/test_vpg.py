""" Tests for vpg agent """
import numpy as np
import tensorflow as tf
import gym

from deeprl.agents.vpg import VPG
from deeprl.utils import tf_utils
from deeprl.utils.tf_utils import tfph


class TestVPG(tf.test.TestCase):
    """ Tests for VPG Agent """

    def setUp(self):
        super().setUp()
        self.env = gym.make('MountainCarContinuous-v0')
        self.vpg = VPG(hidden_sizes=(4,))

    def test_create_placeholders_smoke(self):
        """ smoke test placeholder creation """
        self.vpg.create_placeholders(self.env.observation_space,
                                     self.env.action_space)
        self.assertSetEqual(set(list(self.vpg.placeholders.keys())),
                            {'obs', 'ret', 'adv', 'act'})

    def test_build_value_function_smoke(self):
        """ check the num of trainable params and output shape make sense for
        build_value_func """
        obs_dim = self.env.observation_space.shape[0]
        obs_ph = tf_utils.tfph(obs_dim)
        obs = np.random.rand(8, obs_dim)
        val = self.vpg.build_value_function(
            obs_ph, hidden_sizes=(4,), activation=None)
        n_params = (obs_dim+1)*4 + (4 + 1)*1
        with self.cached_session() as sess:
            ret_n_params = tf_utils.trainable_count(scope='val')
            sess.run(tf.global_variables_initializer())
            sess_val = sess.run(val, feed_dict={obs_ph: obs})
        self.assertEqual(n_params, ret_n_params)
        self.assertEqual(sess_val.shape, (8,))

    def test_build_val_loss_smoke(self):
        """ Make sure the loss goes down when training, and that training
        brings val closer to rets """
        batch_size = 4
        ret_ph = tf_utils.tfph(None)
        ret = np.ones(batch_size)
        val = tf.get_variable(
            'val', dtype=tf.float32, trainable=True,
            initializer=4*[0.])
        loss, train_op = self.vpg.build_val_loss(val, ret_ph, 1e-3)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            old_loss = sess.run(loss, feed_dict={ret_ph: ret})
            sess.run(train_op, feed_dict={ret_ph: ret})
            new_loss = sess.run(loss, feed_dict={ret_ph: ret})
            new_val = sess.run(val)
        self.assertEqual(new_loss.shape, tuple())
        self.assertLess(new_loss, old_loss)
        self.assertTrue(all(new_val > 0))

    def test_build_policy_loss_smoke(self):
        """ Make sure the loss goes down when training, and that training
        changes logp in the expected direction """
        batch_size = 4
        adv_ph = tfph(None)
        adv = np.ones(batch_size)
        logp = tf.get_variable(
            'adv', dtype=tf.float32, trainable=True,
            initializer=batch_size*[0.])
        loss, train_op = self.vpg.build_policy_loss(
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

    def test_build_graph_smoke(self):
        """ Smoke test for building the graph """
        self.vpg.build_graph(self.env.observation_space, self.env.action_space)
