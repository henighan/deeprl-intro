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

    def test_build_graph_smoke(self):
        """ Smoke test for building the graph """
        self.vpg.build_graph(self.env.observation_space, self.env.action_space)
