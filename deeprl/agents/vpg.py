import numpy as np
import tensorflow as tf
from gym.spaces import Box
from deeprl.policies.gaussian import mlp_gaussian_policy
from deeprl.tf_utils import mlp


class VPG():

    def __init__(self, pi_lr=1e-3, val_lr=1e-3, hidden_sizes=(32, 32),
                 activation=tf.tanh, val_train_iters=80, sess=None,
                 close_sess=True):
        self.pi_lr, self.val_lr = pi_lr, val_lr
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.val_train_iters = val_train_iters
        self.sess = sess or tf.Session()
        self.close_sess = close_sess
        self.placeholders = {}

    def step(self, obs):
        """ sample action given observation of environment """
        feed_dict = {self.placeholders['obs']: obs.reshape(1, -1)}
        act, val_t, logp_t = self.sess.run(
            (self.pi, self.val, self.logp_pi), feed_dict=feed_dict)
        return act[0], val_t[0], logp_t[0]

    def train(self, obs_buf, act_buf, adv_buf, ret_buf, logp_buf):
        """ Train after epoch """
        feed_dict = {self.placeholders['obs']: obs_buf,
                     self.placeholders['act']: act_buf,
                     self.placeholders['adv']: adv_buf,
                     self.placeholders['ret']: ret_buf}
        # calculate losses before training
        old_pi_loss, old_val_loss = self.sess.run(
            (self.pi_loss, self.val_loss), feed_dict=feed_dict)
        # update policy
        self.sess.run(self.pi_train_op, feed_dict=feed_dict)
        # update value function
        for ii in range(self.val_train_iters):
            self.sess.run(self.val_train_op, feed_dict=feed_dict)
        # calculate change in loss
        new_pi_loss, new_val_loss = self.sess.run(
            (self.pi_loss, self.val_loss), feed_dict=feed_dict)
        delta_pi_loss = new_pi_loss - old_pi_loss
        delta_val_loss = new_val_loss - old_val_loss
        return old_pi_loss, old_val_loss, delta_pi_loss, delta_val_loss

    def build_graph(self, obs_space, act_space):
        self.create_placeholders(obs_space, act_space)
        """ sampled action, logprob of input action, and logprob of
        sampled action """
        self.pi, self.logp, self.logp_pi = self.build_policy(
            act_space, self.placeholders['obs'], self.placeholders['act'],
            self.hidden_sizes, self.activation)
        # policy loss and train op
        self.pi_loss, self.pi_train_op = self.build_policy_loss(
            self.logp, self.placeholders['adv'], self.pi_lr)
        # value function estimator
        self.val = self.build_value_function(self.placeholders['obs'],
                                             hidden_sizes=self.hidden_sizes,
                                             activation=self.activation)
        # value function estimator loss and train op
        self.val_loss, self.val_train_op = self.build_val_loss(
            self.val, self.placeholders['ret'], lr=self.val_lr)
        self.sess.run(tf.global_variables_initializer())

    def create_placeholders(self, obs_space, act_space):
        """ Build the placeholders required for this agent """
        self.placeholders['obs'] = tf.placeholder(
            dtype=tf.float32, shape=[None, obs_space.shape[-1]], name='obs')
        self.placeholders['act'] = tf.placeholder(
            dtype=tf.float32, shape=[None, act_space.shape[-1]], name='act')
        for name in ('ret', 'adv'):
            self.placeholders[name] = tf.placeholder(
                dtype=tf.float32, shape=[None], name=name)

    @staticmethod
    def build_value_function(obs_ph, hidden_sizes, activation):
        with tf.variable_scope('val'):
            val = mlp(obs_ph, hidden_sizes=hidden_sizes + (1,),
                      activation=activation)
        return tf.reshape(val, [-1])

    @staticmethod
    def build_policy(act_space, obs_ph, act_ph, hidden_sizes, activation):
        if isinstance(act_space, Box):
            with tf.variable_scope('policy'):
                pi, logp, logp_pi = mlp_gaussian_policy(
                    obs_ph, act_ph, hidden_sizes=hidden_sizes,
                    activation=activation, action_space=act_space)
            return pi, logp, logp_pi
        raise NotImplementedError(
            "Only Policies for Box action space implemented")

    @staticmethod
    def build_policy_loss(logp, adv_ph, lr):
        pi_loss = tf.reduce_mean(-logp*adv_ph)
        pi_train_op = tf.train.AdamOptimizer(
            learning_rate=lr).minimize(pi_loss)
        return pi_loss, pi_train_op

    @staticmethod
    def build_val_loss(val, ret_ph, lr):
        val_loss = tf.losses.mean_squared_error(val, ret_ph)
        val_train_op = tf.train.AdamOptimizer(
            learning_rate=lr).minimize(val_loss)
        return val_loss, val_train_op

    # def __del__(self):
    #     if self.sess and self.close_sess:
    #         self.sess.close()
