""" Vanilla Policy Gradient Agent """
import tensorflow as tf

from gym.spaces import Box, Discrete

from deeprl.utils.tf_utils import mlp, adam_opt
from deeprl.agents.base import Base
from deeprl.policies.gaussian import mlp_gaussian_policy
from deeprl.policies.categorical import mlp_categorical_policy


class VPG(Base):
    """ Vanilla Policy Gradient Agent """

    # things to log at each timestep, and what column name to use
    log_each_step = {'val': 'VVals', 'logp': 'LogP'}
    # estimators we want to send to the buffer on each step
    to_buffer_each_step_dict = {'val': 'val', 'act': 'pi', 'logp': 'logp_pi'}
    # kwargs for logger.log_tabular
    log_tabular_kwargs = {'VVals': {'with_min_and_max': True},
                          'LogP': {'with_min_and_max': True}}
    # since this is an on-policy algorithm, we instruct the learner
    # to not run additional episodes for evaluation, but evaluate performance
    # directly on the same episodes used for training
    eval_after_epoch = False

    def __init__(self, pi_lr=3e-4, val_lr=1e-3, hidden_sizes=(64, 64),
                 activation=tf.tanh, val_train_iters=80, sess=None,
                 close_sess=True):
        super().__init__(hidden_sizes=hidden_sizes, activation=activation,
                         sess=sess, close_sess=close_sess)
        self.pi_lr, self.val_lr = pi_lr, val_lr
        self.val_train_iters = val_train_iters

    def train(self, replay_buffer):
        """ Train after epoch. Returns dict of things we want to have logged,
        including the Policy and Value losses, and the change in the losses
        due to training """
        buffer_dump = replay_buffer.dump()
        feed_dict = {self.placeholders[key]: buffer_dump[key]
                     for key in self.train_ph_keys}
        # calculate losses before training
        initial_losses = self.sess.run(self.losses, feed_dict=feed_dict)
        # update model parameters
        update_to_logs = self.update_parameters(feed_dict)
        # get updated losses, use to calculate the change in losses
        new_losses = self.sess.run(self.losses, feed_dict=feed_dict)
        delta_losses = self.delta_losses(initial_losses, new_losses)
        return {**update_to_logs,
                **initial_losses,
                **delta_losses}

    def update_parameters(self, feed_dict):
        """ update agent model parameters and return dict of things
        we want logged """
        policy_to_logs = self.update_policy(feed_dict)
        value_to_logs = self.update_value_function(feed_dict)
        return {**policy_to_logs, **value_to_logs}

    def update_policy(self, feed_dict):
        """ update the policy based on the replay-buffer data (stored in
        feed_dict) """
        self.sess.run(self.train_ops['LossPi'], feed_dict=feed_dict)
        return {}

    def update_value_function(self, feed_dict):
        """ update the policy based on the replay-buffer data (stored in
        feed_dict) """
        for _ in range(self.val_train_iters):
            self.sess.run(self.train_ops['LossVVals'], feed_dict=feed_dict)
        return {}

    def build_estimators(self, placeholders, obs_space, act_space):
        """ build the policy and value function """
        pi, logp, logp_pi = self.build_policy(
            act_space, placeholders['obs'], placeholders['act'],
            self.hidden_sizes, self.activation)
        val = self.build_value_function(placeholders['obs'],
                                        hidden_sizes=self.hidden_sizes,
                                        activation=self.activation)
        return {'pi': pi, 'logp': logp, 'logp_pi': logp_pi, 'val': val}

    def build_losses(self, estimators, placeholders):
        """ build the value and policy losses and training ops """
        pi_loss, pi_train_op = self.build_policy_loss(
            estimators['logp'], placeholders, self.pi_lr)
        val_loss, val_train_op = self.build_mse_loss(
            estimators['val'], placeholders['ret'], self.val_lr)
        losses = {'LossPi': pi_loss, 'LossVVals': val_loss}
        train_ops = {'LossPi': pi_train_op, 'LossVVals': val_train_op}
        return losses, train_ops

    @staticmethod
    def build_value_function(obs_ph, hidden_sizes, activation):
        """ build the graph for the value function """
        with tf.variable_scope('val'):
            val = mlp(obs_ph, hidden_sizes=hidden_sizes + (1,),
                      activation=activation)
        return tf.reshape(val, [-1])

    @staticmethod
    def build_policy(act_space, obs_ph, act_ph, hidden_sizes, activation):
        """ build the graph for the policy """
        if isinstance(act_space, Box):
            with tf.variable_scope('pi'):
                pi, logp, logp_pi = mlp_gaussian_policy(
                    obs_ph, act_ph, hidden_sizes=hidden_sizes,
                    activation=activation, action_space=act_space)
            return pi, logp, logp_pi
        if isinstance(act_space, Discrete):
            with tf.variable_scope('pi'):
                pi, logp, logp_pi = mlp_categorical_policy(
                    obs_ph, act_ph, hidden_sizes=hidden_sizes,
                    activation=activation, action_space=act_space)
            return pi, logp, logp_pi
        raise NotImplementedError('action space {} not implemented'.format(
            act_space))

    @staticmethod
    def build_policy_loss(logp, placeholders, learning_rate):
        """ build the graph for the policy loss """
        pi_loss = tf.reduce_mean(-logp*placeholders['adv'])
        return pi_loss, adam_opt(pi_loss, learning_rate)
