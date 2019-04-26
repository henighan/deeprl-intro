""" Deep Deterministic Policy Gradient Agent """
import tensorflow as tf
import numpy as np
from gym.spaces import Box

from deeprl.utils import tf_utils
from deeprl.agents.base import Base
from deeprl.policies.deterministic import mlp_deterministic_policy

TARGET = 'target'
MAIN = 'main'


class DDPG(Base):
    """ Deep deterministic policy gradient. The first (start_steps) steps are
    sampled randomly and uniformly from the action space to force exploration
    early on. After this, its normal DDPG """

    train_ph_keys = ['obs', 'next_obs', 'act', 'is_term', 'rew']
    to_buffer_each_step_dict = {'act': 'main_pi', 'qval': 'main_qval_pi'}
    log_each_step = {'qval': 'QVals'}
    log_tabular_kwargs = {'QVals': {'average_only': True}}
    polyak = 0.95

    def __init__(self, hidden_sizes=(64, 64), activation=tf.tanh,
                 sess=None, close_sess=True, pi_lr=1e-3, q_lr=1e-3,
                 act_noise=0.1, gamma=0.99, n_batches=5000,
                 start_steps=10000):
        super().__init__(hidden_sizes=hidden_sizes, activation=activation,
                         sess=None, close_sess=True)
        self.pi_lr, self.q_lr = pi_lr, q_lr
        self.act_noise = act_noise
        self.act_space = None
        self.target_update_op = None
        self.gamma = gamma
        self.n_batches = n_batches
        self.step_ctr = 0

    def step(self, obs, testing=False):
        """ sample action given observation """
        to_buffer, to_log = super().step(obs, testing=testing)
        # when not in testing mode, add noise to sampled action
        if not testing:
            self.step_ctr += 1
            if self.step_ctr < self.start_steps:
                to_buffer['act'] = self.action_space.sample()
            else:
                to_buffer['act'] = self.add_noise_and_clip(
                    to_buffer['act'], self.act_noise, self.act_space)
        return to_buffer, to_log

    @staticmethod
    def add_noise_and_clip(act, act_noise, act_space):
        """ add noise to action, then clip to ensure it's within
        the action space """
        noise = act_noise*np.random.randn(*act.shape)
        return np.clip(act + noise, act_space.low, act_space.high)

    @staticmethod
    def build_policy(act_space, obs_ph, hidden_sizes, activation):
        """ build the deterministic policy """
        if isinstance(act_space, Box):
            pi = mlp_deterministic_policy(obs_ph, hidden_sizes,
                                          activation, act_space)
            return pi
        raise NotImplementedError(
            "DDPG can only be used with continuous action spaces")

    @staticmethod
    def build_action_value_function(obs_ph, act_ph, hidden_sizes, activation):
        """ build the action-value function estimator, Q """
        features = tf.concat([obs_ph, act_ph], 1)
        qval = tf_utils.mlp(features, hidden_sizes=hidden_sizes + (1,),
                            activation=activation)
        return tf.reshape(qval, [-1])

    def build_graph(self, obs_space, act_space):
        """ build the tensorflow graph """
        tf_saver_kwargs = super().build_graph(obs_space, act_space)
        # we need to keep the act_space for adding noise to the actions
        # during training episodes
        self.act_space = act_space
        # initialize target parameters to equal main parameters
        self.sess.run(self.target_var_init())
        # define op for polyak updating of target parameters
        self.target_update_op = self.build_target_update_op()
        return tf_saver_kwargs

    def build_estimators(self, placeholders, obs_space, act_space):
        """ build the estimators for the policy and the action-value
        function """
        with tf.variable_scope(MAIN):
            main_pi, main_qval, main_qval_pi = self.build_policy_and_qval(
                placeholders['obs'], placeholders['act'], act_space)
        # use next_obs as input for target policy
        with tf.variable_scope(TARGET):
            target_pi_next, _, target_qval_pi_next = \
                    self.build_policy_and_qval(
                        placeholders['next_obs'], placeholders['act'],
                        act_space)
        return {'main_pi': main_pi, 'main_qval': main_qval,
                'main_qval_pi': main_qval_pi,
                'target_pi_next': target_pi_next,
                'target_qval_pi_next': target_qval_pi_next}

    def build_policy_and_qval(self, obs, act, act_space):
        """ build policy and q-value estimators """
        with tf.variable_scope('pi'):
            # policy
            pi = self.build_policy(act_space, obs,
                                   self.hidden_sizes, self.activation)
        with tf.variable_scope('qval'):
            # qval of action taken during episode
            qval = self.build_action_value_function(
                obs, act, self.hidden_sizes, self.activation)
        with tf.variable_scope('qval', reuse=True):
            # qval of current policy action
            qval_pi = self.build_action_value_function(
                obs, pi, self.hidden_sizes, self.activation)
        return pi, qval, qval_pi

    @staticmethod
    def target_var_init():
        """ returns tensorflow op to initialize target variables to be equal
        to the updated variables """
        op_list = [tf.assign(target_var, updated_var)
                   for target_var, updated_var in
                   zip(tf_utils.var_list(TARGET),
                       tf_utils.var_list(MAIN))]
        return tf.group(op_list)

    def build_target_update_op(self):
        """ returns tensorflow operation to update target parameters
        based on updated parameters and polyak """
        op_list = [tf.assign(target_var,
                             self.polyak*target_var
                             + (1 - self.polyak)*updated_var)
                   for (target_var, updated_var) in
                   zip(tf_utils.var_list(TARGET),
                       tf_utils.var_list(MAIN))]
        return tf.group(op_list)

    def build_losses(self, estimators, placeholders):
        """ build losses and train ops for action-value
        function and policy estimators """
        qval_loss, qval_train_op = self.build_qval_loss(
            estimators, placeholders)
        pi_loss, pi_train_op = self.build_policy_loss(estimators)
        losses = {'LossQ': qval_loss, 'LossPi': pi_loss}
        train_ops = {'LossQ': qval_train_op, 'LossPi': pi_train_op}
        return losses, train_ops

    def build_qval_loss(self, estimators, placeholders):
        """ build loss for action-value function """
        is_term_mask = 1 - placeholders['is_term']
        next_qval_pi = estimators['target_qval_pi_next']
        target = placeholders['rew'] \
            + self.gamma*is_term_mask*next_qval_pi
        loss = tf.losses.mean_squared_error(
            estimators['main_qval'], target)
        train_op = tf.train.AdamOptimizer(
            learning_rate=self.q_lr).minimize(
                loss, var_list=tf_utils.var_list(MAIN + '/qval'))
        return loss, train_op

    def build_policy_loss(self, estimators):
        """ build loss function and train op for deterministic policy """
        loss = -1*tf.reduce_mean(estimators['main_qval_pi'])
        train_op = tf.train.AdamOptimizer(
            learning_rate=self.pi_lr).minimize(
                loss, var_list=tf_utils.var_list(MAIN + '/pi'))
        return loss, train_op

    def train(self, replay_buffer):
        """ train after epoch. returns a dict of things we want to
        have logged including the policy and Qval losses """
        # train action-value function
        q_to_logs = self.update_params('LossQ', replay_buffer)
        # train policy
        pi_to_logs = self.update_params('LossPi', replay_buffer)
        # polyak update target parameters
        self.sess.run(self.target_update_op)
        return {**q_to_logs, **pi_to_logs}

    def update_params(self, loss_key, replay_buffer):
        """ trains a loss parameters after an epoch """
        for batch in replay_buffer.batches(self.n_batches):
            feed_dict = {self.placeholders[key]: batch[key]
                         for key in self.train_ph_keys}
            losses, _ = self.sess.run(
                (self.losses[loss_key], self.train_ops[loss_key]))
            loss_sum += np.sum(losses)
        return {loss_key: loss_sum/self.n_batches}
