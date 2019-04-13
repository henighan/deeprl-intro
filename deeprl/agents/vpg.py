""" Vanilla Policy Gradient Agent """
import tensorflow as tf
from gym.spaces import Box, Discrete
from deeprl.policies.gaussian import mlp_gaussian_policy
from deeprl.policies.categorical import mlp_categorical_policy
from deeprl.tf_utils import mlp, tfph


class VPG():
    """ Vanilla Policy Gradient Agent """

    # placeholder keys for training
    train_ph_keys = ['obs', 'act', 'adv', 'ret']
    # things to log at each timestep, and what column name to use
    log_each_step = {'val': 'VVals', 'logp': 'LogP'}
    # kwargs for logger.log_tabular
    log_tabular_kwargs = {'VVals': {'with_min_and_max': True},
                          'LogP': {'with_min_and_max': True},
                          'LossPi': {'average_only': True},
                          'LossV': {'average_only': True},
                          'DeltaLossPi': {'average_only': True},
                          'DeltaLossV': {'average_only': True}}

    def __init__(self, pi_lr=3e-4, val_lr=1e-3, hidden_sizes=(64, 64),
                 activation=tf.tanh, val_train_iters=80, sess=None,
                 close_sess=True):
        self.pi_lr, self.val_lr = pi_lr, val_lr
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.val_train_iters = val_train_iters
        self.sess = sess or tf.Session()
        self.close_sess = close_sess
        self.placeholders = {}
        self.step_to_buffer = None
        self.pi_loss, self.pi_train_op = None, None
        self.val_loss, self.val_train_op = None, None

    def step(self, obs):
        """ sample action given observation of environment. returns two
        dictionaries. The first conains values we want stored in the buffer.
        The second is values we want sent to the log """
        feed_dict = {self.placeholders['obs']: obs.reshape(1, -1)}
        to_buffer = self.sess.run(
            self.step_to_buffer, feed_dict=feed_dict)
        # sqeeze since this is a single timestep, not a batch
        to_buffer = {key: val[0] for key, val in to_buffer.items()}
        to_log = {log_column: to_buffer[key]
                  for key, log_column in self.log_each_step.items()}
        return to_buffer, to_log

    def train(self, buf):
        """ Train after epoch. Returns dict of things we want to have logged,
        including the Policy and Value losses, and the change in the losses
        due to training """
        feed_dict = {self.placeholders[key]: buf[key]
                     for key in self.train_ph_keys}
        # calculate losses before training
        old_pi_loss, old_val_loss = self.sess.run(
            (self.pi_loss, self.val_loss), feed_dict=feed_dict)
        # update policy
        policy_to_logs = self.update_policy(feed_dict)
        # update value function
        value_to_logs = self.update_value_function(feed_dict)
        # calculate change in loss
        new_pi_loss, new_val_loss = self.sess.run(
            (self.pi_loss, self.val_loss), feed_dict=feed_dict)
        delta_pi_loss = new_pi_loss - old_pi_loss
        delta_val_loss = new_val_loss - old_val_loss
        return {**policy_to_logs,
                **value_to_logs,
                'LossPi': old_pi_loss,
                'LossV': old_val_loss,
                'DeltaLossPi': delta_pi_loss,
                'DeltaLossV': delta_val_loss}

    def update_policy(self, feed_dict):
        """ update the policy based on the replay-buffer data (stored in
        feed_dict) """
        self.sess.run(self.pi_train_op, feed_dict=feed_dict)
        return {}

    def update_value_function(self, feed_dict):
        """ update the policy based on the replay-buffer data (stored in
        feed_dict) """
        for _ in range(self.val_train_iters):
            self.sess.run(self.val_train_op, feed_dict=feed_dict)
        return {}

    def build_graph(self, obs_space, act_space):
        """ Build the tensorflow graph """
        self.create_placeholders(obs_space, act_space)
        """ sampled action, logprob of input action, and logprob of
        sampled action """
        pi, logp, logp_pi = self.build_policy(
            act_space, self.placeholders['obs'], self.placeholders['act'],
            self.hidden_sizes, self.activation)
        # policy loss and train op
        self.pi_loss, self.pi_train_op = self.build_policy_loss(
            logp, self.placeholders, self.pi_lr)
        # value function estimator
        val = self.build_value_function(self.placeholders['obs'],
                                        hidden_sizes=self.hidden_sizes,
                                        activation=self.activation)
        self.step_to_buffer = {'act': pi,
                               'val': val,
                               'logp': logp_pi}
        # value function estimator loss and train op
        self.val_loss, self.val_train_op = self.build_val_loss(
            val, self.placeholders['ret'], learning_rate=self.val_lr)
        self.sess.run(tf.global_variables_initializer())
        # returns kwargs for tf_saver
        return {'sess': self.sess,
                'inputs': {'x': self.placeholders['obs']},
                'outputs': {'pi': pi, 'v': val}}

    def create_placeholders(self, obs_space, act_space):
        """ Build the placeholders required for this agent """
        self.placeholders['obs'] = tfph(obs_space.shape[-1], name='obs')
        if isinstance(act_space, Box):
            self.placeholders['act'] = tfph(act_space.shape[-1], name='act')
        elif isinstance(act_space, Discrete):
            self.placeholders['act'] = tf.placeholder(
                dtype=tf.int64, shape=[None], name='act')
        else:
            raise NotImplementedError(
                'action space {} not implemented'.format(act_space))
        for name in ('ret', 'adv'):
            self.placeholders[name] = tfph(None, name=name)

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
        pi_train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(pi_loss)
        return pi_loss, pi_train_op

    @staticmethod
    def build_val_loss(val, ret_ph, learning_rate):
        """ build the graph for the value function loss """
        val_loss = tf.losses.mean_squared_error(val, ret_ph)
        val_train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(val_loss)
        return val_loss, val_train_op

    def __del__(self):
        """ close session when garbage collected """
        if self.sess and self.close_sess:
            self.sess.close()
