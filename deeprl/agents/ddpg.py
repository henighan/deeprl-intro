""" Deep Deterministic Policy Gradient Agent """
import tensorflow as tf
import numpy as np
from gym.spaces import Box

from deeprl.policies.deterministic import mlp_deterministic_policy
from deeprl.utils import tf_utils

TARGET = 'target'
MAIN = 'main'
POLICY = 'pi'
QVAL = 'qval'


class DDPG():
    """ Deep Deterministic Policy Gradient Agent """

    # placeholder keys for training
    train_ph_keys = ['obs', 'next_obs', 'act', 'is_term', 'rew']
    # kwargs for logger.log_tabular
    log_tabular_kwargs = {'QVals': {'with_min_and_max': True},
                          'LossPi': {'average_only': True},
                          'LossQ': {'average_only': True}}

    def __init__(self, hidden_sizes=(64, 64), activation=tf.nn.relu,
                 output_activation=tf.tanh, gamma=0.99, polyak=0.995,
                 pi_lr=0.001, q_lr=0.001, batch_size=100,
                 start_steps=10000, act_noise=0.1):
        self.pi_lr, self.q_lr = pi_lr, q_lr
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.start_steps, self.act_noise = start_steps, act_noise
        self.batch_size, self.polyak, self.gamma = batch_size, polyak, gamma
        self.sess = tf.Session()
        self.placeholders = {}
        self.pi_loss, self.pi_train_op = None, None
        self.qval_loss, self.qval_train_op = None, None
        self.pi, self.qval, self.qval_pi = None, None, None
        self.qval_pi_targ = None
        self.act_space = None
        self.target_update_op = None
        self.n_training_steps = 0

    def build_graph(self, obs_space, act_space):
        """ Build the tensorflow graph """
        self.act_space = act_space
        self.create_placeholders(obs_space, act_space)
        # build estimators of policy and Q
        est_ret = self.build_estimators(self.placeholders, act_space)
        self.pi, self.qval, self.qval_pi, self.qval_pi_targ = est_ret
        # build Q loss and train op
        qval_targ = self.build_qval_target(
            self.qval_pi_targ, self.placeholders)
        self.qval_loss, self.qval_train_op = self.build_qval_loss(
            self.qval, qval_targ)
        # build policy loss and train op
        self.pi_loss, self.pi_train_op = self.build_policy_loss(self.qval_pi)
        # build polyak target updating op
        self.target_update_op = self.build_target_update_op()
        self.sess.run(tf.global_variables_initializer())
        # set target network parameters equal to main network parameters
        self.target_var_init()
        # returns kwargs for tf_saver
        return {'sess': self.sess,
                'inputs': {'x': self.placeholders['obs'],
                           'a': self.placeholders['act']},
                'outputs': {'pi': self.pi, 'q': self.qval}}

    def train(self, train_iter, batch):
        """ run one iteration of training """
        feed_dict = {self.placeholders[name]: batch[name]
                     for name in self.train_ph_keys}
        # update qval network
        qval_loss, qval, _ = self.sess.run(
            (self.qval_loss, self.qval, self.qval_train_op), feed_dict)
        # update policy network
        pi_loss, _ = self.sess.run((self.pi_loss, self.pi_train_op),
                                   feed_dict=feed_dict)
        # update target networks
        self.sess.run(self.target_update_op)
        return {'LossQ': qval_loss, 'LossPi': pi_loss, 'QVals': qval}

    def step(self, obs, testing=False):
        """ sample action given observation of environment. returns two
        dictionaries. The first conains values we want stored in the buffer.
        The second is values we want sent to the log """
        feed_dict = {self.placeholders['obs']: obs.reshape(1, -1)}
        if testing:
            act = self.sess.run(self.pi, feed_dict=feed_dict)[0]
        elif self.n_training_steps < self.start_steps:
            self.n_training_steps += 1
            act = self.act_space.sample()
        else: # not testing and after initial exploration
            self.n_training_steps += 1
            act = self.add_noise_and_clip(
                self.sess.run(self.pi, feed_dict=feed_dict)[0],
                self.act_noise, self.act_space)
        # sqeeze since this is a single timestep, not a batch
        return act

    @staticmethod
    def add_noise_and_clip(act, act_noise, act_space):
        """ add noise to action, then clip to ensure it's within
        the action space """
        noise = act_noise*np.random.randn(*act.shape)
        return np.clip(act + noise, act_space.low, act_space.high)

    def create_placeholders(self, obs_space, act_space):
        """ create placeholders """
        if not isinstance(act_space, Box):
            raise NotImplementedError(
                "action space {} not implemented".format(type(act_space))
                + "NOTE DDPG only compatible with continuous actions spaces")
        act_dim, obs_dim = act_space.shape[-1], obs_space.shape[-1]
        ph_shapes = {'act': act_dim, 'obs': obs_dim, 'next_obs': obs_dim,
                     'is_term': None, 'rew': None}
        self.placeholders = {name: tf_utils.tfph(shape, name=name)
                             for name, shape in ph_shapes.items()}

    @staticmethod
    def build_policy(act_space, obs_ph, hidden_sizes, activation,
                     output_activation=None):
        """ build the deterministic policy """
        if isinstance(act_space, Box):
            pi = act_space.high[0]*mlp_deterministic_policy(
                obs_ph, hidden_sizes, activation, act_space,
                output_activation=output_activation)
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

    def build_policy_and_qval(self, obs, act, act_space):
        """ build policy and q-value estimators """
        with tf.variable_scope(POLICY):
            # policy
            pi = self.build_policy(act_space, obs,
                                   self.hidden_sizes, self.activation,
                                   output_activation=self.output_activation)
        with tf.variable_scope(QVAL):
            # qval of action taken during episode
            qval = self.build_action_value_function(
                obs, act, self.hidden_sizes, self.activation)
        with tf.variable_scope(QVAL, reuse=True):
            # qval of current policy action
            qval_pi = self.build_action_value_function(
                obs, pi, self.hidden_sizes, self.activation)
        return pi, qval, qval_pi

    def build_estimators(self, placeholders, act_space):
        """ build the estimators for the policy and the action-value
        function """
        with tf.variable_scope(MAIN):
            pi, qval, qval_pi = self.build_policy_and_qval(
                placeholders['obs'], placeholders['act'], act_space)
        # use next_obs as input for target policy
        with tf.variable_scope(TARGET):
            _, _, qval_pi_targ = \
                    self.build_policy_and_qval(
                        placeholders['next_obs'], placeholders['act'],
                        act_space)
        return pi, qval, qval_pi, qval_pi_targ


    def target_var_init(self):
        """ returns tensorflow op to initialize target variables to be equal
        to the updated variables """
        op_list = [tf.assign(target_var, updated_var)
                   for target_var, updated_var in
                   zip(tf_utils.var_list(TARGET),
                       tf_utils.var_list(MAIN))]
        self.sess.run(tf.group(op_list))

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

    def build_qval_target(self, qval_pi_targ, placeholders):
        """ build the targets for qval training """
        is_term_mask = 1 - placeholders['is_term']
        target = tf.stop_gradient(placeholders['rew'] \
            + self.gamma*is_term_mask*qval_pi_targ)
        return target

    def build_qval_loss(self, qval, qval_target):
        """ build loss for action-value function """
        loss = tf.losses.mean_squared_error(qval, qval_target)
        train_op = tf.train.AdamOptimizer(
            learning_rate=self.q_lr).minimize(
                loss, var_list=tf_utils.var_list(MAIN + '/' + QVAL))
        return loss, train_op

    def build_policy_loss(self, qval_pi):
        """ build loss function and train op for deterministic policy """
        loss = -1*tf.reduce_mean(qval_pi)
        train_op = tf.train.AdamOptimizer(
            learning_rate=self.pi_lr).minimize(
                loss, var_list=tf_utils.var_list(MAIN + '/' + POLICY))
        return loss, train_op
