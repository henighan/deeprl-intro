""" Agent base class """
import tensorflow as tf

from gym.spaces import Box, Discrete

from deeprl.utils.tf_utils import tfph, adam_opt
from deeprl.policies.gaussian import mlp_gaussian_policy
from deeprl.policies.categorical import mlp_categorical_policy


class Base:
    """ Base Agent """

    # placeholder keys for training
    train_ph_keys = ['obs', 'act', 'adv', 'ret']
    # estimators we want to send to the buffer at each timestep
    to_buffer_each_step_dict = {'val': 'val', 'act': 'pi'}
    # things from the agent to log at each timestep and column name to use
    log_each_step = {'val': 'VVals'}
    # kwargs for logger.log_tabular
    log_tabular_kwargs = {'VVals': {'with_min_and_max': True}}
    # if you want to evaluate the policy after each epoch, set this to true
    eval_after_epoch = False

    def __init__(self, hidden_sizes=(64, 64), activation=tf.tanh,
                 sess=None, close_sess=True):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.sess = sess or tf.Session()
        self.close_sess = close_sess
        self.placeholders = {} # to be filled in by create_placeholders
        self.step_to_buffer = None # to be filled in by build_graph
        self.estimators = None # to be filled in by build_graph
        self.losses = None # to be filled in by build_graph
        self.train_ops = None # to be filled in by build_graph

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

    def train(self, replay_buffer):
        """ Train agent after epoch.
        This method should do 4 things:
            1. calculate losses before updating parameters
            2. update parameters
            3. calculate losses after updating parameters
            4. calculate the change in losses (delta losses)
        returns a dictionary of things to go to the logger """
        raise NotImplementedError('"train" method must be implemented')

    def create_placeholders(self, obs_space, act_space):
        """ Build basic placeholders """
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

    def update_parameters(self, feed_dict):
        """ This function should perform the parameter updates for a single
        epoch and return a dictionary of values we wish to be logged """
        raise NotImplementedError("must implement 'update_parameters'")

    def build_graph(self, obs_space, act_space):
        """ Build the tensorflow graph """
        self.create_placeholders(obs_space, act_space)
        self.estimators = self.build_estimators(
            self.placeholders, obs_space, act_space)
        # things we want to store in the buffer on each step
        self.step_to_buffer = {buff_key: self.estimators[est_key]
                               for buff_key, est_key
                               in self.to_buffer_each_step_dict.items()}
        # build losses and train ops
        self.losses, self.train_ops = self.build_losses(
            self.estimators, self.placeholders)
        self.sess.run(tf.global_variables_initializer())
        # returns kwargs for tf_saver
        return self.tf_saver_kwargs(self.placeholders, self.estimators)

    def build_estimators(placeholders, obs_space, act_space):
        """ build the estimators and return in dictionary. Examples include
        estimators of the value function, action-value function, and
        the optimal policy """
        raise NotImplementedError(
            '"build_estimators" method must be implemented')

    def build_losses(estimators, placeholders):
        """ build the losses and their training ops and return as dicts """
        raise NotImplementedError(
            '"build_losses" method must be implemented')

    def tf_saver_kwargs(self, placeholders, estimators):
        """ the kwargs for the logger tf_saver method """
        outputs = {'pi': estimators['pi'], 'v': estimators['val']}
        return {'sess': self.sess,
                'inputs': {'x': self.placeholders['obs']},
                'outputs': outputs}

    @staticmethod
    def delta_losses(initial_losses, new_losses):
        """ given a dicts of new and old losses, return a dict of
        the change in loss (delta losses) """
        return {'Delta' + key: new_val - initial_losses[key]
                for key, new_val in new_losses.items()}

    @staticmethod
    def build_policy_gradient_loss(logp, placeholders, learning_rate):
        """ build the graph for the policy loss """
        pi_loss = tf.reduce_mean(-logp*placeholders['adv'])
        return pi_loss, adam_opt(pi_loss, learning_rate)

    @staticmethod
    def build_mse_loss(estimates, targets, learning_rate):
        """ build the graph for the value function loss """
        loss = tf.losses.mean_squared_error(estimates, targets)
        return loss, adam_opt(loss, learning_rate)

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

    def __del__(self):
        """ close session when garbage collected """
        if self.sess and self.close_sess:
            self.sess.close()
