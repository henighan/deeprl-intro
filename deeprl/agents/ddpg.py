""" Deep Deterministic Policy Gradient Agent """
import tensorflow as tf

from gym.spaces import Box

from deeprl.utils import tf_utils
from deeprl.agents import VPG
from deeprl.policies.deterministic import mlp_deterministic_policy

TARGET = 'target'
MAIN = 'main'


class DDPG(VPG):
    """ Deep deterministic policy gradient. The first (start_steps) steps are
    sampled randomly and uniformly from the action space to force exploration
    early on. After this, its normal DDPG """

    step_ctr = 0
    train_ph_keys = ['obs', 'next_obs', 'act', 'is_term', 'rew']
    log_each_step = {'qval': 'QVals'}
    log_tabular_kwargs = {'QVals': {'with_min_and_max': True}}
    to_buffer_each_step_dict = {'qval': 'qval', 'act': 'pi'}
    polyak = 0.95

    @staticmethod
    def build_policy(act_space, obs_ph, hidden_sizes, activation):
        if isinstance(act_space, Box):
            with tf.variable_scope('pi'):
                pi = mlp_deterministic_policy(obs_ph, hidden_sizes,
                                              activation, act_space)
            return pi
        raise NotImplementedError(
            "DDPG can only be used with continuous action spaces")

    @staticmethod
    def build_action_value_function(obs_ph, act_ph, hidden_sizes, activation):
        """ build the action-value function estimator, Q """
        with tf.variable_scope('q_val'):
            features = tf.concat([obs_ph, act_ph], 1)
            qval = tf_utils.mlp(features, hidden_sizes=hidden_sizes + (1,),
                                activation=activation)
        return tf.reshape(qval, [-1])

    def build_estimators(self, placeholders, obs_space, act_space):
        """ build the estimators for the policy and the action-value
        function """
        with tf.variable_scope(MAIN):
            main_pi = self.build_policy(act_space, placeholders['obs'],
                                        self.hidden_sizes, self.activation)
            main_qval = self.build_action_value_function(
                placeholders['obs'], placeholders['act'], self.hidden_sizes,
                self.activation)
        with tf.variable_scope(TARGET):
            target_pi = self.build_policy(act_space, placeholders['obs'],
                                          self.hidden_sizes, self.activation)
            target_qval = self.build_action_value_function(
                placeholders['obs'], placeholders['act'], self.hidden_sizes,
                self.activation)

    @staticmethod
    def target_var_init():
        """ returns tensorflow op to initialize target variables to be equal
        to the updated variables """
        op_list = [tf.assign(target_var, updated_var)
                   for target_var, updated_var in
                   zip(tf_utils.var_list(TARGET),
                       tf_utils.var_list(MAIN))]
        return tf.group(op_list)

    def target_update_op(self):
        """ returns tensorflow operation to update target parameters
        based on updated parameters and polyak """
        op_list = [tf.assign(target_var,
                             self.polyak*target_var
                             + (1 - self.polyak)*updated_var)
                   for (target_var, updated_var) in
                   zip(tf_utils.var_list(TARGET),
                       tf_utils.var_list(MAIN))]
        return tf.group(op_list)

    # def build_qval_loss(self, estimators, placeholders):
    #     is_term_mask = 1 - placeholders['is_term']
    #     target = placeholders['rew'] \
    #         + self.gamma*is_term_mask*estimators['target_qval']
