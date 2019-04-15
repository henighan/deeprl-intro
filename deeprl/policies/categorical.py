""" Categorical Policy Utils """
import tensorflow as tf

from deeprl.utils.tf_utils import mlp


def mlp_categorical_policy(obs, act, hidden_sizes, activation, action_space):
    """ Build Stochastic Categorical Policy """
    n_cat = action_space.n # number of categories
    mlp_hidden_sizes = hidden_sizes + (n_cat,)
    logits = mlp(obs, hidden_sizes=mlp_hidden_sizes, activation=activation)
    log_probs = logits_to_log_probs(logits)
    pi = sample_actions(logits)
    logp = log_prob_of_action(log_probs, act)
    logp_pi = log_prob_of_action(log_probs, pi)
    return pi, logp, logp_pi


def sample_actions(logits):
    """ Sample action from unnormalized log probabilities """
    return tf.reshape(tf.random.categorical(logits, num_samples=1), [-1])


def logits_to_log_probs(logits):
    """ apply softmax to unnormalized log probabilities to produce the log of
    the multinomial probability vector """
    return tf.nn.log_softmax(logits)


def log_prob_of_action(log_probs, actions):
    """ return the log probability of a given action """
    return tf.reshape(tf.batch_gather(
        log_probs, tf.reshape(actions, [-1, 1])), [-1])
