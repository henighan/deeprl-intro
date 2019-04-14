""" Functions for Diagonal Gaussian Policy """
import tensorflow as tf
import numpy as np

from deeprl.utils.tf_utils import mlp


def mlp_gaussian_policy(x, a, hidden_sizes, activation, action_space,
                        output_activation=None):
    """ Builds symbols to sample actions and compute log-probs of actions.  """
    act_dim = np.prod(action_space.shape)
    mlp_hidden_sizes = hidden_sizes + (act_dim,)
    mu = mlp(x, hidden_sizes=mlp_hidden_sizes, activation=activation,
             output_activation=output_activation)
    log_std = make_log_std_var(act_dim)
    pi = sample_actions(mu, log_std)
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


def make_log_std_var(dim):
    """ Initialize diagonal entries of covariance matrix """
    init = tf.constant(-0.5, shape=[1, dim], dtype=tf.float32)
    return tf.get_variable('log_std', initializer=init, trainable=True)


def sample_actions(mu, log_std):
    """ Return a sample from a multivariate normal distribution parameterized
    by mean mu and log_std, the logs of the diagonals of the covariance matrix
    """
    std = tf.exp(log_std)
    noise = std*tf.random.normal(shape=tf.shape(mu))
    return mu + noise


def gaussian_likelihood(x, mu, log_std):
    """
    Args:
        x: Tensor with shape [batch, dim]
        mu: Tensor with shape [batch, dim]
        log_std: Tensor with shape [batch, dim] or [dim]
    Returns:
        Tensor with shape [batch]
    """
    dim = tf.cast(tf.shape(x)[1], tf.float32)
    summation_terms = 2*log_std + ((x - mu)**2)/tf.exp(2*log_std)
    return -0.5*(tf.reduce_sum(summation_terms, axis=1)
                 + dim*np.log(2*np.pi))
