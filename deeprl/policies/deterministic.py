""" Deterministic Policy """
import numpy as np

from deeprl.utils import tf_utils


def mlp_deterministic_policy(obs, hidden_sizes, activation, action_space,
                             output_activation=None):
    """ Build a MLP deterministic policy (ie observation in, action out. No
    stochasticity, no sampling from a distribution """
    act_dim = np.prod(action_space.shape)
    mlp_hidden_sizes = hidden_sizes + (act_dim,)
    return tf_utils.mlp(obs, hidden_sizes=mlp_hidden_sizes,
                        activation=activation,
                        output_activation=output_activation)
