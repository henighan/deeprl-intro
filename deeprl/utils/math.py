""" Math utils """
import scipy
import numpy as np
from spinup.algos.vpg import core


def discount_cumsum(sequence, discount):
    """ Compute discounted cumulative sum of vectors """
    #TODO understand this crazy function
    return scipy.signal.lfilter([1],
                                [1, float(-discount)],
                                sequence[::-1], axis=0)[::-1]


def td_residual(rewards, values, gamma=0.99, last_val=0):
    """ Calculate the TD residual of the value function with discount gamma
    """
    rews = np.append(rewards, last_val)
    vals = np.append(values, last_val)
    return rews[:-1] + gamma*vals[1:] - vals[:-1]


def advantage_function(rewards, values, gamma=0.99, lam=0.95, last_val=0):
    """ Given the trajectory rewards and values, calculate the GAE-Lambda
    advantage estimate at each timestep """
    deltas = td_residual(rewards, values, gamma=gamma, last_val=last_val)
    return discount_cumsum(deltas, gamma*lam)


def combined_shape(length, value):
    """ given a scalar length and a value, which may be a scalar or a numpy
    array, and return their combined shape """
    if np.isscalar(value):
        return (length,)
    return (length, *value.shape)
