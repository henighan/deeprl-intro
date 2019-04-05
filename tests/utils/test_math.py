""" Tests for math utils """
import numpy as np

import pytest

from deeprl.utils import math


def test_discount_cumsum_2d():
    """ test discount_cumsum on matrix """
    seq = np.random.rand(3, 4)
    gamma = 0.9
    discount_cumsum = np.zeros([3, 4])
    discount_cumsum[2] = seq[2]
    discount_cumsum[1] = seq[1] + gamma*seq[2]
    discount_cumsum[0] = seq[0] + gamma*seq[1] + gamma**2*seq[2]
    ret = math.discount_cumsum(seq, gamma)
    np.testing.assert_almost_equal(discount_cumsum, ret)


def test_discount_cumsum_1d():
    """ test discount_cumsum on vector """
    seq = np.random.rand(3)
    gamma = 0.9
    discount_cumsum = np.zeros([3])
    discount_cumsum[2] = seq[2]
    discount_cumsum[1] = seq[1] + gamma*seq[2]
    discount_cumsum[0] = seq[0] + gamma*seq[1] + gamma**2*seq[2]
    ret = math.discount_cumsum(seq, gamma)
    np.testing.assert_almost_equal(discount_cumsum, ret)


def test_td_residual_smoke():
    """ smoke test of td_residual calculation """
    rewards = np.array([1, 3, 2, 8])
    values = np.array([2, 3, 5, 2])
    gamma = 0.9
    last_val = -1
    ret = math.td_residual(rewards, values, gamma=gamma, last_val=last_val)
    assert ret[0] == 1 + gamma*3 - 2
    assert ret[3] == 8 + gamma*last_val - 2


def test_advantage_function_smoke():
    """ smoke test advantage function """
    rewards = np.array([1, 3, 2, 8])
    values = np.array([2, 3, 5, 2])
    gamma = 0.9
    lam = 0.8
    last_val = -1
    # lets calculate by hand the long way for a few values of t
    deltas = math.td_residual(rewards, values, gamma=gamma,
                              last_val=last_val)
    ret = math.advantage_function(rewards, values, gamma=gamma,
                                  lam=lam, last_val=last_val)
    t = 3
    At1 = -values[t] + rewards[t] + gamma*last_val
    assert At1 == pytest.approx(deltas[t])
    At_gae = At1
    assert ret[t] == pytest.approx(At_gae)
    t = 2
    At1 = -values[t] + rewards[t] + gamma*values[t+1]
    assert At1 == pytest.approx(deltas[t])
    At2 = -values[t] + rewards[t] + gamma*rewards[t+1] + last_val*gamma**2
    assert At2 == pytest.approx(deltas[t] + gamma*deltas[t+1])
    At_gae = (1 - lam)*(At1 + At2*lam/(1 - lam))
    assert ret[t] == pytest.approx(At_gae)
    t = 1
    At1 = -values[t] + rewards[t] + gamma*values[t+1]
    At2 = (-values[t] + rewards[t] + gamma*rewards[t+1]
           + values[t+2]*gamma**2)
    At3 = (-values[t] + rewards[t] + rewards[t+1]*gamma
           + rewards[t+2]*gamma**2 + last_val*gamma**3)
    At_gae = (1 - lam)*(At1 + lam*At2 + lam**2*At3/(1-lam))
    assert ret[t] == pytest.approx(At_gae)


def test_rewards_to_go():
    """ smoke test rewards-to-go """
    rewards = np.array([2.3, 4.5, 8.9, 0.3])
    last_val = -1.2
    gamma = 0.9
    ret = math.rewards_to_go(rewards, gamma=gamma, last_val=last_val)
    # calcualte by hand the rewards-to-go for a particular t
    t = 2
    rew_2go = (rewards[t]
               + gamma*rewards[t+1]
               + last_val*gamma**2)
    assert rew_2go == ret[t]
