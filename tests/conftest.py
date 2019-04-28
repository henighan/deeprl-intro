""" Pytest conftest """
import pytest
import gym
from deeprl.learners import PolicyGradientLearner

@pytest.fixture
def discrete_env():
    return gym.make('CartPole-v0') 

@pytest.fixture
def continuous_env():
    return gym.make('MountainCarContinuous-v0') 
