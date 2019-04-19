""" Pytest conftest """
import pytest
import gym
from deeprl.learner import Learner

@pytest.fixture
def discrete_env():
    return gym.make('CartPole-v0') 

@pytest.fixture
def continuous_env():
    return gym.make('MountainCarContinuous-v0') 

@pytest.fixture
def learner(mocker, continuous_env):
    mocker.patch('deeprl.learner.EpochLogger.save_config')
    mocker.patch('deeprl.learner.EpochLogger.log')
    mocker.patch('deeprl.learner.EpochLogger.setup_tf_saver')
    agent = mocker.Mock()
    agent.eval_after_epoch = False
    agent.build_graph.return_value = {'foo': 'bar'}
    return Learner(agent, env=continuous_env)
