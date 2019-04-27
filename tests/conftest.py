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

@pytest.fixture
def learner(mocker, continuous_env):
    learner_path = 'deeprl.learners.policy_gradient_learner.'
    mocker.patch(learner_path + 'EpochLogger.save_config')
    mocker.patch(learner_path + 'EpochLogger.log')
    mocker.patch(learner_path + 'EpochLogger.setup_tf_saver')
    agent = mocker.Mock()
    agent.build_graph.return_value = {'foo': 'bar'}
    return PolicyGradientLearner(agent, env=continuous_env)
