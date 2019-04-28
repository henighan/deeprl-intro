""" Tests for DeterministicLearner """
import os
import random
import time

import pytest
import numpy as np

from deeprl.learners import DeterministicLearner


@pytest.fixture
def agent_step_ret():
    """ agent 'step' return value fixture """
    return ({'act': np.array([0]), 'qval': 0}, {'QVals': 0})


@pytest.fixture
def env_step_ret():
    """ environment step return value """
    return (np.array([0]), 0, False, None)


@pytest.fixture
def learner(mocker, continuous_env, agent_step_ret):
    """ learner fixture """
    learner_path = 'deeprl.learners.deterministic_learner.'
    mocker.patch(learner_path + 'EpochLogger.save_config')
    mocker.patch(learner_path + 'EpochLogger.log')
    mocker.patch(learner_path + 'EpochLogger.setup_tf_saver')
    agent = mocker.Mock()
    agent.step.return_value = agent_step_ret
    agent.build_graph.return_value = {'foo': 'bar'}
    return DeterministicLearner(agent, env=continuous_env)


def test_episode_step_testing_false(mocker, learner, agent_step_ret):
    """ smoke test episode step """
    mocker.patch.object(learner, 'logger')
    obs = learner.env.reset()
    ret = learner.episode_step(obs, 0, False, 0, 0, 0, testing=False)
    ret_obs, ret_rew, ret_is_term, ret_ep_len, ret_ep_ret, ret_epoch_ctr = ret
    assert ret_ep_len == 1
    learner.logger.store.assert_called_once()
    assert ret_epoch_ctr == 1
    assert learner.buffer.buf # ensure buffer was initialized


def test_episode_step_testing_True(mocker, learner, agent_step_ret):
    """ smoke test episode step """
    mocker.patch.object(learner, 'logger')
    mocker.patch.object(learner, 'buffer')
    obs = learner.env.reset()
    ret = learner.episode_step(obs, 0, False, 0, 0, 0, testing=True)
    ret_obs, ret_rew, ret_is_term, ret_ep_len, ret_ep_ret, ret_epoch_ctr = ret
    assert ret_ep_len == 1
    learner.logger.store.assert_not_called() # no logging during testing
    assert ret_epoch_ctr == 0
    # no buffer storage during testing
    learner.buffer.store.assert_not_called()


def test_play_episodes_reached_epoch_len(mocker, learner):
    """ test play_eposide when epoch end is reached """
    learner.logger = mocker.Mock()
    learner.epoch_len = 2
    ret_ep_len, _, _ = learner.play_episode()
    assert ret_ep_len == 2
    assert len(learner.logger.store.call_args_list) == 2


def test_play_episode_max_ep_len(mocker, learner):
    """ test play_episode when the max_ep_len is reached """
    learner.max_ep_len = 2
    learner.logger = mocker.Mock()
    ret_ep_len, _, ret_epoch_ctr = learner.play_episode()
    assert ret_ep_len == 2
    assert ret_epoch_ctr == 2
    learner.logger.store.assert_called_with(EpLen=2, EpRet=0.)
    assert len(learner.logger.store.call_args_list) == 3
    assert learner.buffer.ptr == 2


def test_play_episode_episode_ends(mocker, learner, env_step_ret):
    """ test play_episode when the episode terminates """
    learner.logger = mocker.Mock()
    # episode reaches termial state on fourth step
    learner.env.reset = mocker.Mock(return_value=env_step_ret[0])
    learner.env.step = mocker.Mock(
        side_effect=[env_step_ret,
                     env_step_ret,
                     env_step_ret,
                     (np.array([0]), 2., True, None)]) # last one is terminal!
    ret_ep_len, ret_ep_ret, epoch_ctr = learner.play_episode()
    assert ret_ep_len == 4
    # note the return should include the reward from the last step!!!
    assert ret_ep_ret == 2.
    learner.logger.store.assert_called_with(EpRet=2., EpLen=4)
    assert len(learner.logger.store.call_args_list) == 5


def test_test_epoch_smoke(learner):
    """ smoke test test_epoch method """
    learner.test_epoch(1)
