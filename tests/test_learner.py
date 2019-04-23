""" Tests for Learner """
#pylint: disable=redefined-outer-name
import os
import random
import time

import pytest
import numpy as np

from deeprl.learner import Learner


@pytest.fixture
def agent_step_ret():
    """ agent step return """
    return ({'act': np.array([0]), 'val': 0, 'logp': 0},
            {'VVals': 0, 'Logp': 0})

@pytest.fixture
def env_step_ret():
    """ environment step return value """
    return (np.array([0]), 0, False, None)


def test_play_episodes_reached_epoch_len(mocker, learner, agent_step_ret):
    """ test play_eposide when epoch end is reached """
    learner.logger = mocker.Mock()
    learner.max_ep_len = 2
    learner.agent.step.return_value = agent_step_ret
    ret_ep_len, _ = learner.play_episode()
    assert ret_ep_len == 2
    assert len(learner.logger.store.call_args_list) == 3


def test_play_episode_max_ep_len(mocker, learner, agent_step_ret):
    """ test play_episode when the max_ep_len is reached """
    learner.max_ep_len = 2
    learner.logger = mocker.Mock()
    learner.agent.step.return_value = agent_step_ret
    ret_ep_len, _ = learner.play_episode()
    assert ret_ep_len == 2
    learner.logger.store.assert_called_with(EpLen=2, EpRet=0.)
    assert len(learner.logger.store.call_args_list) == 3
    assert learner.buffer.ptr == 2


def test_play_episode_episode_ends(mocker, learner, agent_step_ret,
                                   env_step_ret):
    """ test play_episode when the episode terminates """
    learner.logger = mocker.Mock()
    # episode reaches termial state on fourth step
    learner.env.reset = mocker.Mock(return_value=env_step_ret[0])
    learner.env.step = mocker.Mock(
        side_effect=[env_step_ret,
                     env_step_ret,
                     env_step_ret,
                     (np.array([0]), 0, True, None)])
    learner.agent.step.return_value = agent_step_ret
    ret_ep_len, _ = learner.play_episode()
    print(learner.logger.store.call_args_list)
    assert ret_ep_len == 4
    learner.logger.store.assert_any_call(Logp=0, VVals=0)
    learner.logger.store.assert_called_with(EpRet=0, EpLen=4)
    assert len(learner.logger.store.call_args_list) == 5


def test_play_episode_check_buffer_called(mocker, learner, agent_step_ret):
    """ make sure the bufer is being called properly during play_episode
    when testing=False """
    learner.logger = mocker.Mock()
    learner.max_ep_len = 2
    learner.agent.step.return_value = agent_step_ret
    learner.buffer = mocker.Mock()
    learner.play_episode()
    assert len(learner.buffer.store.call_args_list) == 2


def test_play_episode_testing_true(mocker, learner, agent_step_ret):
    """ smoke test play_episode with testing=True """
    learner.logger = mocker.Mock()
    learner.max_ep_len = 2
    learner.agent.step.return_value = agent_step_ret
    learner.buffer = mocker.Mock()
    learner.play_episode(testing=True)
    # in testing mode, nothing should go to the buffer!
    learner.buffer.store.assert_not_called()
    learner.buffer.finish_path.assert_not_called()
    # no incrementing the train_step_ctr
    assert learner.train_step_ctr == 0
    # everything log should be prefixed with Test
    learner.logger.store.assert_called_once_with(TestEpRet=0., TestEpLen=2)


def test_log_epoch(mocker, learner):
    """ smoke test log_epoch """
    logtab_mock = mocker.patch('deeprl.learner.EpochLogger.log_tabular')
    dump_mock = mocker.patch('deeprl.learner.EpochLogger.dump_tabular')
    learner.agent.losses = {'LossPi': None, 'LossVal': None}
    learner.agent.log_tabular_kwargs = {'VVals': {'with_min_and_max': True}}
    learner.log_epoch(epoch=10, start_time=time.time())
    dump_mock.assert_called_once()
    call_args = logtab_mock.call_args_list
    assert mocker.call('Epoch', 10) in call_args
    assert mocker.call('LossPi', average_only=True) in call_args
    assert mocker.call('DeltaLossPi', average_only=True) in call_args
    assert mocker.call('VVals', with_min_and_max=True) in call_args


def test_learner_smoke(mocker, continuous_env, agent_step_ret):
    """ Give it a a test run witha mock agent, see that is produces logs """
    output_dir = 'tests/tmp_test_outputs'
    agent = mocker.Mock()
    agent.build_graph.return_value = {'saver': 'config'}
    mocker.patch('deeprl.learner.EpochLogger.setup_tf_saver')
    learner = Learner(agent, continuous_env, steps_per_epoch=1000,
                      epochs=4, output_dir=output_dir,
                      exp_name='learner_test')
    learner.logger.save_config = mocker.Mock()
    learner.logger.setup_tf_saver = mocker.Mock()
    learner.logger.save_state = mocker.Mock()
    learner.agent.step.return_value = agent_step_ret
    learner.agent.log_tabular_kwargs = {}
    learner.agent.train.return_value = {'LossPi': random.random(),
                                        'LossV': random.random(),
                                        'DeltaLossPi': -0.01*random.random(),
                                        'DeltaLossV': -0.01*random.random()}
    learner.agent.losses.keys.return_value = ['LossPi', 'LossV']
    output_path = os.path.join(output_dir, 'progress.txt')
    start_time = time.time()
    learner.learn()
    file_modified_time = os.stat(output_path).st_mtime
    # ensure it just craeted the log file
    assert file_modified_time > start_time
    # assert file has one line for each epoch (plus a header line)
    with open(output_path) as fileobj:
        assert len(fileobj.readlines()) == 5
