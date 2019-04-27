""" Tests for Learner """
import os
import random
import time

import pytest
import numpy as np

from deeprl.learner import Learner


@pytest.fixture
def agent_step_ret():
    return ({'act': np.array([0]), 'val': 0, 'logp': 0},
            {'VVals': 0, 'Logp': 0})


def test_play_episode_buffer_full(mocker, learner, agent_step_ret):
    """ test play_episode when the buffer fills up """
    learner.logger = mocker.Mock()
    # mock buffer to be full after two steps
    full_mock = mocker.patch('deeprl.buffers.OnPolicyBuffer.full',
                             new_callable=mocker.PropertyMock)
    full_mock.side_effect = [False, False, False, False, True]
    learner.agent.step.return_value = agent_step_ret
    ret_ep_len, ret_ep_ret = learner.play_episode()
    assert ret_ep_len == 2
    learner.logger.store.assert_called_with(Logp=0, VVals=0)
    assert len(learner.logger.store.call_args_list) == 2
    assert learner.buffer.ptr == 2


def test_play_episode_max_ep_len(mocker, learner, agent_step_ret):
    """ test play_episode when the max_ep_len is reached """
    learner.max_ep_len = 2
    learner.logger = mocker.Mock()
    learner.agent.step.return_value = agent_step_ret
    ret_ep_len, ret_ep_ret = learner.play_episode()
    assert ret_ep_len == 2
    learner.logger.store.assert_called_with(EpLen=2, EpRet=0.)
    assert len(learner.logger.store.call_args_list) == 3
    assert learner.buffer.ptr == 2


def test_play_episode_episode_ends(mocker, learner, agent_step_ret):
    """ test play_episode when the episode terminates """
    learner.logger = mocker.Mock()
    # episode reaches termial state on fourth step
    learner.env.step = mocker.Mock(side_effect=[([0], 0, False, None),
                                                ([0], 0, False, None),
                                                ([0], 0, False, None),
                                                ([0], 2., True, None)])
    learner.agent.step.return_value = agent_step_ret
    ret_ep_len, ret_ep_ret = learner.play_episode()
    assert ret_ep_len == 4
    # note the return should include the reward from the last step!!!
    assert ret_ep_ret == 2.
    learner.logger.store.assert_any_call(Logp=0, VVals=0)
    learner.logger.store.assert_called_with(EpRet=2., EpLen=4)
    assert len(learner.logger.store.call_args_list) == 5


def test_learner_smoke(mocker, continuous_env, agent_step_ret):
    """ Give it a a test run witha mock agent, see that is produces logs """
    output_dir, exp_name = 'tests/tmp_test_outputs', 'learner_test'
    agent = mocker.Mock()
    agent.build_graph.return_value = {'saver': 'config'}
    mocker.patch('deeprl.learner.EpochLogger.setup_tf_saver')
    learner = Learner(agent, continuous_env, steps_per_epoch=1000,
                      epochs=4, output_dir='tests/tmp_test_outputs',
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
    output_path = os.path.join(output_dir, 'progress.txt')
    start_time = time.time()
    learner.learn()
    file_modified_time = os.stat(output_path).st_mtime
    # ensure it just craeted the log file
    assert file_modified_time > start_time
    # assert file has one line for each epoch (plus a header line)
    with open(output_path) as fileobj:
        assert len(fileobj.readlines()) == 5
