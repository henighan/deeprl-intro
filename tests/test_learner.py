""" Tests for Learner """
import os
import random
import time

from deeprl.learner import Learner


def test_learner_init_smoke(mocker, continuous_env):
    """ smoke test learner init """
    learner = Learner(mocker.Mock(), env=continuous_env)


def test_play_episode_buffer_full(mocker, learner):
    """ test play_episode when the buffer fills up """
    learner.logger = mocker.Mock()
    # mock buffer to be full after two steps
    full_mock = mocker.patch('deeprl.replay_buffer.Buffer.full',
                             new_callable=mocker.PropertyMock)
    full_mock.side_effect = [False, False, True]
    learner.agent.take_step.return_value = ([0], 0, 0)
    ret_ep_len, ret_ep_ret = learner.play_episode()
    assert ret_ep_len == 2
    learner.logger.store.assert_called_with(VVals=0)
    assert len(learner.logger.store.call_args_list) == 2
    assert learner.buffer.ptr == 2


def test_play_episode_episode_ends(mocker, learner):
    """ test play_episode when the episode terminates """
    learner.logger = mocker.Mock()
    # episode reaches termial state on fourth step
    learner.env.step = mocker.Mock(side_effect=[([0], 0, False, None),
                                                ([0], 0, False, None),
                                                ([0], 0, False, None),
                                                ([0], 0, True, None)])
    learner.agent.take_step.return_value = ([0], 0, 0)
    ret_ep_len, ret_ep_ret = learner.play_episode()
    assert ret_ep_len == 4
    learner.logger.store.assert_any_call(VVals=0)
    learner.logger.store.assert_called_with(EpRet=0, EpLen=4)
    assert len(learner.logger.store.call_args_list) == 5


def test_learner_smoke(mocker, continuous_env):
    """ Give it a a test run witha mock agent, see that is produces logs """
    output_dir, exp_name = 'tests/tmp_test_outputs', 'learner_test'
    learner = Learner(mocker.Mock(), continuous_env, steps_per_epoch=1000,
                      epochs=4, output_dir='tests/tmp_test_outputs',
                      exp_name='learner_test')
    learner.agent.take_step.return_value = ([0], random.random(),
                                            random.random())
    learner.agent.train.return_value = (random.random(),
                                        random.random(),
                                        -0.01*random.random(),
                                        -0.01*random.random(),)
    output_path = os.path.join(output_dir, 'progress.txt')
    start_time = time.time()
    learner.learn()
    file_modified_time = os.stat(output_path).st_mtime
    # ensure it just craeted the log file
    assert file_modified_time > start_time
    # assert file has one line for each epoch (plus a header line)
    with open(output_path) as fileobj:
        assert len(fileobj.readlines()) == 5
