""" Tests for Replay Buffer """
# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from deeprl.replay_buffer import ReplayBuffer

@pytest.fixture
def size():
    """ buffer size """
    return 10

@pytest.fixture
def replay_buffer(size):
    """ replay buffer object """
    return ReplayBuffer(size)


def test_initialize_buf_smoke(size, replay_buffer):
    """ smoke test initialize_buf """
    to_buffer = {'obs': np.array([1, 2, 3]), 'rew': 1.2, 'logp': 3.4}
    ret = replay_buffer.initialize_buf(to_buffer)
    assert 'adv' in ret
    assert ret['adv'].shape == (size,)
    assert 'rew' in ret
    assert ret['rew'].shape == (size,)
    assert ret['obs'].shape == (size, 3)


def test_store_smoke(size, replay_buffer):
    """ smoke test store """
    to_buffer = {'obs': np.array([1, 2, 3]), 'rew': 1.2, 'val': 3.4}
    assert not replay_buffer.buf
    assert replay_buffer.ptr == 0
    replay_buffer.store(to_buffer)
    assert replay_buffer.buf['obs'].shape == (size, 3)
    assert replay_buffer.ptr == 1
    assert not replay_buffer.full
    replay_buffer.store(to_buffer)
    assert replay_buffer.buf['obs'][1, 1] == 2


def test_finish_path_smoke(replay_buffer):
    """ smoke test finish path """
    to_buffer = {'obs': np.array([1, 2, 3]), 'rew': 1.2, 'val': 3.4}
    for _ in range(3):
        replay_buffer.store(to_buffer)
    replay_buffer.finish_path()
    assert replay_buffer.buf['adv'].shape == (replay_buffer.max_size,)
    assert replay_buffer.buf['ret'].shape == (replay_buffer.max_size,)


def test_get(size, replay_buffer):
    """ smoke test get """
    to_buffer = {'obs': np.array([1, 2, 3]), 'rew': 1.2, 'val': 3.4}
    for _ in range(size):
        assert not replay_buffer.full
        replay_buffer.store(to_buffer)
    assert replay_buffer.full
    buf = replay_buffer.get()
    assert not replay_buffer.full
    rew = 1.2*np.ones(size, dtype=np.float32)
    np.testing.assert_allclose(buf['rew'], rew)
    assert replay_buffer.ptr == 0
