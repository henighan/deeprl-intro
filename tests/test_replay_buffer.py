""" Tests for Replay Buffer """
# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from deeprl.replay_buffer import ReplayBuffer


@pytest.fixture
def buffer_size():
    """ buffer size """
    return 10


@pytest.fixture
def replay_buffer(buffer_size):
    """ replay buffer object """
    return ReplayBuffer(buffer_size)


def test_initialize_buf_smoke(buffer_size, replay_buffer):
    """ smoke test initialize_buf """
    to_buffer = {'obs': np.array([1, 2, 3]), 'rew': 1.2, 'logp': 3.4}
    ret = replay_buffer.initialize_buf(to_buffer)
    assert 'adv' in ret
    assert ret['adv'].shape == (buffer_size,)
    assert 'rew' in ret
    assert ret['rew'].shape == (buffer_size,)
    assert ret['obs'].shape == (buffer_size, 3)


def test_store_smoke(buffer_size, replay_buffer):
    """ smoke test store """
    to_buffer = {'obs': np.array([1, 2, 3]), 'rew': 1.2, 'val': 3.4}
    assert not replay_buffer.buf
    assert replay_buffer.ptr == 0
    replay_buffer.store(to_buffer)
    assert replay_buffer.buf['obs'].shape == (buffer_size, 3)
    assert replay_buffer.ptr == 1
    replay_buffer.store(to_buffer)
    assert replay_buffer.buf['obs'][1, 1] == 2


def test_finish_path_smoke(replay_buffer):
    """ smoke test finish path """
    to_buffer = {'obs': np.array([1, 2, 3]), 'rew': 1.2, 'val': 3.4}
    for _ in range(3):
        replay_buffer.store(to_buffer)
    replay_buffer.finish_path()
    assert replay_buffer.buf['adv'].shape == (replay_buffer.buffer_size,)
    assert replay_buffer.buf['ret'].shape == (replay_buffer.buffer_size,)


def test_dump(buffer_size, replay_buffer):
    """ smoke test dump """
    to_buffer = {'obs': np.array([1, 2, 3]), 'rew': 1.2, 'val': 3.4}
    for _ in range(buffer_size):
        replay_buffer.store(to_buffer)
    assert replay_buffer.ptr == buffer_size
    buf = replay_buffer.dump()
    rew = 1.2*np.ones(buffer_size, dtype=np.float32)
    np.testing.assert_allclose(buf['rew'], rew)
    assert replay_buffer.ptr == 0


def test_ptr_wraps_around(buffer_size, replay_buffer):
    """ For off-policy agents, we'd like to store more than one epoch's
    worth of experience. We may want to keep the 3 most recent epochs, for
    instance. To acheive this, we let the ptr go back to 0 if it reaches
    the end of the buffer """
    epoch0_store = {'obs': np.array([1, 2, 3]), 'rew': 1.2, 'val': 3.4}
    epoch1_store = {'obs': np.array([4, 5, 6]), 'rew': 1.5, 'val': 1.4}
    for _ in range(buffer_size):
        replay_buffer.store(epoch0_store)
    replay_buffer.finish_path()
    assert replay_buffer.ptr == buffer_size
    assert replay_buffer.buf['val'][0] == pytest.approx(3.4)
    replay_buffer.store(epoch1_store)
    assert replay_buffer.ptr == 1
    assert replay_buffer.buf['val'][0] == pytest.approx(1.4)


def test_raises_notimp_if_buffer_not_multiple_of_epoch_size():
    """ If the buffer size is not an integer mutiple of
    the epoch size, this should raise an error. """
    with pytest.raises(NotImplementedError):
        ret = ReplayBuffer(buffer_size=3, epoch_size=2)
