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


@pytest.fixture
def to_store():
    """ argument for replay_buffer.store """
    return {'obs': np.array([1, 2, 3]), 'rew': 1.2, 'val': 3.4}

def test_initialize_buf_smoke(buffer_size, replay_buffer):
    """ smoke test initialize_buf """
    to_buffer = {'obs': np.array([1, 2, 3]), 'rew': 1.2, 'logp': 3.4}
    ret = replay_buffer.initialize_buf(to_buffer)
    assert 'adv' in ret
    assert ret['adv'].shape == (buffer_size,)
    assert 'rew' in ret
    assert ret['rew'].shape == (buffer_size,)
    assert ret['obs'].shape == (buffer_size, 3)
    assert ret['next_obs'].shape == (buffer_size, 3)


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
    for _ in range(5):
        replay_buffer.store(to_buffer)
    replay_buffer.finish_path(last_val=0, last_obs=np.array([4, 5, 6]))
    assert replay_buffer.buf['adv'].shape == (replay_buffer.buffer_size,)
    assert replay_buffer.buf['ret'].shape == (replay_buffer.buffer_size,)
    # ensure next_obs was set properly
    np.testing.assert_equal(
        replay_buffer.buf['obs'][1], replay_buffer.buf['next_obs'][0])
    np.testing.assert_equal(
        replay_buffer.buf['next_obs'][4], np.array([4, 5, 6]))
    for _ in range(5):
        print(replay_buffer.ptr)
        replay_buffer.store(to_buffer)
    print(replay_buffer.ptr)
    replay_buffer.finish_path(last_val=0, last_obs=np.array([4, 5, 6]))
    np.testing.assert_equal(
        replay_buffer.buf['obs'][6], replay_buffer.buf['next_obs'][5])
    np.testing.assert_equal(
        replay_buffer.buf['next_obs'][9], np.array([4, 5, 6]))


def test_finish_path_1d_obs(replay_buffer):
    """ I got an error when using a 1-dimensional state space """
    to_buffer = {'obs': np.array([1]), 'rew': 1.2, 'val': 3.4}
    for _ in range(5):
        replay_buffer.store(to_buffer)
    replay_buffer.finish_path(last_val=0, last_obs=np.array([4]))
    assert replay_buffer.buf['adv'].shape == (replay_buffer.buffer_size,)
    assert replay_buffer.buf['ret'].shape == (replay_buffer.buffer_size,)
    # ensure next_obs was set properly
    np.testing.assert_equal(
        replay_buffer.buf['obs'][1], replay_buffer.buf['next_obs'][0])
    np.testing.assert_equal(
        replay_buffer.buf['next_obs'][4], np.array([4]))


def test_dump(buffer_size, replay_buffer):
    """ smoke test dump """
    to_buffer = {'obs': np.array([1, 2, 3]), 'rew': 1.2, 'val': 3.4}
    for _ in range(buffer_size):
        replay_buffer.store(to_buffer)
    assert replay_buffer.store_ctr == buffer_size
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
    replay_buffer.finish_path(last_val=0, last_obs=np.array([4, 5, 6]))
    assert replay_buffer.store_ctr == buffer_size
    assert replay_buffer.ptr == 0
    assert replay_buffer.buf['val'][0] == pytest.approx(3.4)
    replay_buffer.store(epoch1_store)
    assert replay_buffer.ptr == 1
    assert replay_buffer.buf['val'][0] == pytest.approx(1.4)


def test_raises_notimp_if_buffer_not_multiple_of_epoch_size():
    """ If the buffer size is not an integer mutiple of
    the epoch size, this should raise an error. """
    with pytest.raises(NotImplementedError):
        _ = ReplayBuffer(buffer_size=3, epoch_size=2)


def test_zeros(buffer_size, replay_buffer):
    """ test buffer zeros function for initializing empty arrays """
    val = np.array([0])
    ret = replay_buffer.zeros(val)
    assert ret.shape == (buffer_size, 1)
    val = np.array([0, 0])
    ret = replay_buffer.zeros(val)
    assert ret.shape == (buffer_size, 2)


def test_normalize_advantage(replay_buffer, to_store):
    """ test normalizing the advantage """
    n_stored = 4
    replay_buffer.store(to_store)
    replay_buffer.store_ctr = n_stored
    replay_buffer.buf['adv'] = 12. + 3*np.random.randn(n_stored)
    adv = replay_buffer.buf['adv'][:replay_buffer.n_stored]
    assert adv.mean() != pytest.approx(0.)
    assert adv.std() != pytest.approx(1.)
    replay_buffer.normalize_advantage()
    # after normalizing, mean should be 0, std 1
    adv = replay_buffer.buf['adv'][:replay_buffer.n_stored]
    assert adv.mean() == pytest.approx(0.)
    assert adv.std() == pytest.approx(1.)


def test_n_stored(buffer_size, replay_buffer):
    """ test n_stored property """
    replay_buffer.store_ctr = 3
    assert replay_buffer.n_stored == 3
    replay_buffer.store_ctr = buffer_size
    assert replay_buffer.n_stored == buffer_size
    replay_buffer.store_ctr = 2*buffer_size
    assert replay_buffer.n_stored == buffer_size


def test_sample_batch(buffer_size, replay_buffer, to_store):
    """ test sample_batch """
    batch_size = 2
    for _ in range(4):
        replay_buffer.store(to_store)
    replay_buffer.finish_path(last_val=0, last_obs=to_store['obs'])
    ret = replay_buffer.sample_batch(batch_size)
    assert len(ret['val']) == batch_size
    np.testing.assert_allclose(ret['val'], np.array(2*[to_store['val']]))
