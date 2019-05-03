""" Replay Buffer """
import numpy as np
from deeprl.utils.math import combined_shape


class OffPolicyBuffer:
    """ Stores episode experience (rewards, observations,
    actions, etc), calculates the advantage and rewards-to-go at the end of
    trajectories. These stored values can then be used for training by the
    agent. """

    def __init__(self, buffer_size=int(1e6), epoch_size=5000, batch_size=100):
        self.buffer_size, self.path_start_idx = buffer_size, 0
        self.batch_size = batch_size
        self.epoch_size = epoch_size or buffer_size
        if buffer_size % self.epoch_size != 0:
            raise NotImplementedError("Buffer size which is not integer "
                                      "multiple of epoch size not supported")
        self.store_ctr, self.buf = 0, None

    def store(self, to_buffer):
        """ Push values from a single timestep into the buffer """
        if not self.buf:
            self.buf = self.initialize_buf(to_buffer)
        for key, val in to_buffer.items():
            self.buf[key][self.ptr] = val
        self.store_ctr += 1

    def finish_path(self, last_obs=None):
        """ Upon completing an episode, fill the 'next_obs' field """
        self.update_path_next_obs(self.path_slice, last_obs)
        self.path_start_idx = self.ptr

    def batches(self, n_batches):
        """ generator of randomly-sampled minibatches of experience """
        for _ in range(n_batches):
            yield self.sample_batch(self.batch_size)

    def sample_batch(self, batch_size):
        """ sample a batch of experiences from the buffer """
        self.assert_path_finished()
        batch_inds = np.random.randint(0, self.n_stored, batch_size)
        return {key: val[batch_inds] for key, val in self.buf.items()}

    def assert_path_finished(self):
        """ raises an error if the most recent path has not be finished """
        if self.ptr != self.path_start_idx:
            raise RuntimeError('Most recent path has not been finished. '
                               + 'Please call "finish_path"')

    def update_path_next_obs(self, path_slice, last_obs):
        """ fill in the next observations (which we've been ignoring during
        the episode) """
        path_obs = self.buf['obs'][path_slice]
        path_next_obs = np.vstack([
            path_obs[1:, :],
            last_obs[np.newaxis, :]
        ])
        self.buf['next_obs'][path_slice] = path_next_obs

    def initialize_buf(self, to_buffer):
        """ Initialize the buffer storage based on the shapes of the
        values we want to store. We also always add fields for the
        return-to-go, advantage and next state observation """
        buf = {'next_obs': self.zeros(to_buffer['obs'])}
        return {**buf, **{key: self.zeros(val)
                          for key, val in to_buffer.items()}}

    def zeros(self, val):
        """ Creates numpy array of zeros of
        shape (buffer_size, *(val shape)) """
        return np.zeros(combined_shape(self.buffer_size, val),
                        dtype=np.float32)

    @property
    def ptr(self):
        """ pointer, points to the index in the buffer to store the next
        experience """
        return self.store_ctr % self.buffer_size

    @property
    def n_stored(self):
        """ number of experiences (steps) currently stored in the buffer """
        return min(self.store_ctr, self.buffer_size)

    @property
    def path_slice(self):
        """ slice of current episode """
        # the 'or' is here in case we reach the end of the buffer, in which case
        # self.ptr wraps around. In that case, self.ptr = 0, whereas we
        # want the path to end at self.buffer_size
        return slice(self.path_start_idx, self.ptr or self.buffer_size)
