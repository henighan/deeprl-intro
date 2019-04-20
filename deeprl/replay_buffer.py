""" Replay Buffer """
import numpy as np
from deeprl.utils.math import (advantage_function, discount_cumsum,
                               combined_shape, rewards_to_go)


class ReplayBuffer:
    """ Stores an Epoch's worth of experience (rewards, observations,
    actions, etc), calculates the advantage and rewards-to-go at the end of
    trajectories. These stored values can then be used for training by the
    agent """

    def __init__(self, buffer_size, epoch_size=None, gamma=0.99, lam=0.95):
        self.gamma, self.lam = gamma, lam
        self.buffer_size, self.path_start_idx, self.ptr = buffer_size, 0, 0
        self.epoch_size = epoch_size or buffer_size
        if buffer_size % self.epoch_size != 0:
            raise NotImplementedError("Buffer size which is not integer "
                "multiple of epoch size not supported")
        self.buf = None

    def store(self, to_buffer):
        """ Push values from a single timestep into the buffer """
        self.ptr = self.ptr % self.buffer_size
        if not self.buf:
            self.buf = self.initialize_buf(to_buffer)
        for key, val in to_buffer.items():
            self.buf[key][self.ptr] = val
        self.ptr += 1

    def finish_path(self, last_val=0):
        """ Upon completing an episode, calculate the advantage and
        rewards-to-go """
        path_slice = slice(self.path_start_idx, self.ptr)
        trajectory_rews = self.buf['rew'][path_slice]
        trajectory_vals = self.buf['val'][path_slice]
        self.buf['adv'][path_slice] = advantage_function(
            trajectory_rews, trajectory_vals, gamma=self.gamma,
            lam=self.lam, last_val=last_val)
        # the next line computes rewards-to-go, to be targets
        # for the value function
        self.buf['ret'][path_slice] = rewards_to_go(
            trajectory_rews, gamma=self.gamma, last_val=last_val)
        self.path_start_idx = self.ptr

    def dump(self):
        """ Dump the contents of the buffer, and reset the pointer """
        assert self.ptr == self.buffer_size
        self.ptr, self.path_start_idx = 0, 0 #re-initialize buffer
        # the next line implements the advantage normalization trick
        self.buf['adv'] = (self.buf['adv']
                           - self.buf['adv'].mean())/self.buf['adv'].std()
        return self.buf

    def initialize_buf(self, to_buffer):
        """ Initialize the buffer storage based on the shapes of the
        values we want to store. We also always add fields for the
        return-to-go and advantage  """
        buf = {'adv': self.zeros(1), 'ret': self.zeros(1)}
        return {**buf, **{key: self.zeros(val)
                          for key, val in to_buffer.items()}}

    def zeros(self, val):
        """ Creates numpy array of zeros of
        shape (buffer_size, *(val shape)) """
        return np.zeros(combined_shape(self.buffer_size, val),
                        dtype=np.float32)
