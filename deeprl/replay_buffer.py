""" Replay Buffer """
import numpy as np
from deeprl.utils.math import (advantage_function,
                               combined_shape, rewards_to_go)


class ReplayBuffer:
    """ Stores an episode experience (rewards, observations,
    actions, etc), calculates the advantage and rewards-to-go at the end of
    trajectories. These stored values can then be used for training by the
    agent """

    def __init__(self, buffer_size, epoch_size=None, gamma=0.99, lam=0.95):
        self.gamma, self.lam = gamma, lam
        self.buffer_size, self.path_start_idx = buffer_size, 0
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

    def update_path_advantage(self, path_slice, last_val):
        """ calculate the advantage over the path and store in buffer """
        trajectory_rews = self.buf['rew'][path_slice]
        trajectory_vals = self.buf['val'][path_slice]
        self.buf['adv'][path_slice] = advantage_function(
            trajectory_rews, trajectory_vals, gamma=self.gamma,
            lam=self.lam, last_val=last_val)

    def update_path_rewards_to_go(self, path_slice, last_val):
        """ computes rewards-to-go, to be targets for the value function """
        trajectory_rews = self.buf['rew'][path_slice]
        self.buf['ret'][path_slice] = rewards_to_go(
            trajectory_rews, gamma=self.gamma, last_val=last_val)

    def update_path_next_obs(self, path_slice, last_obs):
        """ fill in the next observations (which we've been ignoring during
        the episode) """
        path_obs = self.buf['obs'][path_slice]
        path_next_obs = np.vstack([
            path_obs[1:, :],
            last_obs[np.newaxis, :]
        ])
        self.buf['next_obs'][path_slice] = path_next_obs

    def finish_path(self, last_val=0, last_obs=None):
        """ Upon completing an episode, calculate the advantage and
        rewards-to-go """
        self.update_path_advantage(self.path_slice, last_val)
        self.update_path_rewards_to_go(self.path_slice, last_val)
        self.update_path_next_obs(self.path_slice, last_obs)
        self.path_start_idx = self.ptr

    def dump(self):
        """ Dump the contents of the buffer, and reset the pointer """
        assert self.store_ctr == self.buffer_size
        self.store_ctr, self.path_start_idx = 0, 0 #re-initialize buffer
        self.normalize_advantage()
        return self.buf

    def normalize_advantage(self):
        """ implements the advantage normalization trick """
        adv = self.buf['adv'][:self.n_stored]
        normalized_adv = (adv - adv.mean()) / adv.std()
        self.buf['adv'][:self.n_stored] = normalized_adv

    def initialize_buf(self, to_buffer):
        """ Initialize the buffer storage based on the shapes of the
        values we want to store. We also always add fields for the
        return-to-go, advantage and next state observation """
        buf = {'adv': self.zeros(1), 'ret': self.zeros(1),
               'next_obs': self.zeros(to_buffer['obs'])}
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
        # the or is here in case we reach the end of the buffer, in which case
        # self.ptr wraps around. In that case, self.ptr = 0, whereas we
        # want the path to end at self.buffer_size
        return slice(self.path_start_idx, self.ptr or self.buffer_size)
