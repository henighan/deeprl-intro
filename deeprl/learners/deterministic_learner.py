""" Learner for Deterministic Policies for DDPG and TD3 Agents """
import time

import numpy as np
import tensorflow as tf

from deeprl.buffers import OffPolicyBuffer
from deeprl.utils import tf_utils
from spinup.utils.logx import EpochLogger


class DeterministicLearner:
    """ Learner for training Agents with deterministic policies,
    and thus have different behavior during training and testing """

    def __init__(self, agent, env, steps_per_epoch=5000, epochs=50, seed=0,
                 max_ep_len=1000, start_steps=10000, replay_size=int(1e6),
                 batch_size=100, n_test_episodes=10,
                 output_dir=None, output_fname='progress.txt', exp_name=None):
        self.epoch_len, self.n_epochs = steps_per_epoch, epochs
        self.max_ep_len, self.start_steps = max_ep_len, start_steps
        self.n_test_episodes = n_test_episodes
        self.logger = EpochLogger(output_dir=output_dir,
                                  output_fname=output_fname,
                                  exp_name=exp_name)
        print('locals')
        for key, val in locals().items():
            print('{}: {}'.format(key, len(str(val))))
        # self.logger.save_config(locals())
        self.env, self.agent = env, agent
        self.buffer = OffPolicyBuffer(buffer_size=replay_size,
                                      epoch_size=steps_per_epoch,
                                      batch_size=batch_size)
        saver_kwargs = agent.build_graph(env.observation_space,
                                         env.action_space)
        self.logger.setup_tf_saver(**saver_kwargs)
        var_counts = tuple(tf_utils.trainable_count(scope)
                           for scope in ['pi', 'v'])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'
                        % var_counts)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def episode_step(self, obs, rew, is_term, ep_len, ep_ret,
                     epoch_ctr, testing=False):
        """ take a single step in the episode """
        # environment variables to store in buffer
        env_to_buffer = dict(obs=obs, rew=rew, is_term=is_term)
        # Take agent step, return values to store in buffer, and in logs
        agent_to_buffer, agent_to_log = self.agent.step(obs, testing=testing)
        if not testing:
            self.buffer.store({**env_to_buffer, **agent_to_buffer})
            self.logger.store(**agent_to_log)
            epoch_ctr += 1
        ep_len += 1
        ep_ret += rew
        obs, rew, is_term, _ = self.env.step(agent_to_buffer['act'])
        return obs, rew, is_term, ep_len, ep_ret, epoch_ctr

    def play_episode(self, epoch_ctr=0, testing=False):
        """ play out an episode until one of these things happens:
        1. episode ends
        2. max episode length is reached
        3. end of epoch is reached """
        obs = self.env.reset()
        rew, ep_len, ep_ret, is_term_state = 0, 0, 0, False
        while ((ep_len < self.max_ep_len)
               and (not is_term_state)
               and (epoch_ctr < self.epoch_len)):
            step_ret = self.episode_step(obs, rew, is_term_state, ep_len,
                                         ep_ret, epoch_ctr, testing=testing)
            obs, rew, is_term_state, ep_len, ep_ret, epoch_ctr = step_ret
        ep_ret += rew # important! add the last reward to the return!
        log_prefix = 'Test' if testing else ''
        if (is_term_state) or (ep_len >= self.max_ep_len):
            self.logger.store(**{
                log_prefix + 'EpRet': ep_ret, log_prefix + 'EpLen': ep_len})
        if not testing:
            self.buffer.finish_path(last_obs=obs)
        return ep_len, ep_ret, epoch_ctr

    def train_episode(self, ep_len):
        """ train agent at the end of episode """
        batches = self.buffer.batches(n_batches=ep_len)
        for train_iter, batch in enumerate(batches):
            self.agent.train(train_iter, batch)
        self.agent.update_targets()

    def run_epoch(self):
        """ run epoch of training + evaluation """
        epoch_ctr = 0
        while epoch_ctr < self.epoch_len:
            ep_len, _, epoch_ctr = self.play_episode(
                epoch_ctr=epoch_ctr, testing=False)
            self.train_episode(ep_len)
        self.test_epoch(self.n_test_episodes)

    def test_epoch(self, n_test_episodes):
        """ perform testing for an epoch """
        for _ in range(n_test_episodes):
            self.play_episode(0, testing=True)

    def learn(self):
        """ Train the agent over n_epochs """
        for epoch in range(self.n_epochs):
            start_time = time.time()
            self.run_epoch()
            self.log_epoch(epoch, start_time)
            self.logger.save_state({'env': self.env}, None)

    def log_epoch(self, epoch, start_time):
        """ Log info about epoch """
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('TestEpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts',
                                (epoch+1)*self.epoch_len)
        self.logger.log_tabular('Time', time.time()-start_time)
        for column_name, kwargs in self.agent.log_tabular_kwargs.items():
            self.logger.log_tabular(column_name, **kwargs)
        self.logger.dump_tabular()
