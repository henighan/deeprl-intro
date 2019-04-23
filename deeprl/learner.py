""" RL Learner, for playing out episodes and training """
import time

import numpy as np
import tensorflow as tf

from deeprl.replay_buffer import ReplayBuffer
from deeprl.utils import tf_utils
from spinup.utils.logx import EpochLogger


class Learner():
    """ Deep RL Learner, plays out episodes and epochs between the environment
    and the agent. At the end of an epoch, it invokes agent training. Also
    handles all logging """

    def __init__(self, agent, env, steps_per_epoch=4000, epochs=50, seed=0,
                 output_dir=None, output_fname='progress.txt',
                 exp_name=None, max_ep_len=1000, gamma=0.99, lam=0.97,
                 n_test_episodes=10):
        self.epoch_len, self.n_epochs = steps_per_epoch, epochs
        self.max_ep_len, self.n_test_episodes = max_ep_len, n_test_episodes
        self.logger = EpochLogger(output_dir=output_dir,
                                  output_fname=output_fname,
                                  exp_name=exp_name)
        self.train_step_ctr = 0
        print('locals')
        for key, val in locals().items():
            print('{}: {}'.format(key, len(str(val))))
        # self.logger.save_config(locals())
        self.env, self.agent = env, agent
        self.buffer = ReplayBuffer(steps_per_epoch, gamma=gamma, lam=lam)
        saver_kwargs = agent.build_graph(env.observation_space,
                                         env.action_space)
        self.logger.setup_tf_saver(**saver_kwargs)
        var_counts = tuple(tf_utils.trainable_count(scope)
                           for scope in ['pi', 'v'])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'
                        % var_counts)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def play_episode(self, testing=False):
        """ Plays out a full episode UNLESS max_ep_len is reached or
        the steps_per_epoch is reached. testing=True is meant to be
        used when we want to evaluate the agent's performance. """
        log_prefix = 'Test' if testing else ''
        obs = self.env.reset()
        rew, ep_len, ep_ret, is_term_state = 0, 0, 0, False
        while ((self.train_step_ctr < self.epoch_len)
                and (not is_term_state)
                and (ep_len < self.max_ep_len)):
            # environment variables to store in buffer
            env_to_buffer = dict(obs=obs, rew=rew, is_term=is_term_state)
            # Take agent step, return values to store in buffer, and in logs
            agent_to_buffer, agent_to_log = self.agent.step(obs)
            if not testing:
                self.buffer.store({**env_to_buffer, **agent_to_buffer})
                self.logger.store(**agent_to_log)
            ep_len += 1
            ep_ret += rew
            obs, rew, is_term_state, _ = self.env.step(agent_to_buffer['act'])
            if not testing:
                self.train_step_ctr += 1
        if (is_term_state) or (ep_len >= self.max_ep_len):
            self.logger.store(**{log_prefix + 'EpRet': ep_ret,
                                 log_prefix + 'EpLen': ep_len})
        else:
            # if trajectory didn't reach terminal state, bootstrap to target
            rew = self.agent.step(obs)[0]['val']
        if not testing:
            # calculate advantage
            self.buffer.finish_path(last_val=rew, last_obs=obs)
        return ep_len, ep_ret

    def play_epoch(self):
        """ play out the episodes for this epoch """
        self.train_step_ctr = 0
        while self.train_step_ctr < self.epoch_len:
            self.play_episode()

    def train_epoch(self):
        """ run agent training for this epoch """
        to_log = self.agent.train(
            self.buffer)
        self.logger.store(**to_log)

    def test_epoch(self):
        """ evaluate agents performance """
        for _ in range(self.n_test_episodes):
            self.play_episode(testing=True)

    def learn(self):
        for epoch in range(self.n_epochs):
            start_time = time.time()
            self.play_epoch()
            self.train_epoch()
            if self.agent.eval_after_epoch:
                self.test_epoch()
            self.log_epoch(epoch, start_time)
            if (epoch % 10 == 0) or (epoch == self.n_epochs - 1):
                self.logger.save_state({'env': self.env}, None)

    def log_epoch(self, epoch, start_time):
        """ Log info about epoch """
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', (epoch+1)*self.epoch_len)
        self.logger.log_tabular('Time', time.time()-start_time)
        for loss_name in self.agent.losses.keys():
            self.logger.log_tabular(loss_name, average_only=True)
            self.logger.log_tabular('Delta' + loss_name, average_only=True)
        for column_name, kwargs in self.agent.log_tabular_kwargs.items():
            self.logger.log_tabular(column_name, **kwargs)
        self.logger.dump_tabular()
