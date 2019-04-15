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
                 exp_name=None, max_ep_len=1000, gamma=0.99, lam=0.97):
        self.epoch_len, self.n_epochs = steps_per_epoch, epochs
        self.max_ep_len = max_ep_len
        self.logger = EpochLogger(output_dir=output_dir,
                                  output_fname=output_fname,
                                  exp_name=exp_name)
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

    def play_episode(self):
        obs = self.env.reset()
        rew, ep_len, ep_ret, is_term_state = 0, 0, 0, False
        while ((not self.buffer.full)
                and (not is_term_state)
                and (ep_len < self.max_ep_len)):
            # environment variables to store in buffer
            env_to_buffer = dict(obs=obs, rew=rew)
            # Take agent step, return values to store in buffer, and in logs
            agent_to_buffer, agent_to_log = self.agent.step(obs)
            self.buffer.store({**env_to_buffer, **agent_to_buffer})
            self.logger.store(**agent_to_log)
            ep_len += 1
            ep_ret += rew
            obs, rew, is_term_state, _ = self.env.step(agent_to_buffer['act'])
        if (is_term_state) or (ep_len >= self.max_ep_len):
            self.logger.store(EpRet=ep_ret, EpLen=ep_len)
        else:
            # if trajectory didn't reach terminal state, bootstrap to target
            rew = self.agent.step(obs)[0]['val']
        self.buffer.finish_path(rew) # calculate advantage
        return ep_len, ep_ret

    def play_epoch(self):
        while not self.buffer.full:
            self.play_episode()

    def train_epoch(self):
        to_log = self.agent.train(
            self.buffer.get())
        self.logger.store(**to_log)

    def learn(self):
        for epoch in range(self.n_epochs):
            start_time = time.time()
            self.play_epoch()
            self.train_epoch()
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
        for column_name, kwargs in self.agent.log_tabular_kwargs.items():
            self.logger.log_tabular(column_name, **kwargs)
        self.logger.dump_tabular()
