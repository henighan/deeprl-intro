""" RL Learner, for playing out episodes and training """
import time

import numpy as np

from deeprl.replay_buffer import ReplayBuffer
from deeprl import tf_utils
from spinup.utils.logx import EpochLogger


class Learner():

    def __init__(self, agent, env, steps_per_epoch=1000, epochs=50, seed=0,
                 output_dir=None, output_fname='progress.txt',
                 exp_name=None, gamma=0.99, lam=0.97):
        self.epoch_len, self.n_epochs = steps_per_epoch, epochs
        self.logger = EpochLogger(output_dir=output_dir,
                                  output_fname=output_fname,
                                  exp_name=exp_name)
        self.logger.save_config(locals())
        self.env, self.agent = env, agent
        self.buffer = ReplayBuffer(steps_per_epoch, gamma=gamma, lam=lam)
        agent.build_graph(env.observation_space, env.action_space)
        self.logger.setup_tf_saver(
            agent.sess, inputs={'x': agent.placeholders.get('obs')},
            outputs={'pi': agent.pi, 'v': agent.val})
        var_counts = tuple(tf_utils.trainable_count(scope)
                           for scope in ['pi', 'v'])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'
                        % var_counts)
        np.random.seed(seed)

    def play_episode(self):
        reset_return = self.env.reset()
        obs, rew = reset_return, 0
        ep_len, ep_ret, is_term_state = 0, 0, False
        while (not self.buffer.full) and (not is_term_state):
            # environment variables to store in buffer
            env_to_buffer = dict(obs=obs, rew=rew)
            # Take agent step, return values to store in buffer, and in logs
            agent_to_buffer, agent_to_log = self.agent.step(obs)
            self.buffer.store({**env_to_buffer, **agent_to_buffer})
            self.logger.store(**agent_to_log)
            ep_len += 1
            ep_ret += rew
            obs, rew, is_term_state, _ = self.env.step(agent_to_buffer['act'])
        if is_term_state:
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
        pi_loss, v_loss, delta_pi_loss, delta_v_loss = self.agent.train(
            self.buffer.get())
        self.logger.store(LossPi=pi_loss, LossV=v_loss,
                          DeltaLossPi=delta_pi_loss, DeltaLossV=delta_v_loss)

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
        self.logger.log_tabular('VVals', with_min_and_max=True)
        self.logger.log_tabular('Logp', with_min_and_max=True)
        self.logger.log_tabular('TotalEnvInteracts', (epoch+1)*self.epoch_len)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossV', average_only=True)
        self.logger.log_tabular('DeltaLossPi', average_only=True)
        self.logger.log_tabular('DeltaLossV', average_only=True)
        self.logger.log_tabular('Time', time.time()-start_time)
        self.logger.dump_tabular()
