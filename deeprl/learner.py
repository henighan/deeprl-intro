""" RL Learner, for playing out episodes and training """
import tensorflow as tf
import numpy as np
from deeprl.replay_buffer import Buffer
from spinup.utils.logx import EpochLogger
import time


class Learner():

    def __init__(self, agent, env, epoch_len=100, n_epochs=2, seed=0, output_dir=None, output_fname='progress.txt', exp_name=None):
        self.env, self.agent = env, agent
        self.epoch_len, self.n_epochs = epoch_len, n_epochs
        self.buffer = Buffer(env.observation_space.shape[0],
                                env.action_space.shape[0],
                                epoch_len, 0.99, 0.95)
        self.logger = EpochLogger(output_dir=output_dir,
                                  output_fname=output_fname, exp_name=None)
        agent.build_graph(env.observation_space, env.action_space)
        tf.set_random_seed(seed)
        np.random.seed(seed)

    def play_episode(self):
        reset_return = self.env.reset()
        obs, rew = reset_return, 0
        # obs, rew = reset_return[0], reset_return[1]
        ep_len, ep_ret, is_term_state = 0, 0, False
        while (not self.buffer.full) and (not is_term_state):
            act, val_t, logp_t = self.agent.step(obs)
            self.logger.store(VVals=val_t, Logp=logp_t)
            self.buffer.store(obs, act, rew, val_t, logp_t)
            ep_len += 1
            ep_ret += rew
            obs, rew, is_term_state, _ = self.env.step(act)
        if is_term_state:
            self.logger.store(EpRet=ep_ret, EpLen=ep_len)
        else:
            # if trajectory didn't reach terminal state, bootstrap to target
            _, rew, _ = self.agent.step(obs)
        self.buffer.finish_path(rew) # calculate advantage
        return ep_len, ep_ret

    def play_epoch(self):
        while not self.buffer.full:
            self.play_episode()

    def train_epoch(self):
        pi_loss, v_loss, delta_pi_loss, delta_v_loss = self.agent.train(
            *self.buffer.get())
        self.logger.store(LossPi=pi_loss, LossV=v_loss,
                          DeltaLossPi=delta_pi_loss, DeltaLossV=delta_v_loss)

    def learn(self):
        for epoch in range(self.n_epochs):
            start_time = time.time()
            self.play_epoch()
            self.train_epoch()
            self.log_epoch(epoch, start_time)

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
