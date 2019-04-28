""" Entry points for algorithms """
import tensorflow as tf
import gym

from deeprl import agents
from deeprl import learners



def vpg(output_dir, seed, env_name='Swimmer-v2', hidden_sizes=(64, 64),
        activation=tf.tanh, steps_per_epoch=4000, epochs=50, num_cpu=2):
    """ function to run vanilla policy gradient """
    agent = agents.VPG(hidden_sizes=hidden_sizes, activation=activation)
    env = gym.make(env_name)
    learner = learners.PolicyGradientLearner(
        agent, env, output_dir=output_dir,
        steps_per_epoch=steps_per_epoch, epochs=epochs, seed=seed)
    learner.learn()
    del agent


def ppo(output_dir, seed, env_name='Swimmer-v2', hidden_sizes=(64, 64),
        activation=tf.tanh, steps_per_epoch=4000, epochs=50, num_cpu=2):
    """ function to run proximal policy optimization """
    agent = agents.PPO(hidden_sizes=hidden_sizes, activation=activation)
    env = gym.make(env_name)
    learner = learners.PolicyGradientLearner(
        agent, env, output_dir=output_dir,
        steps_per_epoch=steps_per_epoch, epochs=epochs, seed=seed)
    learner.learn()
    del agent


def ddpg(output_dir, seed, env_name='Swimmer-v2', hidden_sizes=(400, 300),
         steps_per_epoch=5000, epochs=50):
    agent = agents.DDPG(hidden_sizes=hidden_sizes)
    env = gym.make(env_name)
    learner = learners.DeterministicLearner(
        agent, env, steps_per_epoch=steps_per_epoch, epochs=epochs)
    learner.learn()
    del agent
