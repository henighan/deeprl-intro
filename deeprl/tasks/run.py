""" Compare Experiments """
import os
import gym
from deeprl.learner import Learner
from spinup.utils.run_utils import ExperimentGrid
from spinup import vpg
import tensorflow as tf
from deeprl import log_utils
from deeprl.agents.vpg import VPG
    

def run(exp_name, implementation, num_runs, **kwargs):
    if implementation == 'spinup':
        spinup_datadir = './data/spinup'
        kwargs['num_runs'] = num_runs
        run_spinup_grid(exp_name, spinup_datadir, **kwargs)
    elif implementation == 'tom':
        exp_name = log_utils.kwargs_to_exp_name(exp_name, kwargs)
        output_dir = os.path.join('./data/tom', exp_name, exp_name)
        print('saving to {}'.format(output_dir))
        run_tom_grid(exp_name, output_dir, num_runs=num_runs, **kwargs)


def run_spinup_grid(exp_name, spinup_datadir, num_runs=3,
                    env_name='Swimmer-v2', hidden_sizes=(64,64),
                    activation=tf.tanh, steps_per_epoch=4000,
                    epochs=50, num_cpu=2):
    eg = ExperimentGrid(name=exp_name)
    eg.add('env_name', env_name, '', True)
    eg.add('seed', [10*i for i in range(num_runs)])
    eg.add('epochs', epochs)
    eg.add('steps_per_epoch', steps_per_epoch)
    eg.add('ac_kwargs:hidden_sizes', hidden_sizes, 'hid')
    eg.add('ac_kwargs:activation', activation, '')
    eg.run(vpg, num_cpu=num_cpu, data_dir=spinup_datadir)


def run_tom_grid(exp_name, output_dir, num_runs=3,
                 env_name='Swimmer-v2', hidden_sizes=(64,64),
                 activation=tf.tanh, steps_per_epoch=4000,
                 epochs=50, num_cpu=2):
    for ii in range(num_runs):
        seed = 10*ii
        output_dir_ii = output_dir + '_s{}'.format(seed)
        tf.reset_default_graph()
        agent = VPG(hidden_sizes=hidden_sizes, activation=activation)
        env = gym.make(env_name)
        learner = Learner(agent, env, output_dir=output_dir_ii,
                          steps_per_epoch=steps_per_epoch, epochs=epochs,
                          seed=seed)
        learner.learn()
