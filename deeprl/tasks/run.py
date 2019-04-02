""" Experiment Runners """
import os
import gym
from deeprl.learner import Learner
from spinup.utils.run_utils import ExperimentGrid
from spinup import vpg
import tensorflow as tf
from deeprl import log_utils
from deeprl.agents.vpg import VPG
from deeprl.common import DEFAULT_KWARGS
import logging
logger = logging.getLogger('deeprl')


def maybe_run(exp_name, num_runs, implementations, **kwargs):
    """ Run maybe_run_single_seed for num_runs different seeds. If
    implementation is not specified, run both tom and spinup implementations
    """
    epochs = kwargs.get('epochs', DEFAULT_KWARGS['epochs'])
    for ii in range(num_runs):
        for imp in implementations:
            seed = 10*ii
            maybe_run_single_seed(exp_name, imp, epochs, seed, kwargs)


def maybe_run_single_seed(exp_name, implementation, epochs, seed, kwargs):
    """ Check if experiment has been run and if so, for how many epochs. Run
    only if it hasn't been run, or not for enough epochs """
    n_epochs_already_run = log_utils.num_run_epochs(
        exp_name, implementation, seed, kwargs)
    output_dir = log_utils.output_dir_from_kwargs(
        exp_name, implementation, kwargs, seed=seed)
    if n_epochs_already_run >= epochs:
        logger.info(
            'seed {} for {} already run for {} epochs, skipping'.format(
                seed, output_dir, n_epochs_already_run))
        return
    if n_epochs_already_run > 0:
        logger.warning(
            'only {} epochs run for {}, rerunning with {} epochs'.format(
                n_epochs_already_run, output_dir, epochs))
    logger.info('running {}'.format(output_dir))
    run(exp_name, implementation, seed, **kwargs)
    

def run(exp_name, implementation, seed, **kwargs):
    output_dir = log_utils.output_dir_from_kwargs(
        exp_name, implementation, kwargs, seed=seed)
    if implementation == 'spinup':
        # spinup_datadir = './data/spinup'
        run_spinup(exp_name, seed,
                   output_dir=output_dir, **kwargs)
    elif implementation == 'tom':
        print('saving to {}'.format(output_dir))
        run_tom(output_dir, seed, **kwargs)


def run_spinup(exp_name, seed, output_dir=None,
               env_name='Swimmer-v2', hidden_sizes=(64,64),
               activation=tf.tanh, num_cpu=2, **kwargs):
    logger_kwargs = {'exp_name': exp_name, 'output_dir': output_dir}
    ac_kwargs = {'hidden_sizes': hidden_sizes, 'activation': activation}
    tf.reset_default_graph()
    vpg(get_env_fn(env_name), ac_kwargs=ac_kwargs,
        logger_kwargs=logger_kwargs, seed=seed, **kwargs)


def get_env_fn(env_name):
    return lambda: gym.make(env_name)


def run_tom(output_dir, seed,
            env_name='Swimmer-v2', hidden_sizes=(64,64),
            activation=tf.tanh, steps_per_epoch=4000,
            epochs=50, num_cpu=2):
    tf.reset_default_graph()
    agent = VPG(hidden_sizes=hidden_sizes, activation=activation)
    env = gym.make(env_name)
    learner = Learner(agent, env, output_dir=output_dir,
                      steps_per_epoch=steps_per_epoch, epochs=epochs,
                      seed=seed)
    learner.learn()
    del agent
