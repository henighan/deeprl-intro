""" Experiment Runners """
import logging

from spinup.utils.run_utils import ExperimentGrid
import gym
import spinup
import tensorflow as tf

from deeprl.common import DEFAULT_KWARGS
from deeprl.helpers import algos
from deeprl.utils import logdir

logger = logging.getLogger('deeprl')


def maybe_run(exp_name, num_runs, implementations, **kwargs):
    """ Run maybe_run_single_seed for num_runs different seeds. If
    implementation is not specified, run both tom and spinup implementations
    """
    epochs = kwargs.get('epochs', DEFAULT_KWARGS['epochs'])
    for run_no in range(num_runs):
        for imp in implementations:
            seed = 10*run_no
            maybe_run_single_seed(exp_name, imp, epochs, seed, kwargs)


def maybe_run_single_seed(exp_name, implementation, epochs, seed, kwargs):
    """ Check if experiment has been run and if so, for how many epochs. Run
    only if it hasn't been run, or not for enough epochs """
    n_epochs_already_run = logdir.num_run_epochs(
        exp_name, implementation, seed, kwargs)
    output_dir = logdir.output_dir_from_kwargs(
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
    """ run an algorithm """
    output_dir = logdir.output_dir_from_kwargs(
        exp_name, implementation, kwargs, seed=seed)
    logger.info('saving to {}'.format(output_dir))
    if implementation == 'spinup':
        run_spinup(exp_name, seed,
                   output_dir=output_dir, **kwargs)
    elif implementation == 'tom':
        run_tom(output_dir, seed, **kwargs)


def run_spinup(exp_name, seed, output_dir=None,
               env_name='Swimmer-v2', hidden_sizes=(64, 64),
               activation=tf.tanh, num_cpu=2, algo='vpg', **kwargs):
    """ run spinup's implementation """
    logger_kwargs = {'exp_name': exp_name, 'output_dir': output_dir}
    ac_kwargs = {'hidden_sizes': hidden_sizes, 'activation': activation}
    tf.reset_default_graph()
    thunk = getattr(spinup, algo)
    env_fn = lambda: gym.make(env_name)
    thunk(env_fn, ac_kwargs=ac_kwargs,
          logger_kwargs=logger_kwargs, seed=seed, **kwargs)


def run_tom(output_dir, seed, algo='vpg', **kwargs):
    """ run tom's implementation """
    tf.reset_default_graph()
    thunk = getattr(algos, algo)
    thunk(output_dir, seed, **kwargs)
