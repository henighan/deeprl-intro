""" Click Command-line interface entry point """
import itertools
import logging

import click
import tensorflow as tf
from matplotlib import pyplot as plt

from deeprl.tasks.run import maybe_run
from deeprl.tasks import plotting
from deeprl.common import DEFAULT_KWARGS, LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


def process_cli_kwargs(all_kwargs):
    """ generator yielding all combinations hidden_sizes, activations, and
    environments that were specified. Also processes hidden_arguments and
    activations from strings into tuples and tf activation functions,
    respectively """
    activations = all_kwargs.pop('activation')
    hidden_sizes = all_kwargs.pop('hidden_sizes')
    env_names = all_kwargs.pop('env_name')
    for hid, act, env in itertools.product(hidden_sizes, activations, env_names):
        kwargs = {'hidden_sizes': eval(hid),
                  'activation': getattr(tf.nn, act),
                  'env_name': env,
                  **all_kwargs}
        yield {key: val for key, val in kwargs.items()
               if DEFAULT_KWARGS[key] != val}


@click.group()
@click.option('--exp_name', '-exp', default='',
              help='Prefix added to experiment name')
@click.option('--num_runs', '-n', default=3,
              help='Number of different random seeds to run',
              show_default=True)
@click.option('--epochs', default=50, help='Number of epochs',
              show_default=True)
@click.option('--steps_per_epoch', '-steps', default=4000,
              help='Number of epochs', show_default=True)
@click.option('--env_name', '-env', default=['Swimmer-v2'],
              help='Environment name', show_default=True, multiple=True)
@click.option('--hidden_sizes', '-hid', default=['(64,64)'],
              help='Hidden sizes for actor and critic MLPs',
              show_default=True, multiple=True)
@click.option('--activation', default=['tanh'],
              help='Activation to use in actor-critic MLPs',
              show_default=True, multiple=True)
@click.pass_context
def cli(ctx, exp_name, num_runs, **kwargs):
    """Main entry point"""
    click.echo(ctx)
    ctx.obj = {'exp_name': exp_name, 'num_runs': num_runs,
               'kwargs': process_cli_kwargs(kwargs)}
    click.echo('exp name: {}'.format(exp_name))
    click.echo('num_runs: {}'.format(num_runs))


@cli.command("benchmark")
@click.pass_context
def benchmark(ctx):
    """ Benchmark tom's implementation against spinup and plot """
    imps = ['tom', 'spinup']
    exp_name, num_runs = ctx.obj['exp_name'], ctx.obj['num_runs']
    for kwargs in ctx.obj['kwargs']:
        click.echo('kwargs: {}'.format(kwargs))
        maybe_run(exp_name, num_runs, imps, **kwargs)
        # ensure we don't have the epochs kwarg twice
        plotting.deeprlplot(
            exp_name, imps, num_runs=num_runs, benchmark=True,
            epochs=kwargs.pop('epochs', DEFAULT_KWARGS['epochs']), **kwargs)
    plt.show()


@cli.command("run")
@click.option('--implementation', '-imp', default='tom',
              help='Prefix added to experiment name',
              type=click.Choice(['tom', 'spinup']))
@click.pass_context
def run(ctx, implementation):
    """ Run experiment and plot Episode Return """
    exp_name, num_runs = ctx.obj['exp_name'], ctx.obj['num_runs']
    click.echo('implementation: {}'.format(implementation))
    for kwargs in ctx.obj['kwargs']:
        click.echo('kwargs: {}'.format(kwargs))
        maybe_run(exp_name, num_runs, [implementation], **kwargs)
        # ensure we don't have the epochs kwarg twice
        plotting.deeprlplot(
            exp_name, [implementation], num_runs=num_runs,
            epochs=kwargs.pop('epochs', DEFAULT_KWARGS['epochs']), **kwargs)
    plt.show()


@cli.command("plot")
@click.option('--implementation', '-imp', default=['tom'],
              help="Which implementation to run, spinup's or Tom's",
              type=click.Choice(['tom', 'spinup']), multiple=True)
@click.option('--value', '-v', default='AverageEpRet',
              help='Value to plot', show_default=True)
@click.pass_context
def plot(ctx, implementation, value):
    """ plot Logging Results """
    exp_name, num_runs = ctx.obj['exp_name'], ctx.obj['num_runs']
    click.echo('implementation: {}'.format(implementation))
    for kwargs in ctx.obj['kwargs']:
        click.echo('kwargs: {}'.format(kwargs))
        plotting.deeprlplot(
            exp_name, implementation,
            epochs=kwargs.pop('epochs', DEFAULT_KWARGS['epochs']),
            value=value, **kwargs)
    plt.show()


if __name__ == "__main__":
    cli()
