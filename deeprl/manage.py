""" Click Command-line interface entry point """
import click
import tensorflow as tf
import logging
from deeprl.tasks.run import run as task_run, maybe_run
from deeprl.tasks import plotting
from deeprl.common import DEFAULT_KWARGS, LOGGER_NAME
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


def process_cli_kwargs(kwargs):
    kwargs['activation'] = getattr(tf.nn, kwargs['activation'])
    kwargs['hidden_sizes'] = eval(kwargs['hidden_sizes'])
    return {key: val for key, val in kwargs.items()
            if DEFAULT_KWARGS[key]!=val}


@click.group()
def cli():
    """Main entry point"""


@cli.command("run")
@click.option('--exp_name', '-exp' ,default='',
              help='Prefix added to experiment name')
@click.option('--implementation', '-imp' ,default='tom',
              help='Prefix added to experiment name',
              type=click.Choice(['tom', 'spinup']))
@click.option('--num_runs', '-n', default=3,
              help='Number of different random seeds to run',
              show_default=True)
@click.option('--epochs', default=50, help='Number of epochs',
              show_default=True)
@click.option('--steps_per_epoch', '-steps', default=4000,
              help='Number of epochs', show_default=True)
@click.option('--env_name', '-env', default='Swimmer-v2',
              help='Environment name', show_default=True)
@click.option('--hidden_sizes', '-hid', default='(64,64)',
              help='Hidden sizes for actor and critic MLPs',
              show_default=True)
@click.option('--activation' ,default='tanh',
              help='Activation to use in actor-critic MLPs',
              show_default=True)
def run(exp_name, implementation, num_runs, **kwargs):
    """ Run experiment and plot Episode Return """
    click.echo('exp name: {}'.format(exp_name))
    click.echo('implementation: {}'.format(implementation))
    click.echo('num_runs: {}'.format(num_runs))
    epochs = kwargs['epochs']
    kwargs = process_cli_kwargs(kwargs)
    click.echo('kwargs: {}'.format(kwargs))
    # task_run(exp_name, implementation, num_runs, **kwargs)
    maybe_run(exp_name, num_runs, [implementation], **kwargs)
    # ensure we don't have the epochs kwarg twice
    plotting.deeprlplot(
        exp_name, [implementation], num_runs=num_runs,
        epochs=kwargs.pop('epochs', DEFAULT_KWARGS['epochs']), **kwargs)


@cli.command("plot")
@click.option('--exp_name', '-exp' ,default='',
              help='Prefix added to experiment name')
@click.option('--implementation', '-imp' ,default='tom',
              help="Which implementation to run, spinup's or Tom's",
              type=click.Choice(['tom', 'spinup']))
@click.option('--value', '-v', default='AverageEpRet',
              help='Value to plot', show_default=True)
@click.option('--env_name', '-env', default='Swimmer-v2',
              help='Environment name', show_default=True)
@click.option('--hidden_sizes', '-hid', default='(64,64)',
              help='Hidden sizes for actor and critic MLPs',
              show_default=True)
@click.option('--activation', default='tanh',
              help='Activation to use in actor-critic MLPs',
              show_default=True)
def plot(exp_name, implementation, value, **kwargs):
    """ plot Logging Results """
    click.echo('exp name: {}'.format(exp_name))
    click.echo('implementation: {}'.format(implementation))
    kwargs = process_cli_kwargs(kwargs)
    click.echo('kwargs: {}'.format(kwargs))
    plotting.deeprlplot(
        exp_name, [implementation],
        epochs=kwargs.pop('epochs', DEFAULT_KWARGS['epochs']),
        value=value, **kwargs)


@cli.command("benchmark")
@click.option('--exp_name', '-exp' ,default='',
              help='Prefix added to experiment name')
@click.option('--num_runs', '-n', default=3,
              help='Number of different random seeds to run',
              show_default=True)
@click.option('--epochs', default=50, help='Number of epochs',
              show_default=True)
@click.option('--steps_per_epoch', '-steps', default=4000,
              help='Number of epochs', show_default=True)
@click.option('--env_name', '-env', default='Swimmer-v2',
              help='Environment name', show_default=True)
@click.option('--hidden_sizes', '-hid', default='(64,64)',
              help='Hidden sizes for actor and critic MLPs',
              show_default=True)
@click.option('--activation' ,default='tanh',
              help='Activation to use in actor-critic MLPs',
              show_default=True)
def benchmark(exp_name, num_runs, **kwargs):
    """ Benchmart tom's implementation against spinup and plot """
    imps = ['tom', 'spinup']
    click.echo('exp name: {}'.format(exp_name))
    click.echo('num_runs: {}'.format(num_runs))
    epochs = kwargs['epochs']
    kwargs = process_cli_kwargs(kwargs)
    click.echo('kwargs: {}'.format(kwargs))
    maybe_run(exp_name, num_runs, imps, **kwargs)
    # ensure we don't have the epochs kwarg twice
    plotting.deeprlplot(
        exp_name, imps, num_runs=num_runs,
        epochs=kwargs.pop('epochs', DEFAULT_KWARGS['epochs']), **kwargs)


if __name__ == "__main__":
    cli()
