""" Click Command-line interface entry point """
import click
import tensorflow as tf
from deeprl.tasks.run import run as task_run
from deeprl.tasks.plotting import plot as task_plot
from deeprl.common import DEFAULT_KWARGS


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
    """ plot Logging Results """
    click.echo('exp name: {}'.format(exp_name))
    click.echo('implementation: {}'.format(implementation))
    click.echo('num_runs: {}'.format(num_runs))
    kwargs = process_cli_kwargs(kwargs)
    click.echo('kwargs: {}'.format(kwargs))
    task_run(exp_name, implementation, num_runs, **kwargs)


@cli.command("plot")
@click.option('--exp_name', '-exp' ,default='',
              help='Prefix added to experiment name')
@click.option('--implementation', '-imp' ,default='tom',
              help="Which implementation to run, spinup's or Tom's",
              type=click.Choice(['tom', 'spinup']))
@click.option('--env_name', '-env', default='Swimmer-v2',
              help='Environment name', show_default=True)
@click.option('--hidden_sizes', '-hid', default='(64,64)',
              help='Hidden sizes for actor and critic MLPs',
              show_default=True)
@click.option('--activation', default='tanh',
              help='Activation to use in actor-critic MLPs',
              show_default=True)
def plot(exp_name, implementation, **kwargs):
    """ plot Logging Results """
    click.echo('exp name: {}'.format(exp_name))
    click.echo('implementation: {}'.format(implementation))
    kwargs = process_cli_kwargs(kwargs)
    click.echo('kwargs: {}'.format(kwargs))
    task_plot(exp_name, implementation, **kwargs)

if __name__ == "__main__":
    cli()