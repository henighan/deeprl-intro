""" Click Command-line interface entry point """
import click


@click.group()
def cli():
    """Main entry point"""


@cli.command("plot")
def plot():
    """ Plot Logging Results """
    click.echo('DO THIS!')


if __name__ == "__main__":
    cli()
