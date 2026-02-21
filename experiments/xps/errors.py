import sys

import click


def error_exit(message: str) -> None:
    click.echo(f"ERROR: {message}", err=True)
    sys.exit(1)
