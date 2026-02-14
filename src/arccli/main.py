"""Root CLI group — `arc` command."""

import click

from arccli.llm import llm


@click.group()
def cli() -> None:
    """Arc — unified CLI for Arc products."""


cli.add_command(llm)
