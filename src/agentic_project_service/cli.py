"""Flask CLI commands for the project service.

Currently exposes the ``docs`` group used to manage the hidden Powabase docs RAG
KB on the singleton system docs project:

    flask docs bootstrap-kb   # ensure the hidden docs KB exists, print its id
    flask docs refresh-kb     # pull docs sources and (re)index changed ones
"""

import click
from flask.cli import AppGroup

from .db import db
from .services.docs_refresh import bootstrap_docs_kb, refresh_docs_kb

docs_cli = AppGroup("docs", help="Manage the hidden Powabase docs RAG knowledge base.")


@docs_cli.command("bootstrap-kb")
def bootstrap_kb_cmd():
    """Create the hidden docs KB if absent and print its id (set as DOCS_KB_ID)."""
    kb_id = bootstrap_docs_kb(db.session)
    click.echo(kb_id)


@docs_cli.command("refresh-kb")
def refresh_kb_cmd():
    """Pull llms-full.txt + docs/agent-skills repos and (re)index changed docs."""
    result = refresh_docs_kb()
    click.echo(
        f"kb={result['kb_id']} docs={result['docs']} "
        f"dispatched={result['dispatched']} unchanged={result['unchanged']} "
        f"error={result['error']}"
    )


def register_cli(app) -> None:
    app.cli.add_command(docs_cli)
