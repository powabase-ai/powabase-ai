"""The migration graph must have exactly one head.

Two open PRs can each add a revision whose down_revision is the current head.
Each is single-head in isolation, so nothing on either branch complains --
and whichever merges second produces two heads. Migrations run on startup, so
the result is not a failed CI job: it is every project's service crash-looping
on boot, fleet-wide, after a deploy.

Nothing caught that shape before this test. It reads the script directory only
-- no database, no env.py -- so it belongs in the tier CI already gates on
rather than in a workflow of its own.
"""

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

REPO_ROOT = Path(__file__).resolve().parents[2]


def _script_directory() -> ScriptDirectory:
    # No ini file: script_location is the only option these read, and the
    # repo's alembic.ini lives under migrations/ for Flask-Migrate anyway.
    cfg = Config()
    cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    return ScriptDirectory.from_config(cfg)


def test_migration_graph_has_exactly_one_head():
    heads = _script_directory().get_heads()
    assert len(heads) == 1, (
        f"expected a single migration head, found {len(heads)}: {sorted(heads)}. "
        "Two revisions share a down_revision -- rebase the later one onto the "
        "other so the chain is linear. Left unfixed this crash-loops every "
        "project's service on startup, not just CI."
    )


def test_every_revision_is_reachable_from_the_head():
    """No orphan chains: walking back from the head must visit every revision.

    A revision whose down_revision points at something that was renamed or
    deleted is still 'a head' by some readings but is unreachable in practice;
    the self-heal refuses to start rather than guess.
    """
    script = _script_directory()
    head = script.get_current_head()
    reachable = {rev.revision for rev in script.walk_revisions("base", head)}
    all_revs = {rev.revision for rev in script.walk_revisions()}
    assert reachable == all_revs, (
        f"unreachable revisions: {sorted(all_revs - reachable)}"
    )
