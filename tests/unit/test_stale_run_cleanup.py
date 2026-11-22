"""The stale-run cutoff is a project setting, bound into both UPDATEs."""

from unittest.mock import MagicMock

from agentic_project_service.services.settings_registry import SETTINGS_REGISTRY
from agentic_project_service.tasks import cleanup as cleanup_mod


def test_setting_registered():
    d = SETTINGS_REGISTRY["RUN_STALE_AFTER_SECONDS"]
    assert (d.type, d.default, d.min, d.max, d.category) == ("int", 1800, 600, 86400, "agents")


def _fake_db(monkeypatch, rowcounts=(0, 0)):
    fake_db = MagicMock()
    results = [MagicMock(rowcount=n) for n in rowcounts]
    fake_db.session.execute.side_effect = results
    monkeypatch.setattr(cleanup_mod, "db", fake_db)
    return fake_db


def test_cutoff_is_bound_into_both_updates(monkeypatch):
    fake_db = _fake_db(monkeypatch)
    monkeypatch.setattr(
        cleanup_mod, "get_setting", lambda key: {"RUN_STALE_AFTER_SECONDS": 2400}[key]
    )

    assert cleanup_mod.run_stale_cleanup() == {"agent_runs": 0, "orchestration_runs": 0}

    calls = fake_db.session.execute.call_args_list
    assert len(calls) == 2
    tables = []
    for call in calls:
        sql = str(call.args[0])
        assert "INTERVAL '10 minutes'" not in sql
        assert "(:stale_after * INTERVAL '1 second')" in sql
        assert call.args[1] == {"stale_after": 2400}
        tables.append("agent_runs" if ".agent_runs" in sql else "orchestration_runs")
    assert tables == ["agent_runs", "orchestration_runs"]
    fake_db.session.commit.assert_called_once()


def test_counts_are_reported(monkeypatch, caplog):
    _fake_db(monkeypatch, rowcounts=(3, 1))
    monkeypatch.setattr(cleanup_mod, "get_setting", lambda key: 1800)
    with caplog.at_level("INFO", logger=cleanup_mod.logger.name):
        assert cleanup_mod.run_stale_cleanup() == {"agent_runs": 3, "orchestration_runs": 1}
    assert "Cleaned up 3 stale agent runs and 1 stale orchestration runs" in caplog.text


def test_the_task_runs_the_cleanup(monkeypatch):
    monkeypatch.setattr(
        cleanup_mod, "run_stale_cleanup", lambda: {"agent_runs": 2, "orchestration_runs": 0}
    )
    assert cleanup_mod.cleanup_stale_runs.run() == {"agent_runs": 2, "orchestration_runs": 0}
