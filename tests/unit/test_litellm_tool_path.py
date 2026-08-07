"""litellm 1.92.0/1.92.1 break every tool-calling completion when fastapi is absent.

completion() gained `if not skip_mcp_handler and tools:` which imports the proxy
MCP handler chain, reaching `from fastapi import HTTPException`. fastapi is not a
dependency and is absent from uv.lock, so the failure reaches the shipped image.
Only tool-calling completions trigger it — a probe without `tools=` looks clean on
every version.

This test pins the CONSTRAINT, not the runtime.
"""

import tomllib
from importlib.metadata import version
from pathlib import Path

BROKEN = {"1.92.0", "1.92.1"}


def test_installed_litellm_is_not_in_the_broken_range():
    assert version("litellm") not in BROKEN


def test_pyproject_excludes_the_broken_range():
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    )
    spec = next(
        d for d in pyproject["project"]["dependencies"] if d.startswith("litellm")
    )
    for bad in BROKEN:
        assert f"!={bad}" in spec, f"{bad} not excluded: {spec}"
