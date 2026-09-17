"""The image limits glibc malloc arenas for every process it runs.

A threads-pool worker extracting large files one after another otherwise
keeps growing: each thread that allocates gets its own arena, freed page
buffers stay mapped in them, and the worker ratchets towards its memory limit
across files that each fit comfortably on their own.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_image_sets_malloc_arena_max_to_two():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert re.search(r"^ENV MALLOC_ARENA_MAX=2\s*$", dockerfile, re.MULTILINE)


def test_the_setting_is_documented_for_other_deployments():
    assert "MALLOC_ARENA_MAX=2" in (ROOT / "env.example").read_text()
