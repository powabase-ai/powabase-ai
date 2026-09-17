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


def test_the_image_fixes_the_malloc_mmap_threshold_at_one_mib():
    # Arenas alone still stepped up across 1,500-page files; a fixed 1 MiB
    # mmap threshold returns large page buffers to the OS on free and held the
    # worker flat. The trailing underscore is part of glibc's variable name.
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert re.search(r"^ENV MALLOC_MMAP_THRESHOLD_=1048576\s*$", dockerfile, re.MULTILINE)


def test_the_mmap_threshold_is_documented_for_other_deployments():
    assert "MALLOC_MMAP_THRESHOLD_=1048576" in (ROOT / "env.example").read_text()
