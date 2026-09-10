"""What the baseline generator shares between its parts: the project, the marker
vocabulary, and how git and PyPI spell things."""

from __future__ import annotations

import os
import pathlib

ROOT = pathlib.Path(os.environ.get("GITHUB_WORKSPACE", ".")).resolve()

FIRST = int(False)
ONE = int(True)
INDENT = ONE + ONE

PROJECT = "wisent-gradio"
BASELINE_FILE = ROOT / "released-surface.json"

# The marker vocabulary. Referenced by name here and matched by family in
# .github/workflows/version-check.yml, so the baseline and its check cannot drift
# apart through prose.
MARKER_PYPI_SDIST = "pypi-sdist"
MARKER_PYPI_WHEEL = "pypi-wheel"
MARKER_GIT_ARCHIVE = "git-archive"
MARKER_HEAD = "head"
REGISTRY_CLAIMING_MARKERS = (MARKER_PYPI_SDIST, MARKER_PYPI_WHEEL)

PURE_WHEEL_SUFFIX = "-none-any.whl"

# How `git ls-remote` spells a tag, and how it spells the commit an annotated tag
# peels to. The peeled line names the same tag twice and must not be listed twice.
TAG_REF_PREFIX = "refs/tags/"
PEELED_SUFFIX = "^{}"
