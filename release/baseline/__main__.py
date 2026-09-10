"""Regenerate released-surface.json: the surface of the version actually published.

The baseline every later version decision is measured against must come from an
artifact somebody can install, not from a hand-edited file and not from HEAD when a
release exists. This script recovers it, and stamps *where it came from* into the
document so the check in .github/workflows/version-check.yml can hold the two
together.

The stamp is the first whitespace-delimited token of the "source" field, and the
tokens are minted here, by name, once:

    pypi-sdist:<filename>     recovered from a published sdist
    pypi-wheel:<filename>     recovered from a published pure-Python wheel
    git-archive:<tag>         reproduced with `git archive` from a tag origin serves
    head:<full sha>           last resort: nothing published, origin serves no tag

Tiers are tried strictly best-first in that order. A lower tier is never taken
because a higher one was inconvenient, and a tier this script does not implement is
a loud failure rather than a quiet demotion.

The marker splits into two families, and the workflow asserts both directions:
`pypi-*` claims a registry serves that exact version, so PyPI must serve it; the
others claim nothing was published, so PyPI must not serve this project at all. A
baseline that claims a release it cannot back up, or hides a release it should have
measured against, is worse than no baseline.

One trap this deliberately avoids: the baseline version is resolved from the
registry's *latest published* version, never from the version setup.py declares. The
moment a bump lands ahead of a release those two differ, and looking up the declared
version would 404 and quietly degrade the baseline to HEAD -- throwing away the real
published artifact and measuring every later change against the wrong thing.

wisent-gradio declares no console scripts (setup.py has no entry_points, and the
published wheel's dist-info carries no entry_points.txt), so the surface kinds in
release/surface.py are the whole contract and reading the unpacked .py files is
enough.

Usage:
    python3 release/baseline            # rewrite released-surface.json
    python3 release/baseline --stdout   # print it instead, change nothing
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

# Run as a directory: python3 release/baseline. The reader is the sibling
# release/surface.py, so the baseline and the check read with one program.
sys.path.insert(int(False), str(pathlib.Path(__file__).resolve().parent.parent))

from constants import (  # noqa: E402
    BASELINE_FILE, FIRST, INDENT, MARKER_GIT_ARCHIVE, MARKER_PYPI_SDIST, MARKER_PYPI_WHEEL, ONE,
    REGISTRY_CLAIMING_MARKERS,
)
from surface import surface  # noqa: E402
from tags import from_git_tag, from_head, from_registry  # noqa: E402


def prose_for(marker: str, version: str) -> str:
    """The human tail of "source": why this tier, in one sentence."""
    kind = marker.split(":", ONE)[FIRST]
    if kind == MARKER_PYPI_SDIST:
        return f"sdist of the published {version}, unpacked and read with release/surface.py"
    if kind == MARKER_PYPI_WHEEL:
        return (
            f"pure-Python wheel of the published {version} (that release ships no "
            "sdist), unpacked and read with release/surface.py"
        )
    if kind == MARKER_GIT_ARCHIVE:
        return (
            "reproduced with `git archive` from a tag origin serves, because nothing "
            "is published on PyPI, and read with release/surface.py"
        )
    return (
        "HEAD, because nothing is published on PyPI and origin serves no tags; "
        "this baseline is not installable by anyone"
    )


def build() -> dict:
    """The baseline document, from the best tier that actually exists."""
    with tempfile.TemporaryDirectory(prefix="wisent-gradio-baseline-") as scratch:
        work = pathlib.Path(scratch)
        recovered = from_registry(work) or from_git_tag(work) or from_head()
        version, marker, tree = recovered
        # Tolerant only here: a module that does not parse in an artifact somebody
        # already installed was not importable for them either, so what it declared
        # was never really on offer. Skipped modules are recorded, never swallowed.
        names, skipped = surface(tree, tolerant=marker.startswith(REGISTRY_CLAIMING_MARKERS))
        document = {
            "version": version,
            "source": f"{marker} {prose_for(marker, version)}",
            "surface": names,
        }
        if skipped:
            document["unparseable"] = skipped
        return document


def main(argv: list) -> int:
    document = build()
    rendered = json.dumps(document, indent=INDENT) + "\n"
    if "--stdout" in argv:
        sys.stdout.write(rendered)
        return int(False)
    BASELINE_FILE.write_text(rendered)
    marker = document["source"].split(" ", ONE)[FIRST]
    print(
        f"{BASELINE_FILE.name}: {document['version']} via {marker}, "
        f"{len(document['surface'])} names"
    )
    return int(False)


if __name__ == "__main__":
    sys.exit(main(sys.argv[ONE:]))
