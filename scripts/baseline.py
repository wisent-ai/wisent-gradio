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
    git-archive:<tag>         reproduced from a git tag with `git archive`
    head:<full sha>           last resort: nothing published, no usable tag

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
scripts/surface.py are the whole contract and reading the unpacked .py files is
enough.

Usage:
    python3 scripts/baseline.py            # rewrite released-surface.json
    python3 scripts/baseline.py --stdout   # print it instead, change nothing
"""

from __future__ import annotations

import ast
import http
import json
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(int(False), str(HERE))

from surface import surface  # noqa: E402  (sibling script, deliberately not a package)

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

# urlopen has no timeout of its own, and a release check must fail rather than hang.
# Thirty-two seconds: two raised to the fifth power.
TIMEOUT_SECONDS = float((ONE + ONE) ** (ONE + ONE + ONE + ONE + ONE))


def fetch_json(url: str):
    """A JSON document from the registry, or None when it serves no such thing."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == http.HTTPStatus.NOT_FOUND:
            return None
        raise SystemExit(f"{url}: {error}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"{url}: {error}") from error


def latest_published(project: str):
    """The newest version the registry serves, or None if it serves the project not at all.

    Deliberately the registry's own answer rather than the declared version: those
    two differ exactly when a bump is waiting for a release, and that is the moment a
    declared-version lookup would 404 and lose the real baseline.
    """
    document = fetch_json(f"https://pypi.org/pypi/{project}/json")
    if document is None:
        return None
    return document["info"]["version"]


def choose_artifact(project: str, version: str) -> tuple:
    """The best available published artifact for one version: (marker kind, name, url).

    An sdist is preferred because it is the release's own source tree. A wheel is
    accepted only when it is pure Python, since then its .py files *are* that source;
    a platform wheel could have been built from something the extractor cannot read.
    """
    document = fetch_json(f"https://pypi.org/pypi/{project}/{version}/json")
    if document is None:
        raise SystemExit(
            f"PyPI serves {project} but not version {version}; the registry "
            "contradicted itself between two calls, so refusing to guess"
        )
    urls = document["urls"]
    for entry in urls:
        if entry["packagetype"] == "sdist":
            return MARKER_PYPI_SDIST, entry["filename"], entry["url"]
    for entry in urls:
        if entry["packagetype"] == "bdist_wheel" and entry["filename"].endswith(
            PURE_WHEEL_SUFFIX
        ):
            return MARKER_PYPI_WHEEL, entry["filename"], entry["url"]
    offered = ", ".join(sorted(entry["filename"] for entry in urls)) or "nothing"
    raise SystemExit(
        f"{project} {version} offers no sdist and no pure-Python wheel ({offered}). "
        "Recovering a surface from a platform wheel is not implemented, and "
        "silently falling back to HEAD would measure every later change against an "
        "artifact nobody released"
    )


def download(url: str, into: pathlib.Path) -> pathlib.Path:
    """Fetch one artifact into a directory, keeping its published filename."""
    target = into / url.rsplit("/", ONE)[-ONE]
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            target.write_bytes(response.read())
    except (urllib.error.URLError, OSError) as error:
        raise SystemExit(f"{url}: {error}") from error
    return target


def unpack(archive: pathlib.Path, into: pathlib.Path) -> pathlib.Path:
    """Unpack an artifact and return the directory the extractor should read.

    A wheel unpacks with the import root at the top. An sdist wraps everything in a
    single `<name>-<version>/` directory, which is that root instead.
    """
    into.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".whl"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(into)
        return into
    with tarfile.open(archive) as bundle:
        bundle.extractall(into, filter="data")
    roots = [child for child in into.iterdir() if child.is_dir()]
    if len(roots) != ONE:
        found = ", ".join(sorted(child.name for child in roots)) or "nothing"
        raise SystemExit(
            f"{archive.name}: expected one top-level directory, found {found}"
        )
    return roots[FIRST]


def git(*arguments: str) -> str:
    """One git command's output, or a loud failure."""
    result = subprocess.run(
        ("git", *arguments), cwd=ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != int(False):
        raise SystemExit(f"git {' '.join(arguments)}: {result.stderr.strip()}")
    return result.stdout.strip()


def from_registry(work: pathlib.Path) -> tuple:
    """(version, marker, tree) for the newest published release, or None if unpublished."""
    version = latest_published(PROJECT)
    if version is None:
        return None
    kind, filename, url = choose_artifact(PROJECT, version)
    tree = unpack(download(url, work), work / "unpacked")
    return version, f"{kind}:{filename}", tree


def declared_version(tree: pathlib.Path):
    """The version setup.py declares in one tree, or None if it does not say.

    Read with `ast` rather than a regex over the line, because this decides whether a
    tag is trustworthy and a near-miss match would trust the wrong thing.
    """
    setup = tree / "setup.py"
    if not setup.is_file():
        return None
    tree_of_setup = ast.parse(setup.read_text(), filename=str(setup))
    for node in ast.walk(tree_of_setup):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        named = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if named != "setup":
            continue
        for given in node.keywords:
            if given.arg == "version" and isinstance(given.value, ast.Constant):
                return given.value.value
    return None


def export_tag(tag: str, work: pathlib.Path) -> pathlib.Path:
    """The tree one tag points at, reproduced with `git archive`."""
    tree = work / f"tagged-{tag}"
    tree.mkdir(parents=True, exist_ok=True)
    archive = work / f"{tag}.tar"
    archive.write_bytes(
        subprocess.run(
            ("git", "archive", tag), cwd=ROOT, capture_output=True, check=True
        ).stdout
    )
    with tarfile.open(archive) as bundle:
        bundle.extractall(tree, filter="data")
    return tree


def from_git_tag(work: pathlib.Path) -> tuple:
    """(version, marker, tree) for the newest trustworthy tag, or None if there is none.

    A tag beats HEAD because it names a point somebody deliberately marked. But a tag
    name is only a claim: it is trusted here solely when the tree it points at
    actually declares the version the name claims. A tag that disagrees with its own
    tree has been moved or mis-signed, and filing its surface under the version it
    advertises would measure every later change against a version that never existed.
    Such a tag is reported and skipped, never quietly believed.
    """
    tags = [line for line in git("tag", "-l", "--sort=-v:refname").splitlines() if line]
    for tag in tags:
        tree = export_tag(tag, work)
        claimed = tag.lstrip("v")
        declared = declared_version(tree)
        if declared != claimed:
            print(
                f"skipping tag {tag}: its tree declares {declared}, not {claimed}",
                file=sys.stderr,
            )
            continue
        return claimed, f"{MARKER_GIT_ARCHIVE}:{tag}", tree
    return None


def from_head() -> tuple:
    """(version, marker, tree) for the working tree: the last resort."""
    sha = git("rev-parse", "HEAD")
    return sha, f"{MARKER_HEAD}:{sha}", ROOT


def prose_for(marker: str, version: str) -> str:
    """The human tail of "source": why this tier, in one sentence."""
    kind = marker.split(":", ONE)[FIRST]
    if kind == MARKER_PYPI_SDIST:
        return f"sdist of the published {version}, unpacked and read with scripts/surface.py"
    if kind == MARKER_PYPI_WHEEL:
        return (
            f"pure-Python wheel of the published {version} (that release ships no "
            "sdist), unpacked and read with scripts/surface.py"
        )
    if kind == MARKER_GIT_ARCHIVE:
        return (
            "reproduced from the tag with `git archive` because nothing is published "
            "on PyPI, and read with scripts/surface.py"
        )
    return (
        "HEAD, because nothing is published on PyPI and the repository has no tags; "
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
