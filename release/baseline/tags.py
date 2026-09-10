"""The git half: the version the tree declares, the tags the remote holds, the
best tag exported as an archive, and the working tree when nothing was ever
published."""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys
import tarfile

from constants import (
    MARKER_GIT_ARCHIVE, MARKER_HEAD, ONE, PEELED_SUFFIX, PROJECT, ROOT, TAG_REF_PREFIX,
)
from registry import choose_artifact, download, latest_published, unpack


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


def remote_tags() -> list:
    """Tag names as `origin` serves them, newest version name first.

    Every tag question here goes to the remote, for two independent reasons and
    each one alone would be enough.

    Blindness: `actions/checkout@v4` fetches one commit and no tags, so a local
    listing comes back empty on a runner however many tags origin holds. The
    generator would then report `head:` as the best tier at exactly the moment a
    tag appeared that it exists to notice -- and the empty local listing agrees
    with an empty listing collected by hand, so two blind reads look like
    corroboration.

    Misattribution: a fork or an imported mirror shares the upstream's object
    store, so a local listing can name tags that were never released *here*. That
    would file this repository's baseline under a stranger's release -- the
    false-positive mirror of the same defect.

    `git()` exits loudly when ls-remote fails, which is the point: an unreachable
    remote must never read as "origin serves no tags".
    """
    listing = git("ls-remote", "--tags", "--sort=-v:refname", "origin")
    names = []
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) < ONE + ONE:
            continue
        ref = parts[-ONE]
        if ref.startswith(TAG_REF_PREFIX) and not ref.endswith(PEELED_SUFFIX):
            names.append(ref[len(TAG_REF_PREFIX) :])
    return names


def held_locally(tag: str) -> bool:
    """Whether this clone holds the tag's commit, and not merely its name."""
    result = subprocess.run(
        ("git", "rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == int(False)


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

    The tags are the ones `origin` serves, never the ones this working copy happens
    to hold: see remote_tags() for why a local listing answers a different question
    on a runner and a wrong one on a fork.

    A tag beats HEAD because it names a point somebody deliberately marked. But a tag
    name is only a claim: it is trusted here solely when the tree it points at
    actually declares the version the name claims. A tag that disagrees with its own
    tree has been moved or mis-signed, and filing its surface under the version it
    advertises would measure every later change against a version that never existed.
    Such a tag is reported and skipped, never quietly believed.
    """
    for tag in remote_tags():
        if not held_locally(tag):
            # Refuse rather than skip. Skipping would walk past a real tag and
            # settle on `head:`, which is the quiet demotion this script exists to
            # make impossible -- and it is the ordinary state of a checkout, not an
            # exotic one.
            raise SystemExit(
                f"origin serves tag {tag} but this clone does not hold its commit, so "
                f"the tree it points at cannot be reproduced and the best tier is "
                f"unknown. Fetch first: git fetch --force --tags --unshallow || "
                f"git fetch --force --tags"
            )
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


