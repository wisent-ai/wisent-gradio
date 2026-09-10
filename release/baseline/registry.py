"""The PyPI half: the version the index serves, the artifact chosen for it, and
the surface recovered from that artifact."""

from __future__ import annotations

import http
import json
import pathlib
import tarfile
import urllib.error
import urllib.request
import zipfile

from constants import FIRST, MARKER_PYPI_SDIST, MARKER_PYPI_WHEEL, ONE, PURE_WHEEL_SUFFIX


def fetch_json(url: str):
    """A JSON document from the registry, or None when it serves no such thing."""
    try:
        with urllib.request.urlopen(url) as response:
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
        with urllib.request.urlopen(url) as response:
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


