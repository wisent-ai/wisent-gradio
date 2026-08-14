#!/usr/bin/env python3
import hashlib
import json
import os
import pathlib
import subprocess
import tarfile
import tempfile


def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing {name}")
    return value

archive = pathlib.Path(required("WISENT_RELEASE_ARCHIVE"))
digest = required("WISENT_RELEASE_SHA256")
if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
    raise RuntimeError("canonical Stado archive digest mismatch")
with tempfile.TemporaryDirectory() as temporary:
    root = pathlib.Path(temporary)
    with tarfile.open(archive, "r:gz") as bundle:
        members = [m for m in bundle.getmembers() if m.isfile() and m.name.startswith("python-distributions/")]
        bundle.extractall(root, members=members, filter="data")
    artifacts = sorted(str(path) for path in (root / "python-distributions").iterdir())
    if not artifacts:
        raise RuntimeError("canonical archive contains no Python distributions")

    # PyPI refuses a distribution whose metadata carries a direct reference, and
    # says so with a bare 400 "Can't have direct dependency" after the upload has
    # already started. `twine check` does not catch it either (pypa/twine#726), so
    # without this the first symptom is a failed release rather than a failed
    # build. This product depends on wisent-errors by pinned git URL, which is the
    # fleet's rule for a package that is not on any index; the two are simply
    # incompatible until someone decides where wisent-errors is published.
    for artifact in artifacts:
        if not artifact.endswith(".tar.gz"):
            continue
        with tarfile.open(artifact, "r:gz") as distribution:
            for member in distribution.getmembers():
                if not member.name.endswith("PKG-INFO"):
                    continue
                extracted = distribution.extractfile(member)
                if extracted is None:
                    continue
                for line in extracted.read().decode("utf-8", "replace").splitlines():
                    if line.startswith("Requires-Dist:") and "@ " in line:
                        raise RuntimeError(
                            "this distribution declares a direct dependency, which PyPI"
                            f" rejects: {line.strip()!r}. Either publish that dependency"
                            " to an index and require it by name and version, or keep this"
                            " product off PyPI. Uploading will fail with HTTP 400."
                        )
    env = os.environ.copy()
    env["TWINE_USERNAME"] = "__token__"
    env["TWINE_PASSWORD"] = required("PYPI_TOKEN")
    completed = subprocess.run(["python3", "-m", "twine", "upload", "--non-interactive", *artifacts], check=True, capture_output=True, text=True, env=env)
receipt = {
    "schema_version": 1,
    "channel": "pypi",
    "product": required("WISENT_PRODUCT"),
    "version": required("WISENT_VERSION"),
    "release_uri": required("WISENT_RELEASE_URI"),
    "release_sha256": digest,
    "provider_output": completed.stdout.strip(),
}
out = pathlib.Path(required("WISENT_OUTPUT_DIR"))
out.mkdir(parents=True, exist_ok=True)
(out / "pypi-receipt.json").write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
