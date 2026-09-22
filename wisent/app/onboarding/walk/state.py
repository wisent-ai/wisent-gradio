"""What this device remembers about the walk: where the state file lives,
how a subject is hashed, whether recorded progress still belongs to the
bundle in hand, and how a condition is evaluated against the evidence."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

from ..contract import JOURNEY_ID, PRODUCT_ID


def _state_path() -> Path:
    configured = os.environ.get("WISENT_GRADIO_ONBOARDING_STATE_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".wisent" / "onboarding" / f"{PRODUCT_ID}.json"


def _empty_store() -> Dict[str, Any]:
    return {"schema_version": 1, "bundles": {}, "progress": {}, "events": []}


def _read_store() -> Dict[str, Any]:
    path = _state_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _empty_store()
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return _empty_store()
    if not isinstance(value.get("bundles"), dict) or not isinstance(value.get("progress"), dict) or not isinstance(value.get("events"), list):
        return _empty_store()
    return value


def _write_store(store: Mapping[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    try:
        with handle:
            json.dump(store, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    finally:
        try:
            os.unlink(handle.name)
        except FileNotFoundError:
            pass


def _progress_key(subject_hash: str) -> str:
    return f"{PRODUCT_ID}\0{JOURNEY_ID}\0{subject_hash}"


def _subject_hash(subject: str) -> str:
    normalized = subject.strip() or "anonymous-device"
    return hashlib.sha256(f"{PRODUCT_ID}\0{normalized}".encode("utf-8")).hexdigest()


def _valid_progress(
    progress: Any,
    bundle: Mapping[str, Any],
    subject_hash: str,
) -> bool:
    if not isinstance(progress, dict):
        return False
    screen_ids = {
        screen["screen_id"] for screen in bundle["definition"]["screens"]
    }
    completed = progress.get("completed_screen_ids")
    return (
        isinstance(progress.get("attempt_id"), str)
        and _UUID.fullmatch(progress["attempt_id"]) is not None
        and progress.get("product_id") == PRODUCT_ID
        and progress.get("journey_version_id") == bundle["journey_version_id"]
        and progress.get("subject_hash") == subject_hash
        and progress.get("scope_kind") == _SCOPE_KIND
        and progress.get("current_screen_id") in screen_ids
        and isinstance(completed, list)
        and all(screen_id in screen_ids for screen_id in completed)
        and progress.get("status") in {"in_progress", "completed", "reset"}
        and isinstance(progress.get("answers"), list)
    )


def _evaluate(condition: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    kind = condition["kind"]
    if kind == "all":
        return all(_evaluate(child, evidence) for child in condition["conditions"])
    if kind == "any":
        return any(_evaluate(child, evidence) for child in condition["conditions"])
    if kind == "not":
        return not _evaluate(condition["condition"], evidence)
    fact = condition["fact"]
    present = fact in evidence and evidence[fact] is not None
    operator = condition["operator"]
    if operator == "present":
        return present
    if operator == "absent":
        return not present
    actual = evidence.get(fact)
    expected = condition.get("value")
    if operator == "eq":
        return actual == expected
    if operator == "not_eq":
        return actual != expected
    if operator == "contains":
        return isinstance(actual, (list, tuple)) and expected in actual
    if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
        return False
    return {
        "gt": actual > expected,
        "gte": actual >= expected,
        "lt": actual < expected,
        "lte": actual <= expected,
    }[operator]

