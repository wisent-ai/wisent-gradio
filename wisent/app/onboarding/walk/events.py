"""What a walk does to the store and to the control plane: assigning an
experiment, advancing a screen, saving progress, and emitting each event
exactly once even when the plane could not be reached at the time."""
from __future__ import annotations

import uuid
from typing import Any, Dict, Mapping, Optional

from ..contract import JOURNEY_VERSION, JourneyError, _EVENT_NAMES, _SCOPE_KIND, _utc_now
from .state import _evaluate, _progress_key, _write_store
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import JourneyRuntime

def assign_experiment_if_needed(runtime: "JourneyRuntime") -> None:
    contract = runtime.bundle["definition"].get("experiment_contract")
    if not contract or runtime.progress.get("variant_id") or not runtime.transport.available:
        return
    try:
        assignment = runtime.transport.assign_experiment(runtime.subject_hash)
    except JourneyError:
        return
    if not isinstance(assignment, dict):
        return
    variant = assignment.get("variant")
    experiment_id = assignment.get("experimentId")
    if variant not in contract.get("eligible_variant_ids", []) or not isinstance(experiment_id, str):
        return
    runtime.progress["variant_id"] = variant
    runtime.progress["experiment_id"] = experiment_id

def advance(runtime: "JourneyRuntime", store: Dict[str, Any], evidence: Mapping[str, Any]) -> bool:
    screen = runtime.screen
    completion = screen.get("completion_evidence")
    if completion and not _evaluate(completion, evidence):
        return False
    transitions = sorted(screen["transitions"], key=lambda item: item["priority"])
    selected = None
    for transition in transitions:
        condition = transition.get("condition")
        target = next(
            item for item in runtime.bundle["definition"]["screens"]
            if item["screen_id"] == transition["next_screen_id"]
        )
        target_entry = target.get("entry_conditions")
        if (
            (condition is None or _evaluate(condition, evidence))
            and (target_entry is None or _evaluate(target_entry, evidence))
        ):
            selected = transition
            break
    if selected is None and screen.get("fallback_screen_id"):
        fallback = next(
            item for item in runtime.bundle["definition"]["screens"]
            if item["screen_id"] == screen["fallback_screen_id"]
        )
        fallback_entry = fallback.get("entry_conditions")
        if fallback_entry is None or _evaluate(fallback_entry, evidence):
            selected = {
                "next_screen_id": fallback["screen_id"],
                "reason_code": "fallback_evidence_unavailable",
            }
    if selected is None:
        return False
    completed_screen = runtime.progress["current_screen_id"]
    runtime.progress["completed_screen_ids"] = list(
        dict.fromkeys(runtime.progress["completed_screen_ids"] + [completed_screen])
    )
    runtime.progress["current_screen_id"] = selected["next_screen_id"]
    runtime.progress["evidence_revision"] = JOURNEY_VERSION
    save_progress(runtime, store)
    emit(
        runtime,
        store,
        "onboarding_step_completed",
        {},
        completed_screen,
        selected,
    )
    return True

def save_progress(runtime: "JourneyRuntime", store: Dict[str, Any]) -> None:
    store["progress"][_progress_key(runtime.subject_hash)] = runtime.progress
    _write_store(store)

def event(
    runtime: "JourneyRuntime",
    event_name: str,
    properties: Mapping[str, Any],
    screen_id: Optional[str] = None,
    decision: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if event_name not in _EVENT_NAMES:
        raise JourneyError("unsupported onboarding event")
    event = {
        "event_id": str(uuid.uuid4()),
        "event_name": event_name,
        "attempt_id": runtime.progress["attempt_id"],
        "product_id": PRODUCT_ID,
        "journey_version_id": runtime.progress["journey_version_id"],
        "subject_hash": runtime.subject_hash,
        "scope_kind": _SCOPE_KIND,
        "screen_id": screen_id or runtime.progress["current_screen_id"],
        "occurred_at": _utc_now(),
        "evidence_revision": JOURNEY_VERSION,
        "properties": dict(properties),
        "answers": runtime.progress.get("answers", []),
    }
    if runtime.progress.get("experiment_id"):
        event["experiment_id"] = runtime.progress["experiment_id"]
    if runtime.progress.get("variant_id"):
        event["variant_id"] = runtime.progress["variant_id"]
    if decision:
        event["selected_next_screen_id"] = decision["next_screen_id"]
        event["reason_code"] = decision["reason_code"]
    return event

def emit(
    runtime: "JourneyRuntime",
    store: Dict[str, Any],
    event_name: str,
    properties: Mapping[str, Any],
    screen_id: Optional[str] = None,
    decision: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = event(runtime, event_name, properties, screen_id, decision)
    store["events"].append(payload)
    _write_store(store)
    if not runtime.transport.available:
        return
    try:
        runtime.transport.collect_event(payload)
    except JourneyError:
        return
    store["events"] = [item for item in store["events"] if item.get("event_id") != payload["event_id"]]
    _write_store(store)

def flush(runtime: "JourneyRuntime", store: Dict[str, Any]) -> None:
    if not runtime.transport.available:
        return
    for event in list(store["events"]):
        try:
            runtime.transport.collect_event(event)
        except JourneyError:
            return
        store["events"] = [item for item in store["events"] if item.get("event_id") != event.get("event_id")]
        _write_store(store)
