"""One walk through the first-use journey for one browser device: routing
it, persisting it, and reporting each event exactly once."""
from __future__ import annotations

import uuid
from typing import Any, Dict, Mapping, Optional

from ..contract import (
    CANONICAL_FALLBACK,
    FIRST_SUCCESS_FACT,
    JOURNEY_ID,
    JourneyError,
    PRODUCT_ID,
    _EVENT_NAMES,
    _LOCK,
    _SCOPE_KIND,
    _utc_now,
)
from .state import (
    _empty_store,
    _evaluate,
    _progress_key,
    _read_store,
    _subject_hash,
    _valid_progress,
    _write_store,
)
from .events import (
    advance,
    assign_experiment_if_needed,
    emit,
    flush,
    save_progress,
)
from ..transport import StadoTransport
from ..validate import validate_bundle


class JourneyRuntime:
    """Route, persist, and report the first-use journey for one browser device."""

    def __init__(self, browser_subject: str) -> None:
        self.subject_hash = _subject_hash(browser_subject)
        self.transport = StadoTransport()
        self.bundle: Dict[str, Any] = CANONICAL_FALLBACK
        self.progress: Dict[str, Any] = {}

    def start(self) -> "JourneyRuntime":
        with _LOCK:
            store = _read_store()
            bundle = None
            if self.transport.available:
                try:
                    bundle = validate_bundle(self.transport.read_bundle())
                    store["bundles"][f"{PRODUCT_ID}\0{JOURNEY_ID}"] = bundle
                except JourneyError:
                    bundle = None
            if bundle is None:
                cached = store["bundles"].get(f"{PRODUCT_ID}\0{JOURNEY_ID}")
                try:
                    bundle = validate_bundle(cached) if cached is not None else None
                except JourneyError:
                    bundle = None
            self.bundle = bundle or validate_bundle(CANONICAL_FALLBACK)
            key = _progress_key(self.subject_hash)
            stored = store["progress"].get(key)
            resumed = (
                _valid_progress(stored, self.bundle, self.subject_hash)
                and stored.get("status") != "reset"
            )
            if resumed:
                self.progress = stored
            else:
                self.progress = {
                    "attempt_id": str(uuid.uuid4()),
                    "product_id": PRODUCT_ID,
                    "journey_version_id": self.bundle["journey_version_id"],
                    "subject_hash": self.subject_hash,
                    "scope_kind": _SCOPE_KIND,
                    "current_screen_id": self.bundle["definition"]["entry_screen_id"],
                    "completed_screen_ids": [],
                    "status": "in_progress",
                    "evidence_revision": JOURNEY_VERSION,
                    "answers": [],
                }
            assign_experiment_if_needed(self)
            store["progress"][key] = self.progress
            _write_store(store)
            flush(self, store)
            if resumed and self.transport.available:
                try:
                    self.transport.read_state(self.progress)
                except JourneyError:
                    pass
            if self.progress["status"] != "completed":
                emit(
                    self,
                    store,
                    "onboarding_resumed" if resumed else "onboarding_started",
                    {},
                )
                emit(self, store, "onboarding_step_viewed", {})
            return self

    def open_existing(self) -> "JourneyRuntime":
        """Open persisted progress without producing a page-exposure event."""
        with _LOCK:
            store = _read_store()
            cached = store["bundles"].get(f"{PRODUCT_ID}\0{JOURNEY_ID}")
            try:
                self.bundle = validate_bundle(cached) if cached is not None else validate_bundle(CANONICAL_FALLBACK)
            except JourneyError:
                self.bundle = validate_bundle(CANONICAL_FALLBACK)
            progress = store["progress"].get(_progress_key(self.subject_hash))
            if not _valid_progress(progress, self.bundle, self.subject_hash):
                return self.start()
            self.progress = progress
            return self

    @property
    def screen(self) -> Dict[str, Any]:
        screen_id = self.progress["current_screen_id"]
        for screen in self.bundle["definition"]["screens"]:
            if screen["screen_id"] == screen_id:
                return screen
        raise JourneyError("persisted journey screen is missing")

    def primary_action(self) -> "JourneyRuntime":
        if self.progress.get("status") == "completed":
            return self
        evidence = {"representation_operations_introduced": True}
        with _LOCK:
            store = _read_store()
            current_screen_id = self.progress["current_screen_id"]
            if not self.progress["completed_screen_ids"]:
                emit(
                    self,
                    store,
                    "onboarding_first_action_completed",
                    {"action": "open_steering_visualization"},
                )
            advance(self, store, evidence)
            if self.progress["current_screen_id"] != current_screen_id:
                emit(self, store, "onboarding_step_viewed", {})
            return self

    def observe_representation_result(self, command_name: str) -> "JourneyRuntime":
        if self.progress.get("status") == "completed":
            return self
        evidence = {
            "representation_operations_introduced": True,
            FIRST_SUCCESS_FACT: True,
        }
        with _LOCK:
            store = _read_store()
            if not self.progress["completed_screen_ids"]:
                emit(
                    self,
                    store,
                    "onboarding_first_action_completed",
                    {"action": command_name},
                )
            for _ in self.bundle["definition"]["screens"]:
                if not self.screen.get("transitions"):
                    break
                if not advance(self, store, evidence):
                    return self
            else:
                return self
            completion = self.screen.get("completion_evidence")
            if not completion or not _evaluate(completion, evidence):
                return self
            completed_screen = self.progress["current_screen_id"]
            self.progress["completed_screen_ids"] = list(
                dict.fromkeys(self.progress["completed_screen_ids"] + [completed_screen])
            )
            self.progress["status"] = "completed"
            self.progress["evidence_revision"] = JOURNEY_VERSION
            save_progress(self, store)
            properties = {
                "fact": FIRST_SUCCESS_FACT,
                "command": command_name,
                "rendered": True,
            }
            emit(self, store, "onboarding_step_completed", properties, completed_screen)
            emit(self, store, "onboarding_first_success_observed", properties, completed_screen)
            emit(self, store, "onboarding_completed", properties, completed_screen)
            return self

