"""Product-local Echo journey adapter for Wisent's Gradio first-use flow."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import urllib.error
import urllib.request
from urllib.parse import urlparse
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

PRODUCT_ID = "wisent-gradio"
JOURNEY_ID = "first-use"
JOURNEY_VERSION = "2026-08-04.1"
FIRST_SUCCESS_FACT = "representation_result_observed"
_CLIENT_ID = "wisent-gradio"
_TOKEN_ENV = "WISENT_GRADIO_STADO_INTEGRATION_TOKEN"
_BASE_URL_ENV = "STADO_INTEGRATION_API_URL"
_SCOPE_KIND = "device"
_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,127}$")
_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_FACTS = frozenset(
    {"representation_operations_introduced", FIRST_SUCCESS_FACT}
)
_ALLOWED_TITLE_KEYS = frozenset(
    {"welcome.title", "result.title"}
)
_ALLOWED_BODY_KEYS = frozenset(
    {"welcome.body", "result.body"}
)
_ALLOWED_ACTIONS = frozenset({"open_steering_visualization"})
_EVENT_NAMES = frozenset(
    {
        "onboarding_started",
        "onboarding_step_viewed",
        "onboarding_step_completed",
        "onboarding_resumed",
        "onboarding_first_action_completed",
        "onboarding_first_success_observed",
        "onboarding_completed",
    }
)
_LOCK = threading.RLock()


class JourneyError(RuntimeError):
    """Raised when central journey data or transport violates the contract."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fallback_bundle() -> Dict[str, Any]:
    definition = {
        "schema_version": _SCHEMA_VERSION,
        "product_id": PRODUCT_ID,
        "journey_id": JOURNEY_ID,
        "journey_version": JOURNEY_VERSION,
        "entry_screen_id": "learn-representations",
        "first_success_fact": FIRST_SUCCESS_FACT,
        "published_at": "2026-08-04T00:00:00Z",
        "source_revision": hashlib.sha1(
            f"{PRODUCT_ID}:{JOURNEY_ID}:{JOURNEY_VERSION}".encode("utf-8")
        ).hexdigest(),
        "screens": [
            {
                "screen_id": "learn-representations",
                "screen_kind": "explanation",
                "title_key": "welcome.title",
                "body_key": "welcome.body",
                "required": True,
                "completion_evidence": {
                    "kind": "fact",
                    "fact": "representation_operations_introduced",
                    "operator": "present",
                },
                "actions": ["open_steering_visualization"],
                "transitions": [
                    {
                        "next_screen_id": "observe-representation-result",
                        "reason_code": "representation_operation_selected",
                        "priority": 0,
                        "condition": {
                            "kind": "fact",
                            "fact": "representation_operations_introduced",
                            "operator": "present",
                        },
                    }
                ],
                "presentation": {"placement": "gradio_header", "accent": "mint"},
            },
            {
                "screen_id": "observe-representation-result",
                "screen_kind": "first_success",
                "title_key": "result.title",
                "body_key": "result.body",
                "required": True,
                "completion_evidence": {
                    "kind": "fact",
                    "fact": FIRST_SUCCESS_FACT,
                    "operator": "present",
                },
                "actions": ["open_steering_visualization"],
                "transitions": [],
                "presentation": {"placement": "gradio_header", "accent": "mint"},
            },
        ],
        "analytics_contract": {
            "contract_version": "1",
            "surface": "gradio_web",
            "exposure_event": "onboarding_step_viewed",
            "primary_action_event": "onboarding_first_action_completed",
            "completion_event": "onboarding_completed",
            "first_success_event": "onboarding_first_success_observed",
        },
    }
    canonical_definition = _canonical(definition)
    return {
        "journey_version_id": str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{PRODUCT_ID}:{JOURNEY_ID}:{JOURNEY_VERSION}")
        ),
        "definition": definition,
        "canonical_definition": canonical_definition,
        "content_sha256": hashlib.sha256(canonical_definition.encode("utf-8")).hexdigest(),
        "source_revision": definition["source_revision"],
    }


CANONICAL_FALLBACK = _fallback_bundle()


def _validate_condition(condition: Any) -> None:
    if not isinstance(condition, dict):
        raise JourneyError("journey condition must be an object")
    kind = condition.get("kind")
    if kind in {"all", "any"}:
        if set(condition) != {"kind", "conditions"}:
            raise JourneyError("journey condition fields are invalid")
        children = condition.get("conditions")
        if not isinstance(children, list) or not children or len(children) > 32:
            raise JourneyError("journey condition list is invalid")
        for child in children:
            _validate_condition(child)
        return
    if kind == "not":
        if set(condition) != {"kind", "condition"}:
            raise JourneyError("journey condition fields are invalid")
        _validate_condition(condition.get("condition"))
        return
    if kind != "fact" or condition.get("fact") not in _ALLOWED_FACTS:
        raise JourneyError("journey condition uses an unsupported fact")
    operator = condition.get("operator")
    if operator not in {
        "present", "absent", "eq", "not_eq", "contains", "gt", "gte", "lt", "lte"
    }:
        raise JourneyError("journey condition operator is invalid")
    expected_fields = (
        {"kind", "fact", "operator"}
        if operator in {"present", "absent"}
        else {"kind", "fact", "operator", "value"}
    )
    if set(condition) != expected_fields:
        raise JourneyError("journey condition fields are invalid")


def validate_bundle(bundle: Any) -> Dict[str, Any]:
    """Validate an Echo bundle and its product-owned presentation contract."""
    if not isinstance(bundle, dict):
        raise JourneyError("journey bundle envelope is invalid")
    version_id = bundle.get("journey_version_id")
    canonical_definition = bundle.get("canonical_definition")
    content_hash = bundle.get("content_sha256")
    if not isinstance(version_id, str) or not _UUID.fullmatch(version_id):
        raise JourneyError("journey version id is invalid")
    if not isinstance(content_hash, str) or not _SHA256.fullmatch(content_hash):
        raise JourneyError("journey content hash is invalid")
    if not isinstance(canonical_definition, str):
        raise JourneyError("journey canonical definition is missing")
    definition = bundle.get("definition")
    if not isinstance(definition, dict) or _canonical(definition) != canonical_definition:
        raise JourneyError("journey canonical definition does not match")
    if hashlib.sha256(canonical_definition.encode("utf-8")).hexdigest() != content_hash:
        raise JourneyError("journey content hash does not match")
    if (
        definition.get("schema_version") != _SCHEMA_VERSION
        or definition.get("product_id") != PRODUCT_ID
        or definition.get("journey_id") != JOURNEY_ID
        or definition.get("journey_version") != JOURNEY_VERSION
        or definition.get("first_success_fact") != FIRST_SUCCESS_FACT
    ):
        raise JourneyError("journey identity is invalid")
    definition_fields = {
        "schema_version", "product_id", "journey_id", "journey_version",
        "entry_screen_id", "first_success_fact", "published_at",
        "source_revision", "screens", "analytics_contract",
    }
    if "experiment_contract" in definition:
        definition_fields.add("experiment_contract")
    if set(definition) != definition_fields:
        raise JourneyError("journey definition fields are invalid")
    source_revision = definition.get("source_revision")
    if (
        not isinstance(source_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
        or bundle.get("source_revision") != source_revision
    ):
        raise JourneyError("journey source revision is invalid")
    published_at = definition.get("published_at")
    try:
        datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise JourneyError("journey publication time is invalid") from exc
    analytics = definition.get("analytics_contract")
    if analytics != {
        "contract_version": "1",
        "surface": "gradio_web",
        "exposure_event": "onboarding_step_viewed",
        "primary_action_event": "onboarding_first_action_completed",
        "completion_event": "onboarding_completed",
        "first_success_event": "onboarding_first_success_observed",
    }:
        raise JourneyError("journey analytics contract is invalid")
    experiment = definition.get("experiment_contract")
    if experiment is not None:
        expected_experiment_fields = {
            "experiment_id", "control_variant_id", "eligible_variant_ids",
            "assignment_unit", "reward_event", "guardrail_events", "owner",
            "kill_switch",
        }
        eligible = experiment.get("eligible_variant_ids") if isinstance(experiment, dict) else None
        if (
            not isinstance(experiment, dict)
            or set(experiment) != expected_experiment_fields
            or not isinstance(eligible, list)
            or not eligible
            or any(not isinstance(variant, str) or not _IDENTIFIER.fullmatch(variant) for variant in eligible)
            or len(set(eligible)) != len(eligible)
            or experiment.get("control_variant_id") not in eligible
            or experiment.get("assignment_unit") != "device"
            or not isinstance(experiment.get("kill_switch"), bool)
        ):
            raise JourneyError("journey experiment contract is invalid")
    screens = definition.get("screens")
    if not isinstance(screens, list) or not screens or len(screens) > 128:
        raise JourneyError("journey screen graph is invalid")
    by_id: Dict[str, Dict[str, Any]] = {}
    has_success_terminal = False
    for screen in screens:
        if not isinstance(screen, dict):
            raise JourneyError("journey screen is invalid")
        allowed_screen_fields = {
            "screen_id", "screen_kind", "title_key", "body_key", "required",
            "actions", "transitions", "presentation",
        }
        allowed_screen_fields.update(
            field for field in ("entry_conditions", "completion_evidence", "fallback_screen_id")
            if field in screen
        )
        if set(screen) != allowed_screen_fields:
            raise JourneyError("journey screen fields are invalid")
        screen_kind = screen.get("screen_kind")
        presentation = screen.get("presentation")
        if (
            not isinstance(screen_kind, str)
            or not _IDENTIFIER.fullmatch(screen_kind)
            or not isinstance(screen.get("required"), bool)
            or not isinstance(presentation, dict)
            or any(not isinstance(value, (str, int, float, bool)) and value is not None for value in presentation.values())
        ):
            raise JourneyError("journey screen presentation is invalid")
        screen_id = screen.get("screen_id")
        if not isinstance(screen_id, str) or not _IDENTIFIER.fullmatch(screen_id) or screen_id in by_id:
            raise JourneyError("journey screen id is invalid")
        if screen.get("title_key") not in _ALLOWED_TITLE_KEYS or screen.get("body_key") not in _ALLOWED_BODY_KEYS:
            raise JourneyError("journey requested content not owned by this product")
        actions = screen.get("actions")
        transitions = screen.get("transitions")
        if (
            not isinstance(actions, list)
            or len(actions) > 16
            or any(not isinstance(action, str) for action in actions)
            or len(set(actions)) != len(actions)
            or any(action not in _ALLOWED_ACTIONS for action in actions)
            or not isinstance(transitions, list)
            or len(transitions) > 32
        ):
            raise JourneyError("journey screen actions or transitions are invalid")
        for field in ("entry_conditions", "completion_evidence"):
            if field in screen:
                _validate_condition(screen[field])
        if not transitions and screen.get("completion_evidence") == {
            "kind": "fact", "fact": FIRST_SUCCESS_FACT, "operator": "present"
        }:
            has_success_terminal = True
        by_id[screen_id] = screen
    if definition.get("entry_screen_id") not in by_id or not has_success_terminal:
        raise JourneyError("journey entry or first-success terminal is missing")
    for screen in screens:
        fallback = screen.get("fallback_screen_id")
        if fallback is not None and fallback not in by_id:
            raise JourneyError("journey fallback target is missing")
        for transition in screen["transitions"]:
            if not isinstance(transition, dict):
                raise JourneyError("journey transition is invalid")
            allowed_transition_fields = {"next_screen_id", "reason_code", "priority"}
            if "condition" in transition:
                allowed_transition_fields.add("condition")
            reason_code = transition.get("reason_code")
            if (
                set(transition) != allowed_transition_fields
                or transition.get("next_screen_id") not in by_id
            ):
                raise JourneyError("journey transition target is missing")
            if (
                not isinstance(transition.get("priority"), int)
                or transition["priority"] < 0
                or not isinstance(reason_code, str)
                or not _IDENTIFIER.fullmatch(reason_code)
            ):
                raise JourneyError("journey transition metadata is invalid")
            if "condition" in transition:
                _validate_condition(transition["condition"])
    return bundle


class StadoTransport:
    """Synchronous Stado transport used by Gradio callback workers."""

    def __init__(self) -> None:
        self._base_url = os.environ.get(_BASE_URL_ENV, "").strip().rstrip("/")
        self._token = os.environ.get(_TOKEN_ENV, "").strip()
        self._failed = False
        if self._base_url:
            parsed = urlparse(self._base_url)
            if (
                parsed.scheme != "https"
                or not parsed.netloc
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
                or parsed.path not in {"", "/"}
            ):
                self._failed = True

    @property
    def available(self) -> bool:
        return bool(self._base_url and self._token and not self._failed)

    def _post(self, operation: str, body: Mapping[str, Any]) -> Any:
        if not self.available:
            raise JourneyError("Stado transport is not configured")
        endpoint = (
            f"{self._base_url}/integration/{_CLIENT_ID}/onboarding/"
            f"{PRODUCT_ID}/{operation}"
        )
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=2.5) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError, urllib.error.URLError) as exc:
            self._failed = True
            raise JourneyError("Stado onboarding request failed") from exc
        if not isinstance(envelope, dict) or envelope.get("ok") is not True or "result" not in envelope:
            self._failed = True
            raise JourneyError("Stado onboarding response is invalid")
        return envelope["result"]

    def read_bundle(self) -> Dict[str, Any]:
        return self._post(
            "bundle.read",
            {
                "product_id": PRODUCT_ID,
                "journey_id": JOURNEY_ID,
                "journey_version": JOURNEY_VERSION,
                "if_none_match": None,
            },
        )

    def read_state(self, progress: Mapping[str, Any]) -> Any:
        return self._post(
            "state.read",
            {
                "product_id": PRODUCT_ID,
                "attempt_id": progress["attempt_id"],
                "subject_hash": progress["subject_hash"],
            },
        )

    def assign_experiment(self, subject_hash: str) -> Any:
        return self._post(
            "experiments.assign",
            {
                "product_id": PRODUCT_ID,
                "app_id": PRODUCT_ID,
                "platform": "web",
                "surface": "gradio_web",
                "subject": subject_hash,
            },
        )

    def collect_event(self, event: Mapping[str, Any]) -> None:
        self._post("events.collect", event)


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
            self._assign_experiment_if_needed()
            store["progress"][key] = self.progress
            _write_store(store)
            self._flush(store)
            if resumed and self.transport.available:
                try:
                    self.transport.read_state(self.progress)
                except JourneyError:
                    pass
            if self.progress["status"] != "completed":
                self._emit(
                    store,
                    "onboarding_resumed" if resumed else "onboarding_started",
                    {},
                )
                self._emit(store, "onboarding_step_viewed", {})
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
                self._emit(
                    store,
                    "onboarding_first_action_completed",
                    {"action": "open_steering_visualization"},
                )
            self._advance(store, evidence)
            if self.progress["current_screen_id"] != current_screen_id:
                self._emit(store, "onboarding_step_viewed", {})
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
                self._emit(
                    store,
                    "onboarding_first_action_completed",
                    {"action": command_name},
                )
            for _ in self.bundle["definition"]["screens"]:
                if not self.screen.get("transitions"):
                    break
                if not self._advance(store, evidence):
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
            self._save_progress(store)
            properties = {
                "fact": FIRST_SUCCESS_FACT,
                "command": command_name,
                "rendered": True,
            }
            self._emit(store, "onboarding_step_completed", properties, completed_screen)
            self._emit(store, "onboarding_first_success_observed", properties, completed_screen)
            self._emit(store, "onboarding_completed", properties, completed_screen)
            return self

    def _assign_experiment_if_needed(self) -> None:
        contract = self.bundle["definition"].get("experiment_contract")
        if not contract or self.progress.get("variant_id") or not self.transport.available:
            return
        try:
            assignment = self.transport.assign_experiment(self.subject_hash)
        except JourneyError:
            return
        if not isinstance(assignment, dict):
            return
        variant = assignment.get("variant")
        experiment_id = assignment.get("experimentId")
        if variant not in contract.get("eligible_variant_ids", []) or not isinstance(experiment_id, str):
            return
        self.progress["variant_id"] = variant
        self.progress["experiment_id"] = experiment_id

    def _advance(self, store: Dict[str, Any], evidence: Mapping[str, Any]) -> bool:
        screen = self.screen
        completion = screen.get("completion_evidence")
        if completion and not _evaluate(completion, evidence):
            return False
        transitions = sorted(screen["transitions"], key=lambda item: item["priority"])
        selected = None
        for transition in transitions:
            condition = transition.get("condition")
            target = next(
                item for item in self.bundle["definition"]["screens"]
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
                item for item in self.bundle["definition"]["screens"]
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
        completed_screen = self.progress["current_screen_id"]
        self.progress["completed_screen_ids"] = list(
            dict.fromkeys(self.progress["completed_screen_ids"] + [completed_screen])
        )
        self.progress["current_screen_id"] = selected["next_screen_id"]
        self.progress["evidence_revision"] = JOURNEY_VERSION
        self._save_progress(store)
        self._emit(
            store,
            "onboarding_step_completed",
            {},
            completed_screen,
            selected,
        )
        return True

    def _save_progress(self, store: Dict[str, Any]) -> None:
        store["progress"][_progress_key(self.subject_hash)] = self.progress
        _write_store(store)

    def _event(
        self,
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
            "attempt_id": self.progress["attempt_id"],
            "product_id": PRODUCT_ID,
            "journey_version_id": self.progress["journey_version_id"],
            "subject_hash": self.subject_hash,
            "scope_kind": _SCOPE_KIND,
            "screen_id": screen_id or self.progress["current_screen_id"],
            "occurred_at": _utc_now(),
            "evidence_revision": JOURNEY_VERSION,
            "properties": dict(properties),
            "answers": self.progress.get("answers", []),
        }
        if self.progress.get("experiment_id"):
            event["experiment_id"] = self.progress["experiment_id"]
        if self.progress.get("variant_id"):
            event["variant_id"] = self.progress["variant_id"]
        if decision:
            event["selected_next_screen_id"] = decision["next_screen_id"]
            event["reason_code"] = decision["reason_code"]
        return event

    def _emit(
        self,
        store: Dict[str, Any],
        event_name: str,
        properties: Mapping[str, Any],
        screen_id: Optional[str] = None,
        decision: Optional[Mapping[str, Any]] = None,
    ) -> None:
        event = self._event(event_name, properties, screen_id, decision)
        store["events"].append(event)
        _write_store(store)
        if not self.transport.available:
            return
        try:
            self.transport.collect_event(event)
        except JourneyError:
            return
        store["events"] = [item for item in store["events"] if item.get("event_id") != event["event_id"]]
        _write_store(store)

    def _flush(self, store: Dict[str, Any]) -> None:
        if not self.transport.available:
            return
        for event in list(store["events"]):
            try:
                self.transport.collect_event(event)
            except JourneyError:
                return
            store["events"] = [item for item in store["events"] if item.get("event_id") != event.get("event_id")]
            _write_store(store)
