"""Validating a published bundle against the contract this build knows how
to finish: the conditions, the screens, the actions and the transitions it
may carry, and nothing wider.
"""
from __future__ import annotations

import re
from typing import Any, Dict

from .contract import (
    JourneyError,
    _ALLOWED_ACTIONS,
    _ALLOWED_BODY_KEYS,
    _ALLOWED_FACTS,
    _ALLOWED_TITLE_KEYS,
    _IDENTIFIER,
    _MAX_ACTIONS,
    _MAX_CONDITION_CHILDREN,
    _MAX_SCREENS,
    _MAX_TRANSITIONS,
    _SCHEMA_VERSION,
    _SHA256,
    _UUID,
    _canonical,
    FIRST_SUCCESS_FACT,
    JOURNEY_ID,
    JOURNEY_VERSION,
    PRODUCT_ID,
)


def _validate_condition(condition: Any) -> None:
    if not isinstance(condition, dict):
        raise JourneyError("journey condition must be an object")
    kind = condition.get("kind")
    if kind in {"all", "any"}:
        if set(condition) != {"kind", "conditions"}:
            raise JourneyError("journey condition fields are invalid")
        children = condition.get("conditions")
        if not isinstance(children, list) or not children or len(children) > _MAX_CONDITION_CHILDREN:
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
    if not isinstance(screens, list) or not screens or len(screens) > _MAX_SCREENS:
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
            or len(actions) > _MAX_ACTIONS
            or any(not isinstance(action, str) for action in actions)
            or len(set(actions)) != len(actions)
            or any(action not in _ALLOWED_ACTIONS for action in actions)
            or not isinstance(transitions, list)
            or len(transitions) > _MAX_TRANSITIONS
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

