"""What this product's first-use journey is: the identity it is published
under, the bounds a journey graph is held to, and the definition shipped
with the app for a machine that cannot reach the control plane.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict

PRODUCT_ID = "wisent-gradio"
JOURNEY_ID = "first-use"
JOURNEY_VERSION = "2026-08-04.1"
FIRST_SUCCESS_FACT = "representation_result_observed"
_CLIENT_ID = "wisent-gradio"
_TOKEN_ENV = "WISENT_GRADIO_STADO_INTEGRATION_TOKEN"
_BASE_URL_ENV = "STADO_INTEGRATION_API_URL"
_SCOPE_KIND = "device"
_SCHEMA_VERSION = 1
# Bounds of a journey graph the app will walk: nested conditions, screens, actions and transitions per screen.
_MAX_CONDITION_CHILDREN = 32
_MAX_SCREENS = 128
_MAX_ACTIONS = 16
_MAX_TRANSITIONS = 32
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

