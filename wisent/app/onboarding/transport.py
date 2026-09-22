"""Reaching Stado from a Gradio callback worker: one synchronous transport,
with the token and the base URL read from the environment."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlparse
from typing import Any, Dict, Mapping

from .contract import JourneyError, _CLIENT_ID, _BASE_URL_ENV, _TOKEN_ENV


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

