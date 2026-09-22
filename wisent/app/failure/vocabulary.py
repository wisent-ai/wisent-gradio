"""Failure classification for the operator console.

The vocabulary, the severity/retryable/outage derivation, the upstream-status
ladder and the detail trim rule all come from `wisent_errors`, the fleet's one
failure envelope — so one failure reads the same in a JSON body, in a terminal
and in this UI: seven codes, one retry rule, one answer to "is this ours or the
input's". This module keeps what is its own: which failure points exist, which
services it names, and how a failure is put in front of an operator.

This is an operator tool, so it differs from the product surfaces in two ways:

* **Nothing is reported over the network.** No collector call, no telemetry: a
  console that hangs while reporting that something hung is worse than useless.
  A failure leaves one structured line on the server log and one classified
  sentence in the browser.
* **Nothing is hidden from the operator.** The traceback is not deleted, it is
  *folded*: the first thing on screen is the verdict — ours or yours, worth
  retrying or not — and the technical text sits one click away.

The one thing this module refuses to do is let a broken dependency look like an
empty one. `ArtifactUnavailable` exists for exactly that: "the store did not
answer" is not "the store answered, and there is nothing there".
"""
from __future__ import annotations

import json
import logging
import traceback as _traceback
from dataclasses import dataclass

from wisent_errors import CODES, FALLBACK, failure_or_fallback, from_upstream_status, trim_detail

#: This console's own width for a detail. The rule for how to cut is the
#: package's; the bound is ours.
_MAX_DETAIL_CHARS = 500


def _catalogued(code: str) -> str:
    """One code from the shared catalogue, or a loud failure at import.

    These names are a projection, not a second vocabulary: a code this console
    still believes in but the catalogue has dropped must break the import here,
    where it is one line to read, rather than surface as a classification that
    silently means nothing.
    """
    if code not in CODES:
        raise ImportError(
            f"wisent.app.failure declares error code {code!r}, which is not in the "
            f"wisent_errors catalogue ({', '.join(CODES)})"
        )
    return code


CODE_CONFIG = _catalogued("config")
CODE_AUTH = _catalogued("auth")
CODE_NOT_FOUND = _catalogued("not_found")
CODE_RATE_LIMIT = _catalogued("rate_limit")
CODE_TIMEOUT = _catalogued("timeout")
CODE_INFRA_DOWN = _catalogued("infra_down")
CODE_UNKNOWN = _catalogued("unknown")

#: What the operator has to know, in one sentence: whose problem it is and
#: whether repeating the action can help. No exception text — that is folded
#: away below, not deleted.
MESSAGE_BY_CODE = {
    CODE_CONFIG: "is not configured on this machine — settings, not your input.",
    CODE_AUTH: "rejected our credentials — the token needs refreshing.",
    CODE_NOT_FOUND: "answered, and what was asked for is not there.",
    CODE_RATE_LIMIT: "is rate limiting us — wait a moment and repeat.",
    CODE_TIMEOUT: "did not answer in time — ours, not your input.",
    CODE_INFRA_DOWN: "is unreachable — ours, not your input.",
    CODE_UNKNOWN: "failed in a way this console does not recognise.",
}

SERVICE_HUGGINGFACE = "huggingface"
SERVICE_CLI = "cli"
SERVICE_APP = "app"

logger = logging.getLogger("wisent.app.failure")

_TIMEOUT_TYPE_NAMES = (
    "TimeoutError",
    "ReadTimeout",
    "ReadTimeoutError",
    "ConnectTimeout",
    "ConnectTimeoutError",
    "ConnectionTimeout",
)

#: `huggingface_hub` exception names, matched as strings so this module keeps
#: working when the hub client is absent or upgraded under us.
_HF_AUTH_TYPE_NAMES = ("GatedRepoError", "LocalTokenNotFoundError")
_HF_NOT_FOUND_TYPE_NAMES = (
    "RepositoryNotFoundError",
    "EntryNotFoundError",
    "RevisionNotFoundError",
)

_CONFIG_MARKERS = (
    "is required",
    "not configured",
    "missing env",
    "must be set",
    "no token",
    "token is required",
)

_NETWORK_MARKERS = (
    "connection refused",
    "connection reset",
    "connection aborted",
    "broken pipe",
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname",
    "network is unreachable",
    "no route to host",
    "cannot connect",
    "server disconnected",
    "max retries exceeded",
    "offlinemodeisenabled",
    "couldn't connect to",
)

