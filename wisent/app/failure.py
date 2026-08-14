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
_MAX_DETAIL_CHARS = int("500")


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


@dataclass(frozen=True)
class Classification:
    """The few distinctions an operator actually acts on."""

    code: str
    service: str
    failure_point: str
    severity: str
    retryable: bool
    outage: bool
    #: Status, exception type and text. Shown to the operator behind a fold —
    #: this console has no anonymous audience to protect it from.
    detail: str | None = None


class ArtifactUnavailable(Exception):
    """A store did not answer, so its inventory is unknown — not empty.

    Raised instead of returning `[]` or `{}`, because those mean "there is
    nothing there", and a caller that cannot tell the two apart will render an
    outage as a clean, empty, entirely convincing screen.
    """

    def __init__(self, classification: Classification, cause: BaseException | None = None):
        super().__init__(summary(classification))
        self.classification = classification
        self.cause = cause


def _status_of(error: BaseException | None) -> int | None:
    """Dig the upstream status out of an HTTP client's exception, if any."""
    if error is None:
        return None
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


def _from_status(status: int | None) -> str | None:
    """The catalogue's status ladder, with "no rule for this" kept distinct.

    `from_upstream_status` answers with the fallback code for a status it has no
    rule for. This module's callers need that answer to stay ``None``, because a
    status alone is the weakest evidence here: an unclassifiable one must fall
    through to the exception type and the message markers rather than end the
    search at ``unknown``.
    """
    if status is None:
        return None
    classified = from_upstream_status(status)
    return None if classified == FALLBACK else classified


def _from_exception(error: BaseException | None) -> str | None:
    if error is None:
        return None
    names = {base.__name__ for base in type(error).__mro__}
    if names & set(_TIMEOUT_TYPE_NAMES):
        return CODE_TIMEOUT
    if names & set(_HF_NOT_FOUND_TYPE_NAMES):
        return CODE_NOT_FOUND
    if names & set(_HF_AUTH_TYPE_NAMES):
        return CODE_AUTH
    status = _from_status(_status_of(error))
    if status is not None:
        return status
    # ConnectionError and friends are OSError subclasses, so the checks above
    # have to come first.
    if isinstance(error, (ConnectionError, OSError)):
        return CODE_INFRA_DOWN

    message = str(error).lower()
    if any(marker in message for marker in _CONFIG_MARKERS):
        return CODE_CONFIG
    if any(marker in message for marker in _NETWORK_MARKERS):
        return CODE_INFRA_DOWN
    return None


def _detail(error: BaseException | None, status: int | None, reason: str | None) -> str | None:
    parts: list[str] = []
    if status is not None:
        parts.append(f"http {status}")
    if reason:
        parts.append(reason)
    if error is not None:
        parts.append(f"{type(error).__name__}: {error}")
    if not parts:
        return None
    return trim_detail(" — ".join(parts), _MAX_DETAIL_CHARS)


def classify(
    failure_point: str,
    *,
    service: str = SERVICE_APP,
    error: BaseException | None = None,
    status: int | None = None,
    code: str | None = None,
    reason: str | None = None,
) -> Classification:
    """Turn whatever a dependency did into the contract's vocabulary.

    An explicit ``code`` wins: a call site that already knows what happened
    should not have it guessed back out of an exception type.
    """
    upstream = status if status is not None else _status_of(error)
    detail = _detail(error, upstream, reason)
    # `failure_or_fallback`, not `failure`: this is the reporting path, and an
    # error path that raises while describing an error takes the diagnosis with
    # it. It also owns the coercion of an off-catalogue code, which is what this
    # module used to do by hand.
    envelope = failure_or_fallback(
        failure_point=failure_point,
        code=code or _from_exception(error) or _from_status(upstream) or FALLBACK,
        service=service,
        detail=detail,
    )
    notes = envelope.get("context")
    if notes:
        # The defect travels in the data rather than becoming an exception: a
        # failure point that violates the fleet's pattern is still shown to the
        # operator verbatim, because that is what they have to grep for.
        logger.debug("wisent_errors notes for %s: %s", failure_point, notes)
    return Classification(
        code=envelope["error_code"],
        service=service,
        failure_point=failure_point,
        severity=envelope["severity"],
        retryable=envelope["retryable"],
        outage=envelope["outage"],
        detail=detail,
    )


def log_line(classification: Classification) -> str:
    """The one structured line, greppable in the server log."""
    fields = [
        f"failure_point={classification.failure_point}",
        f"error_code={classification.code}",
        f"service={classification.service}",
        f"severity={classification.severity}",
        f"retryable={'true' if classification.retryable else 'false'}",
        f"outage={'true' if classification.outage else 'false'}",
    ]
    if classification.detail:
        fields.append(f"detail={json.dumps(classification.detail)}")
    return "wisent.failure " + " ".join(fields)


def report(
    failure_point: str,
    *,
    service: str = SERVICE_APP,
    error: BaseException | None = None,
    status: int | None = None,
    code: str | None = None,
    reason: str | None = None,
) -> Classification:
    """Classify and log once. Never raises, never touches the network."""
    classification = classify(
        failure_point,
        service=service,
        error=error,
        status=status,
        code=code,
        reason=reason,
    )
    logger.error(log_line(classification))
    if error is not None:
        logger.debug("traceback for %s", failure_point, exc_info=error)
    return classification


def summary(classification: Classification) -> str:
    """One plain-text line: whose failure it is and whether to repeat it."""
    verdict = MESSAGE_BY_CODE.get(classification.code, MESSAGE_BY_CODE[CODE_UNKNOWN])
    tail = " Safe to retry." if classification.retryable else ""
    return f"{classification.service} {verdict}{tail}"


def technical_text(classification: Classification, error: BaseException | None = None) -> str:
    """Everything the operator needs to debug: the log line and the traceback."""
    blocks = [log_line(classification)]
    if error is not None:
        blocks.append("".join(
            _traceback.format_exception(type(error), error, error.__traceback__)
        ).rstrip())
    return "\n\n".join(blocks)


def as_markdown(
    classification: Classification,
    error: BaseException | None = None,
    *,
    heading: str = "Failed",
) -> str:
    """The verdict in the open, the traceback folded underneath it."""
    lines = [
        f"**{heading}: {summary(classification)}**",
        "",
        f"`error_code={classification.code}` · `service={classification.service}` "
        f"· `failure_point={classification.failure_point}` "
        f"· `retryable={'true' if classification.retryable else 'false'}`",
    ]
    body = technical_text(classification, error)
    lines += [
        "",
        "<details><summary>Technical detail</summary>",
        "",
        "```",
        body,
        "```",
        "",
        "</details>",
    ]
    return "\n".join(lines)
