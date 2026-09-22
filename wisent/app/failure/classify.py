"""Turning whatever was raised into the few distinctions an operator acts
on: the code, whether repeating the action can help, and the short detail a
log line carries. Nothing technical reaches the screen from here."""
from __future__ import annotations

from dataclasses import dataclass

from wisent_errors import failure_or_fallback, from_upstream_status, trim_detail

from .vocabulary import (
    CODE_AUTH,
    CODE_CONFIG,
    CODE_INFRA_DOWN,
    CODE_NOT_FOUND,
    CODE_RATE_LIMIT,
    CODE_TIMEOUT,
    CODE_UNKNOWN,
    SERVICE_APP,
    _CONFIG_MARKERS,
    _HF_AUTH_TYPE_NAMES,
    _HF_NOT_FOUND_TYPE_NAMES,
    _MAX_DETAIL_CHARS,
    _NETWORK_MARKERS,
    _TIMEOUT_TYPE_NAMES,
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

