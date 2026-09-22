"""What a classified failure looks like where it is read: one structured
line in the server log, one plain sentence on screen, and the technical
text behind a disclosure for whoever is debugging it."""
from __future__ import annotations

import json
import logging
import traceback as _traceback

from .classify import Classification, classify
from .vocabulary import CODE_UNKNOWN, MESSAGE_BY_CODE, SERVICE_APP

logger = logging.getLogger("wisent.app.failure")


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
