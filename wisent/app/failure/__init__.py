"""One honest shape for every failure this console shows: the vocabulary it
is named in, the classification of one, and how it is reported."""
from .classify import ArtifactUnavailable, Classification, classify
from .report import as_markdown, log_line, report, summary, technical_text
from .vocabulary import (
    CODE_AUTH,
    CODE_CONFIG,
    CODE_INFRA_DOWN,
    CODE_NOT_FOUND,
    CODE_RATE_LIMIT,
    CODE_TIMEOUT,
    CODE_UNKNOWN,
    MESSAGE_BY_CODE,
    SERVICE_APP,
    SERVICE_CLI,
    SERVICE_HUGGINGFACE,
)

__all__ = [
    "ArtifactUnavailable",
    "CODE_AUTH",
    "CODE_CONFIG",
    "CODE_INFRA_DOWN",
    "CODE_NOT_FOUND",
    "CODE_RATE_LIMIT",
    "CODE_TIMEOUT",
    "CODE_UNKNOWN",
    "Classification",
    "MESSAGE_BY_CODE",
    "SERVICE_APP",
    "SERVICE_CLI",
    "SERVICE_HUGGINGFACE",
    "as_markdown",
    "classify",
    "log_line",
    "report",
    "summary",
    "technical_text",
]
