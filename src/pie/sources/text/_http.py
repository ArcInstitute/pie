"""HTTP GET with bounded retry and the offline cache-miss error shared by the text tools."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping

import requests

MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 0.25
_TRANSIENT = frozenset({429, 500, 502, 503, 504})
_SECRET_PARAM = re.compile(
    r"(?i)\b(api[_-]?key|apikey|access[_-]?token|token|secret|password)=[^&#\s'\"]+"
)


def redact(text: str) -> str:
    """Mask the value of every key-like query parameter (api_key=..., token=...) in `text`."""
    return _SECRET_PARAM.sub(r"\1=***", text)


class CacheMissError(RuntimeError):
    """An offline run needed a record that is not in the cache."""


def cache_miss(label: str, subject: str, path: object) -> CacheMissError:
    return CacheMissError(
        f"offline {label} cache miss for {subject}; run once online to populate {path}"
    )


def get(
    url: str,
    *,
    params: Mapping[str, str] | None = None,
    session: requests.Session | None = None,
    timeout: float = 60.0,
    stream: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    backoff_base: float = BACKOFF_SECONDS,
) -> requests.Response:
    """GET, retrying rate limits, 5xx and connection errors with exponential backoff.

    Error messages have key-like query parameters redacted, so an API key never reaches a log.
    """
    client = session if session is not None else requests.Session()
    last = ""
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.get(url, params=params, timeout=timeout, stream=stream)
        except (requests.ConnectionError, requests.Timeout) as error:
            last = f"exception {type(error).__name__}: {redact(str(error))}"
        else:
            if response.status_code not in _TRANSIENT:
                try:
                    response.raise_for_status()
                except requests.HTTPError as error:
                    raise requests.HTTPError(redact(str(error)), response=error.response) from None
                return response
            last = f"status {response.status_code}"
        if attempt + 1 < MAX_ATTEMPTS:
            sleep(backoff_base * 2**attempt)
    raise requests.RequestException(
        redact(f"GET {url} failed after {MAX_ATTEMPTS} attempts; last {last}")
    )
