from __future__ import annotations

import pytest
import requests
from tests.sources.text.fakes import FakeResponse, FakeSession

from pie.sources.text import _http

URL = "https://example.org/x"


def test_get_returns_first_success_after_transient_failures() -> None:
    queue = [FakeResponse(503), requests.ConnectionError("down"), FakeResponse(200, {"ok": 1})]
    session = FakeSession({URL: queue})
    sleeps: list[float] = []
    response = _http.get(URL, session=session, sleep=sleeps.append)
    assert response.json() == {"ok": 1}
    assert len(session.calls) == 3
    assert sleeps == [0.25, 0.5]


def test_get_raises_immediately_on_client_error() -> None:
    session = FakeSession({URL: [FakeResponse(404)]})
    with pytest.raises(requests.HTTPError):
        _http.get(URL, session=session, sleep=lambda _s: None)
    assert len(session.calls) == 1


def test_get_gives_up_after_max_attempts() -> None:
    session = FakeSession({URL: [FakeResponse(429)]})
    with pytest.raises(requests.RequestException, match="failed after 4 attempts; last status 429"):
        _http.get(URL, session=session, sleep=lambda _s: None)
    assert len(session.calls) == _http.MAX_ATTEMPTS


def test_cache_miss_message_names_subject_and_path() -> None:
    error = _http.cache_miss("Cellosaurus", "accession CVCL_0004", "/c/56.0/CVCL_0004.json")
    assert isinstance(error, RuntimeError)
    assert str(error) == (
        "offline Cellosaurus cache miss for accession CVCL_0004; "
        "run once online to populate /c/56.0/CVCL_0004.json"
    )


SECRET = "s3cr3t-key-value"
KEYED_URL = f"https://example.org/esummary.fcgi?db=gene&id=1&api_key={SECRET}&tool=pie"


class _KeyedErrorResponse(FakeResponse):
    """A 4xx response whose HTTPError names the full URL, as requests does."""

    def raise_for_status(self) -> None:
        message = f"400 Client Error: Bad Request for url: {KEYED_URL}"
        raise requests.HTTPError(message, response=self)


def _chain_text(error: BaseException) -> str:
    """Every message a traceback of `error` would print (cause and unsuppressed context)."""
    parts = []
    current: BaseException | None = error
    while current is not None:
        parts.append(str(current))
        context = None if current.__suppress_context__ else current.__context__
        current = current.__cause__ or context
    return "\n".join(parts)


def test_get_redacts_api_key_from_retry_exhaustion_message() -> None:
    session = FakeSession({URL: [requests.ConnectionError(f"Max retries exceeded: {KEYED_URL}")]})
    with pytest.raises(requests.RequestException) as info:
        _http.get(URL, params={"api_key": SECRET}, session=session, sleep=lambda _s: None)
    assert SECRET not in _chain_text(info.value)
    assert "api_key=***" in str(info.value)


def test_get_redacts_api_key_from_http_error() -> None:
    session = FakeSession({URL: [_KeyedErrorResponse(400)]})
    with pytest.raises(requests.HTTPError) as info:
        _http.get(URL, params={"api_key": SECRET}, session=session, sleep=lambda _s: None)
    assert SECRET not in _chain_text(info.value)
    assert "api_key=***" in str(info.value)
    assert info.value.response is not None


def test_redact_masks_key_like_query_params() -> None:
    text = "u?api_key=A1&token=B2&apikey=C3&access_token=D4&id=5"
    assert _http.redact(text) == "u?api_key=***&token=***&apikey=***&access_token=***&id=5"
