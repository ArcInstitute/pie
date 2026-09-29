"""Recorded-HTTP stand-ins for the text tool tests (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import requests

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class FakeResponse:
    def __init__(
        self, status_code: int = 200, payload: object = None, content: bytes = b""
    ) -> None:
        self.status_code = status_code
        self.content = json.dumps(payload).encode("utf-8") if payload is not None else content

    def json(self) -> object:
        return json.loads(self.content)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}", response=self)

    def iter_content(self, chunk_size: int = 1) -> object:
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]


class FakeSession:
    """Routes GETs by URL to queued responses; the last queued item repeats."""

    def __init__(self, routes: dict[str, list[object]]) -> None:
        self.routes = {url: list(items) for url, items in routes.items()}
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(
        self, url: str, params: object = None, timeout: float | None = None, stream: bool = False
    ) -> FakeResponse:
        self.calls.append((url, dict(params or {})))
        queue = self.routes.get(url)
        if not queue:
            raise AssertionError(f"unexpected GET {url}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item


class NoNetwork:
    """A session that fails the test on any request."""

    def get(self, url: str, *args: object, **kwargs: object) -> FakeResponse:
        raise AssertionError(f"network access attempted: {url}")
