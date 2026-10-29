from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


class FakeEncoder:
    dim = 4

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls: list[tuple[int, str]] = []

    def encode(self, texts, kind):
        self.calls.append((len(texts), kind))
        time.sleep(self.delay)
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def _client(monkeypatch, key="segredo", encoder=None, **env):
    import src.app as mod

    monkeypatch.setenv("EMBED_API_KEY", key)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    enc = encoder or FakeEncoder()
    return TestClient(mod.create_app(lambda: enc, model_name="fake-e5")), enc


AUTH = {"Authorization": "Bearer segredo"}


def test_embed_contract(monkeypatch):
    with _client(monkeypatch)[0] as c:
        r = c.post("/embed", json={"texts": ["a", "b"], "kind": "passage"}, headers=AUTH)
        assert r.status_code == 200
        body = r.json()
        assert body["model"] == "fake-e5" and body["dim"] == 4 and len(body["embeddings"]) == 2


def test_kind_defaults_to_query_and_reaches_encoder(monkeypatch):
    client, enc = _client(monkeypatch)
    with client as c:
        c.post("/embed", json={"texts": ["x"]}, headers=AUTH)
    assert enc.calls[-1] == (1, "query")


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer errado"}, {"Authorization": "Basic segredo"}])
def test_requires_bearer_key(monkeypatch, headers):
    with _client(monkeypatch)[0] as c:
        assert c.post("/embed", json={"texts": ["x"]}, headers=headers).status_code == 401


def test_refuses_to_start_without_key(monkeypatch):
    import src.app as mod

    monkeypatch.delenv("EMBED_API_KEY", raising=False)
    monkeypatch.delenv("EMBED_ALLOW_NO_AUTH", raising=False)
    with pytest.raises(RuntimeError):
        mod.create_app(lambda: FakeEncoder())


@pytest.mark.parametrize("payload", [{"texts": []}, {"texts": [""]}, {"texts": ["  "]}, {"texts": ["x"], "kind": "foo"},
                                     {"texts": ["x" * 9000]}, {"texts": ["x"] * 65}])
def test_validation(monkeypatch, payload):
    with _client(monkeypatch)[0] as c:
        assert c.post("/embed", json=payload, headers=AUTH).status_code == 422


def test_health_is_public(monkeypatch):
    with _client(monkeypatch)[0] as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"


def test_full_queue_returns_503_with_retry_after(monkeypatch):
    client, _ = _client(monkeypatch)
    with client as c:
        c.app.state.max_pending = 0
        r = c.post("/embed", json={"texts": ["x"]}, headers=AUTH)
        assert r.status_code == 503 and r.headers["Retry-After"] == "2"


def test_health_answers_while_inference_runs(monkeypatch):
    """Inferência fora do event loop: /health responde enquanto um /embed lento está em andamento."""
    client, _ = _client(monkeypatch, encoder=FakeEncoder(delay=1.0))
    with client as c:
        t = threading.Thread(target=lambda: c.post("/embed", json={"texts": ["x"]}, headers=AUTH))
        t.start()
        time.sleep(0.2)
        t0 = time.perf_counter()
        assert c.get("/health").status_code == 200
        assert time.perf_counter() - t0 < 0.5
        t.join()
