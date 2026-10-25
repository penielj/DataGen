"""API tests. The agent is replaced with a fake, so no OpenAI calls are made."""
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from datagen import api

TEST_KEY = "test-key"


@pytest.fixture
def client(monkeypatch):
    calls = []

    def fake_run_agent(prompt, history):
        calls.append((prompt, list(history)))
        return AIMessage(content=f"echo: {prompt}"), [{"users": [], "count": 0}]

    monkeypatch.setattr(api.agent, "run_agent", fake_run_agent)
    monkeypatch.setenv("DATAGEN_API_KEY", TEST_KEY)
    api.sessions.clear()
    c = TestClient(api.app, headers={"X-API-Key": TEST_KEY})
    c.calls = calls
    return c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_generate_returns_reply_data_and_session(client):
    res = client.post("/generate", json={"prompt": "make 3 users"})
    assert res.status_code == 200
    body = res.json()
    assert body["reply"] == "echo: make 3 users"
    assert body["data"] == [{"users": [], "count": 0}]
    assert body["session_id"]


def test_session_keeps_history(client):
    sid = client.post("/generate", json={"prompt": "first"}).json()["session_id"]
    client.post("/generate", json={"prompt": "second", "session_id": sid})
    _, history_on_second_call = client.calls[1]
    assert [m.content for m in history_on_second_call] == ["first", "echo: first"]


def test_separate_sessions_do_not_share_history(client):
    client.post("/generate", json={"prompt": "a"})
    client.post("/generate", json={"prompt": "b"})
    assert client.calls[1][1] == []


def test_empty_prompt_rejected(client):
    assert client.post("/generate", json={"prompt": ""}).status_code == 422


def test_clear_session(client):
    sid = client.post("/generate", json={"prompt": "hi"}).json()["session_id"]
    assert client.delete(f"/sessions/{sid}").status_code == 200
    assert client.delete(f"/sessions/{sid}").status_code == 404


def test_get_file(client, isolated_output_dir):
    isolated_output_dir.mkdir(parents=True, exist_ok=True)
    (isolated_output_dir / "users.json").write_text(json.dumps({"count": 1}))
    assert client.get("/files/users.json").json() == {"count": 1}
    assert client.get("/files/missing.json").status_code == 404


def test_get_file_rejects_non_json(client):
    assert client.get("/files/secrets.txt").status_code == 400


# ---------- access control ----------

@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_protected_endpoints_reject_bad_key(client, headers):
    client.headers.clear()
    for method, path in [("post", "/generate"), ("get", "/files/users.json"),
                         ("delete", "/sessions/abc")]:
        kwargs = {"json": {"prompt": "hi"}} if method == "post" else {}
        assert getattr(client, method)(path, headers=headers, **kwargs).status_code == 401
    assert client.calls == []


def test_refuses_everything_when_server_key_unset(client, monkeypatch):
    monkeypatch.delenv("DATAGEN_API_KEY")
    assert client.post("/generate", json={"prompt": "hi"}).status_code == 503
    assert client.calls == []


def test_health_needs_no_key(client):
    client.headers.clear()
    assert client.get("/health").status_code == 200
