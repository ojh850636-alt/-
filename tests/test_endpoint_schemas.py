import pytest
from fastapi.testclient import TestClient

from lucia_ultimate_quantum_integrated_fixed import app


client = TestClient(app)


def test_command_endpoint_accepts_schema():
    resp = client.post("/command", json={"text": "echo hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)


def test_ai_chat_stub_path():
    resp = client.post("/ai/chat", json={"use_stub": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("ok") is True
    assert data.get("response") == "stubbed response"
