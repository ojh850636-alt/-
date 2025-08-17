import os
import requests
import json

# Try to use FastAPI TestClient when available so tests don't need a running uvicorn
CLIENT = None
try:
    from fastapi.testclient import TestClient
    import lucia_ultimate_quantum_integrated as server
    CLIENT = TestClient(server.app)
except Exception:
    CLIENT = None

BASE = os.getenv('LUCIA_BASE', 'http://127.0.0.1:8002')


def _get(path):
    if CLIENT:
        return CLIENT.get(path)
    return requests.get(f"{BASE}{path}", timeout=5)


def _post(path, json_payload):
    if CLIENT:
        return CLIENT.post(path, json=json_payload)
    return requests.post(f"{BASE}{path}", json=json_payload, timeout=5)


def test_ai_status_and_chat_stub():
    r = _get('/ai/status')
    assert r.status_code == 200
    j = r.json()
    assert 'ai_available' in j

    # request chat with forced stub
    r = _post('/ai/chat', {'text': 'hello', 'use_stub': True})
    assert r.status_code == 200
    j = r.json()
    assert j.get('ok') is True
    assert 'response' in j
