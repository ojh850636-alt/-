import os
import time
from pathlib import Path
import requests

# Try to use TestClient when available
CLIENT = None
try:
    from fastapi.testclient import TestClient
    import lucia_ultimate_quantum_integrated_fixed as server
    CLIENT = TestClient(server.app)
except Exception:
    CLIENT = None

BASE = os.getenv("LUCIA_BASE", "http://127.0.0.1:8002")


def _post(path, json_payload):
    if CLIENT:
        return CLIENT.post(path, json=json_payload)
    return requests.post(f"{BASE}{path}", json=json_payload, timeout=5)


def test_create_list_delete_cycle():
    # create a python file
    r = _post('/file', {"action": "create_python"})
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    filename = data.get("filename")
    assert filename

    # list files and ensure file present
    r = _post('/file', {"action": "list"})
    assert r.status_code == 200
    data = r.json()
    names = [f["name"] for f in data.get("files", [])]
    assert filename in names

    # delete the file
    r = _post('/file', {"action": "delete", "filename": filename})
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
