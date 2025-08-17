import os
import requests

# Try to use TestClient when available
CLIENT = None
try:
    from fastapi.testclient import TestClient
    import lucia_ultimate_quantum_integrated_fixed as server
    CLIENT = TestClient(server.app)
except Exception:
    CLIENT = None

BASE = os.getenv('LUCIA_BASE', 'http://127.0.0.1:8002')


def _post(path, json_payload):
    if CLIENT:
        return CLIENT.post(path, json=json_payload)
    return requests.post(f"{BASE}{path}", json=json_payload, timeout=5)


def test_quantum_stub():
    r = _post('/quantum/run', {'circuit': 'H;MEAS'})
    assert r.status_code == 200
    j = r.json()
    assert j.get('ok') is True
    
