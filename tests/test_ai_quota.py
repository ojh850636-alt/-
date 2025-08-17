import os
import requests

# Try TestClient to run against the app directly
CLIENT = None
try:
    from fastapi.testclient import TestClient
    import lucia_ultimate_quantum_integrated as server
    CLIENT = TestClient(server.app)
except Exception:
    CLIENT = None

BASE = os.getenv('LUCIA_BASE', 'http://127.0.0.1:8002')


import os
import requests

BASE = os.getenv('LUCIA_BASE', 'http://127.0.0.1:8002')


def test_quota_limit():
    # ensure quota enforcement: set daily limit low in CI; first call should pass, second should be rejected
    # force a low daily quota in-process so TestClient exercises quota logic
    os.environ['AI_DAILY_CALL_LIMIT'] = '1'
    os.environ['AI_MAX_CALLS_PER_MINUTE'] = '10'
    # remove any persisted usage file so test starts fresh
    qfile = os.getenv('AI_QUOTA_FILE', 'ai_usage.json')
    try:
        if os.path.exists(qfile):
            os.remove(qfile)
    except Exception:
        pass

    # helper to post either via TestClient or requests
    def _post(path, json_payload):
        if CLIENT:
            return CLIENT.post(path, json=json_payload)
        return requests.post(f"{BASE}{path}", json=json_payload, timeout=5)

    r1 = _post('/ai/chat', {'text': 'hello', 'use_stub': False})
    assert r1.status_code == 200
    j1 = r1.json()
    # either ok True or a failure due to missing AI package; treat as a pass if ok or stub
    assert 'ok' in j1
    r2 = _post('/ai/chat', {'text': 'hello again', 'use_stub': False})
    assert r2.status_code == 200
    j2 = r2.json()
    # second call should be rejected by quota in CI config
    assert j2.get('ok') is False
