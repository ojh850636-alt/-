from lucia_ultimate_quantum_integrated_fixed import app
from fastapi.testclient import TestClient
import json

client = TestClient(app)

failed = False

def pp(j):
    try:
        return json.dumps(j, ensure_ascii=False, indent=2)
    except Exception:
        return str(j)

print('=== DIRECT TEST: /command ===')
resp = client.post('/command', json={'text': 'echo hello'})
print('status:', resp.status_code)
print(pp(resp.json()))
if resp.status_code != 200:
    failed = True

print('\n=== DIRECT TEST: /ai/chat (stub) ===')
resp = client.post('/ai/chat', json={'use_stub': True})
print('status:', resp.status_code)
print(pp(resp.json()))
if not (resp.status_code == 200 and resp.json().get('response') == 'stubbed response'):
    failed = True

print('\n=== DIRECT TEST RESULT ===')
print('ALL_OK:', not failed)
if failed:
    raise SystemExit(2)
