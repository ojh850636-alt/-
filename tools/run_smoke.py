from fastapi.testclient import TestClient
from lucia_ultimate_quantum_integrated_fixed import app
import sys
import json

client = TestClient(app)

ok = True

def pp(j):
    try:
        return json.dumps(j, ensure_ascii=False, indent=2)
    except Exception:
        return str(j)

print('--- SMOKE: POST /command ---')
resp = client.post('/command', json={'text': 'echo hello'})
print('status:', resp.status_code)
print(pp(resp.json()))
if resp.status_code != 200:
    ok = False

print('\n--- SMOKE: POST /ai/chat (stub) ---')
resp = client.post('/ai/chat', json={'use_stub': True})
print('status:', resp.status_code)
print(pp(resp.json()))
if not (resp.status_code == 200 and resp.json().get('response') == 'stubbed response'):
    ok = False

print('\n--- SMOKE RESULT ---')
print('SMOKE OK:', ok)
sys.exit(0 if ok else 2)
