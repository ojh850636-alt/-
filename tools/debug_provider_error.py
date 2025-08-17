import os
import json
from fastapi.testclient import TestClient

# ensure test env
os.environ['AI_USE_MOCK'] = 'true'
# import app after setting env to mimic test ordering
import importlib, sys
p = r'C:\Users\Hi\Desktop\루시아 에이전트\2'
if p not in sys.path:
    sys.path.insert(0, p)
mod = importlib.import_module('lucia_ultimate_quantum_integrated_fixed')
client = TestClient(mod.app)

print('AI_USE_MOCK=', os.environ.get('AI_USE_MOCK'))

r = client.post('/ai/chat', json={'prompt':'will error', 'mock_error': True})
print('status_code:', r.status_code)
try:
    print('json:', json.dumps(r.json(), ensure_ascii=False))
except Exception:
    print('text:', r.text)

r2 = client.post('/ai/chat', json={'prompt':'ok', 'mock': True})
print('second status_code:', r2.status_code)
try:
    print('second json:', json.dumps(r2.json(), ensure_ascii=False))
except Exception:
    print('second text:', r2.text)
