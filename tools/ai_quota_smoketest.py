import importlib, sys, os, json
sys.path.insert(0, r'C:\Users\Hi\Desktop\루시아 에이전트\2')
import lucia_ultimate_quantum_enhanced_modular as mod
from fastapi.testclient import TestClient
client = TestClient(mod.app)
# Set low quota
os.environ['AI_DAILY_CALL_LIMIT']='1'
os.environ['AI_MAX_CALLS_PER_MINUTE']='10'
# remove usage file
uf = os.path.join(os.path.dirname(__file__), '..', 'ai_usage.json')
uf = os.path.abspath(uf)
try:
    if os.path.exists(uf): os.remove(uf)
except Exception:
    pass
r1 = client.post('/ai/chat', json={'text':'hello','use_stub':True})
with open(os.path.join(os.path.dirname(__file__), 'ai_quota_smoketest_out.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps({'r1_status': r1.status_code, 'r1': r1.json()}, ensure_ascii=False))
r2 = client.post('/ai/chat', json={'text':'hello again','use_stub':True})
with open(os.path.join(os.path.dirname(__file__), 'ai_quota_smoketest_out.json'), 'a', encoding='utf-8') as f:
    f.write('\n')
    f.write(json.dumps({'r2_status': r2.status_code, 'r2': r2.json()}, ensure_ascii=False))
print('wrote')
