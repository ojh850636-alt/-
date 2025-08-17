import sys
import json
import importlib
from pathlib import Path

# ensure repo root on path
repo_root = Path(r"C:\Users\Hi\Desktop\루시아 에이전트\2").resolve()
sys.path.insert(0, str(repo_root))

out_path = repo_root / 'tools' / 'health_testclient_out.txt'
try:
    mod = importlib.import_module('lucia_ultimate_quantum_enhanced_modular')
    from fastapi.testclient import TestClient
    client = TestClient(mod.app)
    res = client.get('/health')
    out = json.dumps(res.json(), ensure_ascii=False)
except Exception as e:
    out = 'EXCEPTION:\n' + repr(e)

out_path.write_text(out, encoding='utf-8')
print('WROTE', out_path)
