import sys
from pathlib import Path
from json import dump

proj = Path(r"c:\Users\Hi\Desktop\루시아 에이전트\2")
sys.path.insert(0, str(proj))

try:
    import importlib

    mod = importlib.import_module("lucia_ultimate_quantum_enhanced_modular")
    from fastapi.testclient import TestClient

    client = TestClient(mod.app)
    res = client.get("/health")
    out = {"status_code": res.status_code, "json": res.json()}
except Exception:
    import traceback

    out = {"error": True, "exc": traceback.format_exc()}

p = proj / "tools" / "health_testclient_out.json"
with p.open("w", encoding="utf8") as f:
    dump(out, f, ensure_ascii=False, indent=2)
print("wrote", p)
