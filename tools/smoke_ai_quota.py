# Smoke test script for ai quota + provider mock behavior using TestClient
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
mod = importlib.import_module("lucia_ultimate_quantum_integrated_fixed")
from fastapi.testclient import TestClient

client = TestClient(mod.app)
import os


def run():
    repo_root = Path(__file__).parents[1]
    os.environ["AI_QUOTA_FILE"] = str((repo_root / "ai_usage_smoke.json").absolute())
    os.environ["AI_USE_MOCK"] = "true"
    # ensure clean file
    try:
        os.remove(os.environ["AI_QUOTA_FILE"])
    except Exception:
        pass

    out = repo_root / "tools" / "smoke_ai_quota.out.json"
    try:
        results = {}
        r = client.post("/ai/chat", json={"prompt": "will error", "mock_error": True})
        results["first"] = {
            "status": r.status_code,
            "body": r.json()
            if r.headers.get("content-type", "").startswith("application/json")
            else r.text,
        }

        r2 = client.post("/ai/chat", json={"prompt": "ok", "mock": True})
        results["second"] = {
            "status": r2.status_code,
            "body": r2.json()
            if r2.headers.get("content-type", "").startswith("application/json")
            else r2.text,
        }

        # persist report
        import json

        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("wrote report to", out)
    except Exception as e:
        # ensure we always write a failure report
        import json

        with open(out, "w", encoding="utf-8") as f:
            json.dump({"error": repr(e)}, f, ensure_ascii=False, indent=2)
        print("wrote error report to", out)


if __name__ == "__main__":
    run()
