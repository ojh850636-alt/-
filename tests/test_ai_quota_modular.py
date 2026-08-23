import importlib
from fastapi.testclient import TestClient

sys_path = r"C:\Users\Hi\Desktop\루시아 에이전트\2"
import sys

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

import lucia_ultimate_quantum_enhanced_modular as mod

client = TestClient(mod.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is True


def test_quota_enforcement(tmp_path, monkeypatch):
    # set low quota
    monkeypatch.setenv("AI_DAILY_CALL_LIMIT", "1")
    monkeypatch.setenv("AI_MAX_CALLS_PER_MINUTE", "10")
    usage = tmp_path / "ai_usage.json"
    monkeypatch.setenv("AI_QUOTA_FILE", str(usage))
    # remove any existing
    if usage.exists():
        usage.unlink()

    r1 = client.post("/ai/chat", json={"text": "hello", "use_stub": True})
    assert r1.status_code == 200
    j1 = r1.json()
    assert j1.get("ok") is True

    r2 = client.post("/ai/chat", json={"text": "hello again", "use_stub": True})
    assert r2.status_code == 200
    j2 = r2.json()
    assert j2.get("ok") is False


def test_ai_chat_fixed_runner_stub(tmp_path, monkeypatch):
    # Import fixed runner app
    import sys

    p = r"C:\Users\Hi\Desktop\루시아 에이전트\2"
    if p not in sys.path:
        sys.path.insert(0, p)
    mod = importlib.import_module("lucia_ultimate_quantum_integrated_fixed")
    client2 = TestClient(mod.app)

    # ensure fresh usage file and stub providers
    usage = tmp_path / "ai_usage_fixed.json"
    monkeypatch.setenv("AI_QUOTA_FILE", str(usage))
    monkeypatch.setenv("ENABLE_OPENAI", "false")
    monkeypatch.setenv("ENABLE_GROQ", "false")
    monkeypatch.setenv("ENABLE_OPENROUTER", "false")

    r = client2.post("/ai/chat", json={"prompt": "hello"})
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is True
    assert j.get("provider") == "stub"


def test_quota_rollback_on_provider_error(tmp_path, monkeypatch):
    """If the provider raises, the quota reservation must be rolled back."""
    import sys

    p = r"C:\Users\Hi\Desktop\루시아 에이전트\2"
    if p not in sys.path:
        sys.path.insert(0, p)
    mod = importlib.import_module("lucia_ultimate_quantum_integrated_fixed")
    client2 = TestClient(mod.app)

    usage = tmp_path / "ai_usage_rb.json"
    monkeypatch.setenv("AI_QUOTA_FILE", str(usage))
    monkeypatch.setenv("AI_USE_MOCK", "true")

    # first call: mock provider raises -> should return 500 and not consume quota
    r = client2.post("/ai/chat", json={"prompt": "will error", "mock_error": True})
    assert r.status_code == 500

    # second call: with normal mock response should still be allowed (quota rolled back)
    r2 = client2.post("/ai/chat", json={"prompt": "ok", "mock": True})
    assert r2.status_code == 200
    j2 = r2.json()
    assert j2.get("ok") is True
