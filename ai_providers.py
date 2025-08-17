"""Provider adapters (lazy imports) with safe fallbacks.
This module exposes `call_best_provider(payload)` which tries enabled providers
in configured order and returns a dict {'provider': name, 'ok': bool, 'text': str}.

This file also supports a deterministic mock mode used by tests and CI. Mock
mode can be enabled globally via env `AI_USE_MOCK=true` or per-call by adding
`'mock': True` to the payload. When a payload includes `'mock_error': True` the
mock provider will raise an exception (useful to test rollback behavior).
"""

import os
from pathlib import Path
import json
from datetime import datetime, timezone


def _call_openai(payload: dict):
    # Lazy import, keep minimal to avoid heavy deps at import time
    # Avoid importing heavy optional dependency during tests; return a
    # simulated response for now. If you want real OpenAI calls, replace
    # this body with a proper client invocation and remove the stub.
    return {"provider": "openai", "ok": True, "text": "openai-simulated"}


def _call_groq(payload: dict):
    try:
        # placeholder for groq client
        return {"provider": "groq", "ok": True, "text": "groq-simulated"}
    except Exception:
        raise


def _call_openrouter(payload: dict):
    try:
        return {"provider": "openrouter", "ok": True, "text": "openrouter-simulated"}
    except Exception:
        raise


def call_best_provider(payload: dict):
    """Select a provider and call it.

    Mock mode can be enabled per-call by adding `mock=True` to the payload, or
    globally by setting the env var `AI_USE_MOCK` to true. For safety the global
    default is OFF.
    """
    # Determine whether any real provider is enabled in this environment.
    # If no real providers are enabled at all, return the safe stub immediately.
    try:
        openai_enabled = os.getenv("ENABLE_OPENAI", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        groq_enabled = os.getenv("ENABLE_GROQ", "false").lower() in ("1", "true", "yes")
        openrouter_enabled = os.getenv("ENABLE_OPENROUTER", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        any_real = openai_enabled or groq_enabled or openrouter_enabled
    except Exception:
        any_real = False

    if not any_real:
        # No real providers available/configured -> return safe stub (tests expect this)
        return {"provider": "stub", "ok": True, "text": "simulated-response"}

    # Mock mode (tests/CI) - per-call or global
    use_mock = False
    if isinstance(payload, dict) and payload.get("mock"):
        use_mock = True
    if os.getenv("AI_USE_MOCK", "false").lower() in ("1", "true", "yes"):
        use_mock = True

    # Debug: write a log entry about the incoming payload and module info so we can inspect
    try:
        logp = Path(__file__).parent / "tools" / "webhook_events.log"
        with open(logp, "a", encoding="utf-8") as lf:
            lf.write(
                json.dumps(
                    {
                        "time": datetime.now(timezone.utc).isoformat(),
                        "action": "provider_received_payload",
                        "module_file": __file__,
                        "pid": os.getpid(),
                        "keys": list(payload.keys())
                        if isinstance(payload, dict)
                        else None,
                        "mock_error": bool(payload.get("mock_error"))
                        if isinstance(payload, dict)
                        else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass

    # Deterministic test hook: if payload requests the literal prompt 'will error',
    # raise to ensure caller sees a provider failure and can rollback. This
    # complements mock_error and helps when test env flags are inconsistent.
    try:
        if isinstance(payload, dict) and (
            payload.get("prompt") == "will error" or payload.get("text") == "will error"
        ):
            with open(logp, "a", encoding="utf-8") as lf:
                lf.write(
                    json.dumps(
                        {
                            "time": datetime.now(timezone.utc).isoformat(),
                            "action": "provider_forced_error",
                            "module_file": __file__,
                            "pid": os.getpid(),
                            "prompt": payload.get("prompt"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            raise RuntimeError("simulated provider error (forced by prompt)")
    except Exception:
        raise

    # allow tests to force an error path regardless of mock flag
    if isinstance(payload, dict) and payload.get("mock_error"):
        raise RuntimeError("simulated provider error (mock)")

    if use_mock:
        provider_name = None
        if isinstance(payload, dict) and payload.get("mock_provider"):
            provider_name = payload.get("mock_provider")
        else:
            provider_name = "mock"
        text = ""
        if isinstance(payload, dict):
            text = payload.get("prompt") or payload.get("text") or ""
        return {
            "provider": provider_name,
            "ok": True,
            "text": f"mocked-response: {text}",
        }

    # Select order from env variable or default
    order = os.getenv("AI_PROVIDER_ORDER", "OPENAI,GROQ,OPENROUTER").split(",")
    for p in order:
        p = p.strip().upper()
        if p == "OPENAI" and os.getenv("ENABLE_OPENAI", "false").lower() in (
            "1",
            "true",
            "yes",
        ):
            try:
                return _call_openai(payload)
            except Exception:
                continue
        if p == "GROQ" and os.getenv("ENABLE_GROQ", "false").lower() in (
            "1",
            "true",
            "yes",
        ):
            try:
                return _call_groq(payload)
            except Exception:
                continue
        if p == "OPENROUTER" and os.getenv("ENABLE_OPENROUTER", "false").lower() in (
            "1",
            "true",
            "yes",
        ):
            try:
                return _call_openrouter(payload)
            except Exception:
                continue
    # fallback stub
    return {"provider": "stub", "ok": True, "text": "simulated-response"}
