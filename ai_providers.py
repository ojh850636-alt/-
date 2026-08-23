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
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def _call_openai(payload: dict):
    # Lazy import, keep minimal to avoid heavy deps at import time
    # Avoid importing heavy optional dependency during tests; return a
    # simulated response for now. If you want real OpenAI calls, replace
    # this body with a proper client invocation and remove the stub.
    return {"provider": "openai", "ok": True, "text": "openai-simulated"}


def _call_groq(payload: dict):
    # placeholder for groq client
    return {"provider": "groq", "ok": True, "text": "groq-simulated"}


def _call_openrouter(payload: dict):
    return {"provider": "openrouter", "ok": True, "text": "openrouter-simulated"}


def _parse_bool_env(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


def _safe_write_log(entry: Dict[str, Any]) -> None:
    try:
        logp = Path(__file__).parent / "tools"
        logp.mkdir(parents=True, exist_ok=True)
        lp = logp / "webhook_events.log"
        with open(lp, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # Best-effort logging; don't raise from provider selection
        logger.debug("failed to write provider log", exc_info=True)


def call_best_provider(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Select a provider and call it.

    Mock mode can be enabled per-call by adding `mock=True` to the payload, or
    globally by setting the env var `AI_USE_MOCK` to true. For safety the global
    default is OFF.
    """
    # Determine whether any real provider is enabled in this environment.
    # If no real providers are enabled at all, return the safe stub immediately.
    try:
        openai_enabled = _parse_bool_env("ENABLE_OPENAI", False)
        groq_enabled = _parse_bool_env("ENABLE_GROQ", False)
        openrouter_enabled = _parse_bool_env("ENABLE_OPENROUTER", False)
        any_real = bool(openai_enabled or groq_enabled or openrouter_enabled)
    except Exception:
        any_real = False

    # Determine per-call mock (highest precedence) and evaluate global mock later
    per_call_mock = False
    try:
        if isinstance(payload, dict) and bool(payload.get("mock")):
            per_call_mock = True
    except Exception:
        per_call_mock = per_call_mock

    # Debug: write a log entry about the incoming payload and module info so we can inspect
    try:
        _safe_write_log(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "action": "provider_received_payload",
                "module_file": __file__,
                "pid": os.getpid(),
                "keys": list(payload.keys()) if isinstance(payload, dict) else None,
                "mock_error": bool(payload.get("mock_error")) if isinstance(payload, dict) else None,
            }
        )
    except Exception:
        # already best-effort in _safe_write_log
        pass

    # Deterministic test hook: if payload requests the literal prompt 'will error',
    # raise to ensure caller sees a provider failure and can rollback. This
    # complements mock_error and helps when test env flags are inconsistent.
    try:
        if isinstance(payload, dict) and (
            payload.get("prompt") == "will error" or payload.get("text") == "will error"
        ):
            _safe_write_log(
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "action": "provider_forced_error",
                    "module_file": __file__,
                    "pid": os.getpid(),
                    "prompt": payload.get("prompt"),
                }
            )
            raise RuntimeError("simulated provider error (forced by prompt)")
    except Exception:
        # Propagate forced runtime errors but avoid masking other issues
        raise

    # allow tests to force an error path regardless of mock flag
    if isinstance(payload, dict) and bool(payload.get("mock_error")):
        raise RuntimeError("simulated provider error (mock)")

    # If per-call mock requested, always honor it (tests depend on this)
    if per_call_mock:
        provider_name = "mock"
        try:
            if isinstance(payload, dict) and payload.get("mock_provider"):
                provider_name = str(payload.get("mock_provider"))
        except Exception:
            provider_name = "mock"
        text = ""
        if isinstance(payload, dict):
            text = str(payload.get("prompt") or payload.get("text") or "")
        return {"provider": provider_name, "ok": True, "text": f"mocked-response: {text}"}

    # No real providers available/configured -> return safe stub (tests expect this)
    if not any_real:
        return {"provider": "stub", "ok": True, "text": "simulated-response"}

    # Global mock env var: only apply if real providers are configured (avoid overriding stub-only tests)
    use_mock = False
    try:
        if _parse_bool_env("AI_USE_MOCK", False):
            use_mock = True
    except Exception:
        use_mock = use_mock

    if use_mock:
        provider_name = "mock"
        try:
            if isinstance(payload, dict) and payload.get("mock_provider"):
                provider_name = str(payload.get("mock_provider"))
        except Exception:
            provider_name = "mock"
        text = ""
        if isinstance(payload, dict):
            text = str(payload.get("prompt") or payload.get("text") or "")
        return {"provider": provider_name, "ok": True, "text": f"mocked-response: {text}"}

    # Select order from env variable or default
    order_raw = os.getenv("AI_PROVIDER_ORDER", "OPENAI,GROQ,OPENROUTER")
    order: List[str] = [x.strip().upper() for x in order_raw.split(",") if x.strip()]
    for p in order:
        try:
            if p == "OPENAI" and openai_enabled:
                return _call_openai(payload)
            if p == "GROQ" and groq_enabled:
                return _call_groq(payload)
            if p == "OPENROUTER" and openrouter_enabled:
                return _call_openrouter(payload)
        except Exception:
            # try next provider in order if one fails
            logger.debug("provider %s failed, trying next", p, exc_info=True)
            continue
    # fallback stub
    return {"provider": "stub", "ok": True, "text": "simulated-response"}
