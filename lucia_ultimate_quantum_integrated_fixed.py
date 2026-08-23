#!/usr/bin/env python3
# Early shim: suppress Pydantic v2 dict() deprecation warnings and shim BaseModel.dict
try:
    import warnings as _warnings

    # ignore the specific pydantic deprecation if available
    try:
        from pydantic.errors import PydanticDeprecatedSince20 as _PDS

        _warnings.filterwarnings("ignore", category=_PDS)
    except Exception:
        _warnings.filterwarnings(
            "ignore", message=r".*The `dict` method is deprecated.*"
        )
    try:
        from pydantic import BaseModel as _BaseModel

        if hasattr(_BaseModel, "model_dump"):

            def _safe_dict(self, *a, **k):
                try:
                    return self.model_dump(*a, **k)
                except Exception:
                    return getattr(self, "__dict__", {})

            _BaseModel.dict = _safe_dict
    except Exception:
        pass
except Exception:
    pass
"""
Cleaned, minimal version of Lucia Ultimate Quantum Integrated server.
This file keeps a minimal, test-friendly FastAPI app with deterministic
AI quota behavior and simple file operations.
"""

import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
import ai_quota
from lucia_core import command_parser, dispatcher, file_ops
import warnings

# Prefer to silence the Pydantic v2 dict->model_dump deprecation if present
try:
    from pydantic.errors import PydanticDeprecatedSince20

    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
except Exception:
    warnings.filterwarnings("ignore", message=r".*The `dict` method is deprecated.*")

# Warnings are filtered via `pytest.ini` and `tests/conftest.py`.

# App
app = FastAPI(title="Lucia - Cleaned Minimal Server")

# Simple shared state

# runtime state
connected_clients = set()
command_history = []
server_stats = {
    "start_time": datetime.now(timezone.utc),
    "total_commands": 0,
    "successful_commands": 0,
    "failed_commands": 0,
    "peak_clients": 0,
}


# File operations
DOWNLOADS = Path("downloads")
DOWNLOADS.mkdir(exist_ok=True)


def _ai_status_info():
    """Return a small dict describing AI availability and configured providers."""
    enabled_flag = os.getenv("ENABLE_AI", "false").lower() in ("1", "true", "yes")
    # quick probe for installed package
    try:
        import importlib

        ai_package = importlib.util.find_spec("openai") is not None
    except Exception:
        ai_package = False
    groq_available = bool(os.getenv("GROQ_API_KEY") and os.getenv("GROQ_API_URL"))
    openrouter_available = bool(
        os.getenv("OPENROUTER_API_KEY") and os.getenv("OPENROUTER_URL")
    )
    return {
        "enabled_flag": enabled_flag,
        "ai_package": ai_package,
        "ai_available": False,
        "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "groq_available": groq_available,
        "openrouter_available": openrouter_available,
    }


@app.get("/ai/status")
async def ai_status():
    return _ai_status_info()


@app.get("/health")
async def health():
    return JSONResponse(
        {
            "ok": True,
            "uptime_seconds": (
                datetime.now(timezone.utc) - server_stats["start_time"]
            ).total_seconds(),
        }
    )


async def execute_command(command: Dict[str, Any]) -> Dict[str, Any]:
    return await dispatcher.execute_command(command)


@app.post("/ai/chat")
async def ai_chat(request: Request):
    """AI chat endpoint that reads raw JSON from the request to avoid
    creating a Pydantic model instance (prevents `.dict()` calls and
    related deprecation warnings during tests).
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    prompt = payload.get("prompt") or payload.get("text") or ""

    # Stub path used by tests
    if payload.get("use_stub"):
        return JSONResponse(
            {"ok": True, "response": "stubbed response"}, status_code=200
        )

    # Note: provider errors (including mock_error) are handled by the provider
    # adapter (ai_providers) and will raise from `ai_quota.call_provider` when
    # appropriate; we avoid simulating an error before quota reservation here so
    # that the endpoint's rollback logic can be exercised by tests.

    # try to reserve quota
    try:
        ok = ai_quota.reserve_call()
    except Exception as e:
        try:
            with open(
                Path(__file__).parent / "tools" / "webhook_events.log",
                "a",
                encoding="utf-8",
            ) as _lf:
                _lf.write(
                    json.dumps(
                        {
                            "time": datetime.now(timezone.utc).isoformat(),
                            "action": "quota_reserve_error",
                            "error": str(e),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass
        return JSONResponse(
            {"ok": False, "message": "quota system error"}, status_code=500
        )
    if not ok:
        ai_quota.webhook_quota_exceeded({"prompt": prompt})
        return JSONResponse({"ok": False, "message": "quota_exceeded"}, status_code=200)

    try:
        # log before delegating to provider
        try:
            from datetime import datetime as _dt, timezone as _tz
            import json as _json

            _logp = Path(__file__).parent / "tools" / "webhook_events.log"
            with open(_logp, "a", encoding="utf-8") as _lf:
                _lf.write(
                    _json.dumps(
                        {
                            "time": _dt.now(_tz.utc).isoformat(),
                            "action": "endpoint_before_call_provider",
                            "payload_keys": list(payload.keys())
                            if isinstance(payload, dict)
                            else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass

        resp = ai_quota.call_provider(payload)

        try:
            from datetime import datetime as _dt, timezone as _tz
            import json as _json

            _logp = Path(__file__).parent / "tools" / "webhook_events.log"
            with open(_logp, "a", encoding="utf-8") as _lf:
                _lf.write(
                    _json.dumps(
                        {
                            "time": _dt.now(_tz.utc).isoformat(),
                            "action": "endpoint_after_call_provider",
                            "provider_resp_keys": list(resp.keys())
                            if isinstance(resp, dict)
                            else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass
        return JSONResponse(
            {"ok": True, "provider": resp.get("provider"), "result": resp}
        )
    except Exception as e:
        ai_quota.rollback_call()
        return JSONResponse({"ok": False, "message": str(e)}, status_code=500)


@app.get("/downloads/{filename}")
async def download_file(filename: str):
    filepath = DOWNLOADS / filename
    if filepath.exists():
        return FileResponse(filepath, filename=filename)
    return JSONResponse({"ok": False, "message": "Not found"}, status_code=404)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    if len(connected_clients) > server_stats["peak_clients"]:
        server_stats["peak_clients"] = len(connected_clients)

    # send welcome message
    try:
        await websocket.send_json(
            {
                "type": "connection",
                "message": "🚀 Lucia Minimal Server에 연결되었습니다.",
                "timestamp": datetime.now().isoformat(),
                "client_id": len(connected_clients),
            }
        )

        # receive loop
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except Exception:
                await websocket.send_json({"ok": False, "error": "invalid json"})
                continue

            if msg.get("type") == "command":
                text = msg.get("text", "")
                command_data = command_parser.parse_enhanced_commands(text)
                # add to history
                command_history.append(
                    {
                        "text": text,
                        "timestamp": datetime.now().isoformat(),
                        "id": str(uuid.uuid4()),
                    }
                )
                server_stats["total_commands"] += 1
                result = await execute_command(command_data)
                if result.get("ok"):
                    server_stats["successful_commands"] += 1
                else:
                    server_stats["failed_commands"] += 1

                await websocket.send_json(
                    {
                        "type": "command_result",
                        "original_text": text,
                        "result": result,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
    except Exception:
        connected_clients.discard(websocket)


@app.post("/command")
async def command_endpoint(request: Request):
    """Generic command endpoint used by tests.
    It parses free-text commands via `command_parser.parse_enhanced_commands`
    and dispatches to `dispatcher.execute_command`.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    text = payload.get("text") or payload.get("command") or ""
    cmd = command_parser.parse_enhanced_commands(text)
    # include raw text for ai_chat fallback
    if cmd.get("type") == "ai_chat":
        cmd["text"] = text

    # execute and return result
    try:
        result = await execute_command(cmd)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"ok": False, "message": str(e)}, status_code=500)


@app.post("/file")
async def file_endpoint(request: Request):
    """Compatibility /file endpoint used by tests.
    Accepts JSON payloads with an 'action' key: create_python, create_html,
    create_text, list, download, delete. Delegates to lucia_core.file_ops.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    action = payload.get("action")
    try:
        if action == "create_python":
            res = await file_ops.handle_create_python()
            return JSONResponse(res)
        if action == "create_html":
            res = await file_ops.handle_create_html()
            return JSONResponse(res)
        if action == "create_text":
            res = await file_ops.handle_create_text()
            return JSONResponse(res)
        if action == "list":
            res = await file_ops.handle_list_files()
            return JSONResponse(res)
        if action == "download":
            filename = payload.get("filename")
            if not filename:
                return JSONResponse(
                    {"ok": False, "message": "missing filename"}, status_code=400
                )
            return JSONResponse(await file_ops.handle_download(filename))
        if action == "delete":
            filename = payload.get("filename")
            if not filename:
                return JSONResponse(
                    {"ok": False, "message": "missing filename"}, status_code=400
                )
            return JSONResponse(await file_ops.handle_delete(filename))
    except Exception as e:
        return JSONResponse({"ok": False, "message": str(e)}, status_code=500)

    return JSONResponse({"ok": False, "message": "unknown action"}, status_code=400)


@app.post("/quantum/run")
async def quantum_run(request: Request):
    """Simple guarded quantum stub used by tests.
    If payload indicates an explicit 'use_stub' or quantum backend unavailable,
    return a simulated response. Otherwise perform a tiny simulated run.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    circuit = payload.get("circuit") or payload.get("program") or ""

    use_stub = bool(payload.get("use_stub"))
    if use_stub or os.getenv("ENABLE_QUANTUM", "false").lower() not in (
        "1",
        "true",
        "yes",
    ):
        stub = f"(quantum-stub) simulated result for circuit: {str(circuit)[:120]}"
        return JSONResponse({"ok": True, "response": stub})

    # In future this would delegate to a quantum backend; for now return stub
    stub = f"(quantum-stub) simulated result for circuit: {str(circuit)[:120]}"
    return JSONResponse({"ok": True, "response": stub})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
