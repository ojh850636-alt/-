#!/usr/bin/env python3
"""Lucia: minimal server with file operations restored.

This file provides a small, safe set of APIs:
- /health
- /command (simple parser -> file ops)
- /file  (explicit file actions)
- /downloads/{filename}

File IO uses threads to avoid blocking the event loop.
"""

from datetime import datetime, timedelta
import os
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import asyncio

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import json
import threading
import time
from urllib import request as _urllib_request, error as _urllib_error
try:
    import requests
except Exception:
    requests = None
import warnings

# Suppress Pydantic v2 deprecation messages about `.dict()` to keep CI logs clean.
try:
    from pydantic.errors import PydanticDeprecatedSince20
    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
except Exception:
    warnings.filterwarnings("ignore", message=r".*The `dict` method is deprecated.*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lucia.server")

app = FastAPI(title="Lucia FileOps")
templates = Jinja2Templates(directory="templates")

connected_clients = set()


async def _send_safe(ws, msg: Dict[str, Any]):
    try:
        await ws.send_json(msg)
    except Exception:
        try:
            connected_clients.discard(ws)
        except Exception:
            pass


def broadcast_json(msg: Dict[str, Any]):
    # schedule sends to all connected websockets without blocking
    for ws in list(connected_clients):
        try:
            asyncio.create_task(_send_safe(ws, msg))
        except Exception:
            connected_clients.discard(ws)

START = datetime.now()
DOWNLOADS_DIR = Path("downloads")
DOWNLOADS_DIR.mkdir(exist_ok=True)

# Feature flags (enable heavy integrations via environment variables)
ENABLE_AI = os.getenv("ENABLE_AI", "false").lower() in ("1", "true", "yes")
ENABLE_QUANTUM = os.getenv("ENABLE_QUANTUM", "false").lower() in ("1", "true", "yes")

# guarded imports
ai_client = None
ai_available = False
if ENABLE_AI:
    try:
        # try to import lightweight OpenAI client; keep optional and safe
        import openai
        ai_client = openai
        ai_available = True
    except Exception:
        ai_client = None
        ai_available = False

# External provider flags (Groq / OpenRouter for LLaMA3)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_API_URL = os.getenv('GROQ_API_URL')  # full endpoint, e.g. https://api.groq.ai/v1/...
GROQ_AVAILABLE = bool(GROQ_API_KEY and GROQ_API_URL and requests is not None)

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_URL = os.getenv('OPENROUTER_URL')  # full endpoint for model
OPENROUTER_AVAILABLE = bool(OPENROUTER_API_KEY and OPENROUTER_URL and requests is not None)


def ai_status() -> Dict[str, Any]:
    return {
        "enabled_flag": ENABLE_AI,
        "ai_package": bool(ai_client),
        "ai_available": ai_available,
        "openai_api_key_present": bool(os.getenv('OPENAI_API_KEY')),
        "groq_available": GROQ_AVAILABLE,
        "openrouter_available": OPENROUTER_AVAILABLE,
    }


@app.get('/ai/status')
async def _ai_status():
    return ai_status()


# --- Quota & monitoring helpers ---
AI_QUOTA_FILE = os.getenv('AI_QUOTA_FILE', 'ai_usage.json')
_ai_usage_lock = threading.Lock()

def _load_ai_usage() -> Dict[str, Any]:
    try:
        if not os.path.exists(AI_QUOTA_FILE):
            return {'daily':{}, 'monthly':{}, 'last_minute': {'ts':0,'count':0}, 'total_calls':0}
        with open(AI_QUOTA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {'daily':{}, 'monthly':{}, 'last_minute': {'ts':0,'count':0}, 'total_calls':0}

def _save_ai_usage(data: Dict[str, Any]) -> None:
    try:
        # prune old daily entries according to retention policy before saving
        try:
            retention = int(os.getenv('AI_USAGE_RETENTION_DAYS', '30'))
        except Exception:
            retention = 30
        if 'daily' in data:
            cutoff = datetime.now() - timedelta(days=retention)
            keep = {}
            for dstr, cnt in data['daily'].items():
                try:
                    dt = datetime.strptime(dstr, '%Y-%m-%d')
                    if dt >= cutoff:
                        keep[dstr] = cnt
                except Exception:
                    # keep unknown formats
                    keep[dstr] = cnt
            data['daily'] = keep

        with open(AI_QUOTA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception('failed to save ai usage')


def _send_alert_webhook(payload: Dict[str, Any]) -> None:
    """Send a JSON payload to configured webhook URL (non-blocking)."""
    url = os.getenv('AI_ALERT_WEBHOOK_URL')
    if not url:
        return
    try:
        data = json.dumps(payload).encode('utf-8')
        req = _urllib_request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
        # run synchronous urllib in a thread to avoid blocking
        def _do():
            try:
                with _urllib_request.urlopen(req, timeout=5) as resp:
                    resp.read()
            except _urllib_error.HTTPError as e:
                logger.warning(f'alert webhook HTTPError: {e.code}')
            except Exception:
                logger.exception('alert webhook failed')
        threading.Thread(target=_do, daemon=True).start()
    except Exception:
        logger.exception('failed to prepare webhook')


async def _call_groq_async(text: str, timeout: float = 10.0) -> Tuple[bool, Any]:
    """Async-friendly Groq HTTP adapter. Returns (ok, result_or_error)."""
    if not GROQ_AVAILABLE:
        return False, 'groq_not_configured'
    try:
        headers = {'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'}
        payload = {'prompt': text}

        def _do():
            r = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            try:
                return True, r.json()
            except Exception:
                return True, r.text

        ok, result = await asyncio.to_thread(_do)
        return ok, result
    except Exception as e:
        logger.exception('groq call failed')
        return False, str(e)


async def _call_openrouter_async(text: str, timeout: float = 10.0) -> Tuple[bool, Any]:
    """Async-friendly OpenRouter adapter. Returns (ok, result_or_error)."""
    if not OPENROUTER_AVAILABLE:
        return False, 'openrouter_not_configured'
    try:
        headers = {'Authorization': f'Bearer {OPENROUTER_API_KEY}', 'Content-Type': 'application/json'}
        payload = {'input': text}

        def _do():
            r = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            try:
                return True, r.json()
            except Exception:
                return True, r.text

        ok, result = await asyncio.to_thread(_do)
        return ok, result
    except Exception as e:
        logger.exception('openrouter call failed')
        return False, str(e)


def _maybe_alert_quota_exceeded(message: str) -> None:
    url = os.getenv('AI_ALERT_WEBHOOK_URL')
    if not url:
        return
    payload = {
        'type': 'ai_quota_exceeded',
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'host': os.getenv('HOSTNAME') or os.getenv('COMPUTERNAME') or 'unknown'
    }
    _send_alert_webhook(payload)

def _reserve_ai_call() -> Tuple[bool, str]:
    """Try to reserve one AI call; returns (ok, message)."""
    with _ai_usage_lock:
        data = _load_ai_usage()
        now = datetime.now()
        minute_ts = int(now.timestamp()//60)
        date = now.strftime('%Y-%m-%d')
        month = now.strftime('%Y-%m')

        # read limits from env
        max_min = int(os.getenv('AI_MAX_CALLS_PER_MINUTE', '0')) or 0
        daily_max = int(os.getenv('AI_DAILY_CALL_LIMIT', '0')) or 0
        monthly_max = int(os.getenv('AI_MONTHLY_CALL_LIMIT', '0')) or 0

        # minute window
        lm = data.get('last_minute', {'ts':0,'count':0})
        if lm.get('ts') == minute_ts:
            minute_count = int(lm.get('count',0))
        else:
            minute_count = 0
        if max_min and minute_count + 1 > max_min:
            msg = f'max per-minute quota exceeded ({minute_count}/{max_min})'
            # alert operator if webhook configured
            try:
                _maybe_alert_quota_exceeded(msg)
            except Exception:
                logger.exception('failed to call alert webhook')
            return False, msg

        # daily
        daily_count = int(data.get('daily',{}).get(date,0))
        if daily_max and daily_count + 1 > daily_max:
            msg = f'daily quota exceeded ({daily_count}/{daily_max})'
            try:
                _maybe_alert_quota_exceeded(msg)
            except Exception:
                logger.exception('failed to call alert webhook')
            return False, msg

        # monthly
        monthly_count = int(data.get('monthly',{}).get(month,0))
        if monthly_max and monthly_count + 1 > monthly_max:
            msg = f'monthly quota exceeded ({monthly_count}/{monthly_max})'
            try:
                _maybe_alert_quota_exceeded(msg)
            except Exception:
                logger.exception('failed to call alert webhook')
            return False, msg

        # reserve
        # update minute
        data.setdefault('last_minute', {})
        data['last_minute']['ts'] = minute_ts
        data['last_minute']['count'] = minute_count + 1
        # update daily/monthly/total
        data.setdefault('daily', {})
        data['daily'][date] = daily_count + 1
        data.setdefault('monthly', {})
        data['monthly'][month] = monthly_count + 1
        data['total_calls'] = int(data.get('total_calls',0)) + 1

        _save_ai_usage(data)
        return True, 'reserved'

def _rollback_ai_call():
    with _ai_usage_lock:
        data = _load_ai_usage()
        now = datetime.now()
        minute_ts = int(now.timestamp()//60)
        date = now.strftime('%Y-%m-%d')
        month = now.strftime('%Y-%m')

        lm = data.get('last_minute', {'ts':0,'count':0})
        if lm.get('ts') == minute_ts:
            data['last_minute']['count'] = max(0, int(lm.get('count',0)) - 1)
        # daily
        if data.get('daily',{}).get(date):
            data['daily'][date] = max(0, int(data['daily'][date]) - 1)
        # monthly
        if data.get('monthly',{}).get(month):
            data['monthly'][month] = max(0, int(data['monthly'][month]) - 1)
        data['total_calls'] = max(0, int(data.get('total_calls',0)) - 1)
        _save_ai_usage(data)


@app.get('/ai/usage')
async def ai_usage():
    data = _load_ai_usage()
    return {'ok': True, 'usage': data}

# quantum stub availability
quantum_available = False
if ENABLE_QUANTUM:
    try:
        # optional quantum lib placeholder
        import random
        quantum_available = True
    except Exception:
        quantum_available = False


def quantum_status() -> Dict[str, Any]:
    return {"enabled_flag": ENABLE_QUANTUM, "quantum_available": quantum_available}


@app.post('/quantum/run')
async def quantum_run(payload: Dict[str, Any]):
    # simple guarded stub for quantum workloads
    circuit = payload.get('circuit') if isinstance(payload, dict) else None
    if quantum_available:
        try:
            # placeholder: run a simple random measurement if random imported
            res = {"ok": True, "result": random.choice([0, 1])}
            return res
        except Exception:
            return {"ok": False, "message": "quantum execution failed"}
    # stubbed response
    stub = f"(quantum-stub) simulated result for circuit: {str(circuit)[:120]}"
    return {"ok": True, "response": stub}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


async def _write_file(path: Path, content: str) -> None:
    # run blocking file write in thread
    await asyncio.to_thread(path.write_text, content, "utf-8")


async def create_python_file() -> Dict[str, Any]:
    ts = _timestamp()
    fname = f"lucia_python_{ts}.py"
    fp = DOWNLOADS_DIR / fname
    content = f"""#!/usr/bin/env python3
# Generated by Lucia on {datetime.now().isoformat()}

def main():
    print('Hello from Lucia')


if __name__ == '__main__':
    main()
"""
    await _write_file(fp, content)
    res = {"ok": True, "filename": fname}
    broadcast_json({"type": "file_update", "action": "create", "result": res})
    return res


async def create_html_file() -> Dict[str, Any]:
    ts = _timestamp()
    fname = f"lucia_webpage_{ts}.html"
    fp = DOWNLOADS_DIR / fname
    html = """<!doctype html>
<html lang="ko">
<head><meta charset="utf-8"><title>Lucia</title></head>
<body><h1>Lucia</h1><p>Generated file.</p></body>
</html>"""
    await _write_file(fp, html)
    res = {"ok": True, "filename": fname}
    broadcast_json({"type": "file_update", "action": "create", "result": res})
    return res


async def create_text_file() -> Dict[str, Any]:
    ts = _timestamp()
    fname = f"lucia_notes_{ts}.txt"
    fp = DOWNLOADS_DIR / fname
    text = f"Lucia notes\ncreated: {datetime.now().isoformat()}\n"
    await _write_file(fp, text)
    res = {"ok": True, "filename": fname}
    broadcast_json({"type": "file_update", "action": "create", "result": res})
    return res


async def list_files() -> Dict[str, Any]:
    files = []
    for p in DOWNLOADS_DIR.iterdir():
        if p.is_file():
            stat = p.stat()
            files.append({"name": p.name, "size": stat.st_size, "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()})
    res = {"ok": True, "files": files}
    # don't broadcast list by default
    return res


async def file_action(action: str, filename: str | None = None) -> Dict[str, Any]:
    if action == "create_python":
        return await create_python_file()
    if action == "create_html":
        return await create_html_file()
    if action == "create_text":
        return await create_text_file()
    if action == "list":
        return await list_files()
    if action == "download":
        if not filename:
            return {"ok": False, "message": "filename required"}
        fp = DOWNLOADS_DIR / filename
        if not fp.exists():
            return {"ok": False, "message": "not found"}
        return {"ok": True, "filename": filename, "download_url": f"/downloads/{filename}"}
    if action == "delete":
        if not filename:
            return {"ok": False, "message": "filename required"}
        fp = DOWNLOADS_DIR / filename
        if not fp.exists():
            return {"ok": False, "message": "not found"}
        try:
            fp.unlink()
            res = {"ok": True, "message": "deleted", "filename": filename}
            broadcast_json({"type": "file_update", "action": "delete", "result": res})
            return res
        except Exception as e:
            return {"ok": False, "message": str(e)}
    return {"ok": False, "message": "unknown action"}


def simple_parse(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    if any(x in t for x in ["파이썬", ".py"]):
        return {"type": "file", "action": "create_python"}
    if any(x in t for x in ["html", ".html"]):
        return {"type": "file", "action": "create_html"}
    if any(x in t for x in ["텍스트", ".txt"]):
        return {"type": "file", "action": "create_text"}
    if any(x in t for x in ["목록", "다운로드"]):
        return {"type": "file", "action": "list"}
    return {"type": "echo", "text": text}


@app.get("/health")
async def health() -> Dict[str, Any]:
    uptime = (datetime.now() - START).total_seconds()
    return {"ok": True, "uptime_seconds": uptime}


@app.get("/")
async def index(request: Request):
    tpl = Path("templates") / "index.html"
    if tpl.exists():
        return templates.TemplateResponse("index.html", {"request": request, "ai_available": ai_available})
    return HTMLResponse("<h1>Lucia FileOps</h1><p>No UI installed.</p>")


@app.post("/command")
async def command(payload: Dict[str, Any]):
    text = payload.get("text") if isinstance(payload, dict) else None
    if not text:
        return JSONResponse({"ok": False, "message": "text required"}, status_code=400)
    parsed = simple_parse(text)
    if parsed.get("type") == "file":
        res = await file_action(parsed.get("action"))
        return res
    return {"ok": True, "received": text}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    try:
        await ws.send_json({"type": "connected", "msg": "welcome"})
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except Exception:
                msg = {"type": "raw", "text": data}

            # handle message types
            if msg.get("type") == "command":
                parsed = simple_parse(msg.get("text", ""))
                if parsed.get("type") == "file":
                    res = await file_action(parsed.get("action"))
                    await ws.send_json({"type": "file_result", "result": res})
                else:
                    await ws.send_json({"type": "echo", "text": msg.get("text")})
            elif msg.get("type") == "file_action":
                action = msg.get("action")
                filename = msg.get("filename")
                res = await file_action(action, filename)
                await ws.send_json({"type": "file_result", "result": res})
            else:
                await ws.send_json({"type": "unknown", "payload": msg})
    except WebSocketDisconnect:
        connected_clients.discard(ws)
    except Exception as e:
        connected_clients.discard(ws)
        logger.exception("ws error")


from lucia_core.schemas import FileActionRequest


@app.post("/file")
async def file_endpoint(req: FileActionRequest):
    action = req.action
    filename = req.filename
    res = await file_action(action, filename)
    return res


@app.post('/ai/chat')
async def ai_chat(payload: Dict[str, Any]):
    text = payload.get('text') if isinstance(payload, dict) else None
    use_stub = bool(payload.get('use_stub')) if isinstance(payload, dict) else False
    if not text:
        return JSONResponse({'ok': False, 'message': 'text required'}, status_code=400)
    # If AI client is available use it, otherwise return a safe stub
    if use_stub:
        # explicit client-side request to force stub
        stub = f"(stub-forced) I received: {text[:200]}"
        return {'ok': True, 'response': stub}

    # reserve quota before attempting real call
    ok, msg = _reserve_ai_call()
    if not ok:
        return {'ok': False, 'message': msg}

    if ai_available and ai_client is not None:
        try:
            # minimal OpenAI completion example (guarded)
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                ai_client.api_key = api_key
            model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
            max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '150'))
            ai_timeout = float(os.getenv('AI_TIMEOUT', '10'))

            def call_openai():
                return ai_client.ChatCompletion.create(model=model, messages=[{"role": "user", "content": text}], max_tokens=max_tokens)

            # run blocking call in thread and enforce timeout
            try:
                resp = await asyncio.wait_for(asyncio.to_thread(call_openai), timeout=ai_timeout)
            except asyncio.TimeoutError:
                logger.warning('ai call timed out')
                _rollback_ai_call()
                return {'ok': False, 'message': 'ai timeout'}

            out = None
            try:
                out = resp.choices[0].message.content if hasattr(resp, 'choices') else str(resp)
            except Exception:
                out = str(resp)
            return {'ok': True, 'response': out}
        except Exception as e:
            logger.exception('ai call failed')
            _rollback_ai_call()
            return {'ok': False, 'message': 'ai call failed, check server logs for details'}
    # If OpenAI not available or not chosen, try other providers (OpenRouter -> Groq)
    # First try OpenRouter if configured
    if OPENROUTER_AVAILABLE:
        try:
            ok, res = await asyncio.wait_for(_call_openrouter_async(text, timeout=float(os.getenv('AI_TIMEOUT', '10'))), timeout=float(os.getenv('AI_TIMEOUT', '10')))
            if ok:
                return {'ok': True, 'response': res}
        except Exception:
            logger.exception('openrouter call error')
            _rollback_ai_call()

    # Next try Groq
    if GROQ_AVAILABLE:
        try:
            ok, res = await asyncio.wait_for(_call_groq_async(text, timeout=float(os.getenv('AI_TIMEOUT', '10'))), timeout=float(os.getenv('AI_TIMEOUT', '10')))
            if ok:
                return {'ok': True, 'response': res}
        except Exception:
            logger.exception('groq call error')
            _rollback_ai_call()

    # final fallback: stub
    stub = f"(stub) I received: {text[:200]}"
    return {'ok': True, 'response': stub}


@app.get("/downloads/{filename}")
async def download_file(filename: str):
    fp = DOWNLOADS_DIR / filename
    if not fp.exists():
        return JSONResponse({"ok": False, "message": "not found"}, status_code=404)
    return FileResponse(fp)


if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv("SERVER_PORT", "8000"))
    logger.info(f"Starting Lucia FileOps server on 0.0.0.0:{port}")
    logger.info(f"AI flags: {ai_status()}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")