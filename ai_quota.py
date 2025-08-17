import os
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()
# compute paths lazily so tests can monkeypatch env vars after import
DEFAULT_USAGE = os.path.join(os.path.dirname(__file__), 'ai_usage.json')
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(__file__), 'tools')
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
WEBHOOK_LOG = os.path.join(DEFAULT_LOG_DIR, 'webhook_events.log')


def _usage_file_path() -> str:
    """Return the current usage file path, checking env var each call."""
    return os.getenv('AI_QUOTA_FILE', DEFAULT_USAGE)


def _read_usage():
    path = _usage_file_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _write_usage(data):
    path = _usage_file_path()
    parent = Path(path).parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    tmp = path + '.tmp'
    try:
        # write to temp file, flush and fsync, then atomically replace
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        try:
            os.replace(tmp, path)
            return
        except Exception:
            # fallback: try direct write to final path
            pass
    except Exception:
        # if temp write failed, fall through to direct write
        pass

    # final fallback: write directly to the target file
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def _today_key():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


def reserve_call():
    """Attempt to reserve a single AI call. Returns True if reserved, False if quota exceeded."""
    with _lock:
        path = _usage_file_path()
        # minimal debug: append an event so test runs can be inspected when stdout is unavailable
        try:
            dbg = {'time': datetime.now(timezone.utc).isoformat(), 'action': 'reserve_attempt', 'usage_path': path,
                   'AI_DAILY_CALL_LIMIT': os.getenv('AI_DAILY_CALL_LIMIT'), 'AI_MAX_CALLS_PER_MINUTE': os.getenv('AI_MAX_CALLS_PER_MINUTE')}
            with open(WEBHOOK_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps(dbg, ensure_ascii=False) + '\n')
        except Exception:
            pass

        data = _read_usage()
        today = _today_key()
        if data.get('date') != today:
            data = {'date': today, 'daily': 0, 'minutes': {}}
        daily_limit = int(os.getenv('AI_DAILY_CALL_LIMIT', '100'))
        per_minute = int(os.getenv('AI_MAX_CALLS_PER_MINUTE', '10'))

        # minute bucket
        minute_key = datetime.now(timezone.utc).strftime('%Y%m%d%H%M')
        mins = data.get('minutes', {})
        minute_count = int(mins.get(minute_key, 0))
        if minute_count + 1 > per_minute:
            # write debug file next to usage to help tests diagnose
            try:
                dbg_path = Path(path).with_name(Path(path).name + '.debug.json')
                with open(dbg_path, 'w', encoding='utf-8') as df:
                    json.dump({'reason': 'per_minute_exceeded', 'minute_key': minute_key, 'minute_count': minute_count, 'per_minute': per_minute, 'data': data}, df)
            except Exception:
                pass
            return False

        if data.get('daily', 0) + 1 > daily_limit:
            try:
                dbg_path = Path(path).with_name(Path(path).name + '.debug.json')
                with open(dbg_path, 'w', encoding='utf-8') as df:
                    json.dump({'reason': 'daily_exceeded', 'daily': data.get('daily', 0), 'daily_limit': daily_limit, 'data': data}, df)
            except Exception:
                pass
            return False

        # reserve
        mins[minute_key] = minute_count + 1
        data['minutes'] = mins
        data['daily'] = data.get('daily', 0) + 1
        data['date'] = today
        _write_usage(data)
        try:
            with open(WEBHOOK_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'action': 'reserve_committed', 'usage_path': path, 'data': data}, ensure_ascii=False) + '\n')
        except Exception:
            pass
        try:
            dbg_path = Path(path).with_name(Path(path).name + '.debug.json')
            with open(dbg_path, 'w', encoding='utf-8') as df:
                json.dump({'action': 'reserve_committed', 'data': data}, df)
        except Exception:
            pass
        return True


def rollback_call():
    """Rollback a previous reservation (decrement counters)."""
    with _lock:
        path = _usage_file_path()
        try:
            with open(WEBHOOK_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'action': 'rollback_attempt', 'usage_path': path}, ensure_ascii=False) + '\n')
        except Exception:
            pass
        data = _read_usage()
        if not data:
            return
        minute_key = datetime.now(timezone.utc).strftime('%Y%m%d%H%M')
        mins = data.get('minutes', {})
        # Prefer to decrement current minute bucket, but if it's zero or missing,
        # decrement the most recent non-zero bucket to be resilient to minor timing drift.
        if mins.get(minute_key):
            mins[minute_key] = max(0, int(mins[minute_key]) - 1)
            data['minutes'] = mins
        else:
            # find latest minute key with count > 0
            nonzeros = sorted((k for k, v in mins.items() if int(v) > 0), reverse=True)
            if nonzeros:
                k = nonzeros[0]
                mins[k] = max(0, int(mins.get(k, 0)) - 1)
                data['minutes'] = mins
        data['daily'] = max(0, int(data.get('daily', 0)) - 1)
        _write_usage(data)
        try:
            with open(WEBHOOK_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'action': 'rollback_committed', 'usage_path': path, 'data': data}, ensure_ascii=False) + '\n')
        except Exception:
            pass
        try:
            dbg_path = Path(path).with_name(Path(path).name + '.debug.json')
            with open(dbg_path, 'w', encoding='utf-8') as df:
                json.dump({'action': 'rollback_committed', 'data': data}, df)
        except Exception:
            pass


def webhook_quota_exceeded(details: dict):
    """Stub for webhook: append an event to a local log to avoid external network at import/run."""
    event = {'time': datetime.now(timezone.utc).isoformat(), 'event': 'quota_exceeded', 'details': details}
    try:
        with open(WEBHOOK_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
    except Exception:
        pass


def call_provider(payload: dict):
    """Delegate to provider adapters (ai_providers) with safe fallback."""
    # Log payload details to help debug tests where mock_error should trigger
    try:
        with open(WEBHOOK_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'action': 'call_provider_payload', 'keys': list(payload.keys()) if isinstance(payload, dict) else None, 'payload': payload}, ensure_ascii=False) + '\n')
    except Exception:
        pass

    # Defensive: if tests enable mock mode and request asks for mock_error,
    # raise here to guarantee the caller sees a provider failure and can
    # rollback. This layered check (endpoint/provider/quota) ensures
    # deterministic behavior in test environments.
    try:
        if isinstance(payload, dict) and payload.get('mock_error') and os.getenv('AI_USE_MOCK', 'false').lower() in ('1', 'true', 'yes'):
            raise RuntimeError('simulated provider error (ai_quota)')
    except Exception:
        # propagate to caller
        raise

    # Don't swallow provider exceptions here — let the caller (HTTP endpoint)
    # handle rollback and error reporting. This ensures real provider
    # failures propagate up and tests relying on exceptions behave correctly.
    try:
        with open(WEBHOOK_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'action': 'ai_quota_delegating_to_providers', 'module_file': __file__, 'pid': os.getpid(), 'payload_keys': list(payload.keys()) if isinstance(payload, dict) else None}, ensure_ascii=False) + '\n')
    except Exception:
        pass
    import ai_providers
    return ai_providers.call_best_provider(payload)
