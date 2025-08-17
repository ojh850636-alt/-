#!/usr/bin/env python3
"""Start the FastAPI app locally and verify /health responds.

This is a small, dependency-light smoke check that starts the app in a
separate process (using multiprocessing) and polls /health. It avoids
PowerShell-specific behaviour by running entirely in Python.

Usage:
    py -3 tools\smoke_check.py

Exit codes:
 - 0: success (health responded)
 - 2: failure (no response)
"""

import multiprocessing
import time
import sys
import urllib.request


def run_uvicorn():
    # Import inside child process so the main process doesn't need uvicorn
    try:
        import uvicorn
    except Exception as e:
        print("uvicorn not installed. Install requirements with: pip install -r requirements.txt")
        return
    try:
        from lucia_app import app
    except Exception as e:
        print("Failed to import app:", e)
        return

    # Run uvicorn in the child process (blocks until terminated)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning", reload=False)


def wait_health(url, timeout=12.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                body = r.read().decode(errors="replace")
                return True, body
        except Exception:
            time.sleep(0.5)
    return False, None


if __name__ == "__main__":
    print("Starting smoke check: launching app and polling /health")
    p = multiprocessing.Process(target=run_uvicorn, daemon=True)
    p.start()

    ok, body = wait_health("http://127.0.0.1:8000/health", timeout=15.0)
    if ok:
        print("SMOKE OK: /health responded")
        print(body)
    else:
        print("SMOKE FAIL: /health did not respond within timeout")

    # Ensure the child process is cleaned up
    try:
        if p.is_alive():
            p.terminate()
            p.join(3.0)
    except Exception:
        pass

    sys.exit(0 if ok else 2)
