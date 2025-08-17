import requests
import json


def call_ai(endpoint: str = "http://127.0.0.1:8002/ai/chat", text: str = "안녕 루시아"):
    """Call the AI endpoint and return the parsed JSON response.

    This function is safe to import (it does not run at import time). Use it from
    a test or invoke the module directly.
    """
    r = requests.post(endpoint, json={"text": text}, timeout=5)
    try:
        data = r.json()
    except Exception:
        data = {"raw_text": r.text}
    return r.status_code, data


if __name__ == "__main__":
    status, payload = call_ai()
    print(status)
    print(json.dumps(payload, ensure_ascii=False))
