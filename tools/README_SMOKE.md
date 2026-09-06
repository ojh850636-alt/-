Smoke check

Run a small smoke test that starts the FastAPI app and checks /health.

Usage:

```
py -3 tools\smoke_check.py
```

Requirements:
- `uvicorn` must be installed. Install with `py -3 -m pip install -r requirements.txt`.

Exit codes:
- 0 success
- 2 failure
