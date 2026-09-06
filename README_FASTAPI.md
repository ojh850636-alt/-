FastAPI quick start

This small wrapper makes it easy to run the project's FastAPI app locally.

Requirements
- Python 3.10+ (the repo targets newer, but 3.10+ works)
- install project requirements: pip install -r requirements.txt

Run locally

PowerShell:
```powershell
python .\fastapi_entry.py
```

Or use uvicorn directly:

```powershell
& "C:\Program Files\Python\python.exe" -m uvicorn lucia_app:app --reload
```

Health and sample endpoints
- GET /health
- GET /ai/status
- POST /ai/chat (JSON body: {"prompt": "..."})
- POST /command
- POST /file
- WebSocket /ws

Notes
- The app is a thin wrapper around `lucia_ultimate_quantum_integrated_fixed.py` which is intentionally minimal and test-friendly.
- If you get Pydantic deprecation warnings, they are suppressed in the app for test stability.
