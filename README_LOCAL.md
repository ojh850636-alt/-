Local run & test guide

This repository contains a cleaned, modularized FastAPI server and supporting
helpers. The following steps help run tests and the server locally in a safe
way (no network calls during tests by default).

Prerequisites
- Python 3.11+ (recommended)
## Quick start (PowerShell helper)

If you want a one-command local startup (no GitHub auth), use the provided
PowerShell helper in `tools/quick_start.ps1` which starts the server,
waits for the port and opens the OpenAPI UI in your browser.

From PowerShell:

```powershell
# run the helper (defaults to port 8002)
.\tools\quick_start.ps1

# run on a different port
.\tools\quick_start.ps1 -Port 8000
```

Notes:
- This helper uses the local Python and uvicorn in your PATH — ensure the
   virtualenv is activated if you're using one.
- To stop the server, run `Stop-Process -Id <PID>` (the helper prints the PID).

Developer quick commands
------------------------

If you want a one-command flow that starts the server, runs tests, and then
stops the server locally, use the helper in `tools/run_server_and_tests.ps1`:

```powershell
.\tools\run_server_and_tests.ps1
```

To keep your local work isolated before pushing, create a branch and commit
the changes locally:

```powershell
git checkout -b fix/quick-start
git add -A
git commit -m "Add quick start and import test cleanup"
git status --short
```


- pip

Quick start (local)

1) Create and activate a virtual environment (optional but recommended):

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # PowerShell on Windows

2) Install deps:

   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

3) Run tests (the test suite uses a mocked AI provider by default):

   python -m pytest -q

4) Run concurrency smoke locally (standalone script):

   python .\tools\run_concurrency_smoke.py

Run the server

To run the cleaned runner locally:

   set PORT=8000; python lucia_ultimate_quantum_integrated_fixed.py

Try it (alternative)
---------------------

You can also run the test-friendly app via the new `lucia_app` entrypoint:

```powershell
# install uvicorn once
python -m pip install uvicorn
# run via the small helper
python run_uvicorn.py
``` 

To create a PR, open `PR_BODY.md`, copy the contents and paste into a new
pull request description on GitHub for branch `fix/quick-start`.

Notes
- Tests and CI set `AI_USE_MOCK=true` to avoid network calls. To enable a real
  provider, set provider-specific environment variables and keys and edit
  `ai_providers.py` to enable the provider you want.
- Secrets should be stored by `store_keys.ps1` or CI secrets; never commit
  secrets into the repository.

CI

A GitHub Actions workflow is included at `.github/workflows/ci.yml` which runs
pytest and the concurrency smoke script on push/PR (Ubuntu + Windows).
