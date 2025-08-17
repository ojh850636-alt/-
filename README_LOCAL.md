Local run & test guide

This repository contains a cleaned, modularized FastAPI server and supporting
helpers. The following steps help run tests and the server locally in a safe
way (no network calls during tests by default).

Prerequisites
- Python 3.11+ (recommended)
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

Notes
- Tests and CI set `AI_USE_MOCK=true` to avoid network calls. To enable a real
  provider, set provider-specific environment variables and keys and edit
  `ai_providers.py` to enable the provider you want.
- Secrets should be stored by `store_keys.ps1` or CI secrets; never commit
  secrets into the repository.

CI

A GitHub Actions workflow is included at `.github/workflows/ci.yml` which runs
pytest and the concurrency smoke script on push/PR (Ubuntu + Windows).
