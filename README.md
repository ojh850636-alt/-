# Lucia - Cleaned Minimal Server

Quick start and developer notes for the cleaned Lucia FastAPI app used in tests.

Quickstart
- Create a virtualenv and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Run the app (development):

```powershell
python -m uvicorn lucia_ultimate_quantum_integrated_fixed:app --reload
```

Run tests

```powershell
python -m pytest -q
```

Docker

Build and run locally:

```powershell
docker build -t lucia-cleaned .
docker run -p 8000:8000 lucia-cleaned
```

CI

This repository includes a GitHub Actions workflow that installs dependencies and runs the test suite on push and pull_request.

Notes

- The `patches/` directory is intentionally ignored and kept as local diagnostic artifacts by developers.
Lucia Minimal FileOps Server

Quick start

1. Create a venv and activate it.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install requirements

```powershell
python -m pip install -r requirements.txt
```

3. Run server

```powershell
python -m uvicorn lucia_ultimate_quantum_integrated_fixed:app --host 127.0.0.1 --port 8002
```

Open http://127.0.0.1:8002/ to use the UI.

Optional features

To enable optional AI or quantum stubs (safe, guarded imports), set environment variables before running the server:

PowerShell example:

```powershell
$env:ENABLE_AI = 'true'; $env:ENABLE_QUANTUM = 'true'
```

If the optional packages are not installed the server will continue to run without those features.

Developer notes

To install developer/test dependencies (used by the included tests and helper scripts):

```powershell
python -m pip install -r requirements-dev.txt
```

OpenAI (optional)

To enable real OpenAI responses instead of the built-in stub:

1. Install OpenAI package (optional):

```powershell
python -m pip install openai
```

2. Set your OpenAI API key in the environment before starting the server:

```powershell
$env:OPENAI_API_KEY = 'sk-...'
$env:ENABLE_AI = 'true'
python -m uvicorn lucia_ultimate_quantum_integrated_fixed:app --host 127.0.0.1 --port 8002
```

If `OPENAI_API_KEY` or the OpenAI package is missing the server will continue to run and the `/ai/chat` endpoint will return a safe stub response.

Secrets and timeouts

We recommend storing secrets in environment variables or a local `.env` file and not committing them. Example using a `.env` loader (python-dotenv):

```powershell
python -m pip install python-dotenv
# create a .env file with OPENAI_API_KEY=sk-...
```

You can also configure AI timeouts and model with environment variables:

```powershell
$env:OPENAI_MODEL = 'gpt-4o-mini'
$env:OPENAI_MAX_TOKENS = '200'
$env:AI_TIMEOUT = '8' # seconds
```

Recommended quota examples

```powershell
# gentle defaults you can adjust
$env:AI_MAX_CALLS_PER_MINUTE = '20'
$env:AI_DAILY_CALL_LIMIT = '1000'
$env:AI_MONTHLY_CALL_LIMIT = '20000'
```

Optional dependencies

If you want to enable optional integrations (for example OpenAI or quantum packages) you can install the optional list:

```powershell
python -m pip install -r requirements-optional.txt
```

This will install packages listed in `requirements-optional.txt` such as `openai`.

Webhook alert receiver (local test)

To test webhook alerts locally, run the simple receiver included in `tools/webhook_receiver.py`:

```powershell
python tools\webhook_receiver.py
# then set AI_ALERT_WEBHOOK_URL=http://127.0.0.1:9001
```

The receiver prints incoming alerts to the console.

Running tests
-------------

Recommended: run tests from the repository root so pytest discovers the test suite correctly.

PowerShell (repo root):

```powershell
Set-Location 'C:\Users\Hi\Desktop\루시아 에이전트\2'
python -m pytest -q
```

To run the test helper which starts the server, waits for the port and runs pytest:

```powershell
Set-Location 'C:\Users\Hi\Desktop\루시아 에이전트\2'
& .\tools\run_tests_with_server.ps1
```

If you hit a PowerShell `-File` error when invoking the helper, ensure you provide the correct (root-relative or absolute) path to the script.

## Local dev & checks

These helper commands make it easy to run the project's tests, linter and an optional Docker build locally.

PowerShell helper:

```powershell
# run all checks (tests + ruff + optional docker build)
.
tools\run_all.ps1

# skip docker build
.
tools\run_all.ps1 -SkipDocker
```

Python helper script:

```bash
python tools/dev_checks.py
```

Manual commands:

```powershell
# run tests
C:/Users/Hi/AppData/Local/Programs/Python/Python313/python.exe -m pytest -q

# run ruff autofix
C:/Users/Hi/AppData/Local/Programs/Python/Python313/python.exe -m ruff . --fix

# build docker image (if docker installed)
docker build -t lucia-app .
```

If you run into permission or environment issues, run the commands manually to capture the full console output and paste it here for debugging.

