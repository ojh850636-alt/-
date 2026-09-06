Title: Fix/quick-start — modularize monolith, add test-friendly FastAPI app, provider fallback

Summary

This branch refactors the large monolithic entrypoints into a modular, test-friendly
FastAPI app and supporting `lucia_core` modules. Key user-facing changes:

- Adds `lucia_app.py` which re-exports the test-friendly `app` from
  `lucia_ultimate_quantum_integrated_fixed.py` for easier local runs and CI.
- Ensures deterministic provider behavior:
  - If no real providers are enabled, returns `provider: stub` (tests rely on this).
  - Supports per-call and global mock mode via `mock` in payload or `AI_USE_MOCK` env.
  - `mock_error` and forced-prompt `will error` trigger provider exceptions for rollback tests.
- Adds `/file` and `/quantum/run` endpoints in the fixed server used by tests.
- Makes several safe linting improvements (replace bare excepts, small cleanups).
- Adds `run_uvicorn.py` and `start_uvicorn.ps1` for local testing runs.

Validation

- All tests pass locally: `python -m pytest -q` → `.............. [100%]`

Notes / Follow-ups

- A number of very large monolithic files still contain complex issues flagged by
  linters; these were intentionally left untouched except for safe changes.
- Recommend running CI and reviewing remaining ruff warnings manually.
- If desired, we can convert stubbed provider calls into real provider clients behind
  clear, injectable interfaces in a follow-up.
