PR: fix/quick-start — Modularize providers, stabilize tests, and add run helpers

Summary

This PR stabilizes provider mocking and testing behavior, extracts test-friendly entrypoints, and adds developer run helpers and CI artifacts collection.

What changed (high level)
- ai_providers.py: Reordered mock/forced-error handling so per-call mock and mock_error are honored even when no real providers are configured; global AI_USE_MOCK only applies when real providers are available.
- tests/test_ai_providers_custom.py: Added guard to ensure global mock env does not interfere with provider-order tests.
- Added developer helpers and artifacts: PR_COMMANDS_HELPER.ps1, PR_BODY_DRAFT.md, tools/collect_test_artifacts.py, PR convenience scripts, smoke runner outputs.
- Small logging improvements to help CI collect provider and quota events (webhook_events.log).

Why

Tests and local CI needed deterministic mock behavior for testing provider failures and quota rollback without requiring network calls. Previously global mock settings could unintentionally override tests expecting stub providers.

How to create the PR

Option A — use GitHub CLI (preferred). If `gh` is installed and on PATH, run:

```powershell
gh pr create --fill --body-file PR_BODY_FINAL.md
```

If `gh` is not on PATH but installed in `C:\Program Files\GitHub CLI`, use:

```powershell
& 'C:\Program Files\GitHub CLI\gh.exe' pr create --fill --body-file PR_BODY_FINAL.md
```

Option B — open the PR page in your browser and paste the PR body:

https://github.com/ojh850636-alt/-/pull/new/fix/quick-start?expand=1

Notes

- `tests/conftest.py` sets `AI_USE_MOCK=true` for deterministic test runs; adjust when running against real providers.
- If you want me to open the PR automatically via the API, provide a GitHub token with repo:status and repo scope (best done manually for security).

Files changed (quick list)
- ai_providers.py (logic rework)
- tests/test_ai_providers_custom.py (test fix)
- PR_COMMANDS_HELPER.ps1 (helper)
- PR_BODY_DRAFT.md -> PR_BODY_FINAL.md (this file)
- tools/collect_test_artifacts.py

If you want, I can also prepare the PR description with more granular commit/file lists.
