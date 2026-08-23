# Lucia Consolidation Context

## Invariants
- Public APIs remain stable unless explicitly approved.
- Single-source implementations per capability; duplicates are deprecated.
- All changes must keep tests/build green at each step.
- Deterministic behavior for same inputs; no hidden global state.
- Clear separation: orchestration vs. model I/O vs. tools vs. data.

## Lucia-specific Notes (draft)
- Agent core = planner + executor + memory store + tool router.
- Memory: short-term (conversation), long-term (vector/kv), episodic (worklog).
- Tooling: shell, file I/O, web fetch; each behind a safe adapter.
- Orchestration: token threshold ~120k → snapshot + handoff package.
- Config via env + `config/*.json|yaml`; secrets not committed.

## Architecture Summary
- TBD — will be completed after initial inventory.

## Module Index
| Path | One-line Summary |
|---|---|
| (populate during scans) | |

## Key Decisions
- Logged in `docs/WORKLOG.md` with date and rationale.

