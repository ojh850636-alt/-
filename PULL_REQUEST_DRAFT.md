Title: Fix/quick-start — cleanup, tests, CI, Docker

Summary

This PR prepares the repository for a clearer quick-start experience and CI testing.

- Removes historical diagnostic patch/log artifacts from tracking and ignores the `patches/` directory.
- Adds a `README.md` with quick-start, run, test, and Docker instructions.
- Adds a `Dockerfile` for local containerized runs.
- Simplifies the GitHub Actions CI workflow to a single matrix (Linux/Windows, Python 3.11).
- Adds a `Makefile` to simplify local developer tasks (run, test, docker-build, docker-run).

Why

These changes tidy the repo, make it easier to run locally and in CI, and reduce noisy test output by handling known deprecation warnings in CI.

Testing

- All tests pass locally (pytest) when run in the repo root.
- CI workflow runs tests and treats DeprecationWarning as errors so pydantic deprecations fail fast.

Notes for reviewer

- The `patches/` directory is intentionally ignored and may still exist locally for diagnostic purposes; it is not tracked anymore.
- If you want any of the removed diagnostic files preserved in history, they are still available in prior commits.

Follow-ups

- Consider merging `fix/quick-start` into `main` and running the CI to validate matrix builds on GitHub Actions.
- Optionally enable Docker image builds in CI if desired.
