Release notes for `fix/quick-start` changes

What's included

- Repository cleanup: removed tracked diagnostic patches/logs and added `.gitignore` to prevent re-tracking.
- Developer experience: added `README.md`, `Makefile`, and `Dockerfile` for easy local development and containerization.
- CI: consolidated GitHub Actions workflow for consistent test runs across OSes.

Upgrade notes

- No public API changes; mostly repo maintenance and dev experience improvements.

Testing

- All tests pass locally and the workflow will run on push/PR.

Changelog

- 2025-08-17: initial cleanup and dev tooling additions.
