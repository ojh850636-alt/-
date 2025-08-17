Tools for managing .env and testing in this repo

Files:
- write_env_noninteractive.ps1: non-interactive writer for CI. Reads from env vars or accepts CLI args. Does not echo secrets.
- test_write_env.ps1: smoke test that validates write_env_noninteractive creates a .env with expected keys.
- store_keys.ps1: interactive helper to write .env (backups, SecretManagement option, ACL tightening).

Quick local test:
1. Open PowerShell and cd into tools folder.
2. Run the smoke test:
   .\test_write_env.ps1

Non-interactive usage (CI):
- From environment variables in CI, run the writer:
  .\write_env_noninteractive.ps1
- Or provide explicit keys:
  .\write_env_noninteractive.ps1 -OpenAIKey '<key>' -GroqKey '<key>' -Port 8002 -LocalIP '127.0.0.1'

Interactive usage (developer):
  From repo root run:
    .\store_keys.ps1
  Or for automation in scripts call:
    .\store_keys.ps1 -NonInteractive

Notes:
- The scripts will attempt to add .env to .gitignore and create backups when overwriting.
- Do NOT commit .env to source control. Rotate keys if accidentally exposed.
