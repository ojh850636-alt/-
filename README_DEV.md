Development checks

Run the automated checks from the project root (PowerShell):

```powershell
cd 'C:\Users\Hi\Desktop\루시아 에이전트\2'
python tools/run_all_checks.py
```

This will run:
- import module sanity checks
- TestClient smoke tests for key endpoints
- direct TestClient-based tests
- (optionally) pytest for full test suite

If any step fails, inspect the printed output; paste failures here and I'll fix them.
